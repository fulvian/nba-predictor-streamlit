import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add src to path
sys.path.append("/Users/fulvioventura/nba-predictor-streamlit/src")

from nba_predictor.core.unified_hybrid_pipeline import (
    UnifiedHybridPipeline,
    UnifiedPredictionResult,
)


class TestBayesianFusion(unittest.TestCase):
    def setUp(self):
        # Mock dependencies to avoid loading real data/models
        self.mock_repo = MagicMock()
        self.mock_client = MagicMock()

        # Patch the initialization of UnifiedHybridPipeline to skip component loading
        with (
            patch("nba_predictor.core.unified_hybrid_pipeline.UnifiedDataStore"),
            patch("nba_predictor.core.unified_hybrid_pipeline.RobustScaler"),
            patch("nba_predictor.core.unified_hybrid_pipeline.EVCalculator"),
            patch("nba_predictor.core.unified_hybrid_pipeline.BayesianUpdater"),
            patch("nba_predictor.core.unified_hybrid_pipeline.CompositeNewsAggregator"),
            patch(
                "nba_predictor.core.unified_hybrid_pipeline.NanoGPTClient"
            ) as MockClient,
        ):
            # Create instance without invoking real __init__ potentially expensive parts
            # But simpler: just instantiate and replace attributes.
            # The __init__ does file system checks, we might need to mock Path
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir"),
            ):
                self.pipeline = UnifiedHybridPipeline(
                    data_path="dummy_data",
                    model_path="dummy_models",
                    use_stacked_ensemble=False,
                    enable_explainability=False,
                )

        # Inject Mock Client
        self.pipeline.consensus_client = self.mock_client
        self.pipeline.news_aggregator = MagicMock()
        self.pipeline.news_aggregator.get_latest_news.return_value = []

        # Mock predict_unified to return a stable baseline
        self.baseline_prediction = UnifiedPredictionResult(
            predicted_total=220.0,
            confidence_interval=(215.0, 225.0),
            recommendation="NO BET",
            confidence=0.75,
            over_probability=0.6,
            under_probability=0.4,
            model_weights={"test": 1.0},
            team_analysis={},
            prediction_metadata={},
            # Required non-default fields
            injury_impact={},
            roster_changes={},
            player_momentum={},
            head_to_head_analysis={},
            shap_explanation={},
            feature_importance={},
            model_performance={},
            four_factors_analysis={},
            # Hybrid fields
            raw_quant_prediction=220.0,
            consensus_adjustment=0.0,
            unified_prediction=220.0,
        )
        self.pipeline.predict_unified = MagicMock(return_value=self.baseline_prediction)

    def test_bayesian_fusion_standard(self):
        """Test a standard "Low Risk" adjustment that should fully apply (weighted)."""
        # Mock LLM Response: +4.0 Adjustment, High Confidence, Low Risk
        llm_response = {
            "consensus": {
                "point_adjustment": 4.0,
                "confidence": 85,
                "risk_level": "LOW",
                "reasoning": "Bullish news.",
            }
        }
        self.mock_client.query_consensus_sync.return_value = llm_response

        # Formula: Quant + (Weight * Adj). Risk Low -> Weight 0.3.
        # 220 + (0.3 * 4.0) = 221.2
        result = self.pipeline.predict_unified_with_consensus("TeamA", "TeamB", 220.0)

        # Check raw quant remains untouched
        self.assertEqual(result.raw_quant_prediction, 220.0)

        # Check adjustment.
        self.assertAlmostEqual(result.unified_prediction, 221.2, delta=0.5)

        # Check metadata
        analysis = result.consensus_analysis
        self.assertEqual(analysis["proposed_adjustment"], 4.0)
        self.assertEqual(analysis["risk_level"], "LOW")
        self.assertFalse(analysis["circuit_breakers_triggered"])

    def test_circuit_breaker_hard_cap(self):
        """Test that excessive adjustments are capped at +/- 5.0."""
        # Mock LLM Response: +15.0 Adjustment (Insane)
        llm_response = {
            "consensus": {
                "point_adjustment": 15.0,
                "confidence": 90,
                "risk_level": "LOW",
                "reasoning": "Everybody is scoring!",
            }
        }
        self.mock_client.query_consensus_sync.return_value = llm_response

        result = self.pipeline.predict_unified_with_consensus("TeamA", "TeamB", 220.0)

        # Should be capped at 5.0 before weighting
        # Effective adjustment = 5.0 * 0.3 = 1.5
        # Unified = 220 + 1.5 = 221.5
        analysis = result.consensus_analysis
        self.assertEqual(analysis["proposed_adjustment"], 15.0)
        self.assertEqual(analysis["applied_adjustment_pre_weight"], 5.0)  # Cap applied
        self.assertTrue(analysis["circuit_breakers_triggered"])

    def test_circuit_breaker_confidence(self):
        """Test that low confidence LLM predictions are ignored."""
        # Mock LLM Response: +4.0 Adjustment but 40% confidence
        llm_response = {
            "consensus": {
                "point_adjustment": 4.0,
                "confidence": 40,
                "risk_level": "LOW",
                "reasoning": "Not sure.",
            }
        }
        self.mock_client.query_consensus_sync.return_value = llm_response

        result = self.pipeline.predict_unified_with_consensus("TeamA", "TeamB", 220.0)

        # Should be 0 adjustment
        self.assertEqual(result.unified_prediction, 220.0)
        self.assertEqual(result.consensus_adjustment, 0.0)
        analysis = result.consensus_analysis
        self.assertTrue(analysis["circuit_breakers_triggered"])
        self.assertIn("LOW CONFIDENCE", analysis["circuit_breaker_details"])

    def test_circuit_breaker_high_risk(self):
        """Test that HIGH risk LLM predictions are ignored or heavily penalized."""
        # Mock LLM Response: +4.0 Adjustment but HIGH Risk
        llm_response = {
            "consensus": {
                "point_adjustment": 4.0,
                "confidence": 80,
                "risk_level": "HIGH",
                "reasoning": "Chaos.",
            }
        }
        self.mock_client.query_consensus_sync.return_value = llm_response

        result = self.pipeline.predict_unified_with_consensus("TeamA", "TeamB", 220.0)

        # High User Objective stated High Risk -> Ignore (Weight 0.0)
        self.assertEqual(result.unified_prediction, 220.0)
        self.assertEqual(result.consensus_adjustment, 0.0)

        analysis = result.consensus_analysis
        self.assertTrue(analysis["circuit_breakers_triggered"])
        self.assertIn("HIGH RISK", analysis["circuit_breaker_details"])

    def test_json_parsing_fallback(self):
        """Test robustness against malformed LLM output."""
        # Mock LLM Response: Garbage/Error
        llm_response = {"error": "Some error", "fallback": True}
        self.mock_client.query_consensus_sync.return_value = llm_response

        result = self.pipeline.predict_unified_with_consensus("TeamA", "TeamB", 220.0)

        # Should fallback to baseline
        self.assertEqual(result.unified_prediction, 220.0)
        self.assertTrue(result.consensus_analysis.get("fallback", False))


if __name__ == "__main__":
    unittest.main()
