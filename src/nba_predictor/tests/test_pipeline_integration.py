"""
Integration test for UnifiedHybridPipeline with EV and Bayesian components
"""

import pytest
from unittest.mock import MagicMock, patch
from src.nba_predictor.core.unified_hybrid_pipeline import (
    UnifiedHybridPipeline,
    UnifiedPredictionResult,
)


class TestUnifiedHybridPipelineIntegration:
    @pytest.fixture
    def pipeline(self):
        with (
            patch("src.nba_predictor.core.unified_hybrid_pipeline.UnifiedDataStore"),
            patch(
                "src.nba_predictor.core.unified_hybrid_pipeline.EVCalculator"
            ) as MockEV,
            patch(
                "src.nba_predictor.core.unified_hybrid_pipeline.BayesianUpdater"
            ) as MockBayes,
            patch(
                "src.nba_predictor.core.unified_hybrid_pipeline.NewsAggregator"
            ) as MockNews,
            patch.object(UnifiedHybridPipeline, "_load_team_mapping"),
        ):
            pipeline = UnifiedHybridPipeline(
                data_path="data",
                model_path="models",
                use_stacked_ensemble=False,
                enable_explainability=False,
            )

            # Mock components
            pipeline.ev_calculator = MockEV.return_value
            pipeline.bayesian_updater = MockBayes.return_value
            pipeline.news_aggregator = MockNews.return_value

            # Mock trained model
            pipeline.trained_model = MagicMock()
            pipeline.trained_model.predict.return_value = [225.0]
            pipeline.is_trained = True
            pipeline.metrics = {"mse": 100.0}
            pipeline.feature_scaler = MagicMock()

            # Manually set team mapping for test
            pipeline.team_name_to_id = {"Lakers": 1610612747, "Warriors": 1610612744}

            return pipeline

    def test_predict_unified_integration(self, pipeline):
        # Mock internal methods to avoid complex data loading
        pipeline.load_all_integrated_data = MagicMock(return_value={})
        pipeline._create_unified_prediction_features = MagicMock(
            return_value={"feat1": 1.0}
        )
        pipeline._analyze_unified_injury_impact = MagicMock(return_value={})
        pipeline._analyze_unified_roster_changes = MagicMock(return_value={})
        pipeline._analyze_unified_player_momentum = MagicMock(return_value={})
        pipeline._analyze_unified_head_to_head = MagicMock(return_value={})
        pipeline._get_unified_feature_importance = MagicMock(return_value={})
        pipeline._analyze_four_factors_impact = MagicMock(return_value={})
        pipeline._analyze_unified_teams = MagicMock(return_value={})
        pipeline._get_model_weights = MagicMock(return_value={})
        pipeline._calculate_dynamic_envelope = MagicMock(return_value=None)

        # Mock EV analysis return
        pipeline.ev_calculator.calculate_ev.return_value = MagicMock(
            ev_percentage=5.0, edge=0.05, kelly_stake_percentage=0.01, is_value_bet=True
        )

        # Mock Bayesian update return
        pipeline.news_aggregator.get_latest_news.return_value = [{"impact_score": 1.0}]
        pipeline.bayesian_updater.update_prediction.return_value = MagicMock(
            updated_score_dist=(222.0, 12.0), confidence_interval=(200, 244)
        )

        # Run prediction
        result = pipeline.predict_unified(team1="Lakers", team2="Warriors", line=220.0)

        # Verify EV analysis was called
        assert pipeline.ev_calculator.calculate_ev.called
        assert result.ev_analysis is not None
        assert result.ev_analysis["is_value"] is True

        # Verify Bayesian update was called
        assert pipeline.news_aggregator.get_latest_news.called
        assert pipeline.bayesian_updater.update_prediction.called
        assert result.bayesian_update is not None
        assert result.bayesian_update["updated_total"] == 222.0
