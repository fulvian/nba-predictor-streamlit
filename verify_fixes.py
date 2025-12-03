import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import classes to test
from nba_predictor.features.unified_feature_pipeline import FourFactorsEngineer
from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline


class TestFixes(unittest.TestCase):
    def test_feature_clipping_widened(self):
        """Verify that feature clipping ranges have been widened."""
        engineer = FourFactorsEngineer()

        # Test case with high eFG% (e.g., 0.70) which was previously clipped at 0.58
        game_data = {
            "field_goals_made": 45,
            "field_goals_attempted": 80,
            "three_pointers_made": 15,
            "turnovers": 10,
            "free_throws_attempted": 20,
            "offensive_rebounds": 10,
            "free_throws_made": 15,
        }

        # Calculate expected eFG% manually: (45 + 0.5 * 15) / 80 = 52.5 / 80 = 0.65625
        # Previous clip was 0.580. New clip is 0.750.

        features = engineer.extract_features(game_data)
        efg_pct = features["efg_pct"]

        logger.info(f"Calculated eFG%: {efg_pct}")

        self.assertGreater(
            efg_pct, 0.580, "eFG% should not be clipped at 0.580 anymore"
        )
        self.assertLessEqual(efg_pct, 0.750, "eFG% should be clipped at 0.750")

        # Test extreme case
        game_data_extreme = {
            "field_goals_made": 70,
            "field_goals_attempted": 80,  # eFG% > 0.8
            "three_pointers_made": 0,
        }
        features_extreme = engineer.extract_features(game_data_extreme)
        self.assertEqual(
            features_extreme["efg_pct"], 0.750, "eFG% should be clipped at 0.750 max"
        )

    def test_dynamic_envelope_calculation(self):
        """Verify dynamic envelope calculation logic."""
        # Mock data sources
        mock_df = pd.DataFrame(
            {
                "home_team": ["Team A"] * 10 + ["Team B"] * 10,
                "away_team": ["Team B"] * 10 + ["Team A"] * 10,
                "HOME_SCORE": [120] * 20,
                "AWAY_SCORE": [110] * 20,
                "TOTAL_SCORE": [230] * 20,
                "HOME_ORtg": [115] * 20,
                "AWAY_ORtg": [105] * 20,
            }
        )

        data_sources = {"nba_games": mock_df}

        # Instantiate pipeline (mocking dependencies)
        with patch(
            "nba_predictor.core.unified_hybrid_pipeline.UnifiedHybridPipeline.__init__",
            return_value=None,
        ):
            pipeline = UnifiedHybridPipeline()
            pipeline.NBA_REALISTIC_RANGES = {
                "team_score": (80, 160),
                "total_score": (180, 320),
                "efg_pct": (0.350, 0.750),
                "tov_pct": (0.050, 0.250),
                "orb_pct": (0.100, 0.400),
                "ftr": (0.100, 0.450),
            }

            # Mock _get_team_adjustments since it's used in envelope calc
            pipeline._get_team_adjustments = MagicMock(
                return_value={"team1_score": 2.0, "team2_score": -2.0}
            )

            # Test envelope calculation
            envelope = pipeline._calculate_dynamic_envelope(
                "Team A", "Team B", data_sources
            )

            self.assertIsNotNone(envelope)
            min_score, max_score = envelope
            logger.info(f"Envelope: [{min_score}, {max_score}]")

            # Base total is 230 + 2 - 2 = 230
            # Std dev of TOTAL_SCORE is 0 in mock data, so envelope should be tight?
            # Wait, std dev of constant is 0.
            # The code uses 1.5 * global_std.
            # If std is 0, envelope is [230, 230].

            self.assertEqual(min_score, 230.0)
            self.assertEqual(max_score, 230.0)


if __name__ == "__main__":
    unittest.main()
