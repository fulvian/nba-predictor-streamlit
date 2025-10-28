#!/usr/bin/env python3
"""
🧪 Research Features Unit Tests
Test suite for research-based feature engineering functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from nba_predictor.features.research_features import (
    enhance_nba_features,
    validate_input_data,
    calculate_four_factors_features,
    calculate_team_differentials,
    calculate_pace_features,
    calculate_efficiency_features,
    calculate_situational_features,
    integrate_momentum_features,
    get_feature_importance_ranking,
    validate_feature_engineering_pipeline
)


class TestResearchFeatures(unittest.TestCase):
    """Test cases for research features module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample NBA data with comprehensive statistics
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            # Basic scoring
            'team1_score': np.random.randint(80, 140, 100),
            'team2_score': np.random.randint(80, 140, 100),
            'total_score': np.random.randint(160, 280, 100),

            # Field goals
            'team1_field_goals_made': np.random.randint(30, 50, 100),
            'team1_field_goals_attempted': np.random.randint(60, 100, 100),
            'team2_field_goals_made': np.random.randint(30, 50, 100),
            'team2_field_goals_attempted': np.random.randint(60, 100, 100),

            # Three pointers
            'team1_three_pointers_made': np.random.randint(5, 20, 100),
            'team1_three_pointers_attempted': np.random.randint(15, 40, 100),
            'team2_three_pointers_made': np.random.randint(5, 20, 100),
            'team2_three_pointers_attempted': np.random.randint(15, 40, 100),

            # Two pointers (derived)
            'team1_two_pointers_made': lambda x: x['team1_field_goals_made'] - x['team1_three_pointers_made'],
            'team2_two_pointers_made': lambda x: x['team2_field_goals_made'] - x['team2_three_pointers_made'],

            # Free throws
            'team1_free_throws_made': np.random.randint(10, 25, 100),
            'team1_free_throws_attempted': np.random.randint(15, 35, 100),
            'team2_free_throws_made': np.random.randint(10, 25, 100),
            'team2_free_throws_attempted': np.random.randint(15, 35, 100),

            # Team stats
            'team1_points': lambda x: x['team1_score'],
            'team2_points': lambda x: x['team2_score'],
            'team1_rebounds': np.random.randint(30, 60, 100),
            'team2_rebounds': np.random.randint(30, 60, 100),
            'team1_assists': np.random.randint(15, 35, 100),
            'team2_assists': np.random.randint(15, 35, 100),
            'team1_steals': np.random.randint(5, 15, 100),
            'team2_steals': np.random.randint(5, 15, 100),
            'team1_blocks': np.random.randint(2, 10, 100),
            'team2_blocks': np.random.randint(2, 10, 100),
            'team1_turnovers': np.random.randint(10, 25, 100),
            'team2_turnovers': np.random.randint(10, 25, 100),
            'team1_fouls': np.random.randint(15, 30, 100),
            'team2_fouls': np.random.randint(15, 30, 100),

            # Rebounds
            'team1_offensive_rebounds': np.random.randint(5, 15, 100),
            'team2_offensive_rebounds': np.random.randint(5, 15, 100),
            'team1_defensive_rebounds': lambda x: x['team1_rebounds'] - x['team1_offensive_rebounds'],
            'team2_defensive_rebounds': lambda x: x['team2_rebounds'] - x['team2_offensive_rebounds'],

            # Four Factors (base metrics)
            'efg_pct': np.random.uniform(0.45, 0.65, 100),
            'tov_pct': np.random.uniform(0.10, 0.20, 100),
            'orb_pct': np.random.uniform(0.20, 0.35, 100),
            'ftr': np.random.uniform(0.15, 0.35, 100),

            # Possessions (derived)
            'team1_possessions': lambda x: (
                x['team1_field_goals_attempted'] +
                x['team1_free_throws_attempted'] * 0.44 +
                x['team1_offensive_rebounds'] -
                x['team1_turnovers']
            ),
            'team2_possessions': lambda x: (
                x['team2_field_goals_attempted'] +
                x['team2_free_throws_attempted'] * 0.44 +
                x['team2_offensive_rebounds'] -
                x['team2_turnovers']
            )
        })

        # Apply lambda functions
        self.sample_data['team1_two_pointers_made'] = (
            self.sample_data['team1_field_goals_made'] - self.sample_data['team1_three_pointers_made']
        )
        self.sample_data['team2_two_pointers_made'] = (
            self.sample_data['team2_field_goals_made'] - self.sample_data['team2_three_pointers_made']
        )
        self.sample_data['team1_points'] = self.sample_data['team1_score']
        self.sample_data['team2_points'] = self.sample_data['team2_score']
        self.sample_data['team1_defensive_rebounds'] = (
            self.sample_data['team1_rebounds'] - self.sample_data['team1_offensive_rebounds']
        )
        self.sample_data['team2_defensive_rebounds'] = (
            self.sample_data['team2_rebounds'] - self.sample_data['team2_offensive_rebounds']
        )
        self.sample_data['team1_possessions'] = (
            self.sample_data['team1_field_goals_attempted'] +
            self.sample_data['team1_free_throws_attempted'] * 0.44 +
            self.sample_data['team1_offensive_rebounds'] -
            self.sample_data['team1_turnovers']
        )
        self.sample_data['team2_possessions'] = (
            self.sample_data['team2_field_goals_attempted'] +
            self.sample_data['team2_free_throws_attempted'] * 0.44 +
            self.sample_data['team2_offensive_rebounds'] -
            self.sample_data['team2_turnovers']
        )

    def test_enhance_nba_features_basic(self):
        """Test basic feature enhancement with Four Factors."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']

        enhanced = enhance_nba_features(self.sample_data, four_factors_cols)

        # Check that we have more columns
        self.assertGreater(len(enhanced.columns), len(self.sample_data.columns))

        # Check that original columns are preserved
        for col in self.sample_data.columns:
            self.assertIn(col, enhanced.columns)

        # Check for new Four Factors features
        expected_new_features = [
            'four_factors_product',
            'four_factors_weighted',
            'shooting_efficiency',
            'possession_efficiency',
            'rebounding_contribution',
            'free_throw_contribution'
        ]

        for feature in expected_new_features:
            self.assertIn(feature, enhanced.columns)

    def test_enhance_nba_features_with_momentum(self):
        """Test feature enhancement with momentum data."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']

        # Create mock momentum data
        momentum_data = pd.DataFrame({
            'team_momentum': np.random.uniform(-1, 1, 100),
            'player_form': np.random.uniform(0, 1, 100)
        })

        enhanced = enhance_nba_features(self.sample_data, four_factors_cols, momentum_data)

        # Check that momentum features are added
        self.assertIn('team1_momentum', enhanced.columns)
        self.assertIn('team2_momentum', enhanced.columns)
        self.assertIn('avg_player_form', enhanced.columns)

    def test_validate_input_data_success(self):
        """Test successful input data validation."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']

        # Should not raise exception
        validate_input_data(self.sample_data, four_factors_cols)

    def test_validate_input_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']

        with self.assertRaises(ValueError) as context:
            validate_input_data(empty_df, four_factors_cols)

        self.assertIn("empty", str(context.exception))

    def test_validate_input_data_insufficient_columns(self):
        """Test validation with insufficient Four Factors columns."""
        insufficient_cols = ['efg_pct', 'tov_pct']  # Only 2 columns

        with self.assertRaises(ValueError) as context:
            validate_input_data(self.sample_data, insufficient_cols)

        self.assertIn("At least 4", str(context.exception))

    def test_validate_input_data_missing_columns(self):
        """Test validation with missing required columns."""
        missing_cols = ['nonexistent_col1', 'nonexistent_col2', 'nonexistent_col3', 'nonexistent_col4']

        with self.assertRaises(ValueError) as context:
            validate_input_data(self.sample_data, missing_cols)

        self.assertIn("Missing required columns", str(context.exception))

    def test_validate_input_data_non_numeric_columns(self):
        """Test validation with non-numeric columns."""
        # Add non-numeric column
        df_with_text = self.sample_data.copy()
        df_with_text['text_column'] = ['text'] * len(df_with_text)

        non_numeric_cols = ['text_column', 'efg_pct', 'tov_pct', 'orb_pct']

        with self.assertRaises(ValueError) as context:
            validate_input_data(df_with_text, non_numeric_cols)

        self.assertIn("must be numeric", str(context.exception))

    def test_calculate_four_factors_features(self):
        """Test Four Factors features calculation."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']

        enhanced = calculate_four_factors_features(self.sample_data, four_factors_cols)

        # Check Four Factors product calculation
        expected_product = (
            self.sample_data['efg_pct'] *
            (1 - self.sample_data['tov_pct']) *
            self.sample_data['orb_pct'] *
            self.sample_data['ftr']
        )

        np.testing.assert_array_almost_equal(
            enhanced['four_factors_product'], expected_product
        )

        # Check that all features are positive (product of positive numbers)
        self.assertTrue(all(enhanced['four_factors_product'] > 0))

    def test_calculate_team_differentials(self):
        """Test team differential calculations."""
        enhanced = calculate_team_differentials(self.sample_data)

        # Check score differential
        expected_score_diff = self.sample_data['team1_score'] - self.sample_data['team2_score']
        np.testing.assert_array_equal(enhanced['score_differential'], expected_score_diff)

        # Check scoring ratio
        expected_ratio = self.sample_data['team1_score'] / (self.sample_data['team2_score'] + 1e-6)
        np.testing.assert_array_almost_equal(enhanced['scoring_ratio'], expected_ratio)

    def test_calculate_pace_features(self):
        """Test pace-related features."""
        enhanced = calculate_pace_features(self.sample_data)

        # Check total possessions calculation
        expected_total_possessions = self.sample_data['team1_possessions'] + self.sample_data['team2_possessions']
        np.testing.assert_array_equal(enhanced['total_possessions'], expected_total_possessions)

        # Check pace per team
        expected_pace = expected_total_possessions / 2
        np.testing.assert_array_equal(enhanced['pace_possessions'], expected_pace)

    def test_calculate_efficiency_features(self):
        """Test efficiency features calculation."""
        enhanced = calculate_efficiency_features(self.sample_data)

        # Check True Shooting Percentage calculation
        expected_ts1 = self.sample_data['team1_points'] / (
            2 * (self.sample_data['team1_field_goals_attempted'] +
                 0.44 * self.sample_data['team1_free_throws_attempted'])
        )

        np.testing.assert_array_almost_equal(enhanced['team1_ts_percentage'], expected_ts1)

        # Check that TS% is reasonable (between 0 and 1)
        self.assertTrue(all(enhanced['team1_ts_percentage'] >= 0))
        self.assertTrue(all(enhanced['team1_ts_percentage'] <= 1))

    def test_calculate_situational_features(self):
        """Test situational features calculation."""
        enhanced = calculate_situational_features(self.sample_data)

        # Check three point ratio calculation
        total_fg1 = self.sample_data['team1_two_pointers_made'] + self.sample_data['team1_three_pointers_made']
        expected_ratio1 = self.sample_data['team1_three_pointers_made'] / (total_fg1 + 1e-6)

        np.testing.assert_array_almost_equal(enhanced['team1_three_point_ratio'], expected_ratio1)

        # Check assist ratio
        expected_assist_ratio = self.sample_data['team1_assists'] / (self.sample_data['team1_field_goals_made'] + 1e-6)
        np.testing.assert_array_almost_equal(enhanced['team1_assist_ratio'], expected_assist_ratio)

    def test_integrate_momentum_features(self):
        """Test momentum features integration."""
        momentum_data = pd.DataFrame({
            'team_momentum': np.random.uniform(-1, 1, 100),
            'player_form': np.random.uniform(0, 1, 100)
        })

        enhanced = integrate_momentum_features(self.sample_data, momentum_data)

        # Check that momentum features are added
        self.assertIn('team1_momentum', enhanced.columns)
        self.assertIn('avg_player_form', enhanced.columns)

    def test_get_feature_importance_ranking(self):
        """Test feature importance ranking."""
        # Create enhanced data first
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']
        enhanced = enhance_nba_features(self.sample_data, four_factors_cols)

        # Calculate importance ranking
        importance = get_feature_importance_ranking(enhanced, 'total_score')

        # Check that we get a dictionary
        self.assertIsInstance(importance, dict)

        # Check that we have importance scores
        self.assertGreater(len(importance), 0)

        # Check that scores are floats
        for feature, score in importance.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)

    def test_get_feature_importance_ranking_missing_target(self):
        """Test feature importance with missing target column."""
        enhanced = enhance_nba_features(self.sample_data, ['efg_pct', 'tov_pct', 'orb_pct', 'ftr'])

        with self.assertRaises(ValueError) as context:
            get_feature_importance_ranking(enhanced, 'nonexistent_column')

        self.assertIn("not found", str(context.exception))

    def test_validate_feature_engineering_pipeline_success(self):
        """Test successful feature engineering pipeline validation."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']
        enhanced = enhance_nba_features(self.sample_data, four_factors_cols)

        # Should pass validation
        result = validate_feature_engineering_pipeline(
            self.sample_data, enhanced, expected_feature_count=15
        )

        self.assertTrue(result)

    def test_validate_feature_engineering_pipeline_no_new_features(self):
        """Test validation when no new features are created."""
        # Don't enhance the data
        with self.assertRaises(ValueError) as context:
            validate_feature_engineering_pipeline(
                self.sample_data, self.sample_data, expected_feature_count=5
            )

        self.assertIn("No new features", str(context.exception))

    def test_validate_feature_engineering_pipeline_all_nan(self):
        """Test validation when new features contain only NaN."""
        # Create enhanced DataFrame with NaN feature
        enhanced = self.sample_data.copy()
        enhanced['test_feature'] = np.nan

        with self.assertRaises(ValueError) as context:
            validate_feature_engineering_pipeline(
                self.sample_data, enhanced, expected_feature_count=1
            )

        self.assertIn("contains only NaN", str(context.exception))

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        # Create simple test data
        test_df = pd.DataFrame({
            'efg%': np.random.uniform(0.45, 0.65, 50),
            'TOV%': np.random.uniform(0.10, 0.20, 50),
            'ORB%': np.random.uniform(0.20, 0.35, 50),
            'FTR%': np.random.uniform(0.15, 0.35, 50),
            'other_col': np.random.randn(50)
        })

        four_factors_cols = ['efg%', 'TOV%', 'ORB%', 'FTR%']

        enhanced = enhance_nba_features(test_df, four_factors_cols)

        # Check that enhancement worked
        self.assertGreater(len(enhanced.columns), len(test_df.columns))

        # Check that example columns are present
        self.assertIn('four_factors_product', enhanced.columns)
        self.assertIn('shooting_efficiency', enhanced.columns)

    def test_feature_combinations(self):
        """Test that feature combinations are calculated correctly."""
        four_factors_cols = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']
        enhanced = enhance_nba_features(self.sample_data, four_factors_cols)

        # Test that Four Factors weighted sum is reasonable
        # Weighted sum should be between 0 and 1 for normalized inputs
        self.assertTrue(all(enhanced['four_factors_weighted'] >= 0))
        self.assertTrue(all(enhanced['four_factors_weighted'] <= 1))

    def test_defensive_features(self):
        """Test that defensive features are calculated correctly."""
        enhanced = calculate_situational_features(self.sample_data)

        # Check defensive rebound rates
        expected_def_rate1 = self.sample_data['team1_defensive_rebounds'] / (self.sample_data['team1_possessions'] + 1e-6)
        np.testing.assert_array_almost_equal(enhanced['team1_defensive_rebound_rate'], expected_def_rate1)

        # Should be between 0 and 1
        self.assertTrue(all(enhanced['team1_defensive_rebound_rate'] >= 0))
        self.assertTrue(all(enhanced['team1_defensive_rebound_rate'] <= 1))


if __name__ == '__main__':
    unittest.main()