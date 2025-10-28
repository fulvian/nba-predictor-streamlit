#!/usr/bin/env python3
"""
🧪 LightGBM Model Unit Tests
Test suite for LightGBM model functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from nba_predictor.models.lightgbm_model import (
    create_nba_lightgbm_model,
    validate_lightgbm_parameters,
    get_nba_optimized_params,
    create_lightgbm_for_time_series
)


class TestLightGBMModel(unittest.TestCase):
    """Test cases for LightGBM model module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample NBA data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'team1_score': np.random.randint(80, 140, 100),
            'team2_score': np.random.randint(80, 140, 100),
            'total_score': np.random.randint(160, 280, 100),
            'efg_pct': np.random.uniform(0.45, 0.65, 100),
            'tov_pct': np.random.uniform(0.10, 0.20, 100),
            'orb_pct': np.random.uniform(0.20, 0.35, 100),
            'ftr': np.random.uniform(0.15, 0.35, 100)
        })
        self.X = self.sample_data.drop('total_score', axis=1)
        self.y = self.sample_data['total_score']

    def test_create_nba_lightgbm_model_default(self):
        """Test creating LightGBM model with default parameters."""
        model = create_nba_lightgbm_model()

        # Check model type and parameters
        self.assertEqual(model.__class__.__name__, 'LGBMRegressor')
        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.num_leaves, 31)
        self.assertEqual(model.max_depth, 6)
        self.assertEqual(model.random_state, 42)

    def test_create_nba_lightgbm_model_custom(self):
        """Test creating LightGBM model with custom parameters."""
        model = create_nba_lightgbm_model(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=25,
            max_depth=8,
            random_state=123
        )

        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(model.learning_rate, 0.03)
        self.assertEqual(model.num_leaves, 25)
        self.assertEqual(model.max_depth, 8)
        self.assertEqual(model.random_state, 123)

    def test_create_nba_lightgbm_model_fit_predict(self):
        """Test that model can fit and predict."""
        model = create_nba_lightgbm_model(random_state=42)

        # Fit model
        model.fit(self.X, self.y)

        # Make predictions
        predictions = model.predict(self.X)

        # Check predictions
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(pred > 0 for pred in predictions))

    def test_create_nba_lightgbm_model_missing_dependency(self):
        """Test error handling when LightGBM is not installed."""
        # Patch the import at the module level
        with patch('nba_predictor.models.lightgbm_model.lgb', None):
            with self.assertRaises(ImportError) as context:
                create_nba_lightgbm_model()

            self.assertIn("LightGBM is required but not installed", str(context.exception))
            self.assertIn("pip install lightgbm", str(context.exception))

    def test_validate_lightgbm_parameters_valid(self):
        """Test validation with valid parameters."""
        # Should not raise exception
        validate_lightgbm_parameters(200, 0.05, 31, 6, 42)

    def test_validate_lightgbm_parameters_invalid_n_estimators(self):
        """Test validation with invalid n_estimators."""
        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(5, 0.05, 31, 6, 42)  # Too small

        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(0, 0.05, 31, 6, 42)  # Zero

    def test_validate_lightgbm_parameters_invalid_learning_rate(self):
        """Test validation with invalid learning_rate."""
        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.0, 31, 6, 42)  # Zero

        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, -0.1, 31, 6, 42)  # Negative

        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 1.5, 31, 6, 42)  # Too high

    def test_validate_lightgbm_parameters_invalid_num_leaves(self):
        """Test validation with invalid num_leaves."""
        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.05, 1, 6, 42)  # Too small

        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.05, 0, 6, 42)  # Zero

    def test_validate_lightgbm_parameters_invalid_max_depth(self):
        """Test validation with invalid max_depth."""
        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.05, 31, 0, 42)  # Zero

        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.05, 31, -1, 42)  # Negative

    def test_validate_lightgbm_parameters_invalid_random_state(self):
        """Test validation with invalid random_state."""
        with self.assertRaises(ValueError):
            validate_lightgbm_parameters(200, 0.05, 31, 6, -1)  # Negative

    def test_get_nba_optimized_params(self):
        """Test getting NBA-optimized parameters."""
        params = get_nba_optimized_params()

        # Check required parameters
        expected_keys = [
            'objective', 'metric', 'n_estimators', 'learning_rate',
            'num_leaves', 'max_depth', 'min_child_samples',
            'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
            'n_jobs', 'verbose'
        ]

        for key in expected_keys:
            self.assertIn(key, params)

        # Check specific values
        self.assertEqual(params['objective'], 'regression')
        self.assertEqual(params['n_estimators'], 200)
        self.assertEqual(params['learning_rate'], 0.05)
        self.assertEqual(params['num_leaves'], 31)
        self.assertEqual(params['max_depth'], 6)

    def test_create_lightgbm_for_time_series_default(self):
        """Test creating time series LightGBM model with default parameters."""
        model = create_lightgbm_for_time_series()

        # Check model type and conservative parameters
        self.assertEqual(model.__class__.__name__, 'LGBMRegressor')
        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.num_leaves, 15)  # More conservative
        self.assertEqual(model.max_depth, 4)    # More conservative
        self.assertEqual(model.reg_alpha, 0.2)  # Stronger regularization
        self.assertEqual(model.reg_lambda, 0.2) # Stronger regularization

    def test_create_lightgbm_for_time_series_custom(self):
        """Test creating time series LightGBM model with custom parameters."""
        model = create_lightgbm_for_time_series(
            n_estimators=150,
            learning_rate=0.08,
            random_state=456
        )

        self.assertEqual(model.n_estimators, 150)
        self.assertEqual(model.learning_rate, 0.08)
        self.assertEqual(model.num_leaves, 15)  # Should remain conservative
        self.assertEqual(model.max_depth, 4)    # Should remain conservative
        self.assertEqual(model.random_state, 456)

    def test_create_lightgbm_for_time_series_fit_predict(self):
        """Test that time series model can fit and predict."""
        model = create_lightgbm_for_time_series(random_state=42)

        # Fit model
        model.fit(self.X, self.y)

        # Make predictions
        predictions = model.predict(self.X)

        # Check predictions
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(pred > 0 for pred in predictions))

    def test_create_lightgbm_for_time_series_missing_dependency(self):
        """Test error handling when LightGBM is not installed for time series."""
        # Patch the import at the module level
        with patch('nba_predictor.models.lightgbm_model.lgb', None):
            with self.assertRaises(ImportError) as context:
                create_lightgbm_for_time_series()

            self.assertIn("LightGBM is required but not installed", str(context.exception))

    def test_model_parameters_comparison(self):
        """Test that regular model is more aggressive than time series model."""
        regular_model = create_nba_lightgbm_model(random_state=42)
        ts_model = create_lightgbm_for_time_series(random_state=42)

        # Regular model should be more aggressive (higher complexity)
        self.assertGreater(regular_model.num_leaves, ts_model.num_leaves)
        self.assertGreater(regular_model.max_depth, ts_model.max_depth)
        # Time series model should have stronger regularization
        self.assertGreater(ts_model.reg_alpha, regular_model.reg_alpha)
        self.assertGreater(ts_model.reg_lambda, regular_model.reg_lambda)

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        # Create sample data
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randn(50))

        model = create_nba_lightgbm_model(n_estimators=200, learning_rate=0.05)

        # Should fit without error
        model.fit(X, y)

        # Should predict without error
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_parameter_validation_warnings(self):
        """Test that warnings are logged for extreme parameters."""
        import logging
        with self.assertLogs(level=logging.WARNING) as log:
            validate_lightgbm_parameters(
                n_estimators=15000,  # Very large
                learning_rate=0.4,    # High learning rate
                num_leaves=1500,      # Very high
                max_depth=25,         # Very deep
                random_state=42
            )

        # Check that warnings were logged
        self.assertTrue(any("Very large n_estimators" in message for message in log.output))
        self.assertTrue(any("High learning rate" in message for message in log.output))
        self.assertTrue(any("Very high num_leaves" in message for message in log.output))
        self.assertTrue(any("Very deep trees" in message for message in log.output))


if __name__ == '__main__':
    unittest.main()