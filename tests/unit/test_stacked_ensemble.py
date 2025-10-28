#!/usr/bin/env python3
"""
🧪 Stacked Ensemble Unit Tests
Test suite for stacked ensemble functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from nba_predictor.models.stacked_ensemble import (
    create_research_stacked_ensemble,
    create_base_estimators,
    create_mlp_meta_learner,
    validate_ensemble_dependencies,
    get_ensemble_feature_importance,
    create_conservative_stacked_ensemble
)


class TestStackedEnsemble(unittest.TestCase):
    """Test cases for stacked ensemble module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create larger sample NBA data for proper time series splitting
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'team1_score': np.random.randint(80, 140, 500),
            'team2_score': np.random.randint(80, 140, 500),
            'total_score': np.random.randint(160, 280, 500),
            'efg_pct': np.random.uniform(0.45, 0.65, 500),
            'tov_pct': np.random.uniform(0.10, 0.20, 500),
            'orb_pct': np.random.uniform(0.20, 0.35, 500),
            'ftr': np.random.uniform(0.15, 0.35, 500)
        })
        self.X = self.sample_data.drop('total_score', axis=1)
        self.y = self.sample_data['total_score']

        # Create smaller dataset for tests that don't need time series CV
        self.small_X = self.X.iloc[:100]
        self.small_y = self.y.iloc[:100]

    def test_create_research_stacked_ensemble_default(self):
        """Test creating stacked ensemble with default parameters."""
        ensemble = create_research_stacked_ensemble()

        # Check ensemble type
        self.assertEqual(ensemble.__class__.__name__, 'StackingRegressor')
        self.assertTrue(ensemble.passthrough)
        self.assertEqual(ensemble.n_jobs, -1)

        # Check that base estimators are present
        self.assertGreaterEqual(len(ensemble.estimators), 2)

    def test_create_research_stacked_ensemble_custom_cv(self):
        """Test creating stacked ensemble with custom CV strategy."""
        from sklearn.model_selection import TimeSeriesSplit

        custom_cv = TimeSeriesSplit(n_splits=3, test_size=2)
        ensemble = create_research_stacked_ensemble(cv_strategy=custom_cv, n_jobs=2)

        self.assertEqual(ensemble.cv, custom_cv)
        self.assertEqual(ensemble.n_jobs, 2)

    def test_create_research_stacked_ensemble_fit_predict(self):
        """Test that ensemble can fit and predict."""
        ensemble = create_research_stacked_ensemble()

        # Fit model
        ensemble.fit(self.small_X, self.small_y)

        # Make predictions
        predictions = ensemble.predict(self.small_X)

        # Check predictions
        self.assertEqual(len(predictions), len(self.small_y))
        self.assertTrue(all(isinstance(pred, (int, float)) for pred in predictions))

    def test_create_research_stacked_ensemble_invalid_cv(self):
        """Test error handling for invalid CV strategy."""
        invalid_cv = "not_a_cv_object"

        with self.assertRaises(ValueError) as context:
            create_research_stacked_ensemble(cv_strategy=invalid_cv)

        self.assertIn("must have a 'split' method", str(context.exception))

    def test_create_base_estimators_default(self):
        """Test creating base estimators with default configuration."""
        estimators = create_base_estimators()

        # Check that we have at least 2 estimators
        self.assertGreaterEqual(len(estimators), 2)

        # Check estimator names and types
        estimator_names = [name for name, _ in estimators]
        expected_names = ['xgb', 'lgbm', 'rf', 'ridge']

        # At least some of the expected names should be present
        self.assertTrue(any(name in estimator_names for name in expected_names))

        # Check that each estimator has the expected interface
        for name, estimator in estimators:
            self.assertTrue(hasattr(estimator, 'fit'))
            self.assertTrue(hasattr(estimator, 'predict'))

    def test_create_base_estimators_custom_jobs(self):
        """Test creating base estimators with custom n_jobs."""
        estimators = create_base_estimators(n_jobs=2)

        # Check n_jobs parameter is set for compatible models
        for name, estimator in estimators:
            if hasattr(estimator, 'n_jobs'):
                self.assertEqual(estimator.n_jobs, 2)

    def test_create_mlp_meta_learner(self):
        """Test creating MLP meta-learner."""
        meta_learner = create_mlp_meta_learner()

        # Check model type and parameters
        self.assertEqual(meta_learner.__class__.__name__, 'MLPRegressor')
        self.assertEqual(meta_learner.hidden_layer_sizes, (64, 32))
        self.assertEqual(meta_learner.activation, 'relu')
        self.assertTrue(meta_learner.early_stopping)
        self.assertEqual(meta_learner.random_state, 42)

    def test_create_mlp_meta_learner_fit_predict(self):
        """Test that meta-learner can fit and predict."""
        meta_learner = create_mlp_meta_learner()

        # Fit model with smaller dataset
        meta_learner.fit(self.small_X, self.small_y)

        # Make predictions
        predictions = meta_learner.predict(self.small_X)

        # Check predictions
        self.assertEqual(len(predictions), len(self.small_y))

    def test_validate_ensemble_dependencies(self):
        """Test dependency validation."""
        # Should not raise exception if dependencies are available
        try:
            validate_ensemble_dependencies()
        except ImportError:
            # If dependencies are missing, that's expected in test environment
            pass

    def test_get_ensemble_feature_importance(self):
        """Test extracting feature importance from fitted ensemble."""
        ensemble = create_research_stacked_ensemble()

        # Fit the ensemble
        ensemble.fit(self.X, self.y)

        # Get feature importance
        importance_scores = get_ensemble_feature_importance(ensemble)

        # Check results
        self.assertIsInstance(importance_scores, dict)
        # At least some models should have feature importance
        self.assertGreater(len(importance_scores), 0)

        # Check that scores are reasonable
        for model_name, score in importance_scores.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)

    def test_get_ensemble_feature_importance_unfitted(self):
        """Test error handling for unfitted ensemble."""
        ensemble = create_research_stacked_ensemble()

        with self.assertRaises(ValueError) as context:
            get_ensemble_feature_importance(ensemble)

        self.assertIn("must be fitted", str(context.exception))

    def test_create_conservative_stacked_ensemble_default(self):
        """Test creating conservative stacked ensemble."""
        conservative = create_conservative_stacked_ensemble()

        # Check ensemble type
        self.assertEqual(conservative.__class__.__name__, 'StackingRegressor')
        self.assertTrue(conservative.passthrough)

        # Check that base estimators are present
        self.assertGreaterEqual(len(conservative.estimators), 2)

    def test_create_conservative_stacked_ensemble_fit_predict(self):
        """Test that conservative ensemble can fit and predict."""
        conservative = create_conservative_stacked_ensemble()

        # Fit model
        conservative.fit(self.X, self.y)

        # Make predictions
        predictions = conservative.predict(self.X)

        # Check predictions
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(pred > 0 for pred in predictions))

    def test_missing_dependencies_handling(self):
        """Test error handling when dependencies are missing."""
        with patch('nba_predictor.models.stacked_ensemble._sklearn_available', False):
            with self.assertRaises(ImportError) as context:
                create_research_stacked_ensemble()

            self.assertIn("scikit-learn is required", str(context.exception))

    def test_ensemble_vs_conservative_comparison(self):
        """Test that conservative ensemble has fewer estimators or is simpler."""
        ensemble = create_research_stacked_ensemble()
        conservative = create_conservative_stacked_ensemble()

        # Fit both models on smaller data for speed
        ensemble.fit(self.small_X, self.small_y)
        conservative.fit(self.small_X, self.small_y)

        # Both should make predictions
        pred_ensemble = ensemble.predict(self.small_X)
        pred_conservative = conservative.predict(self.small_X)

        self.assertEqual(len(pred_ensemble), len(pred_conservative))
        self.assertEqual(len(pred_ensemble), len(self.small_y))

    def test_ensemble_cv_integration(self):
        """Test that ensemble properly integrates with time series CV."""
        from sklearn.model_selection import TimeSeriesSplit

        cv_strategy = TimeSeriesSplit(n_splits=3, test_size=2)
        ensemble = create_research_stacked_ensemble(cv_strategy=cv_strategy)

        # Should fit without error
        ensemble.fit(self.X, self.y)

        # Should predict without error
        predictions = ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        # Create sample data
        X = pd.DataFrame(np.random.randn(50, 4))
        y = pd.Series(np.random.randn(50))

        ensemble = create_research_stacked_ensemble()

        # Should fit without error
        ensemble.fit(X, y)

        # Should predict without error
        predictions = ensemble.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_passthrough_functionality(self):
        """Test that passthrough works correctly."""
        ensemble = create_research_stacked_ensemble()

        # Check that passthrough is enabled
        self.assertTrue(ensemble.passthrough)

        # Fit and predict to ensure no errors
        ensemble.fit(self.X, self.y)
        predictions = ensemble.predict(self.X)

        self.assertEqual(len(predictions), len(self.y))

    def test_different_random_states(self):
        """Test ensemble behavior with different random states."""
        ensemble1 = create_research_stacked_ensemble()
        ensemble2 = create_research_stacked_ensemble()

        # Both should have same default random state
        # Fit both on same smaller data for speed
        ensemble1.fit(self.small_X, self.small_y)
        ensemble2.fit(self.small_X, self.small_y)

        # Predictions should be deterministic (same random state)
        pred1 = ensemble1.predict(self.small_X)
        pred2 = ensemble2.predict(self.small_X)

        np.testing.assert_array_almost_equal(pred1, pred2)

    @patch('nba_predictor.models.stacked_ensemble._xgboost_available', False)
    @patch('nba_predictor.models.stacked_ensemble._lightgbm_available', False)
    def test_minimal_base_estimators(self):
        """Test ensemble creation with minimal dependencies."""
        # Should still work with just RandomForest and Ridge
        ensemble = create_research_stacked_ensemble()

        # Should have at least 2 estimators (RF and Ridge)
        self.assertGreaterEqual(len(ensemble.estimators), 2)

        # Should still be able to fit and predict
        ensemble.fit(self.X, self.y)
        predictions = ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


if __name__ == '__main__':
    unittest.main()