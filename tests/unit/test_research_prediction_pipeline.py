#!/usr/bin/env python3
"""
🧪 Research Prediction Pipeline Unit Tests
Test suite for research-based prediction pipeline functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from nba_predictor.core.research_prediction_pipeline import (
    ResearchPredictionPipeline,
    create_research_prediction_pipeline
)


class TestResearchPredictionPipeline(unittest.TestCase):
    """Test cases for research prediction pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "data"
        self.models_path = Path(self.temp_dir) / "models"

        # Create directories
        self.data_path.mkdir(parents=True)
        self.models_path.mkdir(parents=True)

        # Create sample data file
        self.sample_data = self._create_sample_data()
        data_file = self.data_path / "nba_games.csv"
        self.sample_data.to_csv(data_file, index=False)

        # Initialize pipeline
        self.pipeline = ResearchPredictionPipeline(
            data_path=str(self.data_path),
            models_path=str(self.models_path),
            use_stacked_ensemble=True,
            enable_explainability=False  # Disable for faster testing
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample NBA data for testing."""
        np.random.seed(42)
        n_samples = 200

        return pd.DataFrame({
            # Team scoring
            'team1_score': np.random.randint(80, 140, n_samples),
            'team2_score': np.random.randint(80, 140, n_samples),

            # Four Factors
            'efg_pct': np.random.uniform(0.45, 0.65, n_samples),
            'tov_pct': np.random.uniform(0.10, 0.20, n_samples),
            'orb_pct': np.random.uniform(0.20, 0.35, n_samples),
            'ftr': np.random.uniform(0.15, 0.35, n_samples),

            # Additional stats
            'team1_field_goals_made': np.random.randint(30, 50, n_samples),
            'team1_field_goals_attempted': np.random.randint(60, 100, n_samples),
            'team1_three_pointers_made': np.random.randint(5, 20, n_samples),
            'team1_three_pointers_attempted': np.random.randint(15, 40, n_samples),
            'team1_free_throws_made': np.random.randint(10, 25, n_samples),
            'team1_free_throws_attempted': np.random.randint(15, 35, n_samples),
            'team1_rebounds': np.random.randint(30, 60, n_samples),
            'team1_assists': np.random.randint(15, 35, n_samples),
            'team1_steals': np.random.randint(5, 15, n_samples),
            'team1_blocks': np.random.randint(2, 10, n_samples),
            'team1_turnovers': np.random.randint(10, 25, n_samples),
            'team1_fouls': np.random.randint(15, 30, n_samples),

            'team2_field_goals_made': np.random.randint(30, 50, n_samples),
            'team2_field_goals_attempted': np.random.randint(60, 100, n_samples),
            'team2_three_pointers_made': np.random.randint(5, 20, n_samples),
            'team2_three_pointers_attempted': np.random.randint(15, 40, n_samples),
            'team2_free_throws_made': np.random.randint(10, 25, n_samples),
            'team2_free_throws_attempted': np.random.randint(15, 35, n_samples),
            'team2_rebounds': np.random.randint(30, 60, n_samples),
            'team2_assists': np.random.randint(15, 35, n_samples),
            'team2_steals': np.random.randint(5, 15, n_samples),
            'team2_blocks': np.random.randint(2, 10, n_samples),
            'team2_turnovers': np.random.randint(10, 25, n_samples),
            'team2_fouls': np.random.randint(15, 30, n_samples),

            'team1_offensive_rebounds': np.random.randint(5, 15, n_samples),
            'team2_offensive_rebounds': np.random.randint(5, 15, n_samples),

            # Total score (target variable)
            'total_score': np.random.randint(160, 280, n_samples),
        })

    def test_create_research_prediction_pipeline(self):
        """Test creating research prediction pipeline."""
        pipeline = create_research_prediction_pipeline(
            data_path=str(self.data_path),
            models_path=str(self.models_path),
            use_stacked_ensemble=True,
            enable_explainability=False
        )

        self.assertIsInstance(pipeline, ResearchPredictionPipeline)
        self.assertEqual(pipeline.use_stacked_ensemble, True)
        self.assertEqual(pipeline.enable_explainability, False)
        self.assertFalse(pipeline.is_trained)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(str(self.pipeline.data_path), str(self.data_path))
        self.assertEqual(str(self.pipeline.models_path), str(self.models_path))
        self.assertTrue(self.pipeline.use_stacked_ensemble)
        self.assertFalse(self.pipeline.enable_explainability)
        self.assertFalse(self.pipeline.is_trained)

    def test_pipeline_initialization_invalid_data_path(self):
        """Test pipeline initialization with invalid data path."""
        invalid_path = Path(self.temp_dir) / "nonexistent"

        with self.assertRaises(FileNotFoundError):
            ResearchPredictionPipeline(
                data_path=str(invalid_path),
                models_path=str(self.models_path)
            )

    def test_load_data_success(self):
        """Test successful data loading."""
        X, y = self.pipeline.load_data()

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertIn('total_score', self.sample_data.columns)

    def test_load_data_missing_file(self):
        """Test data loading when file is missing."""
        # Remove data file
        data_file = self.data_path / "nba_games.csv"
        data_file.unlink()

        # Should still work by creating sample data
        X, y = self.pipeline.load_data()

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertGreater(len(X), 0)

    def test_train_model_success(self):
        """Test successful model training."""
        # Load data first
        X, y = self.pipeline.load_data()

        # Train model
        metrics = self.pipeline.train_model(X, y, validation_split=0.2)

        # Check results
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('train_samples', metrics)
        self.assertIn('val_samples', metrics)
        self.assertIn('features', metrics)

        # Check pipeline state
        self.assertTrue(self.pipeline.is_trained)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.feature_scaler)
        self.assertGreater(len(self.pipeline.feature_columns), 0)

    def test_train_model_without_data(self):
        """Test model training without providing data."""
        metrics = self.pipeline.train_model(validation_split=0.2)

        self.assertIsInstance(metrics, dict)
        self.assertTrue(self.pipeline.is_trained)

    def test_predict_success(self):
        """Test successful prediction."""
        # Train model first
        self.pipeline.train_model()

        # Make prediction
        result = self.pipeline.predict(
            team1_name="Boston Celtics",
            team2_name="Los Angeles Lakers",
            line=225.5
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('team1', result)
        self.assertIn('team2', result)
        self.assertIn('line', result)
        self.assertIn('predicted_total', result)
        self.assertIn('recommendation', result)
        self.assertIn('confidence', result)
        self.assertIn('difference', result)
        self.assertIn('model_metrics', result)

        # Check values
        self.assertEqual(result['team1'], "Boston Celtics")
        self.assertEqual(result['team2'], "Los Angeles Lakers")
        self.assertEqual(result['line'], 225.5)
        self.assertIn(result['recommendation'], ['OVER', 'UNDER'])
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_predict_untrained_model(self):
        """Test prediction with untrained model."""
        with self.assertRaises(ValueError) as context:
            self.pipeline.predict("Team A", "Team B", 200.0)

        self.assertIn("must be trained", str(context.exception))

    def test_predict_with_custom_features(self):
        """Test prediction with custom features."""
        # Train model first
        self.pipeline.train_model()

        # Create custom features
        custom_features = {
            'efg_pct': 0.55,
            'tov_pct': 0.12,
            'orb_pct': 0.30,
            'ftr': 0.25,
            'team1_score': 115.0,
            'team2_score': 110.0
        }

        result = self.pipeline.predict(
            team1_name="Team A",
            team2_name="Team B",
            line=225.0,
            features=custom_features
        )

        self.assertIsInstance(result, dict)
        self.assertIn('predicted_total', result)

    def test_explain_prediction_disabled(self):
        """Test explanation when explainability is disabled."""
        # Train model first
        self.pipeline.train_model()

        with self.assertRaises(ValueError) as context:
            self.pipeline.explain_prediction("Team A", "Team B", 200.0)

        self.assertIn("explainability not enabled", str(context.exception))

    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Train model first
        self.pipeline.train_model()

        # Save model
        saved_path = self.pipeline.save_model("test_model.pkl")
        self.assertTrue(Path(saved_path).exists())

        # Create new pipeline and load model
        new_pipeline = ResearchPredictionPipeline(
            data_path=str(self.data_path),
            models_path=str(self.models_path),
            use_stacked_ensemble=True,
            enable_explainability=False
        )

        new_pipeline.load_model("test_model.pkl")

        # Check that loaded pipeline has same state
        self.assertTrue(new_pipeline.is_trained)
        self.assertIsNotNone(new_pipeline.model)
        self.assertEqual(new_pipeline.use_stacked_ensemble, self.pipeline.use_stacked_ensemble)
        self.assertEqual(new_pipeline.feature_columns, self.pipeline.feature_columns)
        self.assertEqual(new_pipeline.metrics, self.pipeline.metrics)

    def test_save_untrained_model(self):
        """Test saving untrained model."""
        with self.assertRaises(ValueError) as context:
            self.pipeline.save_model()

        self.assertIn("must be trained", str(context.exception))

    def test_load_nonexistent_model(self):
        """Test loading nonexistent model."""
        with self.assertRaises(FileNotFoundError):
            self.pipeline.load_model("nonexistent_model.pkl")

    def test_get_model_info_untrained(self):
        """Test getting model info for untrained model."""
        info = self.pipeline.get_model_info()

        expected_keys = [
            'is_trained', 'use_stacked_ensemble', 'enable_explainability',
            'feature_columns_count', 'four_factors_columns', 'metrics'
        ]

        for key in expected_keys:
            self.assertIn(key, info)

        self.assertFalse(info['is_trained'])
        self.assertEqual(info['feature_columns_count'], 0)

    def test_get_model_info_trained(self):
        """Test getting model info for trained model."""
        # Train model first
        self.pipeline.train_model()

        info = self.pipeline.get_model_info()

        self.assertTrue(info['is_trained'])
        self.assertGreater(info['feature_columns_count'], 0)
        self.assertIn('model_type', info)
        self.assertIn('metrics', info)

    def test_predict_with_high_scoring_teams(self):
        """Test prediction with high-scoring teams."""
        # Train model first
        self.pipeline.train_model()

        # Test with high-scoring team
        result = self.pipeline.predict(
            team1_name="Golden State Warriors",  # High scoring team
            team2_name="Boston Celtics",
            line=240.0
        )

        # Should have higher predicted total due to team adjustments
        self.assertGreater(result['predicted_total'], 220.0)

    def test_team_adjustments(self):
        """Test team-specific feature adjustments."""
        # Test with high-scoring team
        adjustments = self.pipeline._get_team_adjustments(
            "Golden State Warriors", "Boston Celtics"
        )

        self.assertIsInstance(adjustments, dict)
        self.assertIn('team1_score', adjustments)
        self.assertIn('efg_pct', adjustments)

    def test_default_features_creation(self):
        """Test default feature creation."""
        features = self.pipeline._create_default_features("Team A", "Team B")

        self.assertIsInstance(features, dict)
        self.assertIn('efg_pct', features)
        self.assertIn('tov_pct', features)
        self.assertIn('orb_pct', features)
        self.assertIn('ftr', features)
        self.assertIn('team1_score', features)
        self.assertIn('team2_score', features)

    def test_preprocess_data_validation(self):
        """Test data preprocessing validation."""
        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'team1_score': [100, 110],
            'team2_score': [95, 105]
            # Missing Four Factors columns
        })

        with self.assertRaises(ValueError):
            self.pipeline._preprocess_data(invalid_data)

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        pipeline = create_research_prediction_pipeline(
            data_path=str(self.data_path),
            models_path=str(self.models_path)
        )

        # Should train without error
        metrics = pipeline.train_model()
        self.assertIsInstance(metrics, dict)

        # Should predict without error
        result = pipeline.predict(
            "Boston Celtics", "New Orleans Pelicans", 233.5
        )
        self.assertIsInstance(result, dict)

    def test_pipeline_with_lightgbm_only(self):
        """Test pipeline with LightGBM instead of stacked ensemble."""
        pipeline = ResearchPredictionPipeline(
            data_path=str(self.data_path),
            models_path=str(self.models_path),
            use_stacked_ensemble=False,
            enable_explainability=False
        )

        # Train model
        metrics = pipeline.train_model()

        # Should work with LightGBM
        self.assertIsInstance(metrics, dict)
        self.assertTrue(pipeline.is_trained)

        # Should predict
        result = pipeline.predict("Team A", "Team B", 200.0)
        self.assertIsInstance(result, dict)

    def test_pipeline_with_explainability(self):
        """Test pipeline with explainability enabled."""
        with patch('nba_predictor.core.research_prediction_pipeline.create_nba_shap_explainer') as mock_shap:
            mock_explainer = MagicMock()
            mock_shap.return_value = mock_explainer

            pipeline = ResearchPredictionPipeline(
                data_path=str(self.data_path),
                models_path=str(self.models_path),
                use_stacked_ensemble=True,
                enable_explainability=True
            )

            # Train model
            pipeline.train_model()

            # Should attempt to create SHAP explainer
            mock_shap.assert_called_once()

    def test_feature_scaling(self):
        """Test that features are properly scaled."""
        # Train model first
        self.pipeline.train_model()

        # Create features with different scales
        features = {
            'efg_pct': 0.55,  # Small scale
            'team1_score': 120.0,  # Large scale
            'team1_rebounds': 45.0,  # Medium scale
        }

        result = self.pipeline.predict("Team A", "Team B", 220.0, features)

        # Should produce reasonable prediction despite different scales
        self.assertIsInstance(result['predicted_total'], float)
        self.assertGreater(result['predicted_total'], 150.0)
        self.assertLess(result['predicted_total'], 300.0)

    def test_time_series_validation(self):
        """Test time series validation during training."""
        # Load data
        X, y = self.pipeline.load_data()

        # Train with small validation split to test time series split
        metrics = self.pipeline.train_model(X, y, validation_split=0.1)

        # Should succeed and provide metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertTrue(self.pipeline.is_trained)

    def test_metrics_calculation(self):
        """Test that training metrics are calculated correctly."""
        # Train model
        metrics = self.pipeline.train_model(validation_split=0.2)

        # Check metric calculations
        self.assertGreater(metrics['mae'], 0.0)
        self.assertGreater(metrics['mse'], 0.0)
        self.assertGreater(metrics['rmse'], 0.0)
        self.assertGreater(metrics['mape'], 0.0)

        # Check relationship between metrics
        self.assertAlmostEqual(
            metrics['rmse'], np.sqrt(metrics['mse']), places=5
        )

    def test_confidence_calculation(self):
        """Test confidence calculation in predictions."""
        # Train model
        self.pipeline.train_model()

        # Test with clear OVER case
        result_over = self.pipeline.predict("Team A", "Team B", 180.0)
        self.assertEqual(result_over['recommendation'], 'OVER')
        self.assertGreater(result_over['difference'], 0)
        self.assertGreater(result_over['confidence'], 0)

        # Test with clear UNDER case
        result_under = self.pipeline.predict("Team A", "Team B", 280.0)
        self.assertEqual(result_under['recommendation'], 'UNDER')
        self.assertLess(result_under['difference'], 0)
        self.assertGreater(result_under['confidence'], 0)


if __name__ == '__main__':
    unittest.main()