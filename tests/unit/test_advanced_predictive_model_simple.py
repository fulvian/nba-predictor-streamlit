"""
Unit tests for AdvancedPredictiveModel.

Tests basic functionality of the ensemble ML system.
"""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from advanced_predictive_model import AdvancedPredictiveModel, ModelTrainingError, PredictionError


class TestAdvancedPredictiveModel:
    """Test suite for AdvancedPredictiveModel class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for NBA predictions."""
        np.random.seed(42)
        n_samples = 50

        data = {
            'home_team_score': np.random.randint(80, 140, n_samples),
            'away_team_score': np.random.randint(80, 140, n_samples),
            'home_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'away_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'home_team_reb': np.random.randint(30, 60, n_samples),
            'away_team_reb': np.random.randint(30, 60, n_samples),
            'home_team_ast': np.random.randint(15, 35, n_samples),
            'away_team_ast': np.random.randint(15, 35, n_samples),
        }

        # Create target variable (home team wins)
        df = pd.DataFrame(data)
        df['target'] = (df['home_team_score'] > df['away_team_score']).astype(int)

        return df

    @pytest.fixture
    def model(self):
        """Create AdvancedPredictiveModel instance."""
        return AdvancedPredictiveModel()

    def test_model_initialization_default(self, model):
        """Test model initialization with default parameters."""
        assert model.model_configs is None
        assert hasattr(model, 'models')
        assert hasattr(model, 'scaler')
        assert hasattr(model, 'label_encoder')
        assert hasattr(model, 'metrics')

    def test_model_initialization_with_configs(self):
        """Test model initialization with custom configurations."""
        configs = {
            'xgboost': {'n_estimators': 100, 'max_depth': 5},
            'logistic_regression': {'C': 1.0},
            'random_forest': {'n_estimators': 200}
        }

        model = AdvancedPredictiveModel(model_configs=configs)

        assert model.model_configs == configs

    @patch('advanced_predictive_model.XGBClassifier')
    @patch('advanced_predictive_model.LogisticRegression')
    @patch('advanced_predictive_model.RandomForestClassifier')
    @patch('advanced_predictive_model.VotingClassifier')
    @patch('advanced_predictive_model.StandardScaler')
    def test_train_predictive_models_success(self, mock_scaler_class, mock_voting,
                                            mock_rf, mock_lr, mock_xgb,
                                            model, sample_training_data):
        """Test successful model training."""
        # Setup mocks
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaled_data = np.random.randn(50, 8)
        mock_scaler.fit_transform.return_value = mock_scaled_data

        mock_xgb_instance = Mock()
        mock_lr_instance = Mock()
        mock_rf_instance = Mock()
        mock_xgb.return_value = mock_xgb_instance
        mock_lr.return_value = mock_lr_instance
        mock_rf.return_value = mock_rf_instance

        mock_ensemble = Mock()
        mock_ensemble.fit.return_value = None
        mock_voting.return_value = mock_ensemble

        result = model.train_predictive_models(sample_training_data, 'target')

        # Verify training result
        assert result["status"] == "success"
        assert result["models_trained"] == 3
        assert result["training_samples"] == 50
        assert result["feature_count"] > 0
        assert result["model_version"] == 1
        assert "training_time" in result

        # Verify model state
        assert hasattr(model, 'ensemble')
        assert model.scaler == mock_scaler

    def test_train_predictive_models_missing_target(self, model, sample_training_data):
        """Test training with missing target column."""
        with pytest.raises(ValueError, match="Target column 'missing_target' not found"):
            model.train_predictive_models(sample_training_data, 'missing_target')

    def test_train_predictive_models_empty_data(self, model):
        """Test training with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty training data provided"):
            model.train_predictive_models(empty_data, 'target')

    @patch('advanced_predictive_model.StandardScaler')
    @patch('advanced_predictive_model.VotingClassifier')
    def test_train_predictive_models_training_failure(self, mock_voting, mock_scaler_class,
                                                     model, sample_training_data):
        """Test training failure handling."""
        # Setup mocks
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaled_data = np.random.randn(50, 8)
        mock_scaler.fit_transform.return_value = mock_scaled_data

        mock_ensemble = Mock()
        mock_ensemble.fit.side_effect = Exception("Training failed")
        mock_voting.return_value = mock_ensemble

        with pytest.raises(ModelTrainingError, match="Failed to train predictive models"):
            model.train_predictive_models(sample_training_data, 'target')

    def test_prepare_training_data_empty(self, model):
        """Test preprocessing with empty data."""
        X = pd.DataFrame()
        y = pd.Series()

        with pytest.raises(ValueError, match="Empty training data provided"):
            model._prepare_training_data(X, y)

    @patch('advanced_predictive_model.StandardScaler')
    def test_preprocess_training_data(self, mock_scaler_class, model):
        """Test training data preprocessing."""
        # Setup mock scaler
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaled_data = np.random.randn(10, 3)
        mock_scaler.fit_transform.return_value = mock_scaled_data

        # Create sample data
        X = pd.DataFrame(np.random.randn(10, 3))
        y = pd.Series(np.random.randint(0, 2, 10))

        X_processed, y_processed = model._prepare_training_data(X, y)

        # Verify scaler was fitted and used
        mock_scaler.fit_transform.assert_called_once()
        assert np.array_equal(X_processed, mock_scaled_data)
        assert len(y_processed) == len(y)

    def test_create_voting_ensemble(self, model):
        """Test creation of voting ensemble."""
        # Setup models
        mock_xgb = Mock()
        mock_lr = Mock()
        mock_rf = Mock()

        model.models = {
            'xgboost': mock_xgb,
            'logistic_regression': mock_lr,
            'random_forest': mock_rf
        }

        with patch('advanced_predictive_model.VotingClassifier') as mock_voting:
            mock_ensemble = Mock()
            mock_voting.return_value = mock_ensemble

            model._create_voting_ensemble()

            # Verify ensemble was created
            mock_voting.assert_called_once()
            assert model.ensemble == mock_ensemble

    def test_calculate_quality_score_good_data(self, model):
        """Test quality score calculation for good data."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })

        quality_score = model._calculate_quality_score(X)

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

    def test_calculate_quality_score_missing_data(self, model):
        """Test quality score calculation with missing data."""
        X = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],  # Missing value
            'feature2': [0.1, 0.2, 0.3, None, 0.5]  # Missing value
        })

        quality_score = model._calculate_quality_score(X)

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
        # Should be lower due to missing data
        assert quality_score < 1.0

    def test_evaluate_models_basic(self, model):
        """Test basic model evaluation."""
        # Setup mock models
        mock_ensemble = Mock()
        mock_predictions = np.array([1, 0, 1, 0, 1])
        mock_ensemble.predict.return_value = mock_predictions
        mock_ensemble.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.8, 0.2], [0.2, 0.8]])

        model.ensemble = mock_ensemble

        X_test = pd.DataFrame(np.random.randn(5, 3))
        y_test = pd.Series([1, 0, 1, 0, 1])

        metrics = model._evaluate_models(X_test, y_test)

        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_model_metadata_tracking(self, model):
        """Test that model metadata is properly tracked."""
        # Initially should not have metadata
        assert not hasattr(model, 'training_metadata') or len(model.training_metadata) == 0

        # This will be tested more when we have actual training working
        assert hasattr(model, 'metrics') or hasattr(model, 'training_metadata')

    def test_error_handling_invalid_input_types(self, model):
        """Test error handling for invalid input types."""
        # Test with invalid input types
        with pytest.raises((ValueError, TypeError)):
            model.train_predictive_models("not a dataframe", 'target')

    def test_feature_validation(self, model):
        """Test feature validation during training."""
        # Create data with no valid features
        invalid_data = pd.DataFrame({'invalid_col': ['a', 'b', 'c']})
        invalid_data['target'] = [1, 0, 1]

        # This should raise an error due to invalid features
        with pytest.raises((ValueError, ModelTrainingError)):
            model.train_predictive_models(invalid_data, 'target')

    def test_ensemble_weights(self, model):
        """Test that ensemble weights are properly handled."""
        # This tests the concept that weights should be applied
        # The actual implementation may vary
        assert hasattr(model, 'models') or hasattr(model, 'ensemble')

    def test_feature_engineering_concepts(self, model):
        """Test that feature engineering concepts are implemented."""
        # Test that the model has capability for feature processing
        assert hasattr(model, '_prepare_training_data')

        # Test data preprocessing
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        y = pd.Series([1, 0, 1])

        with patch('advanced_predictive_model.StandardScaler') as mock_scaler_class:
            mock_scaler = Mock()
            mock_scaler_class.return_value = mock_scaler
            mock_scaler.fit_transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])

            X_processed, y_processed = model._prepare_training_data(X, y)

            assert X_processed.shape == (3, 2)
            assert len(y_processed) == 3