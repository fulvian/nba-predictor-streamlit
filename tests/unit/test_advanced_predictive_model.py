"""
Unit tests for AdvancedPredictiveModel.

Tests ensemble model functionality, training methods,
and prediction accuracy for NBA games.
"""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from advanced_predictive_model import AdvancedPredictiveModel, ModelTrainingError, PredictionError


class TestAdvancedPredictiveModel:
    """Test suite for AdvancedPredictiveModel class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for NBA predictions."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'home_team_score': np.random.randint(80, 140, n_samples),
            'away_team_score': np.random.randint(80, 140, n_samples),
            'home_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'away_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'home_team_reb': np.random.randint(30, 60, n_samples),
            'away_team_reb': np.random.randint(30, 60, n_samples),
            'home_team_ast': np.random.randint(15, 35, n_samples),
            'away_team_ast': np.random.randint(15, 35, n_samples),
            'home_team_turnovers': np.random.randint(8, 20, n_samples),
            'away_team_turnovers': np.random.randint(8, 20, n_samples),
            'home_team_wins_last_10': np.random.randint(0, 11, n_samples),
            'away_team_wins_last_10': np.random.randint(0, 11, n_samples),
            'home_team_rest_days': np.random.randint(0, 5, n_samples),
            'away_team_rest_days': np.random.randint(0, 5, n_samples),
            'home_point_diff': np.random.randint(-30, 30, n_samples),
            'away_point_diff': np.random.randint(-30, 30, n_samples)
        }

        # Create target variable (home team wins)
        df = pd.DataFrame(data)
        df['target'] = (df['home_team_score'] > df['away_team_score']).astype(int)

        # Add feature engineering columns
        df['home_team_ppg'] = df['home_team_score']  # Simplified
        df['away_team_ppg'] = df['away_team_score']  # Simplified
        df['home_team_opp_ppg'] = df['away_team_score']  # Simplified
        df['away_team_opp_ppg'] = df['home_team_score']  # Simplified

        return df

    @pytest.fixture
    def sample_prediction_data(self):
        """Create sample data for predictions."""
        np.random.seed(42)
        n_samples = 10

        data = {
            'home_team_score': np.random.randint(80, 140, n_samples),
            'away_team_score': np.random.randint(80, 140, n_samples),
            'home_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'away_team_fg_pct': np.random.uniform(0.35, 0.55, n_samples),
            'home_team_reb': np.random.randint(30, 60, n_samples),
            'away_team_reb': np.random.randint(30, 60, n_samples),
            'home_team_ast': np.random.randint(15, 35, n_samples),
            'away_team_ast': np.random.randint(15, 35, n_samples),
            'home_team_turnovers': np.random.randint(8, 20, n_samples),
            'away_team_turnovers': np.random.randint(8, 20, n_samples),
            'home_team_wins_last_10': np.random.randint(0, 11, n_samples),
            'away_team_wins_last_10': np.random.randint(0, 11, n_samples),
            'home_team_rest_days': np.random.randint(0, 5, n_samples),
            'away_team_rest_days': np.random.randint(0, 5, n_samples),
            'home_point_diff': np.random.randint(-30, 30, n_samples),
            'away_point_diff': np.random.randint(-30, 30, n_samples),
            'home_team_ppg': np.random.randint(80, 140, n_samples),
            'away_team_ppg': np.random.randint(80, 140, n_samples),
            'home_team_opp_ppg': np.random.randint(80, 140, n_samples),
            'away_team_opp_ppg': np.random.randint(80, 140, n_samples)
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def model(self):
        """Create AdvancedPredictiveModel instance."""
        return AdvancedPredictiveModel()

    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = AdvancedPredictiveModel()

        assert model.models == {}
        assert model.ensemble is None
        assert model.ensemble_weights is None
        assert model.feature_columns == []
        assert model.scaler is None
        assert model.is_trained is False
        assert model.model_metadata == {}

    def test_model_initialization_with_configs(self):
        """Test model initialization with custom configurations."""
        configs = {
            'xgboost': {'n_estimators': 100, 'max_depth': 5},
            'logistic_regression': {'C': 1.0},
            'random_forest': {'n_estimators': 200}
        }

        model = AdvancedPredictiveModel(model_configs=configs)

        assert len(model.model_configs) == 3
        assert 'xgboost' in model.model_configs
        assert 'logistic_regression' in model.model_configs
        assert 'random_forest' in model.model_configs

    @patch('advanced_predictive_model.XGBClassifier')
    @patch('advanced_predictive_model.LogisticRegression')
    @patch('advanced_predictive_model.RandomForestClassifier')
    def test_create_individual_models(self, mock_rf, mock_lr, mock_xgb):
        """Test creation of individual ML models."""
        # Setup mocks
        mock_xgb_instance = Mock()
        mock_lr_instance = Mock()
        mock_rf_instance = Mock()
        mock_xgb.return_value = mock_xgb_instance
        mock_lr.return_value = mock_lr_instance
        mock_rf.return_value = mock_rf_instance

        model = AdvancedPredictiveModel()
        model._create_individual_models()

        # Verify models were created
        assert len(model.models) == 3
        assert 'xgboost' in model.models
        assert 'logistic_regression' in model.models
        assert 'random_forest' in model.models

        # Verify XGBoost was configured correctly
        mock_xgb.assert_called_once_with(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

    @patch('advanced_predictive_model.VotingClassifier')
    def test_create_voting_ensemble(self, mock_voting):
        """Test creation of voting ensemble."""
        # Setup models
        mock_xgb = Mock()
        mock_lr = Mock()
        mock_rf = Mock()

        model = AdvancedPredictiveModel()
        model.models = {
            'xgboost': mock_xgb,
            'logistic_regression': mock_lr,
            'random_forest': mock_rf
        }

        mock_ensemble = Mock()
        mock_voting.return_value = mock_ensemble

        model._create_voting_ensemble()

        # Verify ensemble was created with correct parameters
        mock_voting.assert_called_once_with(
            estimators=[
                ('xgboost', mock_xgb),
                ('logistic_regression', mock_lr),
                ('random_forest', mock_rf)
            ],
            voting='soft',
            weights=[2.0, 1.5, 1.0]
        )

        assert model.ensemble == mock_ensemble
        assert model.ensemble_weights == {'xgboost': 2.0, 'logistic_regression': 1.5, 'random_forest': 1.0}

    def test_prepare_features_basic(self):
        """Test basic feature preparation."""
        model = AdvancedPredictiveModel()

        # Create sample data
        data = pd.DataFrame({
            'home_team_score': [100, 95],
            'away_team_score': [95, 100],
            'home_team_fg_pct': [0.45, 0.42],
            'away_team_fg_pct': [0.43, 0.47]
        })

        features = model._prepare_features(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 2
        assert 'point_diff' in features.columns
        assert 'fg_pct_diff' in features.columns
        assert features.iloc[0]['point_diff'] == 5  # 100 - 95

    def test_prepare_features_engineering(self):
        """Test advanced feature engineering."""
        model = AdvancedPredictiveModel()

        # Create sample data with team stats
        data = pd.DataFrame({
            'home_team_score': [100, 95],
            'away_team_score': [95, 100],
            'home_team_ppg': [110, 105],
            'away_team_ppg': [105, 110],
            'home_team_opp_ppg': [100, 98],
            'away_team_opp_ppg': [98, 102],
            'home_team_wins_last_10': [7, 6],
            'away_team_wins_last_10': [8, 5]
        })

        features = model._prepare_features(data)

        # Check engineered features
        assert 'home_offensive_rating' in features.columns
        assert 'away_offensive_rating' in features.columns
        assert 'home_defensive_rating' in features.columns
        assert 'away_defensive_rating' in features.columns
        assert 'home_form' in features.columns
        assert 'away_form' in features.columns

    def test_prepare_features_with_empty_data(self):
        """Test feature preparation with empty data."""
        model = AdvancedPredictiveModel()

        empty_data = pd.DataFrame()
        features = model._prepare_features(empty_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0

    @patch('advanced_predictive_model.StandardScaler')
    def test_preprocess_training_data(self, mock_scaler_class):
        """Test training data preprocessing."""
        # Setup mock scaler
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaled_data = np.random.randn(100, 5)
        mock_scaler.fit_transform.return_value = mock_scaled_data

        model = AdvancedPredictiveModel()

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        X_processed, y_processed = model._preprocess_training_data(X, y)

        # Verify scaler was fitted and used
        mock_scaler.fit_transform.assert_called_once()
        assert model.scaler == mock_scaler
        assert np.array_equal(X_processed, mock_scaled_data)
        assert len(y_processed) == len(y)

    def test_preprocess_training_data_empty(self):
        """Test preprocessing with empty data."""
        model = AdvancedPredictiveModel()

        X = pd.DataFrame()
        y = pd.Series()

        with pytest.raises(ValueError, match="Empty training data provided"):
            model._preprocess_training_data(X, y)

    def test_preprocess_prediction_data(self):
        """Test prediction data preprocessing."""
        model = AdvancedPredictiveModel()

        # Setup mock scaler
        mock_scaler = Mock()
        mock_scaled_data = np.random.randn(10, 5)
        mock_scaler.transform.return_value = mock_scaled_data
        model.scaler = mock_scaler
        model.feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

        # Create prediction data
        X = pd.DataFrame(np.random.randn(10, 5), columns=model.feature_columns)

        X_processed = model._preprocess_prediction_data(X)

        mock_scaler.transform.assert_called_once()
        assert np.array_equal(X_processed, mock_scaled_data)

    def test_preprocess_prediction_data_no_feature_columns(self):
        """Test prediction preprocessing with no feature columns."""
        model = AdvancedPredictiveModel()
        model.feature_columns = []

        X = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(ValueError, match="No feature columns available"):
            model._preprocess_prediction_data(X)

    @patch('advanced_predictive_model.StandardScaler')
    @patch.object(AdvancedPredictiveModel, '_create_individual_models')
    @patch.object(AdvancedPredictiveModel, '_create_voting_ensemble')
    def test_train_predictive_models_success(self, mock_ensemble, mock_models, mock_scaler_class):
        """Test successful model training."""
        # Setup mocks
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaled_data = np.random.randn(100, 10)
        mock_scaler.fit_transform.return_value = mock_scaled_data

        mock_ensemble_model = Mock()
        mock_ensemble.return_value = mock_ensemble_model
        mock_models.return_value = None

        model = AdvancedPredictiveModel()

        # Create sample data
        training_data = self.sample_training_data()

        result = model.train_predictive_models(training_data, 'target')

        # Verify training result
        assert result["status"] == "success"
        assert result["models_trained"] == 3
        assert result["training_samples"] == 100
        assert result["feature_count"] > 0
        assert result["model_version"] == 1
        assert "training_time" in result

        # Verify model state
        assert model.is_trained is True
        assert len(model.feature_columns) > 0

    def test_train_predictive_models_missing_target(self):
        """Test training with missing target column."""
        model = AdvancedPredictiveModel()
        training_data = pd.DataFrame({'feature1': [1, 2, 3]})

        with pytest.raises(ValueError, match="Target column 'missing_target' not found"):
            model.train_predictive_models(training_data, 'missing_target')

    def test_train_predictive_models_empty_data(self):
        """Test training with empty data."""
        model = AdvancedPredictiveModel()
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty training data provided"):
            model.train_predictive_models(empty_data, 'target')

    @patch('advanced_predictive_model.StandardScaler')
    @patch.object(AdvancedPredictiveModel, '_create_individual_models')
    @patch.object(AdvancedPredictiveModel, '_create_voting_ensemble')
    def test_train_predictive_models_training_failure(self, mock_ensemble, mock_models, mock_scaler_class):
        """Test training failure handling."""
        # Setup mocks
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_ensemble_model = Mock()
        mock_ensemble_model.fit.side_effect = Exception("Training failed")
        mock_ensemble.return_value = mock_ensemble_model
        mock_models.return_value = None

        model = AdvancedPredictiveModel()
        training_data = self.sample_training_data()

        with pytest.raises(ModelTrainingError, match="Failed to train predictive models"):
            model.train_predictive_models(training_data, 'target')

    @patch.object(AdvancedPredictiveModel, '_preprocess_prediction_data')
    def test_predict_game_outcome_success(self, mock_preprocess):
        """Test successful game prediction."""
        # Setup mocks
        mock_preprocessed_data = np.random.randn(10, 5)
        mock_preprocess.return_value = mock_preprocessed_data

        mock_ensemble = Mock()
        mock_predictions_proba = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.8, 0.2], [0.2, 0.8],
                                          [0.5, 0.5], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        mock_ensemble.predict_proba.return_value = mock_predictions_proba
        mock_ensemble.predict.return_value = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

        model = AdvancedPredictiveModel()
        model.is_trained = True
        model.ensemble = mock_ensemble
        model.feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

        prediction_data = self.sample_prediction_data()

        result = model.predict_game_outcome(prediction_data)

        # Verify prediction result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns
        assert result['prediction'].isin([0, 1]).all()
        assert (result['confidence'] >= 0).all() and (result['confidence'] <= 1).all()

    @patch.object(AdvancedPredictiveModel, '_preprocess_prediction_data')
    def test_predict_game_outcome_no_confidence(self, mock_preprocess):
        """Test prediction without confidence intervals."""
        # Setup mocks
        mock_preprocessed_data = np.random.randn(5, 5)
        mock_preprocess.return_value = mock_preprocessed_data

        mock_ensemble = Mock()
        mock_predictions = np.array([1, 0, 1, 0, 1])
        mock_ensemble.predict.return_value = mock_predictions

        model = AdvancedPredictiveModel()
        model.is_trained = True
        model.ensemble = mock_ensemble
        model.feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

        prediction_data = self.sample_prediction_data().head(5)

        result = model.predict_game_outcome(prediction_data, return_confidence=False)

        assert 'prediction' in result.columns
        assert 'confidence' not in result.columns

    def test_predict_game_outcome_not_trained(self):
        """Test prediction with untrained model."""
        model = AdvancedPredictiveModel()
        model.is_trained = False

        prediction_data = self.sample_prediction_data()

        with pytest.raises(PredictionError, match="Model must be trained before making predictions"):
            model.predict_game_outcome(prediction_data)

    def test_predict_game_outcome_empty_data(self):
        """Test prediction with empty data."""
        model = AdvancedPredictiveModel()
        model.is_trained = True

        empty_data = pd.DataFrame()

        with pytest.raises(PredictionError, match="Empty prediction data provided"):
            model.predict_game_outcome(empty_data)

    def test_get_feature_importance_trained_model(self):
        """Test getting feature importance from trained model."""
        model = AdvancedPredictiveModel()
        model.is_trained = True
        model.feature_columns = ['feature1', 'feature2', 'feature3']

        # Create mock models with feature importance
        mock_xgb = Mock()
        mock_xgb.feature_importances_ = np.array([0.5, 0.3, 0.2])

        mock_rf = Mock()
        mock_rf.feature_importances_ = np.array([0.4, 0.4, 0.2])

        mock_lr = Mock()
        mock_lr.coef_ = np.array([[0.1, -0.2, 0.3]])

        model.models = {
            'xgboost': mock_xgb,
            'random_forest': mock_rf,
            'logistic_regression': mock_lr
        }

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 3
        assert 'xgboost' in importance.columns
        assert 'random_forest' in importance.columns
        assert 'logistic_regression' in importance.columns
        assert importance.index.tolist() == model.feature_columns

    def test_get_feature_importance_untrained_model(self):
        """Test getting feature importance from untrained model."""
        model = AdvancedPredictiveModel()
        model.is_trained = False

        with pytest.raises(PredictionError, match="Model must be trained before getting feature importance"):
            model.get_feature_importance()

    def test_get_model_metadata(self):
        """Test getting model metadata."""
        model = AdvancedPredictiveModel()
        model.model_metadata = {
            'training_date': datetime.now(),
            'model_version': 1,
            'feature_count': 10
        }

        metadata = model.get_model_metadata()

        assert isinstance(metadata, dict)
        assert 'training_date' in metadata
        assert 'model_version' in metadata
        assert 'feature_count' in metadata

    def test_save_model_success(self, tmp_path):
        """Test successful model saving."""
        model = AdvancedPredictiveModel()
        model.is_trained = True
        model.scaler = Mock()
        model.models = {'xgboost': Mock()}
        model.ensemble = Mock()

        with patch('advanced_predictive_model.joblib.dump') as mock_dump:
            file_path = tmp_path / "test_model.pkl"

            model.save_model(str(file_path))

            # Verify save was called
            assert mock_dump.call_count == 3  # scaler, models, ensemble

    def test_save_model_not_trained(self, tmp_path):
        """Test saving untrained model."""
        model = AdvancedPredictiveModel()
        model.is_trained = False

        file_path = tmp_path / "test_model.pkl"

        with pytest.raises(PredictionError, match="No trained model to save"):
            model.save_model(str(file_path))

    @patch('advanced_predictive_model.joblib.load')
    def test_load_model_success(self, mock_load, tmp_path):
        """Test successful model loading."""
        # Setup mocks
        mock_scaler = Mock()
        mock_models = {'xgboost': Mock()}
        mock_ensemble = Mock()
        mock_metadata = {'model_version': 2}

        def load_side_effect(filepath):
            if 'scaler' in filepath:
                return mock_scaler
            elif 'models' in filepath:
                return mock_models
            elif 'ensemble' in filepath:
                return mock_ensemble
            else:
                return mock_metadata

        mock_load.side_effect = load_side_effect

        model = AdvancedPredictiveModel()
        file_path = tmp_path / "test_model"

        model.load_model(str(file_path))

        assert model.scaler == mock_scaler
        assert model.models == mock_models
        assert model.ensemble == mock_ensemble
        assert model.model_metadata == mock_metadata
        assert model.is_trained is True

    def test_load_model_file_not_found(self):
        """Test loading non-existent model."""
        model = AdvancedPredictiveModel()

        with pytest.raises(PredictionError, match="Model files not found"):
            model.load_model("nonexistent_model")

    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        model = AdvancedPredictiveModel()

        # Create sample probabilities
        probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])

        confidence = model._calculate_confidence_intervals(probabilities)

        assert len(confidence) == 3
        assert all(0 <= c <= 1 for c in confidence)
        # Confidence should be max probability for binary classification
        expected_confidence = np.array([0.8, 0.6, 0.6])
        np.testing.assert_array_almost_equal(confidence, expected_confidence)

    def test_calculate_confidence_intervals_multiclass(self):
        """Test confidence interval calculation for multiclass."""
        model = AdvancedPredictiveModel()

        # Create sample multiclass probabilities
        probabilities = np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])

        confidence = model._calculate_confidence_intervals(probabilities)

        assert len(confidence) == 3
        assert all(0 <= c <= 1 for c in confidence)
        # Should be max probability for each sample
        expected_confidence = np.array([0.6, 0.7, 0.4])
        np.testing.assert_array_almost_equal(confidence, expected_confidence)

    def test_update_model_metadata(self):
        """Test updating model metadata."""
        model = AdvancedPredictiveModel()
        model.feature_columns = ['feature1', 'feature2']

        model._update_model_metadata()

        assert 'training_date' in model.model_metadata
        assert 'model_version' in model.model_metadata
        assert 'feature_count' in model.model_metadata
        assert model.model_metadata['feature_count'] == 2

    def test_error_handling_training_exception(self):
        """Test error handling for training exceptions."""
        model = AdvancedPredictiveModel()

        with patch.object(model, '_create_individual_models', side_effect=Exception("Model creation failed")):
            training_data = self.sample_training_data()

            with pytest.raises(ModelTrainingError, match="Failed to train predictive models"):
                model.train_predictive_models(training_data, 'target')

    def test_error_handling_prediction_exception(self):
        """Test error handling for prediction exceptions."""
        model = AdvancedPredictiveModel()
        model.is_trained = True
        model.ensemble = Mock()
        model.ensemble.predict.side_effect = Exception("Prediction failed")
        model.feature_columns = ['feature1', 'feature2']

        prediction_data = self.sample_prediction_data()

        with pytest.raises(PredictionError, match="Failed to make predictions"):
            model.predict_game_outcome(prediction_data)