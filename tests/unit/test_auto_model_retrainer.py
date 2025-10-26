"""
Unit tests for AutoModelRetrainer.

Tests model retraining functionality, performance monitoring,
and automated retraining triggers.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from auto_model_retrainer import AutoModelRetrainer, RetrainingError
from advanced_predictive_model import AdvancedPredictiveModel


class TestAutoModelRetrainer:
    """Test suite for AutoModelRetrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock AdvancedPredictiveModel."""
        model = Mock(spec=AdvancedPredictiveModel)
        model.is_trained = True
        model.train_predictive_models.return_value = {
            "status": "success",
            "models_trained": 3
        }
        model.predict_game_outcome.return_value = pd.DataFrame({
            "prediction": [1, 0, 1, 0, 1],
            "confidence": [0.8, 0.6, 0.9, 0.7, 0.85]
        })
        return model

    @pytest.fixture
    def retrainer(self, mock_model):
        """Create AutoModelRetrainer instance with mock model."""
        return AutoModelRetrainer(
            model=mock_model,
            performance_threshold=0.75,
            retrain_interval=7
        )

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data."""
        return pd.DataFrame({
            "prediction": [1, 0, 1, 0, 1],
            "confidence": [0.8, 0.6, 0.9, 0.7, 0.85]
        })

    @pytest.fixture
    def sample_actuals(self):
        """Create sample actual results."""
        return pd.DataFrame({
            "actual": [1, 0, 0, 0, 1]
        })

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [1, 0, 1, 0, 1]
        })

    def test_retrainer_initialization(self, retrainer):
        """Test AutoModelRetrainer initialization."""
        assert retrainer.performance_threshold == 0.75
        assert retrainer.retrain_interval == 7
        assert retrainer._retrain_counter == 0
        assert retrainer._model_version == 1
        assert len(retrainer.performance_history) == 0
        assert retrainer.last_retrain_date is None

    def test_check_retrain_needed_high_accuracy(self, retrainer, sample_predictions, sample_actuals):
        """Test retrain check when accuracy is above threshold."""
        # Mock accuracy calculation to return high value
        with patch.object(retrainer, '_calculate_accuracy', return_value=0.85):
            result = retrainer.check_retrain_needed(sample_predictions, sample_actuals)
            assert result is False

    def test_check_retrain_needed_low_accuracy(self, retrainer, sample_predictions, sample_actuals):
        """Test retrain check when accuracy is below threshold."""
        # Mock accuracy calculation to return low value
        with patch.object(retrainer, '_calculate_accuracy', return_value=0.65):
            result = retrainer.check_retrain_needed(sample_predictions, sample_actuals)
            assert result is True

    def test_check_retrain_needed_time_based(self, retrainer, sample_predictions, sample_actuals):
        """Test time-based retraining trigger."""
        # Set last retrain date to 10 days ago
        retrainer.last_retrain_date = datetime.now() - timedelta(days=10)

        # Mock accuracy calculation to return high value (shouldn't trigger based on accuracy)
        with patch.object(retrainer, '_calculate_accuracy', return_value=0.85):
            result = retrainer.check_retrain_needed(sample_predictions, sample_actuals)
            assert result is True  # Should trigger due to time interval

    def test_check_retrain_needed_empty_dataframes(self, retrainer):
        """Test retrain check with empty dataframes."""
        empty_preds = pd.DataFrame()
        empty_actuals = pd.DataFrame()

        result = retrainer.check_retrain_needed(empty_preds, empty_actuals)
        assert result is False

    def test_check_retrain_needed_mismatched_sizes(self, retrainer):
        """Test retrain check with mismatched dataframe sizes."""
        preds = pd.DataFrame({"prediction": [1, 0, 1]})
        actuals = pd.DataFrame({"actual": [1, 0]})  # Different size

        with pytest.raises(ValueError, match="Dataframe size mismatch"):
            retrainer.check_retrain_needed(preds, actuals)

    def test_retrain_models_success(self, retrainer, sample_training_data):
        """Test successful model retraining."""
        # Reset mock call count
        retrainer.model.train_predictive_models.reset_mock()

        with patch('auto_model_retrainer.joblib.dump') as mock_dump:
            result = retrainer.retrain_models(sample_training_data, "target")

            assert result["status"] == "success"
            assert result["training_samples"] == 5
            assert result["feature_count"] == 2
            assert result["model_version"] == 2
            assert result["retrain_counter"] == 1
            assert "retraining_time" in result

            # Verify model training was called
            retrainer.model.train_predictive_models.assert_called_once_with(
                training_data=sample_training_data, target_column="target"
            )

            # Verify model was saved
            mock_dump.assert_called_once()

    def test_retrain_models_missing_target(self, retrainer):
        """Test retraining with missing target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Target column 'target' not found"):
            retrainer.retrain_models(data)

    def test_retrain_models_empty_data(self, retrainer):
        """Test retraining with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Target column 'target' not found in data"):
            retrainer.retrain_models(empty_data)

    def test_validate_retrained_models_success(self, retrainer, sample_training_data):
        """Test successful model validation."""
        # Mock sklearn.metrics.accuracy_score function instead
        with patch('auto_model_retrainer.accuracy_score', return_value=0.85):
            result = retrainer.validate_retrained_models(sample_training_data, "target")

            assert result["accuracy"] == 0.85
            assert result["validation_samples"] == 5
            assert result["model_version"] == 1
            assert "classification_report" in result
            assert "confusion_matrix" in result
            assert result["performance_vs_threshold"]["meets_threshold"] is True

    def test_validate_retrained_models_below_threshold(self, retrainer, sample_training_data):
        """Test validation when accuracy is below threshold."""
        # Mock sklearn.metrics.accuracy_score function instead
        with patch('auto_model_retrainer.accuracy_score', return_value=0.65):
            result = retrainer.validate_retrained_models(sample_training_data, "target")

            assert result["accuracy"] == 0.65
            assert result["performance_vs_threshold"]["meets_threshold"] is False

    def test_validate_retrained_models_missing_target(self, retrainer):
        """Test validation with missing target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Target column 'target' not found"):
            retrainer.validate_retrained_models(data)

    def test_get_performance_history_empty(self, retrainer):
        """Test getting performance history when empty."""
        history = retrainer.get_performance_history()
        assert history.empty
        assert list(history.columns) == ["timestamp", "accuracy", "sample_size", "threshold"]

    def test_get_performance_history_with_data(self, retrainer):
        """Test getting performance history with data."""
        # Add some performance records
        retrainer.performance_history = [
            {"timestamp": datetime.now(), "accuracy": 0.8, "sample_size": 10, "threshold": 0.75},
            {"timestamp": datetime.now(), "accuracy": 0.85, "sample_size": 12, "threshold": 0.75}
        ]

        history = retrainer.get_performance_history()
        assert len(history) == 2
        assert "accuracy" in history.columns
        assert history["accuracy"].tolist() == [0.8, 0.85]

    def test_load_model_success(self, retrainer):
        """Test successful model loading."""
        mock_loaded_model = Mock(spec=AdvancedPredictiveModel)

        with patch('auto_model_retrainer.joblib.load', return_value=mock_loaded_model):
            with patch('auto_model_retrainer.Path.exists', return_value=True):
                result = retrainer.load_model("test_model.joblib", version=5)

                assert result == mock_loaded_model
                assert retrainer._model_version == 5

    def test_load_model_file_not_found(self, retrainer):
        """Test model loading when file doesn't exist."""
        with patch('auto_model_retrainer.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                retrainer.load_model("nonexistent_model.joblib")

    def test_get_retrainer_status(self, retrainer):
        """Test getting retrainer status."""
        status = retrainer.get_retrainer_status()

        assert status["model_version"] == 1
        assert status["performance_threshold"] == 0.75
        assert status["retrain_interval"] == 7
        assert status["retrain_counter"] == 0
        assert status["performance_history_count"] == 0
        assert "last_retrain_date" in status
        assert "current_model_path" in status
        assert "model_save_path" in status

    def test_calculate_accuracy_success(self, retrainer):
        """Test accuracy calculation with valid data."""
        predictions = pd.DataFrame({"prediction": [1, 0, 1, 0]})
        actuals = pd.DataFrame({"actual": [1, 0, 0, 0]})

        accuracy = retrainer._calculate_accuracy(predictions, actuals)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_calculate_accuracy_no_columns(self, retrainer):
        """Test accuracy calculation with empty dataframes."""
        empty_preds = pd.DataFrame()
        empty_actuals = pd.DataFrame()

        accuracy = retrainer._calculate_accuracy(empty_preds, empty_actuals)
        assert accuracy == 0.0

    def test_get_days_since_last_retrain_never_trained(self, retrainer):
        """Test days calculation when never trained."""
        days = retrainer._get_days_since_last_retrain()
        assert days == 0  # New models return 0 to avoid immediate time-based retraining

    def test_get_days_since_last_retrain_recent(self, retrainer):
        """Test days calculation with recent training."""
        retrainer.last_retrain_date = datetime.now() - timedelta(days=2)
        days = retrainer._get_days_since_last_retrain()
        assert days == 2

    def test_get_days_since_last_retrain_with_performance_history(self, retrainer):
        """Test days calculation with performance history but no last retrain date."""
        # Add performance history to simulate recent activity
        retrainer.performance_history = [
            {"timestamp": datetime.now() - timedelta(days=5), "accuracy": 0.8}
        ]
        days = retrainer._get_days_since_last_retrain()
        assert days == 5  # Should use performance history timestamp

    def test_check_performance_degradation_insufficient_data(self, retrainer):
        """Test performance degradation with insufficient history."""
        # Add only 2 records (less than required 3)
        retrainer.performance_history = [
            {"accuracy": 0.8},
            {"accuracy": 0.75}
        ]

        result = retrainer._check_performance_degradation()
        assert result is False

    def test_check_performance_degradation_detected(self, retrainer):
        """Test performance degradation detection."""
        # Add decreasing performance records
        retrainer.performance_history = [
            {"accuracy": 0.85},
            {"accuracy": 0.75},
            {"accuracy": 0.65}  # 20% degradation
        ]

        result = retrainer._check_performance_degradation()
        assert result is True

    def test_check_performance_degradation_not_detected(self, retrainer):
        """Test when performance degradation is not detected."""
        # Add stable performance records
        retrainer.performance_history = [
            {"accuracy": 0.8},
            {"accuracy": 0.82},
            {"accuracy": 0.81}
        ]

        result = retrainer._check_performance_degradation()
        assert result is False

    def test_retraining_error_handling(self, retrainer, sample_training_data):
        """Test error handling during retraining."""
        # Make model training fail
        retrainer.model.train_predictive_models.side_effect = Exception("Training failed")

        with pytest.raises(RetrainingError, match="Failed to retrain models"):
            retrainer.retrain_models(sample_training_data)

    def test_validation_error_handling(self, retrainer, sample_training_data):
        """Test error handling during validation."""
        # Make prediction fail
        retrainer.model.predict_game_outcome.side_effect = Exception("Prediction failed")

        with pytest.raises(RetrainingError, match="Failed to validate retrained models"):
            retrainer.validate_retrained_models(sample_training_data)