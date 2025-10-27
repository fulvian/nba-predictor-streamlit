"""
Auto Model Retrainer for NBA Predictive Analytics System.

This module implements automated model retraining with performance monitoring,
using Context7 compliant patterns for model persistence and incremental learning.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from advanced_predictive_model import AdvancedPredictiveModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingError(Exception):
    """Custom exception for model retraining failures."""
    pass


class AutoModelRetrainer:
    """
    Automated model retraining system for NBA predictions.

    Monitors model performance and triggers retraining
    when accuracy degrades below specified thresholds.

    Attributes:
        model: Current trained model
        performance_threshold: Minimum accuracy threshold
        retrain_interval: Days between retraining checks
        model_save_path: Path to save trained models
        performance_history: Historical performance metrics
    """

    def __init__(
        self,
        model: AdvancedPredictiveModel,
        performance_threshold: float = 0.75,
        retrain_interval: int = 7,
        model_save_path: str = "saved_models"
    ) -> None:
        """
        Initialize the auto retrainer.

        Args:
            model: Current trained model instance
            performance_threshold: Minimum accuracy threshold (default: 0.75)
            retrain_interval: Days between retraining checks (default: 7)
            model_save_path: Directory path to save models (default: "saved_models")

        Example:
            >>> model = AdvancedPredictiveModel()
            >>> retrainer = AutoModelRetrainer(
            ...     model=model,
            ...     performance_threshold=0.80,
            ...     retrain_interval=5
            ... )
            >>> retrainer.check_retrain_needed(predictions, actuals)
        """
        self.model = model
        self.performance_threshold = performance_threshold
        self.retrain_interval = retrain_interval
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.last_retrain_date: Optional[datetime] = None
        self._retrain_counter = 0
        self._initialization_date = datetime.now()

        # Model versioning
        self._model_version = 1
        self._current_model_path = self.model_save_path / f"model_v{self._model_version}.joblib"

        logger.info(
            "AutoModelRetrainer initialized",
            extra={
                "performance_threshold": performance_threshold,
                "retrain_interval": retrain_interval,
                "model_save_path": str(self.model_save_path)
            }
        )

    def check_retrain_needed(
        self,
        recent_predictions: pd.DataFrame,
        actual_results: pd.DataFrame
    ) -> bool:
        """
        Check if model retraining is needed.

        Args:
            recent_predictions: DataFrame with recent predictions
            actual_results: DataFrame with actual outcomes

        Returns:
            True if retraining is needed, False otherwise

        Raises:
            ValueError: If input dataframes are empty or mismatched

        Example:
            >>> predictions = pd.DataFrame({"prediction": [1, 0, 1]})
            >>> actuals = pd.DataFrame({"actual": [1, 0, 0]})
            >>> retrainer.check_retrain_needed(predictions, actuals)
            False
        """
        try:
            if recent_predictions.empty or actual_results.empty:
                logger.warning(
                    "Empty dataframes provided for retraining check",
                    extra={
                        "predictions_shape": recent_predictions.shape,
                        "actuals_shape": actual_results.shape
                    }
                )
                return False

            if len(recent_predictions) != len(actual_results):
                raise ValueError(
                    f"Dataframe size mismatch: predictions {len(recent_predictions)} "
                    f"vs actuals {len(actual_results)}"
                )

            # Calculate current performance
            current_accuracy = self._calculate_accuracy(
                recent_predictions, actual_results
            )

            # Store performance metrics
            performance_record = {
                "timestamp": datetime.now(),
                "accuracy": current_accuracy,
                "sample_size": len(recent_predictions),
                "threshold": self.performance_threshold
            }
            self.performance_history.append(performance_record)

            logger.info(
                "Performance evaluation completed",
                extra={
                    "current_accuracy": current_accuracy,
                    "threshold": self.performance_threshold,
                    "sample_size": len(recent_predictions),
                    "performance_drop": self.performance_threshold - current_accuracy
                }
            )

            # Check if retraining conditions are met
            retrain_needed = False

            # Condition 1: Accuracy below threshold
            if current_accuracy < self.performance_threshold:
                retrain_needed = True
                logger.info(
                    "Retraining needed: accuracy below threshold",
                    extra={
                        "current_accuracy": current_accuracy,
                        "threshold": self.performance_threshold
                    }
                )

            # Condition 2: Time-based retraining
            days_since_last_retrain = self._get_days_since_last_retrain()
            if days_since_last_retrain >= self.retrain_interval:
                retrain_needed = True
                logger.info(
                    "Retraining needed: time interval reached",
                    extra={
                        "days_since_last": days_since_last_retrain,
                        "interval": self.retrain_interval
                    }
                )

            # Condition 3: Performance degradation trend
            if self._check_performance_degradation():
                retrain_needed = True
                logger.info(
                    "Retraining needed: performance degradation detected"
                )

            return retrain_needed

        except ValueError:
            # Re-raise ValueError for mismatched sizes to allow test validation
            raise
        except Exception as e:
            logger.error(
                "Error checking retrain conditions",
                extra={
                    "error": str(e),
                    "predictions_shape": recent_predictions.shape,
                    "actuals_shape": actual_results.shape
                }
            )
            raise RetrainingError("Failed to check retrain conditions") from e

    def retrain_models(
        self,
        new_data: pd.DataFrame,
        target_column: str = "target"
    ) -> Dict[str, Any]:
        """
        Retrain models with new data.

        Args:
            new_data: New training data with features and target
            target_column: Name of target column (default: "target")

        Returns:
            Dictionary with retraining results and metrics

        Raises:
            ValueError: If target column not found in data
            RetrainingError: If retraining fails

        Example:
            >>> new_training_data = pd.DataFrame({
            ...     "feature1": [1, 2, 3],
            ...     "feature2": [4, 5, 6],
            ...     "target": [0, 1, 0]
            ... })
            >>> result = retrainer.retrain_models(new_training_data, "target")
            >>> print(result["status"])
            'success'
        """
        try:
            if target_column not in new_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            if new_data.empty:
                raise ValueError("Training data is empty")

            logger.info(
                "Starting model retraining",
                extra={
                    "data_shape": new_data.shape,
                    "target_column": target_column,
                    "model_version": self._model_version + 1
                }
            )

            start_time = time.time()

            # Save current model as backup
            if self._current_model_path.exists():
                backup_path = self.model_save_path / f"model_v{self._model_version}_backup.joblib"
                joblib.dump(self.model, backup_path)
                logger.info(f"Current model backed up to {backup_path}")

            # Retrain the model
            retraining_results = self.model.train_predictive_models(
                training_data=new_data,
                target_column=target_column
            )

            # Update model version
            self._model_version += 1
            self._current_model_path = self.model_save_path / f"model_v{self._model_version}.joblib"

            # Save new model using joblib (Context7 compliant)
            joblib.dump(self.model, self._current_model_path, protocol=5)

            # Update retraining metadata
            self.last_retrain_date = datetime.now()
            self._retrain_counter += 1

            retraining_time = time.time() - start_time

            # Prepare results
            results = {
                "status": "success",
                "model_version": self._model_version,
                "retraining_time": retraining_time,
                "training_samples": len(new_data),
                "feature_count": len(new_data.columns) - 1,  # Exclude target
                "retrain_counter": self._retrain_counter,
                "model_path": str(self._current_model_path),
                "training_metrics": retraining_results,
                "timestamp": self.last_retrain_date.isoformat()
            }

            logger.info(
                "Model retraining completed successfully",
                extra={
                    "model_version": self._model_version,
                    "retraining_time": retraining_time,
                    "training_samples": len(new_data)
                }
            )

            return results

        except ValueError:
            # Re-raise ValueError for missing target and empty data to allow test validation
            raise
        except Exception as e:
            logger.error(
                "Model retraining failed",
                extra={
                    "data_size": len(new_data),
                    "error": str(e)
                }
            )
            raise RetrainingError("Failed to retrain models") from e

    def validate_retrained_models(
        self,
        validation_data: pd.DataFrame,
        target_column: str = "target"
    ) -> Dict[str, Any]:
        """
        Validate newly retrained models.

        Args:
            validation_data: Validation dataset with features and target
            target_column: Name of target column (default: "target")

        Returns:
            Dictionary with validation results and metrics

        Raises:
            ValueError: If target column not found or data is empty

        Example:
            >>> validation_data = pd.DataFrame({
            ...     "feature1": [7, 8, 9],
            ...     "feature2": [10, 11, 12],
            ...     "target": [1, 0, 1]
            ... })
            >>> results = retrainer.validate_retrained_models(validation_data)
            >>> print(results["accuracy"])
            0.67
        """
        try:
            if target_column not in validation_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in validation data")

            if validation_data.empty:
                raise ValueError("Validation data is empty")

            logger.info(
                "Starting model validation",
                extra={
                    "validation_samples": len(validation_data),
                    "feature_count": len(validation_data.columns) - 1,
                    "model_version": self._model_version
                }
            )

            # Prepare validation data
            X_val = validation_data.drop(columns=[target_column])
            y_val = validation_data[target_column]

            # Generate predictions
            predictions = self.model.predict_game_outcome(
                game_features=X_val,
                return_confidence=True
            )

            # Extract predicted classes
            if 'prediction' in predictions.columns:
                y_pred = predictions['prediction']
            else:
                # Fallback to first column if 'prediction' not found
                y_pred = predictions.iloc[:, 0]

            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)

            # Detailed classification report
            class_report = classification_report(y_val, y_pred, output_dict=True)

            # Confusion matrix
            conf_matrix = confusion_matrix(y_val, y_pred)

            # Prepare validation results
            validation_results = {
                "accuracy": accuracy,
                "validation_samples": len(validation_data),
                "model_version": self._model_version,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "performance_vs_threshold": {
                    "current_accuracy": accuracy,
                    "threshold": self.performance_threshold,
                    "meets_threshold": accuracy >= self.performance_threshold
                },
                "timestamp": datetime.now().isoformat()
            }

            # Check if validation meets performance threshold
            if accuracy >= self.performance_threshold:
                logger.info(
                    "Model validation passed",
                    extra={
                        "accuracy": accuracy,
                        "threshold": self.performance_threshold,
                        "model_version": self._model_version
                    }
                )
            else:
                logger.warning(
                    "Model validation below threshold",
                    extra={
                        "accuracy": accuracy,
                        "threshold": self.performance_threshold,
                        "model_version": self._model_version
                    }
                )

            return validation_results

        except ValueError:
            # Re-raise ValueError for missing target to allow test validation
            raise
        except Exception as e:
            logger.error(
                "Model validation failed",
                extra={
                    "validation_samples": len(validation_data),
                    "error": str(e)
                }
            )
            raise RetrainingError("Failed to validate retrained models") from e

    def get_performance_history(self) -> pd.DataFrame:
        """
        Get historical performance metrics.

        Returns:
            DataFrame with performance history

        Example:
            >>> history = retrainer.get_performance_history()
            >>> print(history.columns.tolist())
            ['timestamp', 'accuracy', 'sample_size', 'threshold']
        """
        if not self.performance_history:
            return pd.DataFrame(columns=["timestamp", "accuracy", "sample_size", "threshold"])

        return pd.DataFrame(self.performance_history)

    def load_model(
        self,
        model_path: str,
        version: Optional[int] = None
    ) -> AdvancedPredictiveModel:
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model file
            version: Optional model version number

        Returns:
            Loaded model instance

        Example:
            >>> model = retrainer.load_model("saved_models/model_v1.joblib")
            >>> print(model.is_trained)
            True
        """
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model using joblib (Context7 compliant)
            loaded_model: AdvancedPredictiveModel = joblib.load(path)

            if version:
                self._model_version = version
                self._current_model_path = path

            logger.info(
                "Model loaded successfully",
                extra={
                    "model_path": str(path),
                    "version": version or "unknown"
                }
            )

            return loaded_model

        except FileNotFoundError:
            # Re-raise FileNotFoundError directly for test expectations
            raise
        except Exception as e:
            logger.error(
                "Failed to load model",
                extra={
                    "model_path": model_path,
                    "error": str(e)
                }
            )
            raise RetrainingError("Failed to load model") from e

    def _calculate_accuracy(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> float:
        """Calculate accuracy between predictions and actual results."""
        try:
            # Extract prediction column (assume first column is prediction)
            if predictions.shape[1] > 0:
                pred_col = predictions.iloc[:, 0]
            else:
                raise ValueError("No prediction column found")

            # Extract actual column (assume first column is actual)
            if actuals.shape[1] > 0:
                actual_col = actuals.iloc[:, 0]
            else:
                raise ValueError("No actual column found")

            accuracy: float = accuracy_score(actual_col, pred_col)
            return accuracy

        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0

    def _get_days_since_last_retrain(self) -> int:
        """Calculate days since last retraining."""
        if self.last_retrain_date is None:
            # For never-trained models, check if this is initial setup
            # If we have recent performance data, assume model was recently trained
            if self.performance_history:
                last_performance = self.performance_history[-1]
                time_since_performance = (datetime.now() - last_performance["timestamp"]).days
                return int(time_since_performance)
            # For truly new models, return 0 to avoid immediate time-based retraining
            return 0

        return (datetime.now() - self.last_retrain_date).days

    def _check_performance_degradation(self) -> bool:
        """Check if performance is degrading over time."""
        if len(self.performance_history) < 3:
            return False

        # Get last 3 performance records
        recent_performances = self.performance_history[-3:]
        accuracies = [record["accuracy"] for record in recent_performances]

        # Check if performance is consistently decreasing
        if accuracies[0] > accuracies[1] > accuracies[2]:
            degradation: float = accuracies[0] - accuracies[2]
            is_degraded: bool = degradation > 0.05  # 5% degradation threshold
            return is_degraded

        return False

    def get_retrainer_status(self) -> Dict[str, Any]:
        """
        Get current status of the auto retrainer.

        Returns:
            Dictionary with retrainer status information

        Example:
            >>> status = retrainer.get_retrainer_status()
            >>> print(status["model_version"])
            1
        """
        return {
            "model_version": self._model_version,
            "performance_threshold": self.performance_threshold,
            "retrain_interval": self.retrain_interval,
            "retrain_counter": self._retrain_counter,
            "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            "current_model_path": str(self._current_model_path),
            "performance_history_count": len(self.performance_history),
            "model_save_path": str(self.model_save_path)
        }