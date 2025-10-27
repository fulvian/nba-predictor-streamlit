#!/usr/bin/env python3
"""
🚀 Advanced Predictive Model for NBA Analytics

Advanced ensemble ML system using XGBoost and multi-model architecture
with weighted voting for improved prediction accuracy.

Author: NBA Predictive Analytics System
Task ID: nba-predictive-analytics-2024
"""

from __future__ import annotations
import logging
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainingError(Exception):
    """Custom exception for model training operations."""
    pass


class PredictionError(Exception):
    """Custom exception for model prediction operations."""
    pass


@dataclass
class PredictionResult:
    """Results from model prediction."""
    predicted_class: int
    predicted_probability: float
    confidence_interval: Tuple[float, float]
    model_weights: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class ModelMetrics:
    """Metrics for model performance evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_scores: List[float]
    training_time: float


class AdvancedPredictiveModel:
    """
    Advanced predictive model using ensemble methods for NBA games.

    Combines multiple ML models with weighted voting to improve
    prediction accuracy and provide confidence intervals.

    Attributes:
        models: Dictionary of trained models
        ensemble_weights: Weights for voting
        feature_columns: List of feature column names
        scaler: Feature scaler for preprocessing
        label_encoder: Label encoder for target variables
        metrics: Model performance metrics
    """

    def __init__(
        self,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Initialize the advanced predictive model.

        Args:
            model_configs: Configuration parameters for models

        Example:
            >>> model = AdvancedPredictiveModel()
            >>> metrics = model.train_predictive_models(
            ...     training_data, target_column='home_win'
            ... )
        """
        # Set default model configurations
        default_configs = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'objective': 'binary:logistic'
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000,
                'C': 1.0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        }

        self.model_configs = model_configs or default_configs

        # Initialize models dictionary
        self.models: Dict[str, Any] = {}
        self.ensemble_weights: Optional[Dict[str, float]] = None
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.metrics: Optional[ModelMetrics] = None
        self.is_trained: bool = False

        # Model storage path
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        logger.info("AdvancedPredictiveModel initialized successfully")

    def train_predictive_models(
        self,
        training_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Train multiple predictive models on NBA data.

        Args:
            training_data: DataFrame containing features and target
            target_column: Name of target column

        Returns:
            Dictionary containing training results and metrics

        Raises:
            ModelTrainingError: If training fails

        Example:
            >>> model = AdvancedPredictiveModel()
            >>> metrics = model.train_predictive_models(
            ...     training_data, target_column='home_win'
            ... )
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting ensemble model training for target: {target_column}")

            # Validate input data
            if training_data.empty:
                raise ModelTrainingError("Training data is empty")

            if target_column not in training_data.columns:
                raise ModelTrainingError(f"Target column '{target_column}' not found in data")

            # Prepare features and target
            X, y = self._prepare_training_data(training_data, target_column)

            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train individual models
            self._train_xgboost_model(X_train, y_train)
            self._train_logistic_regression_model(X_train, y_train)
            self._train_random_forest_model(X_train, y_train)

            # Create ensemble with weighted voting
            self._create_voting_ensemble()

            # Evaluate models
            self._evaluate_models(X_test, y_test)

            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()

            # Store metrics
            self.metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, self.ensemble.predict(X_test)),
                precision=precision_score(y_test, self.ensemble.predict(X_test), average='weighted'),
                recall=recall_score(y_test, self.ensemble.predict(X_test), average='weighted'),
                f1_score=f1_score(y_test, self.ensemble.predict(X_test), average='weighted'),
                roc_auc=roc_auc_score(y_test, self.ensemble.predict_proba(X_test)[:, 1]),
                cross_val_scores=cross_val_score(
                    self.ensemble, X, y, cv=5, scoring='accuracy'
                ).tolist(),
                training_time=training_time
            )

            # Mark as trained
            self.is_trained = True

            logger.info(
                f"Ensemble training completed in {training_time:.2f}s - "
                f"Accuracy: {self.metrics.accuracy:.3f}"
            )

            return {
                'status': 'success',
                'metrics': {
                    'accuracy': self.metrics.accuracy,
                    'precision': self.metrics.precision,
                    'recall': self.metrics.recall,
                    'f1_score': self.metrics.f1_score,
                    'roc_auc': self.metrics.roc_auc,
                    'cross_val_scores': self.metrics.cross_val_scores,
                    'training_time': training_time
                },
                'ensemble_weights': self.ensemble_weights
            }

        except Exception as e:
            logger.error(
                "Model training failed",
                extra={
                    "target_column": target_column,
                    "data_shape": training_data.shape,
                    "error": str(e)
                }
            )
            raise ModelTrainingError(f"Failed to train models: {str(e)}") from e

    def predict_game_outcome(
        self,
        game_features: pd.DataFrame,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Predict game outcomes with confidence intervals.

        Args:
            game_features: DataFrame containing game features
            return_confidence: Whether to return confidence intervals

        Returns:
            DataFrame with predictions and optional confidence intervals

        Raises:
            ValueError: If model is not trained or features are invalid

        Example:
            >>> predictions = model.predict_game_outcome(features_df)
            >>> print(predictions['predicted_class'])
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Validate features
            if game_features.empty:
                raise ValueError("Game features DataFrame is empty")

            # Ensure features match training data
            if not set(game_features.columns).issubset(set(self.feature_columns)):
                missing_features = set(self.feature_columns) - set(game_features.columns)
                raise ValueError(f"Missing features: {missing_features}")

            # Prepare features (scaling)
            if self.scaler is not None:
                X = self.scaler.transform(game_features[self.feature_columns])
            else:
                X = game_features[self.feature_columns]

            # Get predictions
            y_pred = self.ensemble.predict(X)
            y_proba = self.ensemble.predict_proba(X)

            # Calculate confidence intervals (using binomial proportion confidence intervals)
            n = len(y_pred)
            confidence_level = 0.95
            z_score = 1.96  # For 95% confidence

            confidence_intervals = []
            for i in range(len(y_pred)):
                p = y_proba[i, 1]  # Probability of positive class
                if p == 0 or p == 1:
                    # Edge case: no variance
                    ci_lower = max(0.0, p - z_score * np.sqrt(p * (1 - p) / n))
                    ci_upper = min(1.0, p + z_score * np.sqrt(p * (1 - p) / n))
                else:
                    ci_lower = max(0.0, p - z_score * np.sqrt(p * (1 - p) / n))
                    ci_upper = min(1.0, p + z_score * np.sqrt(p * (1 - p) / n))

                confidence_intervals.append((ci_lower, ci_upper))

            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_class': y_pred,
                'predicted_probability': y_proba[:, 1],
                'prediction_time': datetime.now().isoformat()
            })

            if return_confidence:
                results['confidence_lower'] = [ci[0] for ci in confidence_intervals]
                results['confidence_upper'] = [ci[1] for ci in confidence_intervals]
                results['confidence_width'] = [
                    ci[1] - ci[0] for ci in confidence_intervals
                ]

            # Add model weights information
            if self.ensemble_weights:
                results['model_contribution'] = {
                    model: weight
                    for model, weight in self.ensemble_weights.items()
                }

            logger.info(f"Generated {len(results)} predictions")
            return results

        except Exception as e:
            logger.error(
                "Prediction failed",
                extra={
                    "features_shape": game_features.shape,
                    "return_confidence": return_confidence,
                    "error": str(e)
                }
            )
            raise ValueError(f"Failed to predict game outcomes: {str(e)}") from e

    def save_model(
        self,
        filepath: Optional[str] = None,
        include_preprocessors: bool = True
    ) -> str:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (default: auto-generated)
            include_preprocessors: Whether to save preprocessing objects

        Returns:
            Path to saved model file

        Example:
            >>> model_path = model.save_model('nba_predictor_v1.pkl')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        try:
            # Generate default filepath if not provided
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = str(self.model_dir / f"nba_predictive_model_{timestamp}.pkl")

            model_data = {
                'models': self.models,
                'ensemble_weights': self.ensemble_weights,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'model_configs': self.model_configs,
                'metrics': self.metrics
            }

            if include_preprocessors:
                model_data.update({
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder
                })

            # Save model
            joblib.dump(model_data, filepath)

            logger.info(f"Model saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(
        self,
        filepath: str,
        include_preprocessors: bool = True
    ) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model file
            include_preprocessors: Whether to load preprocessing objects

        Example:
            >>> model.load_model('nba_predictor_v1.pkl')
        """
        try:
            # Load model data
            model_data = joblib.load(filepath)

            # Restore model components
            self.models = model_data['models']
            self.ensemble_weights = model_data['ensemble_weights']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.model_configs = model_data['model_configs']
            self.metrics = model_data.get('metrics')

            if include_preprocessors:
                self.scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_feature_importance(
        self,
        method: str = 'average'
    ) -> Dict[str, float]:
        """
        Get feature importance from trained models.

        Args:
            method: Method for combining importance ('average', 'max', 'xgboost')

        Returns:
            Dictionary mapping feature names to importance scores

        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance['most_important_feature'])
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        try:
            importance_scores = {}

            # Get importance from XGBoost (tree-based)
            if 'xgboost' in self.models:
                xgb_model = self.models['xgboost']
                importance_scores['xgboost'] = dict(zip(
                    self.feature_columns,
                    xgb_model.feature_importances_
                ))

            # Get importance from Random Forest
            if 'random_forest' in self.models:
                rf_model = self.models['random_forest']
                importance_scores['random_forest'] = dict(zip(
                    self.feature_columns,
                    rf_model.feature_importances_
                ))

            # Combine importance scores
            if method == 'average' and importance_scores:
                # Average across all available models
                avg_importance = {}
                for feature in self.feature_columns:
                    values = [
                        scores.get(feature, 0.0)
                        for scores in importance_scores.values()
                    ]
                    if values:
                        avg_importance[feature] = np.mean(values)

                return avg_importance

            elif method == 'max' and importance_scores:
                # Maximum across all models
                max_importance = {}
                for feature in self.feature_columns:
                    values = [
                        scores.get(feature, 0.0)
                        for scores in importance_scores.values()
                    ]
                    if values:
                        max_importance[feature] = max(values)

                return max_importance

            elif method == 'xgboost' and 'xgboost' in importance_scores:
                return importance_scores['xgboost']

            else:
                logger.warning(f"Unknown importance method: {method}")
                return {}

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def update_model_weights(
        self,
        new_weights: Dict[str, float]
    ) -> None:
        """
        Update ensemble weights.

        Args:
            new_weights: Dictionary mapping model names to weights

        Example:
            >>> model.update_model_weights({'xgboost': 2.0, 'logistic': 1.5, 'random_forest': 1.0})
        """
        try:
            # Validate weights
            for model_name in new_weights:
                if model_name not in self.models:
                    raise ValueError(f"Model '{model_name}' not found in trained models")
                if new_weights[model_name] <= 0:
                    raise ValueError(f"Weight must be positive for model '{model_name}'")

            # Normalize weights
            total_weight = sum(new_weights.values())
            self.ensemble_weights = {
                name: weight / total_weight
                for name, weight in new_weights.items()
            }

            # Update VotingClassifier weights if needed
            if hasattr(self.ensemble, 'weights'):
                self.ensemble.weights = list(self.ensemble_weights.values())

            logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            raise

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary information.

        Returns:
            Dictionary containing model details and performance metrics

        Example:
            >>> summary = model.get_model_summary()
            >>> print(f"Model accuracy: {summary['metrics']['accuracy']:.3f}")
        """
        if not self.is_trained:
            return {
                'status': 'not_trained',
                'message': 'Model must be trained first'
            }

        return {
            'status': 'trained',
            'models': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_columns),
            'metrics': {
                'accuracy': self.metrics.accuracy if self.metrics else 0,
                'precision': self.metrics.precision if self.metrics else 0,
                'recall': self.metrics.recall if self.metrics else 0,
                'f1_score': self.metrics.f1_score if self.metrics else 0,
                'roc_auc': self.metrics.roc_auc if self.metrics else 0,
                'cross_val_mean': np.mean(self.metrics.cross_val_scores) if self.metrics else 0,
                'training_time': self.metrics.training_time if self.metrics else 0
            },
            'model_configs': self.model_configs
        }

    # Private helper methods

    def _prepare_training_data(
        self,
        training_data: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        try:
            logger.info("Preparing training data")

            # Separate features and target
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]

            # Remove non-numeric columns for now
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            # Store feature column names
            self.feature_columns = X.columns.tolist()

            # Initialize preprocessors
            if self.scaler is None:
                self.scaler = StandardScaler()

            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()

            # Encode target variable if needed
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)

            logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise

    def _train_xgboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """Train XGBoost model."""
        try:
            logger.info("Training XGBoost model")

            config = self.model_configs['xgboost']
            # Ensure config is a proper mapping
            if not isinstance(config, dict):
                config = {}

            self.models['xgboost'] = xgb.XGBClassifier(**config)
            self.models['xgboost'].fit(X_train, y_train)

            logger.info("XGBoost model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train XGBoost model: {e}")
            raise ModelTrainingError(f"XGBoost training failed: {str(e)}") from e

    def _train_logistic_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """Train Logistic Regression model."""
        try:
            logger.info("Training Logistic Regression model")

            config = self.model_configs['logistic_regression']

            self.models['logistic_regression'] = LogisticRegression(**config)
            self.models['logistic_regression'].fit(X_train, y_train)

            logger.info("Logistic Regression model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train Logistic Regression model: {e}")
            raise ModelTrainingError(f"Logistic Regression training failed: {str(e)}") from e

    def _train_random_forest_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """Train Random Forest model."""
        try:
            logger.info("Training Random Forest model")

            config = self.model_configs['random_forest']

            self.models['random_forest'] = RandomForestClassifier(**config)
            self.models['random_forest'].fit(X_train, y_train)

            logger.info("Random Forest model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train Random Forest model: {e}")
            raise ModelTrainingError(f"Random Forest training failed: {str(e)}") from e

    def _create_voting_ensemble(self) -> None:
        """Create weighted voting ensemble."""
        try:
            logger.info("Creating voting ensemble")

            # Define default weights (emphasize XGBoost)
            default_weights = {
                'xgboost': 2.0,
                'logistic_regression': 1.5,
                'random_forest': 1.0
            }

            self.ensemble_weights = default_weights

            # Create ensemble with soft voting (probability-based)
            self.ensemble = VotingClassifier(
                estimators=[
                    ('xgboost', self.models['xgboost']),
                    ('logistic_regression', self.models['logistic_regression']),
                    ('random_forest', self.models['random_forest'])
                ],
                voting='soft',
                weights=list(self.ensemble_weights.values())
            )

            logger.info("Voting ensemble created successfully")

        except Exception as e:
            logger.error(f"Failed to create voting ensemble: {e}")
            raise ModelTrainingError(f"Ensemble creation failed: {str(e)}") from e

    def _evaluate_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> None:
        """Evaluate model performance."""
        try:
            logger.info("Evaluating model performance")

            # Scale test features
            if self.scaler is not None:
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_test_scaled = X_test

            # Get predictions
            y_pred = self.ensemble.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Model evaluation completed - Accuracy: {accuracy:.3f}")

        except Exception as e:
            logger.error(f"Failed to evaluate models: {e}")
            raise

    def _calculate_quality_score(
        self,
        data: pd.DataFrame,
        missing_values: Dict[str, int],
        duplicate_rows: int,
        validation_errors: List[str]
    ) -> float:
        """Calculate overall data quality score."""
        try:
            base_score = 1.0

            # Penalize missing values
            total_cells = len(data) * len(data.columns)
            missing_cells = sum(missing_values.values())
            missing_penalty = missing_cells / total_cells if total_cells > 0 else 0
            base_score -= missing_penalty

            # Penalize duplicates
            duplicate_penalty = duplicate_rows / len(data) if len(data) > 0 else 0
            base_score -= duplicate_penalty

            # Penalize validation errors
            error_penalty = len(validation_errors) * 0.1
            base_score -= error_penalty

            return max(0.0, min(1.0, base_score))

        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.0