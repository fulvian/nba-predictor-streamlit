#!/usr/bin/env python3
"""
🤖 NBA Machine Learning Models
Context7-compliant ensemble models for NBA predictive analytics.
Implements XGBoost, Random Forest, and LSTM models with proper evaluation and explainability.
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for NBA ML models."""

    # XGBoost parameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 10
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1

    # LSTM parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32

    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Feature selection
    feature_importance_threshold: float = 0.01
    max_features: Optional[int] = None

class XGBoostModel:
    """XGBoost model for NBA predictions with SHAP explainability."""

    def __init__(self, config: ModelConfig, task_type: str = 'classification'):
        self.config = config
        self.task_type = task_type
        self.model = None
        self.feature_names = None
        self.shap_explainer = None

        logger.info(f"🚀 Initializing XGBoost {task_type} model")

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for XGBoost training."""
        # Handle missing values
        X = X.fillna(0)

        # Feature selection if specified
        if self.config.max_features and len(X.columns) > self.config.max_features:
            # Simple correlation-based selection for demo
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(self.config.max_features).index
            X = X[top_features]

        self.feature_names = list(X.columns)
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info("🎯 Training XGBoost model")

        X, y = self.prepare_data(X, y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y if self.task_type == 'classification' else None
        )

        # Initialize model
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                eval_metric='logloss'
            )
        else:
            self.model = xgb.XGBRegressor(
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                eval_metric='rmse'
            )

        # Train model
        self.model.fit(X_train, y_train)

        # Initialize SHAP explainer - use TreeExplainer for XGBoost
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)

        logger.info(f"✅ XGBoost training completed. Validation {self.task_type} score: {metrics.get('val_score', 'N/A')}")

        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': metrics,
            'shap_explainer': self.shap_explainer
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names].fillna(0)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.model is None or self.task_type != 'classification':
            raise ValueError("Model not trained or not a classification model")

        X = X[self.feature_names].fillna(0)
        return self.model.predict_proba(X)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP explanations for predictions."""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")

        X = X[self.feature_names].fillna(0)
        return self.shap_explainer(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def _calculate_metrics(self, y_train, y_train_pred, y_val, y_val_pred) -> Dict[str, float]:
        """Calculate appropriate metrics based on task type."""
        if self.task_type == 'classification':
            return {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
                'val_score': accuracy_score(y_val, y_val_pred)
            }
        else:
            return {
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'val_mse': mean_squared_error(y_val, y_val_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'val_score': r2_score(y_val, y_val_pred)
            }

class RandomForestModel:
    """Random Forest model for NBA predictions."""

    def __init__(self, config: ModelConfig, task_type: str = 'classification'):
        self.config = config
        self.task_type = task_type
        self.model = None
        self.feature_names = None

        logger.info(f"🌲 Initializing Random Forest {task_type} model")

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for Random Forest training."""
        X = X.fillna(0)

        if self.config.max_features and len(X.columns) > self.config.max_features:
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(self.config.max_features).index
            X = X[top_features]

        self.feature_names = list(X.columns)
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("🎯 Training Random Forest model")

        X, y = self.prepare_data(X, y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y if self.task_type == 'classification' else None
        )

        # Initialize model
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state
            )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)

        logger.info(f"✅ Random Forest training completed. Validation {self.task_type} score: {metrics.get('val_score', 'N/A')}")

        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': metrics
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names].fillna(0)
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def _calculate_metrics(self, y_train, y_train_pred, y_val, y_val_pred) -> Dict[str, float]:
        """Calculate appropriate metrics."""
        if self.task_type == 'classification':
            return {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
                'val_score': accuracy_score(y_val, y_val_pred)
            }
        else:
            return {
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'val_mse': mean_squared_error(y_val, y_val_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'val_score': r2_score(y_val, y_val_pred)
            }

class LSTMModel(nn.Module):
    """LSTM model for time series NBA predictions."""

    def __init__(self, config: ModelConfig, input_size: int):
        super(LSTMModel, self).__init__()
        self.config = config
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(config.lstm_dropout)
        self.fc = nn.Linear(config.lstm_hidden_size, 1)  # Binary classification/regression

        logger.info(f"🔥 Initializing LSTM model with input_size={input_size}")

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        output = self.dropout(last_output)
        output = self.fc(output)

        return output.squeeze()

class ModelTrainer:
    """Trainer for managing model training workflows."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()

    def prepare_time_series_data(self, df: pd.DataFrame, target_col: str,
                               sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM time series training."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        # Scale features
        features = df[numeric_cols].values
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(df)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(df[target_col].iloc[i])

        return np.array(X), np.array(y)

    def train_lstm(self, model: LSTMModel, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model."""
        logger.info("🔥 Training LSTM model")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Loss function and optimizer
        criterion = nn.MSELoss()  # Can be changed to BCEWithLogitsLoss for classification
        optimizer = optim.Adam(model.parameters(), lr=self.config.lstm_learning_rate)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(self.config.lstm_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).numpy()
            val_pred = model(X_val_tensor).numpy()

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)

        logger.info(f"✅ LSTM training completed. Validation MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'val_score': val_r2
            }
        }

class ModelEvaluator:
    """Comprehensive model evaluation with cross-validation."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def cross_validate_model(self, model_class, X: pd.DataFrame, y: pd.Series,
                            task_type: str = 'classification') -> Dict[str, Any]:
        """Perform cross-validation on a model."""
        logger.info(f"📊 Cross-validating {model_class.__name__}")

        # Initialize model
        if task_type == 'classification':
            model = model_class(self.config, task_type)
        else:
            model = model_class(self.config, task_type)

        # Prepare data
        X_prepared, y_prepared = model.prepare_data(X, y)

        # Time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        scores = []
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_prepared)):
            X_train_fold, X_val_fold = X_prepared.iloc[train_idx], X_prepared.iloc[val_idx]
            y_train_fold, y_val_fold = y_prepared.iloc[train_idx], y_prepared.iloc[val_idx]

            # Train model
            result = model.train(X_train_fold, y_train_fold)
            fold_score = result['metrics']['val_score']
            scores.append(fold_score)
            fold_metrics.append(result['metrics'])

            logger.info(f"Fold {fold + 1}: {task_type} score = {fold_score:.4f}")

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        logger.info(f"✅ Cross-validation completed. Mean {task_type} score: {mean_score:.4f} ± {std_score:.4f}")

        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': scores,
            'fold_metrics': fold_metrics,
            'model_class': model_class.__name__
        }

    def evaluate_ensemble(self, models: List[Dict[str, Any]], X_test: pd.DataFrame,
                         y_test: pd.Series, weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        logger.info("🎯 Evaluating ensemble performance")

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Get predictions from all models
        predictions = []
        for model_dict in models:
            model = model_dict['model']
            pred = model.predict(X_test[model_dict['feature_names']].fillna(0))
            predictions.append(pred)

        # Weighted ensemble
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # Calculate metrics
        if len(np.unique(y_test)) == 2:  # Classification
            metrics = {
                'accuracy': accuracy_score(y_test, np.round(ensemble_pred)),
                'precision': precision_score(y_test, np.round(ensemble_pred)),
                'recall': recall_score(y_test, np.round(ensemble_pred)),
                'f1': f1_score(y_test, np.round(ensemble_pred))
            }
        else:  # Regression
            metrics = {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'r2': r2_score(y_test, ensemble_pred)
            }

        logger.info(f"✅ Ensemble evaluation completed")

        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': predictions,
            'weights': weights,
            'metrics': metrics
        }

class NBAEnsembleModel:
    """
    Main ensemble model for NBA predictive analytics.

    Combines XGBoost, Random Forest, and LSTM models with optimized weights
    for comprehensive NBA game predictions.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.ensemble_weights = None

        logger.info("🏀 NBA Ensemble Model initialized")

    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train classification ensemble (win/loss predictions)."""
        logger.info("🏆 Training classification ensemble models")

        # Train XGBoost
        xgb_model = XGBoostModel(self.config, 'classification')
        xgb_result = xgb_model.train(X, y)
        self.models['xgboost'] = xgb_result

        # Train Random Forest
        rf_model = RandomForestModel(self.config, 'classification')
        rf_result = rf_model.train(X, y)
        self.models['random_forest'] = rf_result

        # Cross-validation for robust evaluation
        xgb_cv = self.evaluator.cross_validate_model(XGBoostModel, X, y, 'classification')
        rf_cv = self.evaluator.cross_validate_model(RandomForestModel, X, y, 'classification')

        # Optimize ensemble weights based on CV performance
        weights = self._optimize_weights([xgb_cv['mean_score'], rf_cv['mean_score']])
        self.ensemble_weights = weights

        results = {
            'xgboost': xgb_result,
            'random_forest': rf_result,
            'cross_validation': {
                'xgboost': xgb_cv,
                'random_forest': rf_cv
            },
            'ensemble_weights': weights,
            'task_type': 'classification'
        }

        logger.info(f"✅ Classification ensemble trained. Optimal weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}")

        return results

    def train_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train regression ensemble (point differential predictions)."""
        logger.info("📈 Training regression ensemble models")

        # Train XGBoost
        xgb_model = XGBoostModel(self.config, 'regression')
        xgb_result = xgb_model.train(X, y)
        self.models['xgboost_reg'] = xgb_result

        # Train Random Forest
        rf_model = RandomForestModel(self.config, 'regression')
        rf_result = rf_model.train(X, y)
        self.models['random_forest_reg'] = rf_result

        # Cross-validation
        xgb_cv = self.evaluator.cross_validate_model(XGBoostModel, X, y, 'regression')
        rf_cv = self.evaluator.cross_validate_model(RandomForestModel, X, y, 'regression')

        # Optimize weights
        weights = self._optimize_weights([xgb_cv['mean_score'], rf_cv['mean_score']])

        results = {
            'xgboost': xgb_result,
            'random_forest': rf_result,
            'cross_validation': {
                'xgboost': xgb_cv,
                'random_forest': rf_cv
            },
            'ensemble_weights': weights,
            'task_type': 'regression'
        }

        logger.info(f"✅ Regression ensemble trained. Optimal weights: XGB={weights[0]:.3f}, RF={weights[1]:.3f}")

        return results

    def train_lstm_model(self, df: pd.DataFrame, target_col: str,
                        sequence_length: int = 10) -> Dict[str, Any]:
        """Train LSTM model for time series predictions."""
        logger.info(f"🔥 Training LSTM model for {target_col}")

        # Prepare time series data
        X, y = self.trainer.prepare_time_series_data(df, target_col, sequence_length)

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Initialize and train LSTM
        lstm_model = LSTMModel(self.config, X.shape[2])
        lstm_result = self.trainer.train_lstm(lstm_model, X_train, y_train, X_val, y_val)

        self.models['lstm'] = lstm_result

        logger.info(f"✅ LSTM model trained for {target_col}")

        return lstm_result

    def predict_classification(self, X: pd.DataFrame) -> np.ndarray:
        """Make classification predictions using ensemble."""
        if not self.models or 'xgboost' not in self.models:
            raise ValueError("Classification models not trained yet")

        predictions = []

        # XGBoost prediction
        xgb_pred = self.models['xgboost']['model'].predict(
            X[self.models['xgboost']['feature_names']].fillna(0)
        )
        predictions.append(xgb_pred)

        # Random Forest prediction
        rf_pred = self.models['random_forest']['model'].predict(
            X[self.models['random_forest']['feature_names']].fillna(0)
        )
        predictions.append(rf_pred)

        # Weighted ensemble
        if self.ensemble_weights:
            ensemble_pred = np.average(predictions, axis=0, weights=self.ensemble_weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def predict_regression(self, X: pd.DataFrame) -> np.ndarray:
        """Make regression predictions using ensemble."""
        if not self.models or 'xgboost_reg' not in self.models:
            raise ValueError("Regression models not trained yet")

        predictions = []

        # XGBoost prediction
        xgb_pred = self.models['xgboost_reg']['model'].predict(
            X[self.models['xgboost_reg']['feature_names']].fillna(0)
        )
        predictions.append(xgb_pred)

        # Random Forest prediction
        rf_pred = self.models['random_forest_reg']['model'].predict(
            X[self.models['random_forest_reg']['feature_names']].fillna(0)
        )
        predictions.append(rf_pred)

        # Weighted ensemble
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get consolidated feature importance from all models."""
        importance = {}

        if 'xgboost' in self.models:
            importance['xgboost'] = self.models['xgboost']['model'].get_feature_importance()

        if 'random_forest' in self.models:
            importance['random_forest'] = self.models['random_forest']['model'].get_feature_importance()

        return importance

    def explain_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get SHAP explanations from XGBoost model."""
        if 'xgboost' not in self.models:
            raise ValueError("XGBoost model not trained yet")

        shap_values = self.models['xgboost']['model'].explain(
            X[self.models['xgboost']['feature_names']].fillna(0)
        )

        return {
            'shap_values': shap_values,
            'feature_names': self.models['xgboost']['feature_names']
        }

    def save_model(self, filepath: str) -> bool:
        """Save trained ensemble model."""
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'config': self.config,
                    'ensemble_weights': self.ensemble_weights
                }, f)

            logger.info(f"💾 Ensemble model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained ensemble model."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.models = data['models']
            self.config = data['config']
            self.ensemble_weights = data.get('ensemble_weights')

            logger.info(f"📂 Ensemble model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def _optimize_weights(self, scores: List[float]) -> List[float]:
        """Optimize ensemble weights based on individual model performance."""
        # Simple weight optimization based on performance
        total_score = sum(scores)
        if total_score > 0:
            weights = [score / total_score for score in scores]
        else:
            weights = [1.0 / len(scores)] * len(scores)

        return weights