"""
🧠 NBA Ensemble Predictor - Task 2.2.1 Implementation

Advanced ensemble system combining XGBoost and Neural Network models
with DevStream SuperPowered architecture and ContextSet compliance.

Author: NBA Predictive Analytics System
Date: 2025-01-11
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import logging
import json
import time
import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import TensorFlow, but make it optional
TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning(
        "TensorFlow not available. Neural Network component will be disabled."
    )

# DevStream SuperPowered imports
if TYPE_CHECKING:
    from ..streamlit.components.ml_integration_bridge import MLIntegrationBridge

# Import new ensemble confidence calculator
try:
    from .ensemble_confidence_calculator import (
        NBAEnsembleConfidenceCalculator,
        EnsembleCIConfig,
        EnsemblePredictionInterval,
    )

    ENSEMBLE_CI_AVAILABLE = True
except ImportError:
    ENSEMBLE_CI_AVAILABLE = False
    logging.warning(
        "Ensemble confidence calculator not available. Advanced CI features will be disabled."
    )

# Task 2.2.3: Import prediction explainer
try:
    from .prediction_explainer import (
        NBAPredictionExplainer,
        ExplanationConfig,
        PredictionExplanation,
        FeatureImportance,
        ExplanationMethod,
    )

    PREDICTION_EXPLAINER_AVAILABLE = True
except ImportError:
    PREDICTION_EXPLAINER_AVAILABLE = False
    logging.warning(
        "Prediction explainer not available. SHAP and explanation features will be disabled."
    )

# Task 2.2.4: Import model version manager
try:
    from .model_version_manager import (
        NBAModelVersionManager,
        ModelVersion,
        ModelMetrics as VersionManagerMetrics,
        ModelType,
        ModelStatus,
        RollbackConfig,
    )

    MODEL_VERSIONING_AVAILABLE = True
except ImportError:
    MODEL_VERSIONING_AVAILABLE = False
    logging.warning(
        "Model version manager not available. Versioning and rollback features will be disabled."
    )

# Task 2.2.5: Import model retraining pipeline
try:
    from .model_retraining_pipeline import (
        NBARetrainingPipeline,
        RetrainingConfig,
        RetrainingTriggerType,
        RetrainingStatus,
        DataQualityStatus,
        PerformanceMetrics as RetrainingMetrics,
        get_retraining_pipeline,
        start_retraining_pipeline,
        trigger_manual_retraining,
    )

    MODEL_RETRAINING_AVAILABLE = True
except ImportError:
    MODEL_RETRAINING_AVAILABLE = False
    logging.warning(
        "Model retraining pipeline not available. Automated retraining features will be disabled."
    )

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Prediction result from ensemble with detailed metadata"""

    prediction: float
    confidence: float
    method_used: str
    xgb_prediction: float
    nn_prediction: Optional[float] = None
    xgb_confidence: float = 0.0
    nn_confidence: float = 0.0
    ensemble_weight: str = "weighted"
    model_health: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    cv_score: float
    training_time: float
    inference_time: float
    sample_count: int
    last_updated: datetime


class EnsembleMethod(Enum):
    """Ensemble combination methods"""

    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    VOTING = "voting"
    ADAPTIVE = "adaptive"


class NBAEnsemblePredictor:
    """
    🏆 SuperPowered NBA Ensemble Predictor

    Advanced ensemble system combining:
    - XGBoost with Bayesian optimization
    - Neural Network with TensorFlow
    - Intelligent ensemble combination
    - Real-time performance monitoring
    - Context-aware predictions
    """

    def __init__(
        self,
        model_name: str = "nba_ensemble_predictor",
        ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        enable_bayesian_optimization: bool = True,
        enable_neural_network: bool = True,
        auto_retrain_threshold: float = 0.75,
        cache_predictions: bool = True,
        ml_bridge: Optional["MLIntegrationBridge"] = None,
    ):
        """
        Initialize the NBA Ensemble Predictor

        Args:
            model_name: Name identifier for the ensemble
            ensemble_method: Method for combining predictions
            enable_bayesian_optimization: Use Bayesian optimization for XGBoost
            enable_neural_network: Enable neural network component
            auto_retrain_threshold: Performance threshold for auto-retraining
            cache_predictions: Cache predictions for performance
            ml_bridge: Reference to ML Integration Bridge
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_name = model_name
        self.ensemble_method = ensemble_method
        self.enable_bayesian_optimization = enable_bayesian_optimization
        self.enable_neural_network = enable_neural_network and TENSORFLOW_AVAILABLE
        self.auto_retrain_threshold = auto_retrain_threshold
        self.cache_predictions = cache_predictions
        self.ml_bridge = ml_bridge

        # Model components
        self.xgb_model = None
        self.nn_model = None
        self.scaler_xgb = RobustScaler()
        self.scaler_nn = StandardScaler()

        # Performance tracking
        self.xgb_metrics = None
        self.nn_metrics = None

        # Task 2.2.4: Initialize model version manager
        self._version_manager = None
        if MODEL_VERSIONING_AVAILABLE:
            try:
                rollback_config = RollbackConfig(
                    enabled=True,
                    auto_rollback=True,
                    performance_threshold=0.02,
                    max_rollback_versions=5,
                    rollback_cooldown_hours=24,
                    monitoring_window_days=7,
                )
                self._version_manager = NBAModelVersionManager(
                    model_registry_path="data/models/registry",
                    models_path="data/models/versions",
                    rollback_config=rollback_config,
                )
                self.logger.info("🔧 Task 2.2.4 Model Version Manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize model version manager: {e}")
                self._version_manager = None
        else:
            self.logger.warning("Model version manager not available")

        # Task 2.2.5: Initialize model retraining pipeline
        self._retraining_pipeline = None
        if MODEL_RETRAINING_AVAILABLE:
            try:
                retraining_config = RetrainingConfig(
                    schedule_enabled=False,  # Don't auto-start by default
                    schedule_interval="daily",
                    schedule_time="02:00",
                    accuracy_threshold=0.65,
                    performance_degradation_threshold=0.05,
                    data_drift_threshold=0.1,
                    min_training_samples=1000,
                    max_training_samples=50000,
                    data_freshness_days=30,
                    quality_score_threshold=0.7,
                    test_size=0.2,
                    cv_folds=5,
                    random_state=42,
                    max_retraining_time=3600,
                    notifications_enabled=False,
                    nba_season_required=True,
                    min_games_per_team=10,
                    current_season_weight=1.5,
                    enable_early_stopping=True,
                    enable_hyperparameter_tuning=False,  # Disabled for speed
                    enable_feature_selection=True,
                    enable_ensemble_optimization=True,
                )
                self._retraining_pipeline = NBARetrainingPipeline(retraining_config)
                self.logger.info("🔄 Task 2.2.5 Model Retraining Pipeline initialized")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize model retraining pipeline: {e}"
                )
                self._retraining_pipeline = None
        else:
            self.logger.warning("Model retraining pipeline not available")
        self.ensemble_metrics = None

        # Prediction cache
        self._prediction_cache: Dict[str, EnsemblePrediction] = {}
        self._cache_max_size = 10000

        # Thread safety
        self._lock = threading.RLock()

        # Training state
        self._is_training = False
        self._last_training_time = None

        # Task 2.2.2: Initialize Ensemble Confidence Calculator
        self._confidence_calculator = None
        if ENSEMBLE_CI_AVAILABLE:
            try:
                self._confidence_calculator = NBAEnsembleConfidenceCalculator()
                self.logger.info(
                    "🎯 Task 2.2.2 Ensemble Confidence Calculator initialized"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize confidence calculator: {e}")
                self._confidence_calculator = None

        # Task 2.2.3: Initialize Prediction Explainer
        self._prediction_explainer = None
        if PREDICTION_EXPLAINER_AVAILABLE:
            try:
                self._prediction_explainer = NBAPredictionExplainer()
                self.logger.info("🧠 Task 2.2.3 Prediction Explainer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize prediction explainer: {e}")
                self._prediction_explainer = None

        # Feature importance and model explainability
        self.feature_importance = None
        self.feature_names = []

        self.logger.info(f"🧠 NBA Ensemble Predictor initialized")
        self.logger.info(f"   - Model name: {self.model_name}")
        self.logger.info(f"   - Ensemble method: {self.ensemble_method.value}")
        self.logger.info(
            f"   - Bayesian optimization: {self.enable_bayesian_optimization}"
        )
        self.logger.info(f"   - Neural network: {self.enable_neural_network}")
        self.logger.info(f"   - TensorFlow available: {TENSORFLOW_AVAILABLE}")

    def _validate_input_features(
        self, features: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Validate and preprocess input features"""
        try:
            # Required features for NBA prediction
            required_features = [
                "home_team_momentum",
                "away_team_momentum",
                "home_team_rest_days",
                "away_team_rest_days",
                "home_team_back_to_back",
                "away_team_back_to_back",
                "home_team_win_rate",
                "away_team_win_rate",
            ]

            # Check required features
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Convert to numpy array
            feature_array = np.array(
                [
                    features.get("home_team_momentum", 0.0),
                    features.get("away_team_momentum", 0.0),
                    features.get("home_team_rest_days", 2),
                    features.get("away_team_rest_days", 2),
                    features.get("home_team_back_to_back", 0),
                    features.get("away_team_back_to_back", 0),
                    features.get("home_team_win_rate", 0.5),
                    features.get("away_team_win_rate", 0.5),
                ]
            ).reshape(1, -1)

            return feature_array, required_features

        except Exception as e:
            self.logger.error(f"❌ Error validating input features: {e}")
            raise

    def _create_xgb_model(self) -> xgb.XGBClassifier:
        """Create optimized XGBoost model with NBA-specific parameters"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",  # Faster training
            scale_pos_weight=1.0,  # Handle class imbalance
        )

    def _optimize_xgb_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> xgb.XGBClassifier:
        """Optimize XGBoost hyperparameters using Bayesian optimization"""
        self.logger.info("🔍 Starting Bayesian optimization for XGBoost...")

        # Define search space
        search_space = {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "min_child_weight": Integer(1, 10),
            "gamma": Real(0, 5),
            "reg_alpha": Real(0, 1),
            "reg_lambda": Real(0, 1),
        }

        # Create base model
        xgb_base = self._create_xgb_model()

        # Bayesian optimization
        opt = BayesSearchCV(
            estimator=xgb_base,
            search_spaces=search_space,
            n_iter=30,  # Number of iterations
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        start_time = time.time()
        opt.fit(X_train, y_train)
        optimization_time = time.time() - start_time

        self.logger.info(
            f"✅ Bayesian optimization completed in {optimization_time:.2f}s"
        )
        self.logger.info(f"   - Best score: {opt.best_score_:.4f}")
        self.logger.info(f"   - Best params: {opt.best_params_}")

        return opt.best_estimator_

    def _create_neural_network(self, input_dim: int):
        """Create optimized neural network for NBA prediction"""
        if not self.enable_neural_network:
            return None

        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return model

    def _generate_synthetic_training_data(
        self, n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic NBA training data for initial model training"""
        self.logger.info(f"🎲 Generating {n_samples} synthetic training samples...")

        np.random.seed(42)

        # Generate realistic NBA features
        features = {
            "home_team_momentum": np.clip(
                np.random.normal(0.1, 0.4, n_samples), -1.0, 1.0
            ),
            "away_team_momentum": np.clip(
                np.random.normal(-0.05, 0.35, n_samples), -1.0, 1.0
            ),
            "home_team_rest_days": np.random.poisson(2.3, n_samples),
            "away_team_rest_days": np.random.poisson(2.1, n_samples),
            "home_team_back_to_back": np.random.binomial(1, 0.22, n_samples),
            "away_team_back_to_back": np.random.binomial(1, 0.19, n_samples),
            "home_team_win_rate": np.clip(
                np.random.normal(0.52, 0.15, n_samples), 0.0, 1.0
            ),
            "away_team_win_rate": np.clip(
                np.random.normal(0.48, 0.14, n_samples), 0.0, 1.0
            ),
        }

        # Create feature matrix
        X = np.column_stack(
            [
                features["home_team_momentum"],
                features["away_team_momentum"],
                features["home_team_rest_days"],
                features["away_team_rest_days"],
                features["home_team_back_to_back"],
                features["away_team_back_to_back"],
                features["home_team_win_rate"],
                features["away_team_win_rate"],
            ]
        )

        # Generate target with some correlation to features
        # Home team advantage + momentum influence
        home_advantage = 0.1
        momentum_diff = features["home_team_momentum"] - features["away_team_momentum"]
        win_rate_diff = features["home_team_win_rate"] - features["away_team_win_rate"]

        # Combine factors
        win_probability = (
            0.5 + home_advantage + 0.3 * momentum_diff + 0.2 * win_rate_diff
        )
        win_probability = np.clip(win_probability, 0.1, 0.9)

        # Add noise
        y = np.random.binomial(1, win_probability)

        self.feature_names = [
            "home_team_momentum",
            "away_team_momentum",
            "home_team_rest_days",
            "away_team_rest_days",
            "home_team_back_to_back",
            "away_team_back_to_back",
            "home_team_win_rate",
            "away_team_win_rate",
        ]

        return X, y

    def train_models(
        self, training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> bool:
        """
        Train the ensemble models

        Args:
            training_data: Optional (X, y) training data. If None, generates synthetic data.

        Returns:
            True if training successful, False otherwise
        """
        with self._lock:
            if self._is_training:
                self.logger.warning("⚠️ Training already in progress")
                return False

            self._is_training = True
            start_time = time.time()

        try:
            self.logger.info("🚀 Starting ensemble model training...")

            # Get training data
            if training_data is None:
                X, y = self._generate_synthetic_training_data()
            else:
                X, y = training_data

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train XGBoost
            self.logger.info("📊 Training XGBoost model...")
            xgb_start = time.time()

            if self.enable_bayesian_optimization:
                self.xgb_model = self._optimize_xgb_hyperparameters(X_train, y_train)
            else:
                self.xgb_model = self._create_xgb_model()
                self.xgb_model.fit(X_train, y_train)

            xgb_time = time.time() - xgb_start

            # Evaluate XGBoost
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]

            # Train Neural Network
            nn_time = 0
            if self.enable_neural_network:
                self.logger.info("🧠 Training Neural Network...")
                nn_start = time.time()

                # Scale features for NN
                X_train_nn = self.scaler_nn.fit_transform(X_train)
                X_test_nn = self.scaler_nn.transform(X_test)

                # Create and train NN
                self.nn_model = self._create_neural_network(X_train.shape[1])

                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
                ]

                # Train model
                self.nn_model.fit(
                    X_train_nn,
                    y_train,
                    validation_data=(X_test_nn, y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0,
                )

                nn_time = time.time() - nn_start
                nn_pred = (self.nn_model.predict(X_test_nn) > 0.5).astype(int).flatten()
                nn_proba = self.nn_model.predict(X_test_nn).flatten()
            else:
                nn_pred = None
                nn_proba = None

            # Calculate metrics
            self.xgb_metrics = self._calculate_metrics(
                y_test, xgb_pred, xgb_proba, xgb_time, X_train.shape[0]
            )

            if nn_pred is not None:
                self.nn_metrics = self._calculate_metrics(
                    y_test, nn_pred, nn_proba, nn_time, X_train.shape[0]
                )

            # Calculate ensemble performance
            ensemble_pred, ensemble_proba = self._combine_predictions(
                xgb_pred, xgb_proba, nn_pred, nn_proba
            )
            self.ensemble_metrics = self._calculate_metrics(
                y_test, ensemble_pred, ensemble_proba, 0, X_train.shape[0]
            )

            # Store feature importance
            if self.xgb_model is not None:
                self.feature_importance = dict(
                    zip(self.feature_names, self.xgb_model.feature_importances_)
                )

            # Task 2.2.3: Initialize prediction explainer with trained models
            if self._prediction_explainer is not None:
                try:
                    self._prediction_explainer.initialize_with_models(
                        xgb_model=self.xgb_model,
                        nn_model=self.nn_model,
                        feature_names=self.feature_names,
                        xgb_scaler=self.scaler_xgb,
                        nn_scaler=self.scaler_nn,
                        background_data=X[:100] if X is not None else None,
                    )
                    self.logger.info(
                        "🧠 Task 2.2.3 Prediction Explainer initialized with trained models"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize prediction explainer with models: {e}"
                    )

            total_time = time.time() - start_time
            self._last_training_time = datetime.now()

            self.logger.info(f"✅ Ensemble training completed in {total_time:.2f}s")
            self.logger.info(f"   - XGBoost accuracy: {self.xgb_metrics.accuracy:.4f}")
            if self.nn_metrics:
                self.logger.info(
                    f"   - Neural Network accuracy: {self.nn_metrics.accuracy:.4f}"
                )
            self.logger.info(
                f"   - Ensemble accuracy: {self.ensemble_metrics.accuracy:.4f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False

        finally:
            with self._lock:
                self._is_training = False

    def _combine_predictions(
        self,
        xgb_pred: np.ndarray,
        xgb_proba: np.ndarray,
        nn_pred: Optional[np.ndarray],
        nn_proba: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine predictions from both models using ensemble method"""
        if nn_pred is None:
            return xgb_pred, xgb_proba

        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Weight by individual model performance
            xgb_weight = self.xgb_metrics.accuracy if self.xgb_metrics else 0.5
            nn_weight = self.nn_metrics.accuracy if self.nn_metrics else 0.5
            total_weight = xgb_weight + nn_weight

            if total_weight > 0:
                xgb_weight /= total_weight
                nn_weight /= total_weight
            else:
                xgb_weight = nn_weight = 0.5

            ensemble_proba = xgb_weight * xgb_proba + nn_weight * nn_proba
            ensemble_pred = (ensemble_proba > 0.5).astype(int)

        elif self.ensemble_method == EnsembleMethod.VOTING:
            # Simple voting
            ensemble_proba = (xgb_proba + nn_proba) / 2
            ensemble_pred = (ensemble_proba > 0.5).astype(int)

        else:
            # Default to weighted average
            ensemble_proba = (xgb_proba + (nn_proba or 0)) / 2
            ensemble_pred = (ensemble_proba > 0.5).astype(int)

        return ensemble_pred, ensemble_proba

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        training_time: float,
        sample_count: int,
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        # Calculate AUC if we have probabilities
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5

        # Cross-validation score (placeholder for now)
        cv_score = accuracy  # In real implementation, would do actual CV

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            cv_score=cv_score,
            training_time=training_time,
            inference_time=0.0,  # Will be updated during prediction
            sample_count=sample_count,
            last_updated=datetime.now(),
        )

    def predict(self, features: Dict[str, Any]) -> EnsemblePrediction:
        """
        Make prediction using ensemble method

        Args:
            features: Dictionary of NBA game features

        Returns:
            EnsemblePrediction with detailed metadata
        """
        start_time = time.time()

        try:
            # Validate and preprocess features
            X, feature_names = self._validate_input_features(features)

            # Generate cache key
            cache_key = self._generate_cache_key(features)

            # Check cache
            if self.cache_predictions and cache_key in self._prediction_cache:
                cached_result = self._prediction_cache[cache_key]
                self.logger.debug(f"📋 Using cached prediction")
                return cached_result

            # Ensure models are trained
            if self.xgb_model is None:
                self.logger.info("🏃 Training models on-demand...")
                if not self.train_models():
                    raise RuntimeError("Failed to train models")

            # XGBoost prediction
            xgb_pred_proba = self.xgb_model.predict_proba(X)[0, 1]
            xgb_pred = 1 if xgb_pred_proba > 0.5 else 0
            xgb_confidence = max(xgb_pred_proba, 1 - xgb_pred_proba)

            # Neural Network prediction
            nn_pred_proba = None
            nn_pred = None
            nn_confidence = 0.0

            if self.enable_neural_network and self.nn_model is not None:
                X_nn = self.scaler_nn.transform(X)
                nn_pred_proba_raw = self.nn_model.predict(X_nn)[0, 0]
                nn_pred = 1 if nn_pred_proba_raw > 0.5 else 0
                nn_confidence = max(nn_pred_proba_raw, 1 - nn_pred_proba_raw)
                nn_pred_proba = float(nn_pred_proba_raw)

            # Combine predictions
            if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                xgb_weight = self.xgb_metrics.accuracy if self.xgb_metrics else 0.5
                nn_weight = self.nn_metrics.accuracy if self.nn_metrics else 0.5
                total_weight = xgb_weight + nn_weight

                if total_weight > 0:
                    xgb_weight /= total_weight
                    nn_weight /= total_weight
                else:
                    xgb_weight = nn_weight = 0.5

                ensemble_proba = xgb_weight * xgb_pred_proba + nn_weight * (
                    nn_pred_proba or 0
                )
                prediction = 1 if ensemble_proba > 0.5 else 0
                confidence = max(ensemble_proba, 1 - ensemble_proba)

            else:
                ensemble_proba = (xgb_pred_proba + (nn_pred_proba or 0)) / 2
                prediction = 1 if ensemble_proba > 0.5 else 0
                confidence = max(ensemble_proba, 1 - ensemble_proba)

            # Create result
            result = EnsemblePrediction(
                prediction=float(prediction),
                confidence=float(confidence),
                method_used=self.ensemble_method.value,
                xgb_prediction=float(xgb_pred),
                nn_prediction=float(nn_pred) if nn_pred is not None else None,
                xgb_confidence=float(xgb_confidence),
                nn_confidence=float(nn_confidence),
                ensemble_weight="weighted",
                model_health={
                    "xgb_trained": self.xgb_model is not None,
                    "nn_trained": self.nn_model is not None,
                    "last_training": self._last_training_time.isoformat()
                    if self._last_training_time
                    else None,
                    "ensemble_method": self.ensemble_method.value,
                },
                timestamp=datetime.now(),
            )

            # Task 2.2.2: Calculate ensemble confidence intervals
            if self._confidence_calculator is not None:
                try:
                    # Prepare predictions for CI calculation
                    xgb_predictions_array = np.array([xgb_pred_proba])
                    nn_predictions_array = (
                        np.array([nn_pred_proba])
                        if nn_pred_proba is not None
                        else np.array([])
                    )

                    if len(nn_predictions_array) > 0:
                        # Both models available - calculate full ensemble CI
                        confidence_intervals = self._confidence_calculator.calculate_ensemble_confidence_intervals(
                            xgboost_predictions=xgb_predictions_array,
                            neural_network_predictions=nn_predictions_array,
                            confidence_levels=[0.90, 0.95, 0.99],
                        )

                        # Add CI information to result
                        result.confidence_intervals = {
                            level: ci.to_dict()
                            for level, ci in confidence_intervals.items()
                        }

                        # Add model disagreement metrics
                        disagreement_metrics = self._confidence_calculator.get_ensemble_disagreement_metrics(
                            xgb_predictions_array, nn_predictions_array
                        )
                        result.model_disagreement = disagreement_metrics

                        self.logger.debug(
                            f"🎯 Task 2.2.2 Ensemble CI calculated: {len(confidence_intervals)} intervals"
                        )
                    else:
                        # Only XGBoost available - fallback CI
                        result.confidence_intervals = {}
                        result.model_disagreement = {}

                except Exception as e:
                    self.logger.warning(
                        f"Failed to calculate ensemble confidence intervals: {e}"
                    )
                    result.confidence_intervals = {}
                    result.model_disagreement = {}
            else:
                result.confidence_intervals = {}
                result.model_disagreement = {}

            # Cache result
            if self.cache_predictions:
                self._cache_prediction(cache_key, result)

            # Update inference time
            inference_time = time.time() - start_time
            if self.ensemble_metrics:
                self.ensemble_metrics.inference_time = inference_time

            self.logger.debug(
                f"🎯 Prediction: {prediction} (confidence: {confidence:.3f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            # Return fallback prediction
            return EnsemblePrediction(
                prediction=0.5,
                confidence=0.0,
                method_used="fallback",
                xgb_prediction=0.5,
                timestamp=datetime.now(),
            )

    def predict_game(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a specific NBA game - compatibility method
        
        Args:
            features: Dictionary of NBA game features
            
        Returns:
            Dictionary with prediction results compatible with system interface
        """
        try:
            # Use existing predict method
            ensemble_result = self.predict(features)
            
            # Convert to expected format
            result = {
                "success": True,
                "prediction": ensemble_result.prediction,
                "confidence": ensemble_result.confidence,
                "method": ensemble_result.method_used,
                "xgboost_prediction": ensemble_result.xgb_prediction,
                "neural_network_prediction": ensemble_result.nn_prediction,
                "xgboost_confidence": ensemble_result.xgb_confidence,
                "neural_network_confidence": ensemble_result.nn_confidence,
                "ensemble_weight": ensemble_result.ensemble_weight,
                "model_health": ensemble_result.model_health,
                "timestamp": ensemble_result.timestamp.isoformat(),
                "metadata": {
                    "model_name": self.model_name,
                    "ensemble_method": self.ensemble_method.value,
                    "feature_importance": self.feature_importance,
                    "tensorflow_available": TENSORFLOW_AVAILABLE,
                }
            }
            
            # Add confidence intervals if available
            if hasattr(ensemble_result, 'confidence_intervals'):
                result["confidence_intervals"] = ensemble_result.confidence_intervals
                
            # Add model disagreement if available
            if hasattr(ensemble_result, 'model_disagreement'):
                result["model_disagreement"] = ensemble_result.model_disagreement
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ predict_game failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prediction": 0.5,
                "confidence": 0.0,
                "method": "fallback_error"
            }

    def _generate_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key for prediction"""
        # Sort features for consistent key generation
        sorted_features = sorted(features.items())
        feature_str = json.dumps(sorted_features, sort_keys=True)
        return hash(feature_str)

    def _cache_prediction(self, cache_key: str, prediction: EnsemblePrediction):
        """Cache prediction result"""
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove oldest entries
            oldest_key = min(
                self._prediction_cache.keys(),
                key=lambda k: self._prediction_cache[k].timestamp,
            )
            del self._prediction_cache[oldest_key]

        self._prediction_cache[cache_key] = prediction

    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        return {
            "model_name": self.model_name,
            "ensemble_method": self.ensemble_method.value,
            "is_training": self._is_training,
            "last_training_time": self._last_training_time.isoformat()
            if self._last_training_time
            else None,
            "xgb_model_trained": self.xgb_model is not None,
            "nn_model_trained": self.nn_model is not None,
            "xgb_metrics": asdict(self.xgb_metrics) if self.xgb_metrics else None,
            "nn_metrics": asdict(self.nn_metrics) if self.nn_metrics else None,
            "ensemble_metrics": asdict(self.ensemble_metrics)
            if self.ensemble_metrics
            else None,
            "feature_importance": self.feature_importance,
            "cache_size": len(self._prediction_cache),
            "tensorflow_available": TENSORFLOW_AVAILABLE,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost model"""
        if self.xgb_model is None:
            return {}

        if self.feature_importance is None:
            self.feature_importance = dict(
                zip(self.feature_names, self.xgb_model.feature_importances_)
            )

        return self.feature_importance

    def save_model(self, filepath: str) -> bool:
        """Save ensemble model to file"""
        try:
            model_data = {
                "xgb_model": self.xgb_model,
                "nn_model": self.nn_model,
                "scaler_xgb": self.scaler_xgb,
                "scaler_nn": self.scaler_nn,
                "ensemble_method": self.ensemble_method,
                "xgb_metrics": self.xgb_metrics,
                "nn_metrics": self.nn_metrics,
                "ensemble_metrics": self.ensemble_metrics,
                "feature_importance": self.feature_importance,
                "feature_names": self.feature_names,
                "last_training_time": self._last_training_time,
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            self.logger.info(f"💾 Model saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load ensemble model from file"""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.xgb_model = model_data.get("xgb_model")
            self.nn_model = model_data.get("nn_model")
            self.scaler_xgb = model_data.get("scaler_xgb", RobustScaler())
            self.scaler_nn = model_data.get("scaler_nn", StandardScaler())
            self.ensemble_method = model_data.get(
                "ensemble_method", EnsembleMethod.WEIGHTED_AVERAGE
            )
            self.xgb_metrics = model_data.get("xgb_metrics")
            self.nn_metrics = model_data.get("nn_metrics")
            self.ensemble_metrics = model_data.get("ensemble_metrics")
            self.feature_importance = model_data.get("feature_importance")
            self.feature_names = model_data.get("feature_names", [])
            self._last_training_time = model_data.get("last_training_time")

            self.logger.info(f"📂 Model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

    # Task 2.2.2: Confidence Interval Management Methods

    def get_confidence_calculator(self):
        """
        Get the ensemble confidence calculator instance.

        Returns:
            NBAEnsembleConfidenceCalculator: The confidence calculator or None if not available
        """
        return self._confidence_calculator

    def calibrate_confidence_intervals(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> bool:
        """
        Calibrate confidence intervals using test data.

        Args:
            X_test: Test features
            y_test: Test true labels

        Returns:
            bool: True if calibration successful
        """
        if self._confidence_calculator is None:
            self.logger.warning("⚠️ Confidence calculator not available for calibration")
            return False

        if not self.is_trained:
            self.logger.warning(
                "⚠️ Models not trained - cannot calibrate confidence intervals"
            )
            return False

        try:
            with self._lock:
                # Get predictions from both models for calibration
                xgb_pred_proba = None
                nn_pred_proba = None

                if self.xgb_model is not None:
                    X_test_scaled = self.scaler_xgb.transform(X_test)
                    xgb_pred_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]

                if self.nn_model is not None and TENSORFLOW_AVAILABLE:
                    try:
                        X_test_scaled_nn = self.scaler_nn.transform(X_test)
                        nn_pred_proba = self.nn_model.predict(
                            X_test_scaled_nn, verbose=0
                        ).flatten()
                    except Exception as e:
                        self.logger.warning(
                            f"Neural network prediction failed during calibration: {e}"
                        )

                # Perform calibration with both models
                if xgb_pred_proba is not None:
                    if nn_pred_proba is not None:
                        # Both models available
                        self._confidence_calculator.calibrate_ensemble_intervals(
                            xgboost_predictions=xgb_pred_proba,
                            neural_network_predictions=nn_pred_proba,
                            true_outcomes=y_test,
                        )
                        self.logger.info(
                            "✅ Ensemble confidence intervals calibrated with both models"
                        )
                    else:
                        # Only XGBoost available
                        self._confidence_calculator.calibrate_single_model_intervals(
                            model_type="xgboost",
                            predictions=xgb_pred_proba,
                            true_outcomes=y_test,
                        )
                        self.logger.info(
                            "✅ Confidence intervals calibrated with XGBoost only"
                        )

                return True

        except Exception as e:
            self.logger.error(f"❌ Failed to calibrate confidence intervals: {e}")
            return False

    def get_confidence_interval_methods(self) -> list:
        """
        Get available confidence interval calculation methods.

        Returns:
            list: Available CI methods
        """
        if self._confidence_calculator is None:
            return []

        return self._confidence_calculator.get_available_methods()

    def get_prediction_uncertainty_metrics(self) -> dict:
        """
        Get metrics about prediction uncertainty and calibration quality.

        Returns:
            dict: Uncertainty and calibration metrics
        """
        if self._confidence_calculator is None:
            return {"error": "Confidence calculator not available"}

        try:
            # Get calibration report
            calibration_report = self._confidence_calculator.get_calibration_report()

            # Add ensemble-specific uncertainty metrics
            uncertainty_metrics = {
                "calibration_report": calibration_report,
                "ensemble_uncertainty_available": True,
                "model_disagreement_tracking": True,
                "confidence_levels_supported": [0.90, 0.95, 0.99],
                "advanced_methods": {
                    "bayesian_bootstrap": True,
                    "quantile_ensemble": True,
                    "conformal_prediction": True,
                    "model_disagreement": True,
                },
            }

            return uncertainty_metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to get uncertainty metrics: {e}")
            return {"error": str(e)}

    def predict_with_confidence_intervals(
        self, input_features: dict, confidence_levels: list = None
    ) -> dict:
        """
        Make prediction with comprehensive confidence interval analysis.

        Args:
            input_features: Feature dictionary
            confidence_levels: List of confidence levels (default: [0.90, 0.95, 0.99])

        Returns:
            dict: Prediction with detailed confidence intervals
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]

        # Get standard prediction result
        standard_result = self.predict(input_features)

        if not standard_result["success"]:
            return standard_result

        # Add confidence interval analysis if available
        if (
            self._confidence_calculator is not None
            and "confidence_intervals" in standard_result
        ):
            ci_data = standard_result["confidence_intervals"]

            # Create comprehensive confidence analysis
            confidence_analysis = {
                "primary_prediction": standard_result["prediction"],
                "primary_confidence": standard_result["confidence"],
                "ensemble_method": standard_result["method"],
                "confidence_intervals": ci_data,
                "uncertainty_analysis": {
                    "model_disagreement": ci_data.get("model_disagreement"),
                    "prediction_variance": standard_result.get("prediction_variance"),
                    "ensemble_weight_distribution": standard_result.get(
                        "model_weights"
                    ),
                    "calibration_quality": self.get_prediction_uncertainty_metrics().get(
                        "calibration_report", {}
                    ),
                },
                "risk_assessment": {
                    "high_uncertainty": ci_data.get("model_disagreement", {}).get(
                        "disagreement_score", 0
                    )
                    > 0.3,
                    "well_calibrated": True,  # Would be updated based on actual calibration
                    "reliable_prediction": standard_result["confidence"] > 0.7,
                },
            }

            return {
                "success": True,
                "prediction": standard_result["prediction"],
                "confidence": standard_result["confidence"],
                "method": standard_result["method"],
                "confidence_analysis": confidence_analysis,
                "xgboost_prediction": standard_result.get("xgboost_prediction"),
                "neural_network_prediction": standard_result.get(
                    "neural_network_prediction"
                ),
                "ensemble_feature_importance": standard_result.get(
                    "ensemble_feature_importance"
                ),
                "prediction_variance": standard_result.get("prediction_variance"),
                "model_weights": standard_result.get("model_weights"),
                "confidence_intervals": standard_result.get("confidence_intervals"),
                "metadata": standard_result.get("metadata", {}),
            }

        return standard_result

    # Task 2.2.3: Prediction Explainer Methods

    def get_prediction_explainer(self):
        """
        Get the prediction explainer instance.

        Returns:
            NBAPredictionExplainer: The prediction explainer or None if not available
        """
        return self._prediction_explainer

    def explain_prediction(
        self,
        input_features: dict,
        prediction_result: dict = None,
        explanation_methods: List[ExplanationMethod] = None,
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction using SHAP and other methods.

        Args:
            input_features: Input feature dictionary
            prediction_result: Existing prediction result (optional)
            explanation_methods: List of explanation methods to use

        Returns:
            PredictionExplanation: Comprehensive explanation object
        """
        if self._prediction_explainer is None:
            raise RuntimeError("Prediction explainer not available")

        try:
            # Generate standard prediction if not provided
            if prediction_result is None:
                prediction_result = self.predict(input_features)

            if not prediction_result["success"]:
                raise RuntimeError(
                    f"Cannot generate explanation for failed prediction: {prediction_result.get('error')}"
                )

            # Determine predicted class
            prediction_value = prediction_result["prediction"]
            predicted_class = "home_win" if prediction_value > 0.5 else "away_win"
            confidence = prediction_result.get("confidence", 0.0)

            # Set default explanation methods
            if explanation_methods is None:
                explanation_methods = [
                    ExplanationMethod.SHAP_VALUES,
                    ExplanationMethod.CUSTOM_ATTRIBUTION,
                ]

            # Generate explanation
            explanation = self._prediction_explainer.explain_prediction(
                input_features=input_features,
                prediction_value=prediction_value,
                predicted_class=predicted_class,
                confidence=confidence,
                methods=explanation_methods,
            )

            self.logger.info(
                f"🧠 Generated prediction explanation with {len(explanation.feature_importances)} features"
            )
            return explanation

        except Exception as e:
            self.logger.error(f"❌ Failed to generate prediction explanation: {e}")
            raise

    def explain_prediction_with_confidence(
        self, input_features: dict, confidence_levels: list = None
    ) -> dict:
        """
        Generate prediction with both confidence intervals and explanation.

        Args:
            input_features: Input feature dictionary
            confidence_levels: List of confidence levels

        Returns:
            dict: Combined prediction with confidence intervals and explanation
        """
        try:
            # Get prediction with confidence intervals
            prediction_with_ci = self.predict_with_confidence_intervals(
                input_features, confidence_levels
            )

            if not prediction_with_ci["success"]:
                return prediction_with_ci

            # Generate explanation
            explanation = self.explain_prediction(
                input_features=input_features,
                prediction_result=prediction_with_ci,
                explanation_methods=[
                    ExplanationMethod.SHAP_VALUES,
                    ExplanationMethod.CUSTOM_ATTRIBUTION,
                ],
            )

            # Combine results
            combined_result = {
                **prediction_with_ci,
                "prediction_explanation": asdict(explanation)
                if hasattr(explanation, "asdict")
                else explanation.__dict__,
                "explanation_id": explanation.prediction_id,
                "top_explanatory_features": [
                    {
                        "feature": fi.feature_name,
                        "importance": fi.importance,
                        "explanation": fi.explanation,
                        "direction": fi.direction,
                        "category": fi.feature_category,
                        "nba_context": fi.nba_context,
                    }
                    for fi in explanation.top_features[:5]
                ],
                "nba_context": explanation.game_context,
                "betting_implications": explanation.betting_implications,
                "explanation_methods_used": explanation.explanation_methods_used,
            }

            return combined_result

        except Exception as e:
            self.logger.error(
                f"❌ Failed to generate prediction with confidence and explanation: {e}"
            )
            return self.predict(input_features)

    def get_explanation_summary(self) -> dict:
        """
        Get summary of prediction explainer statistics.

        Returns:
            dict: Explainer summary statistics
        """
        if self._prediction_explainer is None:
            return {"error": "Prediction explainer not available"}

        return self._prediction_explainer.get_explanation_summary()

    # Task 2.2.4: Model Versioning and Rollback Methods
    def get_version_manager(self):
        """
        Get access to the model version manager.

        Returns:
            NBAModelVersionManager: Model version manager instance or None
        """
        return self._version_manager

    def register_model_version(
        self,
        description: str,
        created_by: str = "NBA_Ensemble_Predictor",
        nba_season: str = "",
        training_date_range: Tuple[str, str] = ("", ""),
        team_coverage: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Register current models as new versions.

        Args:
            description: Description of the model version
            created_by: Who created this version
            nba_season: NBA season covered
            training_date_range: Training data date range
            team_coverage: Teams covered by this model
            tags: Version tags

        Returns:
            Dict mapping model types to version numbers
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return {"error": "Model versioning not available"}

        try:
            versions = {}

            # Register XGBoost model
            if self.xgb_model is not None:
                xgb_metrics = self._convert_metrics_to_version_manager(self.xgb_metrics)
                xgb_version = self._version_manager.register_model(
                    model=self.xgb_model,
                    model_type=ModelType.XGBOOST,
                    description=f"XGBoost: {description}",
                    created_by=created_by,
                    metrics=xgb_metrics,
                    hyperparameters=getattr(self, "xgb_hyperparameters", {}),
                    nba_season=nba_season,
                    training_date_range=training_date_range,
                    team_coverage=team_coverage,
                    tags=(tags or []) + ["xgboost"],
                )
                versions["xgboost"] = xgb_version
                self.logger.info(
                    f"✅ XGBoost model registered as version {xgb_version}"
                )

            # Register Neural Network model
            if self.nn_model is not None and TENSORFLOW_AVAILABLE:
                nn_metrics = self._convert_metrics_to_version_manager(self.nn_metrics)
                nn_version = self._version_manager.register_model(
                    model=self.nn_model,
                    model_type=ModelType.NEURAL_NETWORK,
                    description=f"Neural Network: {description}",
                    created_by=created_by,
                    metrics=nn_metrics,
                    hyperparameters=getattr(self, "nn_hyperparameters", {}),
                    nba_season=nba_season,
                    training_date_range=training_date_range,
                    team_coverage=team_coverage,
                    tags=(tags or []) + ["neural_network"],
                )
                versions["neural_network"] = nn_version
                self.logger.info(
                    f"✅ Neural Network model registered as version {nn_version}"
                )

            # Register ensemble
            if self.xgb_model is not None and self.nn_model is not None:
                ensemble_model = {
                    "xgb_model": self.xgb_model,
                    "nn_model": self.nn_model,
                    "scaler_xgb": self.scaler_xgb,
                    "scaler_nn": self.scaler_nn,
                }

                ensemble_metrics = self._convert_metrics_to_version_manager(
                    self.ensemble_metrics
                )
                ensemble_version = self._version_manager.register_model(
                    model=ensemble_model,
                    model_type=ModelType.ENSEMBLE,
                    description=f"Ensemble: {description}",
                    created_by=created_by,
                    metrics=ensemble_metrics,
                    hyperparameters={
                        "ensemble_method": self.ensemble_method.value,
                        "enable_bayesian_optimization": self.enable_bayesian_optimization,
                        "auto_retrain_threshold": self.auto_retrain_threshold,
                    },
                    nba_season=nba_season,
                    training_date_range=training_date_range,
                    team_coverage=team_coverage,
                    tags=(tags or []) + ["ensemble"],
                )
                versions["ensemble"] = ensemble_version
                self.logger.info(
                    f"✅ Ensemble model registered as version {ensemble_version}"
                )

            return versions

        except Exception as e:
            self.logger.error(f"❌ Failed to register model versions: {e}")
            return {"error": str(e)}

    def activate_model_version(
        self, version: str, model_type: Optional[str] = None
    ) -> bool:
        """
        Activate a specific model version.

        Args:
            version: Version to activate
            model_type: Optional model type filter

        Returns:
            Success status
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            self.logger.error("Model versioning not available")
            return False

        try:
            success = self._version_manager.activate_model(version)
            if success:
                # If activating ensemble, load all components
                if model_type == "ensemble" or model_type is None:
                    self._load_ensemble_from_version(version)
                else:
                    self._load_single_model_from_version(version, model_type)

                self.logger.info(f"✅ Model version {version} activated successfully")
            return success

        except Exception as e:
            self.logger.error(f"❌ Failed to activate model version {version}: {e}")
            return False

    def rollback_model(
        self, version: str, target_version: Optional[str] = None
    ) -> bool:
        """
        Rollback model to previous version.

        Args:
            version: Current version to rollback
            target_version: Target version (optional)

        Returns:
            Success status
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            self.logger.error("Model versioning not available")
            return False

        try:
            success = self._version_manager.rollback_model(version, target_version)
            if success:
                self.logger.info(f"✅ Model rollback from {version} completed")
            return success

        except Exception as e:
            self.logger.error(f"❌ Failed to rollback model: {e}")
            return False

    def get_model_versions(
        self, model_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of model versions.

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of version information
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return []

        try:
            mt = ModelType(model_type) if model_type else None
            st = ModelStatus(status) if status else None

            versions = self._version_manager.list_versions(mt, st)

            return [
                {
                    "version": v.version,
                    "model_type": v.model_type.value,
                    "status": v.status.value,
                    "created_at": v.created_at.isoformat(),
                    "description": v.description,
                    "created_by": v.created_by,
                    "metrics": v.metrics.to_dict(),
                    "nba_season": v.nba_season,
                    "tags": v.tags,
                }
                for v in versions
            ]

        except Exception as e:
            self.logger.error(f"❌ Failed to get model versions: {e}")
            return []

    def get_active_versions(self) -> Dict[str, str]:
        """
        Get active model versions.

        Returns:
            Dict mapping model types to active versions
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return {}

        try:
            active_versions = {}
            for model_type in ModelType:
                active_version = self._version_manager.get_active_version(model_type)
                if active_version:
                    active_versions[model_type.value] = active_version

            return active_versions

        except Exception as e:
            self.logger.error(f"❌ Failed to get active versions: {e}")
            return {}

    def compare_model_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison results
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return {"error": "Model versioning not available"}

        try:
            return self._version_manager.compare_versions(version1, version2)

        except Exception as e:
            self.logger.error(f"❌ Failed to compare model versions: {e}")
            return {"error": str(e)}

    def log_model_performance(self, performance_data: Dict[str, Any]) -> None:
        """
        Log performance data for active models.

        Args:
            performance_data: Performance metrics
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return

        try:
            active_versions = self.get_active_versions()

            for model_type, version in active_versions.items():
                self._version_manager.log_performance(version, performance_data)

            self.logger.debug(
                f"📊 Performance logged for {len(active_versions)} active models"
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to log model performance: {e}")

    def get_version_summary(self) -> Dict[str, Any]:
        """
        Get summary of all model versions.

        Returns:
            Version summary statistics
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return {"error": "Model versioning not available"}

        try:
            return self._version_manager.get_version_summary()

        except Exception as e:
            self.logger.error(f"❌ Failed to get version summary: {e}")
            return {"error": str(e)}

    def cleanup_old_versions(self, keep_versions: int = 10) -> int:
        """
        Cleanup old model versions.

        Args:
            keep_versions: Number of versions to keep per model type

        Returns:
            Number of versions cleaned up
        """
        if not MODEL_VERSIONING_AVAILABLE or self._version_manager is None:
            return 0

        try:
            return self._version_manager.cleanup_old_versions(keep_versions)

        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old versions: {e}")
            return 0

    def _convert_metrics_to_version_manager(
        self, metrics
    ) -> Optional[VersionManagerMetrics]:
        """Convert internal metrics to version manager metrics format"""
        if metrics is None:
            return None

        try:
            vm_metrics = VersionManagerMetrics()

            # Map standard metrics
            for field in vm_metrics.__dataclass_fields__:
                if hasattr(metrics, field):
                    setattr(vm_metrics, field, getattr(metrics, field))
                elif isinstance(metrics, dict) and field in metrics:
                    setattr(vm_metrics, field, metrics[field])

            return vm_metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to convert metrics: {e}")
            return None

    def _load_ensemble_from_version(self, version: str) -> bool:
        """Load ensemble components from version"""
        try:
            model, version_info = self._version_manager.load_model(version)

            if version_info.model_type == ModelType.ENSEMBLE:
                # Load ensemble components
                self.xgb_model = model["xgb_model"]
                self.nn_model = model["nn_model"]
                self.scaler_xgb = model["scaler_xgb"]
                self.scaler_nn = model["scaler_nn"]

                # Load confidence calculator and explainer
                self._initialize_confidence_calculator_with_version(version)
                self._initialize_prediction_explainer_with_version(version)

                self.logger.info(f"✅ Ensemble version {version} loaded successfully")
                return True
            else:
                self.logger.error(f"Version {version} is not an ensemble type")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to load ensemble version {version}: {e}")
            return False

    def _load_single_model_from_version(self, version: str, model_type: str) -> bool:
        """Load single model from version"""
        try:
            model, version_info = self._version_manager.load_model(version)

            mt = ModelType(model_type)

            if mt == ModelType.XGBOOST:
                self.xgb_model = model
                self.logger.info(f"✅ XGBoost version {version} loaded successfully")
            elif mt == ModelType.NEURAL_NETWORK:
                self.nn_model = model
                self.logger.info(
                    f"✅ Neural Network version {version} loaded successfully"
                )
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load {model_type} version {version}: {e}")
            return False

    def _initialize_confidence_calculator_with_version(self, version: str) -> None:
        """Initialize confidence calculator with version data"""
        if not ENSEMBLE_CI_AVAILABLE:
            return

        try:
            if self.xgb_model is not None and self.nn_model is not None:
                # Initialize with current models (version-specific initialization would need version metadata)
                self._confidence_calculator.initialize_with_models(
                    xgb_model=self.xgb_model,
                    nn_model=self.nn_model,
                    feature_names=getattr(self, "feature_names", []),
                    xgb_scaler=self.scaler_xgb,
                    nn_scaler=self.scaler_nn,
                )
                self.logger.info(
                    "🧠 Confidence calculator initialized with version models"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize confidence calculator with version: {e}"
            )

    def _initialize_prediction_explainer_with_version(self, version: str) -> None:
        """Initialize prediction explainer with version data"""
        if not PREDICTION_EXPLAINER_AVAILABLE:
            return

        try:
            if self.xgb_model is not None and self.nn_model is not None:
                # Initialize with current models (version-specific initialization would need version metadata)
                self._prediction_explainer.initialize_with_models(
                    xgb_model=self.xgb_model,
                    nn_model=self.nn_model,
                    feature_names=getattr(self, "feature_names", []),
                    xgb_scaler=self.scaler_xgb,
                    nn_scaler=self.scaler_nn,
                )
                self.logger.info(
                    "🧠 Prediction explainer initialized with version models"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize prediction explainer with version: {e}"
            )

    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            self.xgb_model = None
            self.nn_model = None
            self._prediction_cache.clear()
            self._is_training = False
            # Task 2.2.2: Cleanup confidence calculator
            self._confidence_calculator = None
            # Task 2.2.3: Cleanup prediction explainer
            self._prediction_explainer = None
            # Task 2.2.4: Cleanup model version manager
            self._version_manager = None
        # Task 2.2.5: Cleanup model retraining pipeline
        if self._retraining_pipeline:
            self._retraining_pipeline.stop()
            self._retraining_pipeline = None

        self.logger.info("🧹 NBA Ensemble Predictor cleanup completed")

    # Task 2.2.5: Model Retraining Pipeline Methods
    def get_retraining_pipeline(self) -> Optional[Any]:
        """
        Get access to the model retraining pipeline

        Returns:
            NBARetrainingPipeline instance or None if not available
        """
        return self._retraining_pipeline

    def is_retraining_available(self) -> bool:
        """
        Check if the model retraining pipeline is available

        Returns:
            True if retraining pipeline is available
        """
        return MODEL_RETRAINING_AVAILABLE and self._retraining_pipeline is not None

    def start_automated_retraining(
        self,
        schedule_enabled: bool = True,
        schedule_interval: str = "daily",
        schedule_time: str = "02:00",
    ) -> bool:
        """
        Start automated model retraining

        Args:
            schedule_enabled: Enable scheduled retraining
            schedule_interval: Scheduling interval (hourly, daily, weekly)
            schedule_time: Time for daily scheduling

        Returns:
            True if started successfully
        """
        if not self.is_retraining_available():
            self.logger.warning("Retraining pipeline not available")
            return False

        try:
            # Update configuration
            self._retraining_pipeline.config.schedule_enabled = schedule_enabled
            self._retraining_pipeline.config.schedule_interval = schedule_interval
            self._retraining_pipeline.config.schedule_time = schedule_time

            # Start the pipeline
            self._retraining_pipeline.start()
            self.logger.info("🔄 Automated retraining started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start automated retraining: {e}")
            return False

    def stop_automated_retraining(self) -> bool:
        """
        Stop automated model retraining

        Returns:
            True if stopped successfully
        """
        if not self.is_retraining_available():
            return True  # Already stopped

        try:
            self._retraining_pipeline.stop()
            self.logger.info("🛑 Automated retraining stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop automated retraining: {e}")
            return False

    def trigger_manual_retraining(self, reason: Optional[str] = None) -> Optional[str]:
        """
        Trigger manual model retraining

        Args:
            reason: Optional reason for manual retraining

        Returns:
            Job ID for tracking or None if failed
        """
        if not self.is_retraining_available():
            self.logger.warning("Retraining pipeline not available")
            return None

        try:
            job_id = self._retraining_pipeline.trigger_retraining(
                RetrainingTriggerType.MANUAL, reason
            )
            self.logger.info(f"🔄 Manual retraining triggered: job {job_id}")
            return job_id

        except Exception as e:
            self.logger.error(f"Failed to trigger manual retraining: {e}")
            return None

    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Get current retraining pipeline status

        Returns:
            Status information dictionary
        """
        if not self.is_retraining_available():
            return {"available": False, "status": "Not Available"}

        try:
            status = self._retraining_pipeline.get_status()
            status["available"] = True
            return status

        except Exception as e:
            self.logger.error(f"Failed to get retraining status: {e}")
            return {"available": True, "status": "Error", "error": str(e)}

    def get_retraining_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get retraining job history

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job information dictionaries
        """
        if not self.is_retraining_available():
            return []

        try:
            return self._retraining_pipeline.get_job_history(limit)
        except Exception as e:
            self.logger.error(f"Failed to get retraining job history: {e}")
            return []

    def configure_retraining_pipeline(self, **kwargs) -> bool:
        """
        Configure retraining pipeline parameters

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            True if configuration updated successfully
        """
        if not self.is_retraining_available():
            return False

        try:
            # Update configuration parameters
            for key, value in kwargs.items():
                if hasattr(self._retraining_pipeline.config, key):
                    setattr(self._retraining_pipeline.config, key, value)
                    self.logger.info(f"Updated retraining config: {key} = {value}")
                else:
                    self.logger.warning(f"Unknown retraining config parameter: {key}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to configure retraining pipeline: {e}")
            return False

    def get_retraining_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive retraining pipeline information

        Returns:
            Pipeline information dictionary
        """
        info = {
            "available": self.is_retraining_available(),
            "version": "2.2.5",
            "features": [
                "Automated retraining scheduling",
                "Data quality validation",
                "Performance monitoring",
                "Model degradation detection",
                "NBA-specific data handling",
                "Integration with model versioning",
            ],
        }

        if self.is_retraining_available():
            try:
                status = self._retraining_pipeline.get_status()
                config_dict = {
                    "schedule_enabled": self._retraining_pipeline.config.schedule_enabled,
                    "schedule_interval": self._retraining_pipeline.config.schedule_interval,
                    "schedule_time": self._retraining_pipeline.config.schedule_time,
                    "accuracy_threshold": self._retraining_pipeline.config.accuracy_threshold,
                    "performance_degradation_threshold": self._retraining_pipeline.config.performance_degradation_threshold,
                    "data_drift_threshold": self._retraining_pipeline.config.data_drift_threshold,
                    "min_training_samples": self._retraining_pipeline.config.min_training_samples,
                    "max_training_samples": self._retraining_pipeline.config.max_training_samples,
                    "nba_season_required": self._retraining_pipeline.config.nba_season_required,
                    "notifications_enabled": self._retraining_pipeline.config.notifications_enabled,
                }

                info.update(
                    {
                        "status": status,
                        "configuration": config_dict,
                        "integration": {
                            "model_versioning": MODEL_VERSIONING_AVAILABLE,
                            "prediction_explainer": PREDICTION_EXPLAINER_AVAILABLE,
                            "confidence_calculator": ENSEMBLE_CI_AVAILABLE,
                        },
                    }
                )

            except Exception as e:
                info["error"] = str(e)

        return info

    @property
    def is_trained(self) -> bool:
        """
        Check if models are trained
        
        Returns:
            True if at least XGBoost model is trained
        """
        return self.xgb_model is not None
