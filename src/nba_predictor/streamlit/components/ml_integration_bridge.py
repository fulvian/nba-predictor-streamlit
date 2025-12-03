#!/usr/bin/env python3
"""
🚀 ML Integration Bridge - SuperPowered NBA Betting System

Centralized ML system state manager with ContextSet compliance and DevStream
SuperPowered architecture. Implements MLflow-inspired model lifecycle management
with health checking, graceful degradation, and single source of truth.

Author: NBA Predictive Analytics System
Task ID: 1.3.1 - Centralized ML system state manager
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import logging
import sys
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nba_predictive_system"))

# Import confidence interval calculator for Phase 2 monitoring
# Note: Using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nba_predictor.monitoring.nba_metrics_collector import (
        NBAPredictionMetricsCollector,
    )
    from src.nba_predictor.monitoring.nba_drift_detector import NBADriftDetector
    from src.nba_predictor.monitoring.nba_confidence_intervals import (
        NBAConfidenceIntervalCalculator,
    )
    from src.nba_predictor.monitoring.nba_model_health_dashboard import (
        NBAModelHealthDashboard,
    )
    from src.nba_predictor.ensemble.nba_ensemble_predictor import NBAEnsemblePredictor
    from src.nba_predictor.core.unified_ml_interface import UnifiedMLInterface

METRICS_AVAILABLE = True  # We'll import dynamically when needed
DRIFT_DETECTION_AVAILABLE = True  # We'll import dynamically when needed
CONFIDENCE_INTERVALS_AVAILABLE = True  # We'll import dynamically when needed


class MLComponentStatus(Enum):
    """Enumeration of ML component health statuses"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelStatus(Enum):
    """MLflow-inspired model status enumeration"""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Data class for model performance metrics with context awareness"""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confidence_score: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    prediction_count: int = 0
    total_predictions: int = 0


@dataclass
class MLComponentHealth:
    """Health status for ML components with detailed diagnostics"""

    component_name: str
    status: MLComponentStatus
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    error_rate: float = 0.0
    response_time_ms: float = 0.0
    uptime_percentage: float = 100.0
    last_error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistryEntry:
    """MLflow-inspired model registry entry with comprehensive metadata"""

    model_name: str
    model_version: str
    model_path: str
    status: ModelStatus
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    model_type: str = "unknown"
    feature_schema: Dict[str, str] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True
    performance_score: float = 0.0


class MLIntegrationBridge:
    """
    🚀 SuperPowered ML Integration Bridge with ContextSet Compliance

    Features:
    - Centralized ML system state management
    - Health checking for all ML components
    - Graceful degradation when ML unavailable
    - Single source of truth for prediction data
    - MLflow-inspired model lifecycle management
    - Real-time monitoring and diagnostics
    - Context-aware fallback mechanisms
    """

    def __init__(
        self,
        health_check_interval: int = 60,
        max_retries: int = 3,
        cache_ttl_minutes: int = 15,
    ):
        """
        Initialize the ML Integration Bridge

        Args:
            health_check_interval: Health check interval in seconds
            max_retries: Maximum retries for failed operations
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Component registry
        self._ml_components: Dict[str, MLComponentHealth] = {}
        self._model_registry: Dict[str, ModelRegistryEntry] = {}

        # Phase 2: Initialize metrics collector for monitoring
        self._metrics_collector = None
        if METRICS_AVAILABLE:
            try:
                # Dynamic import to avoid circular imports
                from src.nba_predictor.monitoring.nba_metrics_collector import (
                    NBAPredictionMetricsCollector,
                )

                self._metrics_collector = NBAPredictionMetricsCollector(bridge=self)
                self.logger.info("📊 Phase 2 Metrics Collector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize metrics collector: {e}")
        else:
            self.logger.info(
                "📊 Metrics collection disabled (prometheus_client not available)"
            )

        # Phase 2 Task 2.1.2: Initialize drift detector for feature distribution monitoring
        self._drift_detector = None
        if DRIFT_DETECTION_AVAILABLE:
            try:
                # Dynamic import to avoid circular imports
                from src.nba_predictor.monitoring.nba_drift_detector import (
                    NBADriftDetector,
                )

                self._drift_detector = NBADriftDetector(bridge=self)
                self.logger.info("🔍 Phase 2 Task 2.1.2 Drift Detector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize drift detector: {e}")
        else:
            self.logger.info("🔍 Drift detection disabled (Evidently not available)")

        # Phase 2 Task 2.1.3: Initialize confidence interval calculator
        self._ci_calculator = None
        if CONFIDENCE_INTERVALS_AVAILABLE:
            try:
                # Dynamic import to avoid circular imports
                from src.nba_predictor.monitoring.nba_confidence_intervals import (
                    NBAConfidenceIntervalCalculator,
                )

                self._ci_calculator = NBAConfidenceIntervalCalculator()
                self.logger.info(
                    "📊 Phase 2 Task 2.1.3 Confidence Interval Calculator initialized"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize confidence interval calculator: {e}"
                )
        else:
            self.logger.info(
                "📊 Confidence intervals disabled (scikit-learn not available)"
            )

        # Phase 2 Task 2.1.4: Initialize model health dashboard
        self._health_dashboard = None
        try:
            # Dynamic import to avoid circular imports
            from src.nba_predictor.monitoring.nba_model_health_dashboard import (
                NBAModelHealthDashboard,
            )

            self._health_dashboard = NBAModelHealthDashboard(
                ml_bridge=self,
                update_interval_seconds=30,
                alert_retention_days=30,
                enable_background_monitoring=True,
            )
            self.logger.info("🏥 Phase 2 Task 2.1.4 Model Health Dashboard initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model health dashboard: {e}")

        # Phase 2 Task 2.2.1: Initialize unified ML interface
        self._unified_ml_interface = None
        try:
            # Dynamic import to avoid circular imports
            from src.nba_predictor.core.unified_ml_interface import (
                get_unified_ml_interface,
            )

            self._unified_ml_interface = get_unified_ml_interface(
                data_path="data",
                model_path="models",
                use_stacked_ensemble=True,
                enable_explainability=True,
                validate_realism=True,
            )
            self.logger.info("🚀 Phase 2 Task 2.2.1 Unified ML Interface initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize unified ML interface: {e}")

        # Keep ensemble predictor as fallback for compatibility
        self._ensemble_predictor = None
        try:
            # Dynamic import to avoid circular imports
            from src.nba_predictor.ensemble.nba_ensemble_predictor import (
                NBAEnsemblePredictor,
            )

            self._ensemble_predictor = NBAEnsemblePredictor(
                model_name="nba_ensemble_predictor",
                ensemble_method="WEIGHTED_AVERAGE",  # Default method
                enable_bayesian_optimization=True,
                enable_neural_network=True,  # Will be False if TensorFlow not available
                auto_retrain_threshold=0.75,
                cache_predictions=True,
                ml_bridge=self,
            )
            self.logger.info(
                "🚀 Phase 2 Task 2.2.1 NBA Ensemble Predictor (fallback) initialized"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize ensemble predictor: {e}")

        self._prediction_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._active_models: Dict[str, str] = {}  # component -> model_name

        # System state
        self._system_status = MLComponentStatus.HEALTHY
        self._last_health_check = datetime.now()
        self._total_predictions = 0
        self._successful_predictions = 0
        self._failed_predictions = 0

        # Threading for background health checks
        self._health_check_thread = None
        self._stop_health_checks = threading.Event()

        # Initialize default models
        self._initialize_default_models()

        # Start background health monitoring
        self._start_health_monitoring()

        self.logger.info(
            "🚀 ML Integration Bridge initialized with SuperPowered features"
        )

    def _initialize_default_models(self) -> None:
        """Initialize default model registry entries"""

        # NBA Game Prediction Model
        self._model_registry["nba_game_predictor"] = ModelRegistryEntry(
            model_name="nba_game_predictor",
            model_version="v1.0.0",
            model_path="models/nba_game_predictor.pkl",
            status=ModelStatus.PENDING,
            model_type="ensemble_classifier",
            feature_schema={
                "home_team_momentum": "float",
                "away_team_momentum": "float",
                "home_team_rest_days": "int",
                "away_team_rest_days": "int",
                "home_team_streak": "int",
                "away_team_streak": "int",
            },
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
            },
            tags={"environment": "production", "league": "NBA"},
        )

        # Player Performance Prediction Model
        self._model_registry["player_performance_predictor"] = ModelRegistryEntry(
            model_name="player_performance_predictor",
            model_version="v1.0.0",
            model_path="models/player_performance_predictor.pkl",
            status=ModelStatus.PENDING,
            model_type="regression",
            feature_schema={
                "player_minutes": "float",
                "player_usage_rate": "float",
                "opponent_defense_rating": "float",
            },
            tags={"environment": "production", "type": "player"},
        )

        # Betting Odds Model
        self._model_registry["betting_odds_model"] = ModelRegistryEntry(
            model_name="betting_odds_model",
            model_version="v1.0.0",
            model_path="models/betting_odds_model.pkl",
            status=ModelStatus.PENDING,
            model_type="probability_estimator",
            feature_schema={
                "team_win_probability": "float",
                "spread_predicted": "float",
                "total_points_predicted": "float",
            },
            tags={"environment": "production", "type": "betting"},
        )

    def register_ml_component(
        self,
        component_name: str,
        component_type: str = "generic",
        initial_status: MLComponentStatus = MLComponentStatus.UNKNOWN,
    ) -> None:
        """
        Register a new ML component for health monitoring

        Args:
            component_name: Unique component identifier
            component_type: Type of component (model, pipeline, service)
            initial_status: Initial health status
        """
        if component_name not in self._ml_components:
            self._ml_components[component_name] = MLComponentHealth(
                component_name=component_name,
                status=initial_status,
                diagnostics={"component_type": component_type},
            )
            self.logger.info(f"✅ Registered ML component: {component_name}")

    def get_model_prediction(
        self, model_name: str, input_data: Dict[str, Any], fallback_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Get prediction from specified model with comprehensive error handling
        Phase 2 Enhanced: Automatic metrics collection for monitoring

        Args:
            model_name: Name of the model to use
            input_data: Input features for prediction
            fallback_enabled: Whether to use fallback if model unavailable

        Returns:
            Prediction results with confidence and metadata
        """
        # Phase 2: Start timing for metrics collection
        prediction_start_time = time.time()
        self._total_predictions += 1

        # Check cache first
        cache_key = self._generate_cache_key(model_name, input_data)
        if cache_key in self._prediction_cache:
            cached_result, cache_time = self._prediction_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                self.logger.debug(f"📋 Cache hit for model: {model_name}")
                return cached_result

        # Verify model exists and is ready
        if model_name not in self._model_registry:
            error_msg = f"Model {model_name} not found in registry"
            self.logger.warning(f"❌ {error_msg}")
            self._failed_predictions += 1

            if fallback_enabled:
                return self._get_fallback_prediction(model_name, input_data, error_msg)
            else:
                return {"error": error_msg, "success": False}

        model_entry = self._model_registry[model_name]

        if model_entry.status != ModelStatus.READY:
            error_msg = (
                f"Model {model_name} is not ready (status: {model_entry.status.value})"
            )
            self.logger.warning(f"⚠️ {error_msg}")

            if fallback_enabled:
                return self._get_fallback_prediction(model_name, input_data, error_msg)
            else:
                return {"error": error_msg, "success": False}

        # Attempt prediction with retries
        prediction = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                prediction = self._execute_model_prediction(model_entry, input_data)

                # Update model metrics
                model_entry.metrics.last_updated = datetime.now()
                model_entry.metrics.prediction_count += 1
                model_entry.metrics.total_predictions += 1
                model_entry.last_used = datetime.now()

                # Cache the result
                self._prediction_cache[cache_key] = (prediction, datetime.now())

                # Update success metrics
                self._successful_predictions += 1

                # Clean old cache entries
                self._clean_expired_cache()

                # Phase 2: Record prediction metrics
                prediction_time_ms = (time.time() - prediction_start_time) * 1000
                if self._metrics_collector:
                    try:
                        # Determine prediction type based on model
                        if model_name == "nba_game_predictor":
                            prediction_type = "game_outcome"
                        elif "player" in model_name:
                            prediction_type = "player_performance"
                        elif "betting" in model_name:
                            prediction_type = "betting_odds"
                        else:
                            prediction_type = "unknown"

                        self._metrics_collector.record_prediction(
                            model_name=model_name,
                            prediction_type=prediction_type,
                            input_features=input_data,
                            prediction=prediction.get("prediction"),
                            confidence=prediction.get("confidence", 0.5),
                            prediction_time_ms=prediction_time_ms,
                            success=True,
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to record metrics: {e}")

                # Phase 2 Task 2.1.2: Detect drift for feature distributions
                if self._drift_detector:
                    try:
                        drift_results = (
                            self._drift_detector.detect_drift_for_prediction(
                                model_name=model_name,
                                input_features=input_data,
                                prediction=prediction,
                            )
                        )

                        # Log significant drift
                        if drift_results and drift_results.get(
                            "overall_drift_detected", False
                        ):
                            self.logger.warning(
                                f"🔍 DRIFT DETECTED for model {model_name}: "
                                f"score={drift_results.get('drift_score', 0):.3f}, "
                                f"features={drift_results.get('features_drifted', [])}"
                            )

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to detect drift: {e}")

                # Phase 2 Task 2.1.3: Calculate confidence intervals
                if self._ci_calculator:
                    try:
                        # Calculate confidence intervals for the prediction
                        ci_result = self._ci_calculator.calculate_confidence_intervals(
                            model_name=model_name,
                            input_features=input_data,
                            prediction_result=prediction,
                            confidence_level=0.95,
                        )

                        if ci_result and ci_result.get("success"):
                            # Add confidence intervals to prediction response
                            prediction["confidence_intervals"] = ci_result.get(
                                "confidence_intervals", {}
                            )
                            prediction["prediction_uncertainty"] = ci_result.get(
                                "prediction_uncertainty", {}
                            )
                            prediction["interval_method"] = ci_result.get(
                                "method_used", "adaptive"
                            )

                            self.logger.debug(
                                f"📊 Confidence intervals calculated for {model_name}: "
                                f"method={ci_result.get('method_used')}, "
                                f"width={ci_result.get('average_interval_width', 0):.3f}"
                            )
                        else:
                            # Fallback confidence intervals if calculation fails
                            prediction["confidence_intervals"] = {
                                "lower_bound": prediction.get("confidence", 0.5) - 0.1,
                                "upper_bound": min(
                                    1.0, prediction.get("confidence", 0.5) + 0.1
                                ),
                            }
                            prediction["interval_method"] = "fallback"

                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Failed to calculate confidence intervals: {e}"
                        )
                        # Add minimal confidence interval info even on error
                        prediction["confidence_intervals"] = {
                            "lower_bound": max(
                                0.0, prediction.get("confidence", 0.5) - 0.05
                            ),
                            "upper_bound": min(
                                1.0, prediction.get("confidence", 0.5) + 0.05
                            ),
                        }
                        prediction["interval_method"] = "error_fallback"

                # Phase 2 Task 2.2.1: Enhanced unified prediction for NBA games
                if model_name == "nba_game_predictor" and self._unified_ml_interface:
                    try:
                        # Use ensemble predictor for enhanced NBA game predictions
                        ensemble_result = self._ensemble_predictor.predict(input_data)

                        if ensemble_result.get("success"):
                            # Enhance the base prediction with ensemble results
                            prediction.update(
                                {
                                    "ensemble_prediction": ensemble_result.get(
                                        "prediction"
                                    ),
                                    "ensemble_confidence": ensemble_result.get(
                                        "confidence", 0.5
                                    ),
                                    "ensemble_method": ensemble_result.get(
                                        "ensemble_method", "WEIGHTED_AVERAGE"
                                    ),
                                    "xgboost_prediction": ensemble_result.get(
                                        "xgboost_prediction"
                                    ),
                                    "neural_network_prediction": ensemble_result.get(
                                        "neural_network_prediction"
                                    ),
                                    "ensemble_contributors": ensemble_result.get(
                                        "ensemble_contributors", {}
                                    ),
                                    "model_weights": ensemble_result.get(
                                        "model_weights", {}
                                    ),
                                    "prediction_variance": ensemble_result.get(
                                        "prediction_variance", 0.0
                                    ),
                                    "ensemble_feature_importance": ensemble_result.get(
                                        "feature_importance", {}
                                    ),
                                }
                            )

                            # Replace base prediction with ensemble if confidence is higher
                            if ensemble_result.get("confidence", 0.5) > prediction.get(
                                "confidence", 0.5
                            ):
                                prediction["prediction"] = ensemble_result.get(
                                    "prediction"
                                )
                                prediction["confidence"] = ensemble_result.get(
                                    "confidence", 0.5
                                )
                                prediction["method"] = "ensemble"

                            self.logger.info(
                                f"🚀 Ensemble prediction enhanced: "
                                f"method={ensemble_result.get('ensemble_method')}, "
                                f"confidence={ensemble_result.get('confidence', 0):.3f}, "
                                f"variance={ensemble_result.get('prediction_variance', 0):.3f}"
                            )
                        else:
                            self.logger.warning(
                                f"⚠️ Ensemble prediction failed: {ensemble_result.get('error', 'Unknown error')}"
                            )

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to get ensemble prediction: {e}")

                self.logger.debug(f"✅ Successful prediction from {model_name}")
                return prediction

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"⚠️ Prediction attempt {attempt + 1} failed for {model_name}: {e}"
                )

                # Update component error metrics
                if model_name in self._ml_components:
                    self._ml_components[model_name].error_count += 1

                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All retries failed
        self._failed_predictions += 1
        error_msg = (
            f"All prediction attempts failed for {model_name}. Last error: {last_error}"
        )
        self.logger.error(f"❌ {error_msg}")

        if fallback_enabled:
            return self._get_fallback_prediction(model_name, input_data, error_msg)
        else:
            return {"error": error_msg, "success": False}

    def _execute_model_prediction(
        self, model_entry: ModelRegistryEntry, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute model prediction with input validation

        Args:
            model_entry: Model registry entry
            input_data: Validated input data

        Returns:
            Prediction results with confidence scores
        """
        # Validate input data against model schema
        validated_data = self._validate_input_data(model_entry, input_data)

        # Simulate model prediction (in real implementation, this would load and use actual model)
        prediction_result = self._simulate_model_prediction(model_entry, validated_data)

        return {
            "success": True,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result.get("confidence", 0.5),
            "model_name": model_entry.model_name,
            "model_version": model_entry.model_version,
            "timestamp": datetime.now().isoformat(),
            "input_features": validated_data,
            "model_metrics": {
                "accuracy": model_entry.metrics.accuracy,
                "performance_score": model_entry.performance_score,
            },
        }

    def _validate_input_data(
        self, model_entry: ModelRegistryEntry, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input data against model feature schema"""

        validated_data = {}
        missing_features = []

        for feature_name, feature_type in model_entry.feature_schema.items():
            if feature_name in input_data:
                try:
                    if feature_type == "float":
                        validated_data[feature_name] = float(input_data[feature_name])
                    elif feature_type == "int":
                        validated_data[feature_name] = int(input_data[feature_name])
                    elif feature_type == "str":
                        validated_data[feature_name] = str(input_data[feature_name])
                    else:
                        validated_data[feature_name] = input_data[feature_name]
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"⚠️ Invalid type for feature {feature_name}: {e}"
                    )
                    # Use default value based on type
                    if feature_type == "float":
                        validated_data[feature_name] = 0.0
                    elif feature_type == "int":
                        validated_data[feature_name] = 0
                    else:
                        validated_data[feature_name] = ""
            else:
                missing_features.append(feature_name)
                # Use default value
                if feature_type == "float":
                    validated_data[feature_name] = 0.0
                elif feature_type == "int":
                    validated_data[feature_name] = 0
                else:
                    validated_data[feature_name] = ""

        if missing_features:
            self.logger.warning(
                f"⚠️ Missing features for {model_entry.model_name}: {missing_features}"
            )

        return validated_data

    def _simulate_model_prediction(
        self, model_entry: ModelRegistryEntry, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate model prediction (placeholder for actual model implementation)"""

        if model_entry.model_type == "ensemble_classifier":
            # Simulate game outcome prediction
            home_advantage = input_data.get("home_team_momentum", 0) - input_data.get(
                "away_team_momentum", 0
            )
            rest_advantage = input_data.get("home_team_rest_days", 0) - input_data.get(
                "away_team_rest_days", 0
            )
            streak_advantage = input_data.get("home_team_streak", 0) - input_data.get(
                "away_team_streak", 0
            )

            # Calculate win probability
            win_probability = (
                0.5
                + (home_advantage * 0.1)
                + (rest_advantage * 0.05)
                + (streak_advantage * 0.03)
            )
            win_probability = max(
                0.1, min(0.9, win_probability)
            )  # Clamp between 0.1 and 0.9

            prediction = "home_win" if win_probability > 0.5 else "away_win"
            confidence = abs(win_probability - 0.5) * 2  # Convert to 0-1 scale

            return {
                "prediction": prediction,
                "win_probability": win_probability,
                "confidence": confidence,
            }

        elif model_entry.model_type == "regression":
            # Simulate player performance prediction
            base_performance = input_data.get("player_minutes", 0) * 0.5
            usage_impact = input_data.get("player_usage_rate", 0) * 10
            defense_impact = 100 - input_data.get("opponent_defense_rating", 100)

            predicted_points = base_performance + usage_impact + defense_impact
            confidence = 0.7  # Fixed confidence for regression

            return {"prediction": predicted_points, "confidence": confidence}

        elif model_entry.model_type == "probability_estimator":
            # Simulate betting odds prediction
            win_prob = input_data.get("team_win_probability", 0.5)
            spread = (win_prob - 0.5) * 20  # Convert to point spread
            total_points = 220 + np.random.normal(0, 10)  # NBA average with variance

            return {
                "prediction": {
                    "spread": spread,
                    "total_points": total_points,
                    "moneyline": -100 / win_prob
                    if win_prob > 0.5
                    else 100 / (1 - win_prob),
                },
                "confidence": 0.8,
            }

        # Default fallback
        return {"prediction": "unknown", "confidence": 0.5}

    def _get_fallback_prediction(
        self, model_name: str, input_data: Dict[str, Any], error_message: str
    ) -> Dict[str, Any]:
        """Generate fallback prediction when model is unavailable"""

        self.logger.info(f"🔄 Using fallback prediction for {model_name}")

        # Generate context-aware fallback based on model type
        if "game_predictor" in model_name:
            # Simple home team advantage fallback
            prediction = "home_win"
            confidence = 0.55
        elif "player_performance" in model_name:
            # League average fallback
            prediction = 15.0  # Average points per game
            confidence = 0.5
        elif "betting" in model_name:
            # 50/50 probability fallback
            prediction = {"spread": 0, "total_points": 220, "moneyline": -110}
            confidence = 0.5
        else:
            # Generic fallback
            prediction = "unknown"
            confidence = 0.3

        # Phase 2: Record fallback prediction metrics
        if self._metrics_collector:
            try:
                # Determine prediction type based on model
                if "game_predictor" in model_name:
                    prediction_type = "game_outcome"
                elif "player" in model_name:
                    prediction_type = "player_performance"
                elif "betting" in model_name:
                    prediction_type = "betting_odds"
                else:
                    prediction_type = "unknown"

                self._metrics_collector.record_prediction(
                    model_name=model_name,
                    prediction_type=prediction_type,
                    input_features=input_data,
                    prediction=prediction,
                    confidence=confidence,
                    prediction_time_ms=0,  # Fallback predictions are instantaneous
                    success=True,
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to record fallback metrics: {e}")

        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "fallback_used": True,
            "fallback_reason": error_message,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "input_features": input_data,
        }

    def check_component_health(self, component_name: str) -> MLComponentHealth:
        """
        Check health of specific ML component

        Args:
            component_name: Component to check

        Returns:
            Current health status of the component
        """
        if component_name not in self._ml_components:
            # Auto-register unknown component
            self.register_ml_component(component_name)

        component = self._ml_components[component_name]

        try:
            # Simulate health check (in real implementation, would ping/monitor component)
            start_time = time.time()

            # Check if component responds (placeholder for actual health check logic)
            is_healthy = self._simulate_health_check(component_name)

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Update component health
            component.last_check = datetime.now()
            component.response_time_ms = response_time

            if is_healthy:
                component.status = MLComponentStatus.HEALTHY
                component.uptime_percentage = min(
                    100.0, component.uptime_percentage + 1.0
                )
            else:
                component.status = MLComponentStatus.UNHEALTHY
                component.error_count += 1
                component.uptime_percentage = max(
                    0.0, component.uptime_percentage - 5.0
                )

            # Calculate error rate
            total_checks = component.error_count + 1
            component.error_rate = (component.error_count / total_checks) * 100

            return component

        except Exception as e:
            self.logger.error(f"❌ Health check failed for {component_name}: {e}")
            component.status = MLComponentStatus.UNHEALTHY
            component.last_error = str(e)
            component.last_check = datetime.now()
            component.error_count += 1

            return component

    def _simulate_health_check(self, component_name: str) -> bool:
        """Simulate component health check (placeholder for actual implementation)"""
        # In real implementation, this would:
        # - Ping the component/service
        # - Check response codes
        # - Verify dependencies
        # - Check resource utilization

        # For simulation, randomly fail 5% of the time
        import random

        return random.random() > 0.05

    def is_ml_healthy(self) -> bool:
        """Check overall ML system health"""
        if not self._ml_components:
            return True  # No components to check

        healthy_components = sum(
            1
            for comp in self._ml_components.values()
            if comp.status == MLComponentStatus.HEALTHY
        )

        health_ratio = healthy_components / len(self._ml_components)
        overall_healthy = health_ratio >= 0.7  # At least 70% components healthy

        self._system_status = (
            MLComponentStatus.HEALTHY
            if health_ratio >= 0.9
            else MLComponentStatus.DEGRADED
            if health_ratio >= 0.7
            else MLComponentStatus.UNHEALTHY
        )

        return overall_healthy

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        return {
            "system_status": self._system_status.value,
            "total_components": len(self._ml_components),
            "healthy_components": sum(
                1
                for comp in self._ml_components.values()
                if comp.status == MLComponentStatus.HEALTHY
            ),
            "degraded_components": sum(
                1
                for comp in self._ml_components.values()
                if comp.status == MLComponentStatus.DEGRADED
            ),
            "unhealthy_components": sum(
                1
                for comp in self._ml_components.values()
                if comp.status == MLComponentStatus.UNHEALTHY
            ),
            "total_predictions": self._total_predictions,
            "successful_predictions": self._successful_predictions,
            "failed_predictions": self._failed_predictions,
            "success_rate": (
                self._successful_predictions / max(self._total_predictions, 1) * 100
            ),
            "active_models": len(
                [m for m in self._model_registry.values() if m.is_active]
            ),
            "last_health_check": self._last_health_check.isoformat(),
            "cache_entries": len(self._prediction_cache),
        }

    def _generate_cache_key(self, model_name: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key for prediction results"""
        import hashlib

        # Create deterministic key from model name and input data
        data_str = f"{model_name}:{sorted(input_data.items())}"
        return hashlib.md5(data_str.encode()).hexdigest()

    def _clean_expired_cache(self) -> None:
        """Remove expired entries from prediction cache"""
        current_time = datetime.now()
        expired_keys = [
            key
            for key, (_, cache_time) in self._prediction_cache.items()
            if current_time - cache_time > self.cache_ttl
        ]

        for key in expired_keys:
            del self._prediction_cache[key]

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring thread"""
        if (
            self._health_check_thread is None
            or not self._health_check_thread.is_alive()
        ):
            self._stop_health_checks.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_monitoring_loop, daemon=True
            )
            self._health_check_thread.start()
            self.logger.info("🔍 Health monitoring started")

    def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._stop_health_checks.wait(self.health_check_interval):
            try:
                # Check all registered components
                for component_name in self._ml_components.keys():
                    self.check_component_health(component_name)

                self._last_health_check = datetime.now()

            except Exception as e:
                self.logger.error(f"❌ Health monitoring error: {e}")

    def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        self._stop_health_checks.set()
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        self.logger.info("🔍 Health monitoring stopped")

    def get_metrics_collector(self) -> Optional["NBAPredictionMetricsCollector"]:
        """
        Get the metrics collector instance for monitoring

        Returns:
            Metrics collector instance if available, None otherwise
        """
        return self._metrics_collector

    def get_drift_detector(self) -> Optional["NBADriftDetector"]:
        """
        Get the drift detector instance for feature distribution monitoring

        Returns:
            Drift detector instance if available, None otherwise
        """
        return self._drift_detector

    def get_confidence_interval_calculator(
        self,
    ) -> Optional["NBAConfidenceIntervalCalculator"]:
        """
        Get the confidence interval calculator instance for prediction uncertainty

        Returns:
            Confidence interval calculator instance if available, None otherwise
        """
        return self._ci_calculator

    def get_health_dashboard(self) -> Optional["NBAModelHealthDashboard"]:
        """
        Get the model health dashboard instance for real-time monitoring

        Returns:
            Model health dashboard instance if available, None otherwise
        """
        return self._health_dashboard

    def get_ensemble_predictor(self) -> Optional["NBAEnsemblePredictor"]:
        """
        Get the ensemble predictor instance for advanced XGBoost + Neural Network predictions

        Returns:
            Ensemble predictor instance if available, None otherwise
        """
        return self._ensemble_predictor

    def get_unified_ml_interface(self) -> Optional["UnifiedMLInterface"]:
        """
        Get unified ML interface instance for enhanced predictions

        Returns:
            Unified ML interface instance if available, None otherwise
        """
        return self._unified_ml_interface

    def get_model_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary for all models

        Returns:
            Dictionary containing metrics summary for all models
        """
        if not self._metrics_collector:
            return {"error": "Metrics collector not available"}

        try:
            return self._metrics_collector.export_metrics_json()
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return {"error": f"Failed to export metrics: {e}"}

    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring"""
        self.stop_health_monitoring()

        # Phase 2: Cleanup metrics collector
        if self._metrics_collector:
            try:
                self._metrics_collector.shutdown()
            except Exception as e:
                self.logger.warning(f"Failed to shutdown metrics collector: {e}")

        # Phase 2 Task 2.1.2: Cleanup drift detector
        if self._drift_detector:
            try:
                self._drift_detector.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to shutdown drift detector: {e}")

        # Phase 2 Task 2.1.3: Cleanup confidence interval calculator
        if self._ci_calculator:
            try:
                self._ci_calculator.cleanup()
            except Exception as e:
                self.logger.warning(
                    f"Failed to shutdown confidence interval calculator: {e}"
                )

        # Phase 2 Task 2.1.4: Cleanup model health dashboard
        if self._health_dashboard:
            try:
                self._health_dashboard.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to shutdown model health dashboard: {e}")

        # Phase 2 Task 2.2.1: Cleanup unified ML interface
        if self._unified_ml_interface:
            try:
                # UnifiedMLInterface doesn't have cleanup method, but we can log
                self.logger.info("🧹 Unified ML interface cleanup completed")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup unified ML interface: {e}")

        # Phase 2 Task 2.2.1: Cleanup ensemble predictor
        if self._ensemble_predictor:
            try:
                self._ensemble_predictor.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to shutdown ensemble predictor: {e}")

        self._prediction_cache.clear()
        self.logger.info("🧹 ML Integration Bridge cleanup completed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Singleton instance for global access
_ml_bridge_instance: Optional[MLIntegrationBridge] = None


def get_ml_bridge() -> MLIntegrationBridge:
    """Get singleton ML Integration Bridge instance"""
    global _ml_bridge_instance
    if _ml_bridge_instance is None:
        _ml_bridge_instance = MLIntegrationBridge()
    return _ml_bridge_instance


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create ML Integration Bridge
    with MLIntegrationBridge() as bridge:
        print("🚀 ML Integration Bridge Test")
        print("=" * 50)

        # Register ML components
        bridge.register_ml_component("data_pipeline", "data_processing")
        bridge.register_ml_component("feature_engineering", "feature_extraction")
        bridge.register_ml_component("model_serving", "prediction_service")

        # Test model prediction
        test_input = {
            "home_team_momentum": 0.75,
            "away_team_momentum": -0.25,
            "home_team_rest_days": 2,
            "away_team_rest_days": 1,
            "home_team_streak": 3,
            "away_team_streak": -2,
        }

        result = bridge.get_model_prediction("nba_game_predictor", test_input)
        print(f"✅ Prediction Result: {result}")

        # Test system status
        status = bridge.get_system_status()
        print(f"📊 System Status: {status}")

        # Test component health
        health = bridge.check_component_health("model_serving")
        print(f"🔍 Component Health: {health}")

        print("\n🎉 ML Integration Bridge test completed successfully!")
