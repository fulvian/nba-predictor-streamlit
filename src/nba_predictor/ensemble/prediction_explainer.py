#!/usr/bin/env python3
"""
🧠 NBA Prediction Explainer - Task 2.2.3

Advanced prediction explanation system for NBA Ensemble Predictor using SHAP, LIME,
and custom feature attribution methods with DevStream SuperPowered architecture.

Author: NBA Predictive Analytics System
Date: 2025-01-11
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import joblib
from collections import defaultdict
import threading
from enum import Enum

# Task 2.2.3: Explainability libraries
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance

    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Existing imports
from sklearn.preprocessing import StandardScaler, RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods"""

    SHAP_VALUES = "shap_values"
    SHAP_GLOBAL = "shap_global"
    LIME_LOCAL = "lime_local"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_ABLATION = "feature_ablation"
    CUSTOM_ATTRIBUTION = "custom_attribution"


@dataclass
class ExplanationConfig:
    """Configuration for prediction explanation system"""

    # SHAP configuration
    shap_background_samples: int = 100
    shap_n_samples: int = 1000
    shap_approximate: bool = True

    # LIME configuration
    lime_n_samples: int = 5000
    lime_feature_selection: str = "auto"
    lime_kernel_width: float = 3.0

    # Custom methods configuration
    ablation_n_samples: int = 100
    permutation_n_repeats: int = 10

    # Performance settings
    cache_explanations: bool = True
    max_cache_size: int = 1000
    parallel_processing: bool = True

    # Output settings
    include_visualizations: bool = True
    output_format: str = "dict"  # "dict", "json", "plotly"
    explanation_depth: str = "detailed"  # "basic", "detailed", "comprehensive"


@dataclass
class FeatureImportance:
    """Feature importance with explanation context"""

    feature_name: str
    importance: float
    direction: str  # "positive", "negative", "neutral"
    explanation: str
    confidence: float
    attribution_method: str

    # Additional context for NBA features
    nba_context: Optional[Dict[str, Any]] = None
    feature_category: Optional[str] = None  # "offensive", "defensive", "team_stats"


@dataclass
class PredictionExplanation:
    """Complete prediction explanation"""

    prediction_id: str
    prediction_value: float
    predicted_class: str
    confidence: float

    # Feature importance data
    feature_importances: List[FeatureImportance]
    top_features: List[FeatureImportance]

    # Context and metadata (non-default fields must come first)
    explanation_timestamp: str
    explanation_methods_used: List[str]
    model_type: str
    input_features: Dict[str, float]

    # Method-specific explanations (default values)
    shap_explanation: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    custom_explanation: Optional[Dict[str, Any]] = None

    # Visualization data
    visualizations: Optional[Dict[str, Any]] = None

    # NBA-specific context
    game_context: Optional[Dict[str, Any]] = None
    betting_implications: Optional[Dict[str, Any]] = None


class NBAPredictionExplainer:
    """
    Advanced NBA prediction explanation system with SHAP, LIME, and custom methods.

    DevStream SuperPowered with ContextSet compliance.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None):
        """
        Initialize NBA Prediction Explainer.

        Args:
            config: Configuration for explanation system
        """
        self.config = config or ExplanationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Explainer components
        self._shap_explainer_xgb = None
        self._shap_explainer_nn = None
        self._lime_explainer = None
        self._feature_names = []
        self._feature_categories = self._get_nba_feature_categories()

        # Caching system
        self._explanation_cache = {}
        self._cache_lock = threading.RLock()

        # Background data for SHAP
        self._background_data = None

        # Model references
        self._xgb_model = None
        self._nn_model = None
        self._xgb_scaler = None
        self._nn_scaler = None

        # Performance tracking
        self._explanation_stats = defaultdict(int)

        self.logger.info(
            "🧠 NBA Prediction Explainer initialized with SuperPowered features"
        )
        self.logger.info(f"   - SHAP available: {SHAP_AVAILABLE}")
        self.logger.info(f"   - LIME available: {LIME_AVAILABLE}")
        self.logger.info(f"   - ELI5 available: {ELI5_AVAILABLE}")
        self.logger.info(f"   - Plotting available: {PLOTTING_AVAILABLE}")
        self.logger.info(f"   - Plotly available: {PLOTLY_AVAILABLE}")

    def initialize_with_models(
        self,
        xgb_model,
        nn_model,
        feature_names: List[str],
        xgb_scaler=None,
        nn_scaler=None,
        background_data=None,
    ):
        """
        Initialize explainer with trained models.

        Args:
            xgb_model: Trained XGBoost model
            nn_model: Trained Neural Network model
            feature_names: List of feature names
            xgb_scaler: XGBoost feature scaler
            nn_scaler: Neural Network feature scaler
            background_data: Background data for SHAP explanations
        """
        try:
            with self._cache_lock:
                self._xgb_model = xgb_model
                self._nn_model = nn_model
                self._feature_names = feature_names or []
                self._xgb_scaler = xgb_scaler or RobustScaler()
                self._nn_scaler = nn_scaler or StandardScaler()

                # Set background data
                if background_data is not None:
                    self._background_data = background_data
                else:
                    # Create synthetic background data if none provided
                    self._background_data = self._create_synthetic_background_data()

                # Initialize explainers
                if SHAP_AVAILABLE and xgb_model is not None:
                    self._initialize_shap_explainers()

                if LIME_AVAILABLE:
                    self._initialize_lime_explainer()

                self.logger.info("✅ NBA Prediction Explainer initialized with models")
                self.logger.info(f"   - Feature names: {len(self._feature_names)}")
                self.logger.info(
                    f"   - Background data: {self._background_data.shape if self._background_data is not None else 'None'}"
                )

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize explainer with models: {e}")
            raise

    def _initialize_shap_explainers(self):
        """Initialize SHAP explainers for XGBoost and Neural Network."""
        try:
            if self._background_data is None or len(self._background_data) == 0:
                self.logger.warning("⚠️ No background data available for SHAP")
                return

            # Sample background data for efficiency
            bg_samples = min(
                self.config.shap_background_samples, len(self._background_data)
            )
            background_subset = self._background_data[:bg_samples]

            # Initialize XGBoost SHAP explainer
            if self._xgb_model is not None:
                self._shap_explainer_xgb = shap.TreeExplainer(
                    self._xgb_model,
                    data=background_subset,
                    approximate=self.config.shap_approximate,
                )
                self.logger.info("✅ SHAP TreeExplainer initialized for XGBoost")

            # Initialize Neural Network SHAP explainer
            if self._nn_model is not None and SHAP_AVAILABLE:
                # Create a wrapper function for the neural network
                def nn_predict_fn(X):
                    X_scaled = self._nn_scaler.transform(X)
                    return self._nn_model.predict(X_scaled)

                self._shap_explainer_nn = shap.KernelExplainer(
                    nn_predict_fn, data=background_subset
                )
                self.logger.info(
                    "✅ SHAP KernelExplainer initialized for Neural Network"
                )

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SHAP explainers: {e}")

    def _initialize_lime_explainer(self):
        """Initialize LIME explainer."""
        try:
            if self._background_data is None or len(self._background_data) == 0:
                self.logger.warning("⚠️ No background data available for LIME")
                return

            # Create wrapper function for LIME (use XGBoost as primary)
            def predict_fn(X):
                X_scaled = self._xgb_scaler.transform(X)
                probs = self._xgb_model.predict_proba(X_scaled)
                return probs

            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self._background_data,
                feature_names=self._feature_names,
                mode="classification",
                feature_selection=self.config.lime_feature_selection,
                kernel_width=self.config.lime_kernel_width,
                discretize_continuous=True,
            )
            self.logger.info("✅ LIME TabularExplainer initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LIME explainer: {e}")

    def explain_prediction(
        self,
        input_features: Dict[str, float],
        prediction_value: float,
        predicted_class: str,
        confidence: float = None,
        methods: List[ExplanationMethod] = None,
        explanation_id: str = None,
    ) -> PredictionExplanation:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            input_features: Input feature dictionary
            prediction_value: Model prediction value
            predicted_class: Predicted class label
            confidence: Prediction confidence
            methods: List of explanation methods to use
            explanation_id: Unique identifier for explanation

        Returns:
            PredictionExplanation: Comprehensive explanation object
        """
        try:
            # Generate explanation ID if not provided
            if explanation_id is None:
                explanation_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Check cache first
            if self.config.cache_explanations:
                cache_key = self._generate_cache_key(input_features, methods)
                if cache_key in self._explanation_cache:
                    self.logger.debug(
                        f"🎯 Retrieved explanation from cache: {explanation_id}"
                    )
                    return self._explanation_cache[cache_key]

            # Set default methods
            if methods is None:
                methods = [
                    ExplanationMethod.SHAP_VALUES,
                    ExplanationMethod.CUSTOM_ATTRIBUTION,
                ]

            # Convert features to numpy array
            features_array = self._features_to_array(input_features)

            # Generate explanations using different methods
            explanations = {}

            for method in methods:
                try:
                    if method == ExplanationMethod.SHAP_VALUES and SHAP_AVAILABLE:
                        explanations[method.value] = self._generate_shap_explanation(
                            features_array
                        )
                        self._explanation_stats[f"shap_explanations"] += 1

                    elif method == ExplanationMethod.LIME_LOCAL and LIME_AVAILABLE:
                        explanations[method.value] = self._generate_lime_explanation(
                            features_array
                        )
                        self._explanation_stats[f"lime_explanations"] += 1

                    elif method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                        explanations[method.value] = (
                            self._generate_permutation_importance(features_array)
                        )
                        self._explanation_stats[f"permutation_explanations"] += 1

                    elif method == ExplanationMethod.FEATURE_ABLATION:
                        explanations[method.value] = self._generate_feature_ablation(
                            features_array
                        )
                        self._explanation_stats[f"ablation_explanations"] += 1

                    elif method == ExplanationMethod.CUSTOM_ATTRIBUTION:
                        explanations[method.value] = self._generate_custom_attribution(
                            features_array
                        )
                        self._explanation_stats[f"custom_explanations"] += 1

                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to generate {method.value} explanation: {e}"
                    )
                    continue

            # Combine explanations
            feature_importances = self._combine_explanations(explanations)

            # Get top features
            top_features = sorted(
                feature_importances, key=lambda x: abs(x.importance), reverse=True
            )[:10]

            # Generate NBA-specific context
            nba_context = self._generate_nba_context(input_features, top_features)
            betting_implications = self._generate_betting_implications(
                top_features, prediction_value
            )

            # Generate visualizations if requested
            visualizations = None
            if self.config.include_visualizations and PLOTTING_AVAILABLE:
                visualizations = self._generate_visualizations(
                    feature_importances, explanation_id
                )

            # Create explanation object
            explanation = PredictionExplanation(
                prediction_id=explanation_id,
                prediction_value=prediction_value,
                predicted_class=predicted_class,
                confidence=confidence or 0.0,
                feature_importances=feature_importances,
                top_features=top_features,
                shap_explanation=explanations.get("shap_values"),
                lime_explanation=explanations.get("lime_local"),
                custom_explanation=explanations.get("custom_attribution"),
                explanation_timestamp=datetime.now().isoformat(),
                explanation_methods_used=[method.value for method in methods],
                model_type="nba_ensemble",
                input_features=input_features,
                visualizations=visualizations,
                game_context=nba_context,
                betting_implications=betting_implications,
            )

            # Cache explanation
            if self.config.cache_explanations:
                with self._cache_lock:
                    if len(self._explanation_cache) < self.config.max_cache_size:
                        self._explanation_cache[cache_key] = explanation

            self.logger.info(
                f"✅ Generated comprehensive explanation: {explanation_id}"
            )
            return explanation

        except Exception as e:
            self.logger.error(f"❌ Failed to generate explanation: {e}")
            raise

    def _generate_shap_explanation(self, features_array: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        try:
            shap_data = {}

            # XGBoost SHAP values
            if self._shap_explainer_xgb is not None and self._xgb_model is not None:
                features_scaled = self._xgb_scaler.transform(
                    features_array.reshape(1, -1)
                )
                shap_values = self._shap_explainer_xgb.shap_values(features_scaled)

                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # Multi-class case
                    shap_values = shap_values[1]  # Use positive class

                shap_data["xgboost_shap_values"] = shap_values[0].tolist()
                shap_data["xgboost_base_value"] = (
                    self._shap_explainer_xgb.expected_value
                )

                if not isinstance(self._shap_explainer_xgb.expected_value, np.ndarray):
                    shap_data["xgboost_base_value"] = float(
                        self._shap_explainer_xgb.expected_value
                    )
                else:
                    shap_data["xgboost_base_value"] = float(
                        self._shap_explainer_xgb.expected_value[0]
                    )

            # Neural Network SHAP values
            if self._shap_explainer_nn is not None and self._nn_model is not None:
                features_scaled_nn = self._nn_scaler.transform(
                    features_array.reshape(1, -1)
                )
                shap_values_nn = self._shap_explainer_nn.shap_values(features_scaled_nn)

                # Handle Neural Network SHAP format
                if isinstance(shap_values_nn, list):
                    shap_values_nn = shap_values_nn[0]

                shap_data["neural_network_shap_values"] = shap_values_nn[0].tolist()
                shap_data["neural_network_base_value"] = float(
                    self._shap_explainer_nn.expected_value
                )

            return shap_data

        except Exception as e:
            self.logger.error(f"❌ SHAP explanation generation failed: {e}")
            return {}

    def _generate_lime_explanation(self, features_array: np.ndarray) -> Dict[str, Any]:
        """Generate LIME-based explanation."""
        try:
            if self._lime_explainer is None:
                return {}

            # Use XGBoost prediction function for LIME
            def predict_fn(X):
                X_scaled = self._xgb_scaler.transform(X)
                probs = self._xgb_model.predict_proba(X_scaled)
                return probs

            # Create LIME explainer with current training data
            lime_exp = self._lime_explainer.explain_instance(
                data_row=features_array,
                predict_fn=predict_fn,
                num_features=15,
                num_samples=self.config.lime_n_samples,
            )

            # Extract feature importances
            lime_data = {
                "local_importance": [],
                "intercept": lime_exp.intercept[1],  # Positive class intercept
                "score": lime_exp.score,
                "local_pred": lime_exp.local_pred[1]
                if len(lime_exp.local_pred) > 1
                else lime_exp.local_pred[0],
            }

            for feature, importance in lime_exp.as_list():
                lime_data["local_importance"].append(
                    {"feature": feature, "importance": importance}
                )

            return lime_data

        except Exception as e:
            self.logger.error(f"❌ LIME explanation generation failed: {e}")
            return {}

    def _generate_permutation_importance(
        self, features_array: np.ndarray
    ) -> Dict[str, Any]:
        """Generate permutation importance-based explanation."""
        try:
            if not ELI5_AVAILABLE or self._xgb_model is None:
                return {}

            # Create evaluation dataset (small subset for performance)
            eval_data = (
                self._background_data[:50]
                if self._background_data is not None
                else features_array.reshape(1, -1)
            )

            # Calculate permutation importance
            perm_importance = PermutationImportance(
                self._xgb_model,
                scoring="accuracy",
                n_iter=self.config.permutation_n_repeats,
            ).fit(eval_data, np.zeros(len(eval_data)))  # Dummy targets

            # Extract importance scores
            importance_data = {}
            for i, feature_name in enumerate(self._feature_names):
                if i < len(perm_importance.feature_importances_):
                    importance_data[feature_name] = {
                        "importance": float(perm_importance.feature_importances_[i]),
                        "std": float(perm_importance.feature_importances_std_[i])
                        if hasattr(perm_importance, "feature_importances_std_")
                        else 0.0,
                    }

            return {
                "permutation_importance": importance_data,
                "feature_names": self._feature_names[
                    : len(perm_importance.feature_importances_)
                ],
            }

        except Exception as e:
            self.logger.error(f"❌ Permutation importance generation failed: {e}")
            return {}

    def _generate_feature_ablation(self, features_array: np.ndarray) -> Dict[str, Any]:
        """Generate feature ablation-based explanation."""
        try:
            if self._xgb_model is None:
                return {}

            # Get baseline prediction
            features_scaled = self._xgb_scaler.transform(features_array.reshape(1, -1))
            baseline_pred = self._xgb_model.predict_proba(features_scaled)[0, 1]

            ablation_results = {}

            # Ablate each feature (set to mean/zero)
            for i, feature_name in enumerate(self._feature_names):
                if i >= features_array.shape[1]:
                    continue

                # Create modified features with current feature ablated
                modified_features = features_array.copy()
                # Set to mean of background data or 0
                if self._background_data is not None:
                    modified_features[i] = np.mean(self._background_data[:, i])
                else:
                    modified_features[i] = 0.0

                # Get prediction with ablated feature
                modified_scaled = self._xgb_scaler.transform(
                    modified_features.reshape(1, -1)
                )
                ablated_pred = self._xgb_model.predict_proba(modified_scaled)[0, 1]

                # Calculate importance (difference from baseline)
                importance = baseline_pred - ablated_pred

                ablation_results[feature_name] = {
                    "importance": float(importance),
                    "baseline_prediction": float(baseline_pred),
                    "ablated_prediction": float(ablated_pred),
                }

            return {
                "feature_ablation": ablation_results,
                "baseline_prediction": float(baseline_pred),
            }

        except Exception as e:
            self.logger.error(f"❌ Feature ablation generation failed: {e}")
            return {}

    def _generate_custom_attribution(
        self, features_array: np.ndarray
    ) -> Dict[str, Any]:
        """Generate custom feature attribution explanation."""
        try:
            if self._xgb_model is None:
                return {}

            # Get baseline prediction
            features_scaled = self._xgb_scaler.transform(features_array.reshape(1, -1))
            baseline_pred = self._xgb_model.predict_proba(features_scaled)[0, 1]

            # Use multiple attribution methods
            attribution_results = {}

            # 1. Feature importance from XGBoost
            if hasattr(self._xgb_model, "feature_importances_"):
                for i, feature_name in enumerate(self._feature_names):
                    if i < len(self._xgb_model.feature_importances_):
                        attribution_results[f"{feature_name}_xgb_importance"] = float(
                            self._xgb_model.feature_importances_[i]
                        )

            # 2. Gradient-based attribution (simplified)
            for i, feature_name in enumerate(self._feature_names):
                if i >= features_array.shape[1]:
                    continue

                # Perturb feature slightly
                epsilon = 0.01
                perturbed_features = features_array.copy()
                perturbed_features[i] += epsilon

                perturbed_scaled = self._xgb_scaler.transform(
                    perturbed_features.reshape(1, -1)
                )
                perturbed_pred = self._xgb_model.predict_proba(perturbed_scaled)[0, 1]

                # Approximate gradient
                gradient = (perturbed_pred - baseline_pred) / epsilon
                attribution_results[f"{feature_name}_gradient"] = float(gradient)

            # 3. Feature contribution using partial dependence idea
            for i, feature_name in enumerate(self._feature_names):
                if i >= features_array.shape[1] or self._background_data is None:
                    continue

                # Calculate average prediction when feature is at different values
                feature_values = np.percentile(
                    self._background_data[:, i], [10, 50, 90]
                )
                partial_deps = []

                for value in feature_values:
                    temp_features = features_array.copy()
                    temp_features[i] = value
                    temp_scaled = self._xgb_scaler.transform(
                        temp_features.reshape(1, -1)
                    )
                    pred = self._xgb_model.predict_proba(temp_scaled)[0, 1]
                    partial_deps.append(float(pred))

                attribution_results[f"{feature_name}_partial_dependence"] = partial_deps

            return {
                "custom_attribution": attribution_results,
                "baseline_prediction": float(baseline_pred),
                "methods_used": ["xgb_importance", "gradient", "partial_dependence"],
            }

        except Exception as e:
            self.logger.error(f"❌ Custom attribution generation failed: {e}")
            return {}

    def _combine_explanations(
        self, explanations: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """Combine explanations from different methods into unified feature importance."""
        try:
            feature_scores = defaultdict(
                lambda: {"total": 0.0, "count": 0, "direction": "neutral"}
            )

            # Process SHAP explanations
            shap_exp = explanations.get("shap_values", {})
            if "xgboost_shap_values" in shap_exp and self._feature_names:
                shap_values = shap_exp["xgboost_shap_values"]
                for i, shap_val in enumerate(shap_values):
                    if i < len(self._feature_names):
                        feature_name = self._feature_names[i]
                        feature_scores[feature_name]["total"] += abs(shap_val)
                        feature_scores[feature_name]["count"] += 1
                        feature_scores[feature_name]["direction"] = (
                            "positive" if shap_val > 0 else "negative"
                        )

            # Process LIME explanations
            lime_exp = explanations.get("lime_local", {})
            if "local_importance" in lime_exp:
                for item in lime_exp["local_importance"]:
                    feature_name = item["feature"]
                    importance = abs(item["importance"])
                    feature_scores[feature_name]["total"] += importance
                    feature_scores[feature_name]["count"] += 1
                    feature_scores[feature_name]["direction"] = (
                        "positive" if item["importance"] > 0 else "negative"
                    )

            # Process permutation importance
            perm_exp = explanations.get("permutation_importance", {})
            if "permutation_importance" in perm_exp:
                for feature_name, data in perm_exp["permutation_importance"].items():
                    importance = abs(data["importance"])
                    feature_scores[feature_name]["total"] += importance
                    feature_scores[feature_name]["count"] += 1
                    feature_scores[feature_name]["direction"] = (
                        "positive" if data["importance"] > 0 else "negative"
                    )

            # Process feature ablation
            ablation_exp = explanations.get("feature_ablation", {})
            if "feature_ablation" in ablation_exp:
                for feature_name, data in ablation_exp["feature_ablation"].items():
                    importance = abs(data["importance"])
                    feature_scores[feature_name]["total"] += importance
                    feature_scores[feature_name]["count"] += 1
                    feature_scores[feature_name]["direction"] = (
                        "positive" if data["importance"] > 0 else "negative"
                    )

            # Create FeatureImportance objects
            feature_importances = []
            for feature_name, scores in feature_scores.items():
                if scores["count"] > 0:
                    avg_importance = scores["total"] / scores["count"]

                    # Generate explanation
                    explanation = self._generate_feature_explanation(
                        feature_name, avg_importance, scores["direction"]
                    )

                    # Get NBA category
                    category = self._feature_categories.get(feature_name, "other")

                    fi = FeatureImportance(
                        feature_name=feature_name,
                        importance=avg_importance,
                        direction=scores["direction"],
                        explanation=explanation,
                        confidence=min(scores["count"] / len(explanations), 1.0),
                        attribution_method="combined",
                        feature_category=category,
                        nba_context=self._get_nba_feature_context(feature_name),
                    )
                    feature_importances.append(fi)

            return sorted(feature_importances, key=lambda x: x.importance, reverse=True)

        except Exception as e:
            self.logger.error(f"❌ Failed to combine explanations: {e}")
            return []

    def _generate_feature_explanation(
        self, feature_name: str, importance: float, direction: str
    ) -> str:
        """Generate human-readable explanation for feature importance."""
        try:
            # NBA-specific explanations based on feature name and importance
            if "momentum" in feature_name.lower():
                if direction == "positive":
                    return f"Team momentum contributes positively to prediction (+{importance:.3f})"
                else:
                    return f"Team momentum negatively impacts prediction (-{importance:.3f})"

            elif "win_rate" in feature_name.lower():
                if direction == "positive":
                    return f"Higher win rate increases prediction confidence (+{importance:.3f})"
                else:
                    return f"Lower win rate reduces prediction confidence (-{importance:.3f})"

            elif "rest_days" in feature_name.lower():
                if direction == "positive":
                    return (
                        f"More rest days favor prediction outcome (+{importance:.3f})"
                    )
                else:
                    return f"Fewer rest days impact prediction negatively (-{importance:.3f})"

            elif "back_to_back" in feature_name.lower():
                if direction == "positive":
                    return (
                        f"Back-to-back situation favors prediction (+{importance:.3f})"
                    )
                else:
                    return (
                        f"Back-to-back fatigue impacts prediction (-{importance:.3f})"
                    )

            elif "points_per_game" in feature_name.lower():
                if direction == "positive":
                    return f"Higher scoring performance supports prediction (+{importance:.3f})"
                else:
                    return f"Lower scoring performance weakens prediction (-{importance:.3f})"

            elif "field_goal_percentage" in feature_name.lower():
                if direction == "positive":
                    return f"Better shooting percentage favors prediction (+{importance:.3f})"
                else:
                    return f"Poor shooting percentage impacts prediction (-{importance:.3f})"

            elif "rebounds" in feature_name.lower():
                if direction == "positive":
                    return f"Strong rebounding performance supports prediction (+{importance:.3f})"
                else:
                    return f"Weak rebounding impacts prediction (-{importance:.3f})"

            elif "assists" in feature_name.lower():
                if direction == "positive":
                    return f"Better ball movement favors prediction (+{importance:.3f})"
                else:
                    return f"Poor ball movement impacts prediction (-{importance:.3f})"

            elif "turnovers" in feature_name.lower():
                # Turnovers are typically negative
                if direction == "positive":
                    return f"Fewer turnovers favor prediction (+{importance:.3f})"
                else:
                    return f"More turnovers negatively impact prediction (-{importance:.3f})"

            else:
                return f"Feature {feature_name} {'positively' if direction == 'positive' else 'negatively'} impacts prediction ({importance:.3f})"

        except Exception as e:
            self.logger.error(f"❌ Failed to generate feature explanation: {e}")
            return f"Feature {feature_name} importance: {importance:.3f} ({direction})"

    def _get_nba_feature_categories(self) -> Dict[str, str]:
        """Get NBA feature categorization."""
        return {
            # Momentum features
            "home_team_momentum": "momentum",
            "away_team_momentum": "momentum",
            # Rest and scheduling
            "home_team_rest_days": "scheduling",
            "away_team_rest_days": "scheduling",
            "home_team_back_to_back": "scheduling",
            "away_team_back_to_back": "scheduling",
            # Team performance
            "home_team_win_rate": "performance",
            "away_team_win_rate": "performance",
            "home_team_points_per_game": "offensive",
            "away_team_points_per_game": "offensive",
            # Shooting statistics
            "home_team_field_goal_percentage": "offensive",
            "away_team_field_goal_percentage": "offensive",
            "home_team_three_point_percentage": "offensive",
            "away_team_three_point_percentage": "offensive",
            "home_team_free_throw_percentage": "offensive",
            "away_team_free_throw_percentage": "offensive",
            # Rebounding
            "home_team_offensive_rebounds_per_game": "rebounds",
            "away_team_offensive_rebounds_per_game": "rebounds",
            "home_team_defensive_rebounds_per_game": "rebounds",
            "away_team_defensive_rebounds_per_game": "rebounds",
            # Ball movement
            "home_team_assists_per_game": "ball_control",
            "away_team_assists_per_game": "ball_control",
            # Defense
            "home_team_steals_per_game": "defensive",
            "away_team_steals_per_game": "defensive",
            "home_team_blocks_per_game": "defensive",
            "away_team_blocks_per_game": "defensive",
            # Ball control
            "home_team_turnovers_per_game": "ball_control",
            "away_team_turnovers_per_game": "ball_control",
            # Fouls
            "home_team_personal_fouls_per_game": "fouls",
            "away_team_personal_fouls_per_game": "fouls",
        }

    def _get_nba_feature_context(self, feature_name: str) -> Dict[str, Any]:
        """Get NBA-specific context for a feature."""
        context = {
            "category": self._feature_categories.get(feature_name, "other"),
            "description": self._get_feature_description(feature_name),
            "basketball_relevance": self._get_basketball_relevance(feature_name),
        }
        return context

    def _get_feature_description(self, feature_name: str) -> str:
        """Get descriptive explanation of NBA feature."""
        descriptions = {
            "home_team_momentum": "Recent performance trend of the home team",
            "away_team_momentum": "Recent performance trend of the away team",
            "home_team_rest_days": "Number of days since home team's last game",
            "away_team_rest_days": "Number of days since away team's last game",
            "home_team_back_to_back": "Home team playing second game in consecutive nights",
            "away_team_back_to_back": "Away team playing second game in consecutive nights",
            "home_team_win_rate": "Home team's winning percentage this season",
            "away_team_win_rate": "Away team's winning percentage this season",
            "home_team_points_per_game": "Average points scored per game by home team",
            "away_team_points_per_game": "Average points scored per game by away team",
            "home_team_field_goal_percentage": "Home team's field goal shooting percentage",
            "away_team_field_goal_percentage": "Away team's field goal shooting percentage",
            "home_team_three_point_percentage": "Home team's three-point shooting percentage",
            "away_team_three_point_percentage": "Away team's three-point shooting percentage",
            "home_team_free_throw_percentage": "Home team's free throw shooting percentage",
            "away_team_free_throw_percentage": "Away team's free throw shooting percentage",
            "home_team_offensive_rebounds_per_game": "Home team's average offensive rebounds per game",
            "away_team_offensive_rebounds_per_game": "Away team's average offensive rebounds per game",
            "home_team_defensive_rebounds_per_game": "Home team's average defensive rebounds per game",
            "away_team_defensive_rebounds_per_game": "Away team's average defensive rebounds per game",
            "home_team_assists_per_game": "Home team's average assists per game",
            "away_team_assists_per_game": "Away team's average assists per game",
            "home_team_steals_per_game": "Home team's average steals per game",
            "away_team_steals_per_game": "Away team's average steals per game",
            "home_team_blocks_per_game": "Home team's average blocks per game",
            "away_team_blocks_per_game": "Away team's average blocks per game",
            "home_team_turnovers_per_game": "Home team's average turnovers per game",
            "away_team_turnovers_per_game": "Away team's average turnovers per game",
            "home_team_personal_fouls_per_game": "Home team's average personal fouls per game",
            "away_team_personal_fouls_per_game": "Away team's average personal fouls per game",
        }
        return descriptions.get(
            feature_name, f"NBA statistical feature: {feature_name}"
        )

    def _get_basketball_relevance(self, feature_name: str) -> str:
        """Get basketball-specific relevance of feature."""
        relevance_map = {
            "momentum": "Critical for predicting performance trends",
            "scheduling": "Important for player fatigue and preparation",
            "performance": "Direct indicator of team quality",
            "offensive": "Key factor in scoring potential",
            "defensive": "Key factor in opponent scoring prevention",
            "rebounds": "Crucial for possession control",
            "ball_control": "Important for turnover prevention",
            "fouls": "Affects player availability and strategy",
        }
        category = self._feature_categories.get(feature_name, "other")
        return relevance_map.get(category, "Standard NBA statistical metric")

    def _generate_nba_context(
        self, input_features: Dict[str, float], top_features: List[FeatureImportance]
    ) -> Dict[str, Any]:
        """Generate NBA-specific context for the prediction."""
        try:
            # Analyze key NBA factors
            home_advantage = self._analyze_home_advantage(input_features)
            fatigue_factors = self._analyze_fatigue_factors(input_features)
            momentum_analysis = self._analyze_momentum(input_features)

            # Determine game type context
            game_type = self._classify_game_type(input_features)

            # Key insights
            insights = []

            # Home court advantage analysis
            if home_advantage["significant"]:
                insights.append(
                    {
                        "factor": "Home Court Advantage",
                        "impact": home_advantage["impact"],
                        "description": home_advantage["explanation"],
                    }
                )

            # Fatigue analysis
            if fatigue_factors["high_fatigue_risk"]:
                insights.append(
                    {
                        "factor": "Fatigue Risk",
                        "impact": "negative",
                        "description": fatigue_factors["explanation"],
                    }
                )

            # Momentum analysis
            if momentum_analysis["strong_momentum"]:
                insights.append(
                    {
                        "factor": "Team Momentum",
                        "impact": momentum_analysis["direction"],
                        "description": momentum_analysis["explanation"],
                    }
                )

            return {
                "home_advantage": home_advantage,
                "fatigue_factors": fatigue_factors,
                "momentum_analysis": momentum_analysis,
                "game_type": game_type,
                "key_insights": insights,
                "critical_factors": [f.feature_name for f in top_features[:5]],
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to generate NBA context: {e}")
            return {}

    def _analyze_home_advantage(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze home court advantage factors."""
        try:
            home_momentum = features.get("home_team_momentum", 0)
            away_momentum = features.get("away_team_momentum", 0)
            home_rest = features.get("home_team_rest_days", 0)
            away_rest = features.get("away_team_rest_days", 0)

            # Calculate home advantage score
            momentum_diff = home_momentum - away_momentum
            rest_diff = home_rest - away_rest

            advantage_score = momentum_diff * 0.6 + rest_diff * 0.4

            significant = abs(advantage_score) > 0.2
            impact = "positive" if advantage_score > 0 else "negative"

            explanation = ""
            if significant:
                if advantage_score > 0:
                    explanation = f"Home team shows {'strong' if advantage_score > 0.5 else 'moderate'} advantage due to {'momentum' if momentum_diff > rest_diff else 'better rest'}"
                else:
                    explanation = f"Away team may have advantage due to {'momentum' if abs(momentum_diff) > abs(rest_diff) else 'better rest'}"
            else:
                explanation = "No significant home court advantage detected"

            return {
                "advantage_score": float(advantage_score),
                "significant": significant,
                "impact": impact,
                "explanation": explanation,
                "momentum_difference": float(momentum_diff),
                "rest_difference": float(rest_diff),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze home advantage: {e}")
            return {
                "significant": False,
                "impact": "neutral",
                "explanation": "Unable to analyze",
            }

    def _analyze_fatigue_factors(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze fatigue-related factors."""
        try:
            home_b2b = features.get("home_team_back_to_back", 0)
            away_b2b = features.get("away_team_back_to_back", 0)
            home_rest = features.get("home_team_rest_days", 0)
            away_rest = features.get("away_team_rest_days", 0)

            # Calculate fatigue risk
            b2b_impact = (
                home_b2b * 0.7 - away_b2b * 0.3
            )  # Home back-to-back is more impactful
            rest_impact = (away_rest - home_rest) * 0.2  # Less rest = more fatigue

            fatigue_score = b2b_impact + rest_impact
            high_fatigue_risk = abs(fatigue_score) > 0.5

            explanation = ""
            if high_fatigue_risk:
                if fatigue_score > 0:
                    explanation = f"Home team may be fatigued (back-to-back: {home_b2b}, rest: {home_rest} days)"
                else:
                    explanation = f"Away team may be fatigued (back-to-back: {away_b2b}, rest: {away_rest} days)"
            else:
                explanation = "No significant fatigue concerns"

            return {
                "fatigue_score": float(fatigue_score),
                "high_fatigue_risk": high_fatigue_risk,
                "explanation": explanation,
                "home_b2b": int(home_b2b),
                "away_b2b": int(away_b2b),
                "home_rest_days": int(home_rest),
                "away_rest_days": int(away_rest),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze fatigue factors: {e}")
            return {
                "high_fatigue_risk": False,
                "explanation": "Unable to analyze fatigue",
            }

    def _analyze_momentum(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze team momentum factors."""
        try:
            home_momentum = features.get("home_team_momentum", 0)
            away_momentum = features.get("away_team_momentum", 0)
            home_win_rate = features.get("home_team_win_rate", 0.5)
            away_win_rate = features.get("away_team_win_rate", 0.5)

            momentum_diff = home_momentum - away_momentum
            win_rate_diff = home_win_rate - away_win_rate

            # Overall momentum score
            momentum_score = momentum_diff * 0.6 + win_rate_diff * 0.4
            strong_momentum = abs(momentum_score) > 0.3

            direction = "positive" if momentum_score > 0 else "negative"

            explanation = ""
            if strong_momentum:
                if momentum_score > 0:
                    explanation = f"Home team has strong momentum (recent form: {home_momentum:.2f}, win rate: {home_win_rate:.1%})"
                else:
                    explanation = f"Away team has strong momentum (recent form: {away_momentum:.2f}, win rate: {away_win_rate:.1%})"
            else:
                explanation = "No strong momentum advantage detected"

            return {
                "momentum_score": float(momentum_score),
                "strong_momentum": strong_momentum,
                "direction": direction,
                "explanation": explanation,
                "home_momentum": float(home_momentum),
                "away_momentum": float(away_momentum),
                "home_win_rate": float(home_win_rate),
                "away_win_rate": float(away_win_rate),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze momentum: {e}")
            return {
                "strong_momentum": False,
                "direction": "neutral",
                "explanation": "Unable to analyze momentum",
            }

    def _classify_game_type(self, features: Dict[str, float]) -> str:
        """Classify the type of NBA game based on features."""
        try:
            home_win_rate = features.get("home_team_win_rate", 0.5)
            away_win_rate = features.get("away_team_win_rate", 0.5)

            # Classify based on team quality
            if home_win_rate > 0.6 and away_win_rate > 0.6:
                return "High-Quality Matchup"
            elif home_win_rate > 0.6 or away_win_rate > 0.6:
                return "Favored Team vs Underdog"
            elif home_win_rate < 0.4 and away_win_rate < 0.4:
                return "Struggling Teams"
            else:
                return "Evenly Matched"

        except Exception as e:
            self.logger.error(f"❌ Failed to classify game type: {e}")
            return "Standard Matchup"

    def _generate_betting_implications(
        self, top_features: List[FeatureImportance], prediction_value: float
    ) -> Dict[str, Any]:
        """Generate betting-related implications from feature analysis."""
        try:
            # Analyze confidence factors
            confidence_factors = []
            risk_factors = []

            for feature in top_features[:5]:  # Top 5 features
                if feature.importance > 0.1:
                    if feature.direction == "positive" and prediction_value > 0.5:
                        confidence_factors.append(feature.explanation)
                    elif feature.direction == "negative" and prediction_value < 0.5:
                        confidence_factors.append(feature.explanation)
                    else:
                        risk_factors.append(
                            f"Conflicting signal: {feature.explanation}"
                        )

            # Determine betting recommendation
            if prediction_value > 0.65:
                recommendation = "Strong Bet"
                confidence_level = "High"
            elif prediction_value > 0.55:
                recommendation = "Moderate Bet"
                confidence_level = "Medium"
            elif prediction_value > 0.45:
                recommendation = "No Clear Bet"
                confidence_level = "Low"
            else:
                recommendation = "Consider Against"
                confidence_level = "Medium"

            # Adjust confidence based on conflicting signals
            if len(risk_factors) > len(confidence_factors):
                confidence_level = "Low"
                recommendation = "Caution Recommended"

            return {
                "betting_recommendation": recommendation,
                "confidence_level": confidence_level,
                "prediction_confidence": float(prediction_value),
                "confidence_factors": confidence_factors,
                "risk_factors": risk_factors,
                "key_insights": [
                    f"Top feature: {top_features[0].feature_name if top_features else 'N/A'}",
                    f"Confidence: {confidence_level}",
                    f"Recommendation: {recommendation}",
                ],
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to generate betting implications: {e}")
            return {
                "betting_recommendation": "No Recommendation",
                "confidence_level": "Unknown",
            }

    def _generate_visualizations(
        self, feature_importances: List[FeatureImportance], explanation_id: str
    ) -> Dict[str, Any]:
        """Generate visualization data for explanations."""
        try:
            if not PLOTTING_AVAILABLE:
                return {}

            visualizations = {}

            # Feature importance bar chart data
            if feature_importances:
                top_10 = feature_importances[:10]
                visualizations["feature_importance_bar"] = {
                    "type": "bar",
                    "title": "Top 10 Feature Importance",
                    "features": [fi.feature_name for fi in top_10],
                    "importance": [fi.importance for fi in top_10],
                    "direction": [fi.direction for fi in top_10],
                    "explanations": [fi.explanation for fi in top_10],
                }

                # Feature importance by category
                categories = defaultdict(list)
                for fi in feature_importances:
                    cat = fi.feature_category or "other"
                    categories[cat].append(fi.importance)

                visualizations["importance_by_category"] = {
                    "type": "grouped_bar",
                    "title": "Feature Importance by Category",
                    "categories": list(categories.keys()),
                    "total_importance": [sum(imp) for imp in categories.values()],
                    "feature_count": [len(imp) for imp in categories.values()],
                }

            return visualizations

        except Exception as e:
            self.logger.error(f"❌ Failed to generate visualizations: {e}")
            return {}

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        try:
            if not self._feature_names:
                raise ValueError("Feature names not initialized")

            array = []
            for feature_name in self._feature_names:
                value = features.get(feature_name, 0.0)
                array.append(float(value))

            return np.array(array)

        except Exception as e:
            self.logger.error(f"❌ Failed to convert features to array: {e}")
            raise

    def _create_synthetic_background_data(self) -> np.ndarray:
        """Create synthetic background data for SHAP explanations."""
        try:
            # Create reasonable ranges for NBA features
            synthetic_data = []

            # Generate 100 synthetic samples
            for _ in range(100):
                sample = []

                for feature_name in self._feature_names:
                    if "momentum" in feature_name.lower():
                        sample.append(np.random.normal(0, 0.3))
                    elif "rest_days" in feature_name.lower():
                        sample.append(np.random.poisson(2))
                    elif "back_to_back" in feature_name.lower():
                        sample.append(np.random.binomial(1, 0.2))
                    elif "win_rate" in feature_name.lower():
                        sample.append(
                            np.random.beta(10, 10)
                        )  # Beta distribution for rates
                    elif "percentage" in feature_name.lower():
                        sample.append(np.random.normal(0.45, 0.05))
                    elif "per_game" in feature_name.lower():
                        if "points" in feature_name.lower():
                            sample.append(np.random.normal(110, 15))
                        elif "rebounds" in feature_name.lower():
                            sample.append(np.random.normal(42, 5))
                        elif "assists" in feature_name.lower():
                            sample.append(np.random.normal(25, 4))
                        elif "steals" in feature_name.lower():
                            sample.append(np.random.normal(8, 2))
                        elif "blocks" in feature_name.lower():
                            sample.append(np.random.normal(5, 2))
                        elif "turnovers" in feature_name.lower():
                            sample.append(np.random.normal(14, 3))
                        elif "fouls" in feature_name.lower():
                            sample.append(np.random.normal(20, 3))
                        else:
                            sample.append(np.random.normal(10, 5))
                    else:
                        sample.append(np.random.normal(0, 1))

                synthetic_data.append(sample)

            return np.array(synthetic_data)

        except Exception as e:
            self.logger.error(f"❌ Failed to create synthetic background data: {e}")
            # Return minimal background data
            return (
                np.random.randn(50, len(self._feature_names))
                if self._feature_names
                else np.random.randn(50, 25)
            )

    def _generate_cache_key(
        self, input_features: Dict[str, float], methods: List[ExplanationMethod]
    ) -> str:
        """Generate cache key for explanations."""
        try:
            # Sort features for consistent key generation
            sorted_features = sorted(input_features.items())
            feature_str = str(sorted_features)
            methods_str = str(sorted([m.value for m in methods]))

            # Create simple hash
            import hashlib

            key_str = feature_str + methods_str
            return hashlib.md5(key_str.encode()).hexdigest()[:16]

        except Exception:
            # Fallback to simple key
            return f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of explanation system statistics."""
        try:
            with self._cache_lock:
                return {
                    "total_explanations": sum(self._explanation_stats.values()),
                    "cache_size": len(self._explanation_cache),
                    "methods_available": {
                        "shap": SHAP_AVAILABLE,
                        "lime": LIME_AVAILABLE,
                        "eli5": ELI5_AVAILABLE,
                        "plotting": PLOTTING_AVAILABLE,
                        "plotly": PLOTLY_AVAILABLE,
                    },
                    "explanations_by_method": dict(self._explanation_stats),
                    "feature_categories": self._feature_categories,
                    "cache_enabled": self.config.cache_explanations,
                    "max_cache_size": self.config.max_cache_size,
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get explanation summary: {e}")
            return {}

    def save_explainer(self, filepath: str) -> bool:
        """Save explainer state to file."""
        try:
            state = {
                "config": asdict(self.config),
                "feature_names": self._feature_names,
                "feature_categories": self._feature_categories,
                "explanation_stats": dict(self._explanation_stats),
                "background_data": self._background_data.tolist()
                if self._background_data is not None
                else None,
                "timestamp": datetime.now().isoformat(),
            }

            # Note: We don't save the models or explainers themselves, just the configuration
            joblib.dump(state, filepath)
            self.logger.info(f"💾 Explainer configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save explainer: {e}")
            return False

    def load_explainer(self, filepath: str) -> bool:
        """Load explainer configuration from file."""
        try:
            state = joblib.load(filepath)

            self.config = ExplanationConfig(**state.get("config", {}))
            self._feature_names = state.get("feature_names", [])
            self._feature_categories = state.get("feature_categories", {})
            self._explanation_stats = defaultdict(
                int, state.get("explanation_stats", {})
            )

            bg_data = state.get("background_data")
            if bg_data is not None:
                self._background_data = np.array(bg_data)

            self.logger.info(f"📂 Explainer configuration loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load explainer: {e}")
            return False

    def cleanup(self):
        """Cleanup explainer resources."""
        try:
            with self._cache_lock:
                self._explanation_cache.clear()
                self._shap_explainer_xgb = None
                self._shap_explainer_nn = None
                self._lime_explainer = None
                self._background_data = None
                self._xgb_model = None
                self._nn_model = None

            self.logger.info("🧹 NBA Prediction Explainer cleanup completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup explainer: {e}")
