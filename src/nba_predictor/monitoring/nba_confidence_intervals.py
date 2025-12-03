#!/usr/bin/env python3
"""
📊 NBA Confidence Interval Calculator - Task 2.1.3 Implementation

Sistema di calcolo confidence intervals per NBA predictions usando metodi statistici avanzati.
Implementa quantile regression, bootstrap methods, e prediction intervals robusti.

Author: NBA Predictive Analytics System
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import warnings
from abc import ABC, abstractmethod

# Statistical libraries
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logger = logging.getLogger(__name__)

class ConfidenceIntervalMethod(Enum):
    """Enumeration of confidence interval calculation methods"""
    BOOTSTRAP = "bootstrap"
    QUANTILE_REGRESSION = "quantile_regression"
    PREDICTION_INTERVAL = "prediction_interval"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"

@dataclass
class ConfidenceIntervalConfig:
    """Configuration for confidence interval calculations"""

    # Method settings
    primary_method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.ADAPTIVE
    confidence_levels: List[float] = field(default_factory=lambda: [0.50, 0.80, 0.90, 0.95])

    # Bootstrap settings
    n_bootstrap_samples: int = 1000
    bootstrap_random_state: int = 42

    # Quantile regression settings
    quantile_alpha: float = 0.5  # Median for primary prediction
    n_quantile_estimators: int = 100
    quantile_learning_rate: float = 0.1

    # Ensemble settings
    n_ensemble_models: int = 10
    ensemble_method: str = "bagging"  # bagging, boosting, or voting

    # NBA-specific settings
    min_samples_for_ci: int = 50
    max_ci_width: float = 0.8  # Maximum width for confidence intervals
    adaptive_threshold: float = 100  # Minimum samples for adaptive methods

    # Robustness settings
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    use_robust_statistics: bool = True

@dataclass
class PredictionInterval:
    """Represents a confidence interval for a prediction"""

    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    width: float = field(init=False)

    def __post_init__(self):
        self.width = self.upper_bound - self.lower_bound

    @property
    def center(self) -> float:
        """Get center of interval"""
        return (self.lower_bound + self.upper_bound) / 2

    def contains(self, value: float) -> bool:
        """Check if value is within interval"""
        return self.lower_bound <= value <= self.upper_bound

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "width": self.width,
            "center": self.center
        }

@dataclass
class NBAConfidenceResult:
    """Complete confidence interval result for NBA prediction"""

    prediction: float
    confidence_intervals: Dict[float, PredictionInterval] = field(default_factory=dict)
    method_used: str = ""
    sample_size: int = 0
    prediction_variance: float = 0.0
    prediction_std_error: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    calibration_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_interval(self, confidence_level: float) -> Optional[PredictionInterval]:
        """Get confidence interval for specific level"""
        return self.confidence_intervals.get(confidence_level)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "prediction": self.prediction,
            "confidence_intervals": {
                str(level): interval.to_dict()
                for level, interval in self.confidence_intervals.items()
            },
            "method_used": self.method_used,
            "sample_size": self.sample_size,
            "prediction_variance": self.prediction_variance,
            "prediction_std_error": self.prediction_std_error,
            "feature_importance": self.feature_importance,
            "calibration_score": self.calibration_score,
            "metadata": self.metadata
        }

class AbstractCICalculator(ABC):
    """Abstract base class for confidence interval calculators"""

    def __init__(self, config: ConfidenceIntervalConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def calculate_intervals(self,
                           predictions: np.ndarray,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
        """Calculate confidence intervals for predictions"""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AbstractCICalculator':
        """Fit the calculator on training data"""
        pass

    def _validate_inputs(self,
                         predictions: np.ndarray,
                         features: Optional[pd.DataFrame] = None) -> bool:
        """Validate input data"""
        if len(predictions) == 0:
            self.logger.warning("Empty predictions array")
            return False

        if features is not None and len(predictions) != len(features):
            self.logger.warning("Prediction and feature length mismatch")
            return False

        if len(predictions) < self.config.min_samples_for_ci:
            self.logger.warning(f"Insufficient samples: {len(predictions)} < {self.config.min_samples_for_ci}")
            return False

        return True

class BootstrapCICalculator(AbstractCICalculator):
    """Bootstrap-based confidence interval calculator"""

    def __init__(self, config: ConfidenceIntervalConfig):
        super().__init__(config)
        self._bootstrap_samples = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BootstrapCICalculator':
        """Fit bootstrap calculator (stores original data for resampling)"""
        self._original_X = X.copy()
        self._original_y = y.copy()
        self.logger.info(f"Fitted bootstrap calculator with {len(X)} samples")
        return self

    def calculate_intervals(self,
                           predictions: np.ndarray,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
        """Calculate bootstrap confidence intervals"""

        if not self._validate_inputs(predictions, features):
            return self._create_fallback_result(predictions[0] if len(predictions) > 0 else 0.5)

        confidence_levels = confidence_levels or self.config.confidence_levels
        result = NBAConfidenceResult(
            prediction=float(np.mean(predictions)),
            sample_size=len(predictions)
        )

        # Perform bootstrap sampling
        bootstrap_predictions = self._bootstrap_sample(predictions)

        # Calculate intervals for each confidence level
        intervals = {}
        for level in confidence_levels:
            alpha = 1.0 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(bootstrap_predictions, lower_percentile)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile)

            # Ensure bounds are within valid range [0, 1]
            lower_bound = np.clip(lower_bound, 0.0, 1.0)
            upper_bound = np.clip(upper_bound, 0.0, 1.0)

            intervals[level] = PredictionInterval(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                confidence_level=level,
                method="bootstrap"
            )

        result.confidence_intervals = intervals
        result.method_used = "bootstrap"
        result.prediction_variance = np.var(bootstrap_predictions)
        result.prediction_std_error = np.std(bootstrap_predictions)

        return result

    def _bootstrap_sample(self, predictions: np.ndarray) -> np.ndarray:
        """Perform bootstrap sampling of predictions"""
        n_samples = len(predictions)
        n_bootstraps = self.config.n_bootstrap_samples

        # Generate bootstrap samples
        rng = np.random.RandomState(self.config.bootstrap_random_state)
        bootstrap_indices = rng.randint(0, n_samples, size=(n_bootstraps, n_samples))

        # Calculate mean for each bootstrap sample
        bootstrap_means = np.array([
            np.mean(predictions[indices]) for indices in bootstrap_indices
        ])

        return bootstrap_means

    def _create_fallback_result(self, prediction: float) -> NBAConfidenceResult:
        """Create fallback result for insufficient data"""
        intervals = {}
        for level in self.config.confidence_levels:
            margin = 0.1  # Default margin for insufficient data
            intervals[level] = PredictionInterval(
                lower_bound=max(0.0, prediction - margin),
                upper_bound=min(1.0, prediction + margin),
                confidence_level=level,
                method="fallback"
            )

        return NBAConfidenceResult(
            prediction=prediction,
            confidence_intervals=intervals,
            method_used="fallback",
            sample_size=0
        )

class QuantileRegressionCICalculator(AbstractCICalculator):
    """Quantile regression-based confidence interval calculator"""

    def __init__(self, config: ConfidenceIntervalConfig):
        super().__init__(config)
        self._lower_quantile = None
        self._upper_quantile = None
        self._median_quantile = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'QuantileRegressionCICalculator':
        """Fit quantile regression models"""

        if len(X) < self.config.min_samples_for_ci:
            self.logger.warning("Insufficient data for quantile regression")
            return self

        # Fit models for different quantiles
        quantiles_to_fit = [0.05, 0.25, 0.5, 0.75, 0.95]

        try:
            self._lower_quantile = QuantileRegressor(
                alpha=0.05,
                n_estimators=self.config.n_quantile_estimators,
                learning_rate=self.config.quantile_learning_rate,
                random_state=42
            )

            self._upper_quantile = QuantileRegressor(
                alpha=0.95,
                n_estimators=self.config.n_quantile_estimators,
                learning_rate=self.config.quantile_learning_rate,
                random_state=42
            )

            self._median_quantile = QuantileRegressor(
                alpha=self.config.quantile_alpha,
                n_estimators=self.config.n_quantile_estimators,
                learning_rate=self.config.quantile_learning_rate,
                random_state=42
            )

            # Fit models
            self._lower_quantile.fit(X, y)
            self._upper_quantile.fit(X, y)
            self._median_quantile.fit(X, y)

            self.logger.info(f"Fitted quantile regression models with {len(X)} samples")

        except Exception as e:
            self.logger.error(f"Error fitting quantile regression: {e}")
            self._reset_models()

        return self

    def calculate_intervals(self,
                           predictions: np.ndarray,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
        """Calculate quantile regression confidence intervals"""

        if not self._validate_inputs(predictions, features):
            return self._create_fallback_result(predictions[0] if len(predictions) > 0 else 0.5)

        if features is None or self._median_quantile is None:
            return self._create_fallback_result(np.mean(predictions))

        confidence_levels = confidence_levels or self.config.confidence_levels

        # Make predictions with quantile models
        try:
            lower_predictions = self._lower_quantile.predict(features)
            median_predictions = self._median_quantile.predict(features)
            upper_predictions = self._upper_quantile.predict(features)

            # Calculate intervals
            intervals = {}
            for level in confidence_levels:
                # Adjust bounds based on confidence level
                level_factor = level / 0.95  # Scale to 95% reference

                # Calculate bounds for this confidence level
                spread = (upper_predictions - lower_predictions) * level_factor / 2
                center = median_predictions

                lower_bound = np.clip(center - spread, 0.0, 1.0)
                upper_bound = np.clip(center + spread, 0.0, 1.0)

                intervals[level] = PredictionInterval(
                    lower_bound=float(np.mean(lower_bound)),
                    upper_bound=float(np.mean(upper_bound)),
                    confidence_level=level,
                    method="quantile_regression"
                )

            result = NBAConfidenceResult(
                prediction=float(np.mean(median_predictions)),
                confidence_intervals=intervals,
                method_used="quantile_regression",
                sample_size=len(predictions)
            )

            # Calculate feature importance (simplified)
            if hasattr(self._median_quantile, 'feature_importances_'):
                result.feature_importance = {
                    f"feature_{i}": float(imp)
                    for i, imp in enumerate(self._median_quantile.feature_importances_)
                }

            return result

        except Exception as e:
            self.logger.error(f"Error in quantile regression prediction: {e}")
            return self._create_fallback_result(np.mean(predictions))

    def _reset_models(self):
        """Reset fitted models"""
        self._lower_quantile = None
        self._upper_quantile = None
        self._median_quantile = None

    def _create_fallback_result(self, prediction: float) -> NBAConfidenceResult:
        """Create fallback result for model failure"""
        intervals = {}
        for level in self.config.confidence_levels:
            margin = 0.15  # Conservative margin for model failure
            intervals[level] = PredictionInterval(
                lower_bound=max(0.0, prediction - margin),
                upper_bound=min(1.0, prediction + margin),
                confidence_level=level,
                method="fallback_quantile"
            )

        return NBAConfidenceResult(
            prediction=prediction,
            confidence_intervals=intervals,
            method_used="fallback_quantile",
            sample_size=0
        )

class EnsembleCICalculator(AbstractCICalculator):
    """Ensemble-based confidence interval calculator"""

    def __init__(self, config: ConfidenceIntervalConfig):
        super().__init__(config)
        self._ensemble_models = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleCICalculator':
        """Fit ensemble models"""

        if len(X) < self.config.min_samples_for_ci:
            self.logger.warning("Insufficient data for ensemble method")
            return self

        try:
            self._ensemble_models = []

            for i in range(self.config.n_ensemble_models):
                if self.config.ensemble_method == "bagging":
                    # Create bagging ensemble
                    model = GradientBoostingRegressor(
                        n_estimators=50,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42 + i
                    )
                else:  # boosting
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42 + i
                    )

                # Bootstrap sample for diversity
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]

                model.fit(X_boot, y_boot)
                self._ensemble_models.append(model)

            self.logger.info(f"Fitted {len(self._ensemble_models)} ensemble models")

        except Exception as e:
            self.logger.error(f"Error fitting ensemble models: {e}")
            self._ensemble_models = []

        return self

    def calculate_intervals(self,
                           predictions: np.ndarray,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
        """Calculate ensemble-based confidence intervals"""

        if not self._validate_inputs(predictions, features):
            return self._create_fallback_result(predictions[0] if len(predictions) > 0 else 0.5)

        if features is None or len(self._ensemble_models) == 0:
            return self._create_fallback_result(np.mean(predictions))

        confidence_levels = confidence_levels or self.config.confidence_levels

        try:
            # Get predictions from all ensemble models
            ensemble_predictions = np.array([
                model.predict(features) for model in self._ensemble_models
            ])

            # Calculate intervals from ensemble predictions
            intervals = {}
            for level in confidence_levels:
                alpha = 1.0 - level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                lower_bound = np.percentile(ensemble_predictions, lower_percentile, axis=0)
                upper_bound = np.percentile(ensemble_predictions, upper_percentile, axis=0)

                # Take mean across instances
                lower_mean = np.mean(lower_bound)
                upper_mean = np.mean(upper_bound)

                # Ensure bounds are within valid range
                lower_mean = np.clip(lower_mean, 0.0, 1.0)
                upper_mean = np.clip(upper_mean, 0.0, 1.0)

                intervals[level] = PredictionInterval(
                    lower_bound=float(lower_mean),
                    upper_bound=float(upper_mean),
                    confidence_level=level,
                    method="ensemble"
                )

            # Calculate ensemble mean prediction
            ensemble_mean = np.mean(ensemble_predictions)
            ensemble_std = np.std(ensemble_predictions)

            result = NBAConfidenceResult(
                prediction=float(ensemble_mean),
                confidence_intervals=intervals,
                method_used="ensemble",
                sample_size=len(predictions),
                prediction_variance=float(ensemble_std ** 2),
                prediction_std_error=float(ensemble_std)
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return self._create_fallback_result(np.mean(predictions))

    def _create_fallback_result(self, prediction: float) -> NBAConfidenceResult:
        """Create fallback result for ensemble failure"""
        intervals = {}
        for level in self.config.confidence_levels:
            margin = 0.12  # Conservative margin for ensemble failure
            intervals[level] = PredictionInterval(
                lower_bound=max(0.0, prediction - margin),
                upper_bound=min(1.0, prediction + margin),
                confidence_level=level,
                method="fallback_ensemble"
            )

        return NBAConfidenceResult(
            prediction=prediction,
            confidence_intervals=intervals,
            method_used="fallback_ensemble",
            sample_size=0
        )

class AdaptiveCICalculator(AbstractCICalculator):
    """Adaptive confidence interval calculator that selects best method"""

    def __init__(self, config: ConfidenceIntervalConfig):
        super().__init__(config)
        self._bootstrap_calc = BootstrapCICalculator(config)
        self._quantile_calc = QuantileRegressionCICalculator(config)
        self._ensemble_calc = EnsembleCICalculator(config)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveCICalculator':
        """Fit all available calculators"""

        # Fit bootstrap (always available)
        self._bootstrap_calc.fit(X, y)

        # Fit quantile regression if enough data
        if len(X) >= self.config.adaptive_threshold:
            self._quantile_calc.fit(X, y)

        # Fit ensemble if enough data
        if len(X) >= self.config.adaptive_threshold:
            self._ensemble_calc.fit(X, y)

        self.logger.info(f"Fitted adaptive CI calculator with {len(X)} samples")
        return self

    def calculate_intervals(self,
                           predictions: np.ndarray,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
        """Calculate adaptive confidence intervals"""

        if not self._validate_inputs(predictions, features):
            return self._create_fallback_result(predictions[0] if len(predictions) > 0 else 0.5)

        sample_size = len(predictions)

        # Select method based on data characteristics
        if sample_size >= self.config.adaptive_threshold and features is not None:
            # Use ensemble for larger datasets
            try:
                result = self._ensemble_calc.calculate_intervals(predictions, features, confidence_levels)
                result.method_used = "adaptive_ensemble"
                return result
            except Exception as e:
                self.logger.warning(f"Ensemble method failed: {e}")

        if sample_size >= self.config.adaptive_threshold and features is not None:
            # Use quantile regression for medium datasets
            try:
                result = self._quantile_calc.calculate_intervals(predictions, features, confidence_levels)
                result.method_used = "adaptive_quantile"
                return result
            except Exception as e:
                self.logger.warning(f"Quantile regression method failed: {e}")

        # Fall back to bootstrap
        try:
            result = self._bootstrap_calc.calculate_intervals(predictions, features, confidence_levels)
            result.method_used = "adaptive_bootstrap"
            return result
        except Exception as e:
            self.logger.warning(f"Bootstrap method failed: {e}")

        # Ultimate fallback
        return self._create_fallback_result(np.mean(predictions))

    def _create_fallback_result(self, prediction: float) -> NBAConfidenceResult:
        """Create ultimate fallback result"""
        intervals = {}
        for level in self.config.confidence_levels:
            margin = 0.2  # Very conservative margin
            intervals[level] = PredictionInterval(
                lower_bound=max(0.0, prediction - margin),
                upper_bound=min(1.0, prediction + margin),
                confidence_level=level,
                method="ultimate_fallback"
            )

        return NBAConfidenceResult(
            prediction=prediction,
            confidence_intervals=intervals,
            method_used="ultimate_fallback",
            sample_size=0
        )

class NBAConfidenceIntervalCalculator:
    """Main NBA confidence interval calculator with adaptive method selection"""

    def __init__(self, config: Optional[ConfidenceIntervalConfig] = None):
        """Initialize NBA confidence interval calculator"""

        self.config = config or ConfidenceIntervalConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize calculator based on method
        self._initialize_calculator()

        # Training data for fitting
        self._training_X = None
        self._training_y = None
        self._is_fitted = False

        # Historical confidence intervals for calibration
        self._historical_intervals = deque(maxlen=1000)

        self.logger.info(f"NBA Confidence Interval Calculator initialized with method: {self.config.primary_method.value}")

    def _initialize_calculator(self):
        """Initialize the appropriate calculator"""
        if self.config.primary_method == ConfidenceIntervalMethod.BOOTSTRAP:
            self._calculator = BootstrapCICalculator(self.config)
        elif self.config.primary_method == ConfidenceIntervalMethod.QUANTILE_REGRESSION:
            self._calculator = QuantileRegressionCICalculator(self.config)
        elif self.config.primary_method == ConfidenceIntervalMethod.ENSEMBLE:
            self._calculator = EnsembleCICalculator(self.config)
        elif self.config.primary_method == ConfidenceIntervalMethod.ADAPTIVE:
            self._calculator = AdaptiveCICalculator(self.config)
        else:
            self._calculator = AdaptiveCICalculator(self.config)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NBAConfidenceIntervalCalculator':
        """Fit the confidence interval calculator on training data"""

        # Validate input
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if len(X) < self.config.min_samples_for_ci:
            self.logger.warning(f"Insufficient data for fitting: {len(X)} < {self.config.min_samples_for_ci}")
            return self

        # Store training data
        self._training_X = X.copy()
        self._training_y = y.copy()

        # Fit the calculator
        try:
            self._calculator.fit(X, y)
            self._is_fitted = True
            self.logger.info(f"Successfully fitted CI calculator with {len(X)} samples")
        except Exception as e:
            self.logger.error(f"Error fitting CI calculator: {e}")
            self._is_fitted = False

        return self

    def calculate_intervals(self,
                           prediction: float,
                           features: Optional[pd.DataFrame] = None,
                           confidence_levels: Optional[List[float]] = None,
                           return_single_interval: bool = False,
                           single_confidence_level: float = 0.95) -> Union[NBAConfidenceResult, PredictionInterval]:
        """Calculate confidence intervals for NBA prediction"""

        confidence_levels = confidence_levels or self.config.confidence_levels

        # For single prediction, create array
        predictions = np.array([prediction])

        # Calculate intervals
        try:
            result = self._calculator.calculate_intervals(
                predictions, features, confidence_levels
            )

            # Set the prediction to our input
            result.prediction = prediction

            # Add calibration metadata
            result.metadata.update({
                "calculator_type": type(self._calculator).__name__,
                "is_fitted": self._is_fitted,
                "training_samples": len(self._training_X) if self._training_X is not None else 0,
                "timestamp": datetime.now().isoformat()
            })

            # Store interval for calibration
            self._historical_intervals.append(result)

            # Apply post-processing for NBA-specific constraints
            result = self._apply_nba_constraints(result)

            if return_single_interval:
                # Return single interval
                interval = result.get_interval(single_confidence_level)
                return interval or PredictionInterval(
                    lower_bound=max(0.0, prediction - 0.1),
                    upper_bound=min(1.0, prediction + 0.1),
                    confidence_level=single_confidence_level,
                    method="fallback"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {e}")

            # Return fallback result
            if return_single_interval:
                return PredictionInterval(
                    lower_bound=max(0.0, prediction - 0.2),
                    upper_bound=min(1.0, prediction + 0.2),
                    confidence_level=single_confidence_level,
                    method="error_fallback"
                )

            # Create fallback intervals
            intervals = {}
            for level in confidence_levels:
                intervals[level] = PredictionInterval(
                    lower_bound=max(0.0, prediction - 0.2),
                    upper_bound=min(1.0, prediction + 0.2),
                    confidence_level=level,
                    method="error_fallback"
                )

            return NBAConfidenceResult(
                prediction=prediction,
                confidence_intervals=intervals,
                method_used="error_fallback",
                sample_size=0,
                metadata={"error": str(e)}
            )

    def _apply_nba_constraints(self, result: NBAConfidenceResult) -> NBAConfidenceResult:
        """Apply NBA-specific constraints to confidence intervals"""

        # Ensure all intervals are within [0, 1] range for probabilities
        for level, interval in result.confidence_intervals.items():
            interval.lower_bound = max(0.0, interval.lower_bound)
            interval.upper_bound = min(1.0, interval.upper_bound)

            # Ensure interval width is reasonable
            if interval.width > self.config.max_ci_width:
                center = interval.center
                half_width = self.config.max_ci_width / 2
                interval.lower_bound = max(0.0, center - half_width)
                interval.upper_bound = min(1.0, center + half_width)

        # Ensure prediction is within [0, 1]
        result.prediction = max(0.0, min(1.0, result.prediction))

        return result

    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration metrics for the confidence interval calculator"""

        if len(self._historical_intervals) == 0:
            return {"error": "No historical intervals available"}

        # Calculate calibration metrics
        avg_interval_widths = {}
        for level in self.config.confidence_levels:
            widths = [
                interval.width for interval_result in self._historical_intervals
                for interval in interval_result.confidence_intervals.values()
                if interval.confidence_level == level
            ]

            if widths:
                avg_interval_widths[level] = np.mean(widths)

        return {
            "total_calculations": len(self._historical_intervals),
            "is_fitted": self._is_fitted,
            "training_samples": len(self._training_X) if self._training_X is not None else 0,
            "average_interval_widths": avg_interval_widths,
            "method_used": self.config.primary_method.value,
            "config": {
                "min_samples_for_ci": self.config.min_samples_for_ci,
                "max_ci_width": self.config.max_ci_width,
                "confidence_levels": self.config.confidence_levels
            }
        }

    def update_config(self, **kwargs) -> None:
        """Update calculator configuration"""

        # Update config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Reinitialize calculator if method changed
        if 'primary_method' in kwargs:
            self._initialize_calculator()

        self.logger.info(f"Updated CI calculator config: {kwargs}")

    def cleanup(self) -> None:
        """Cleanup resources used by the confidence interval calculator"""
        try:
            # Clear fitted models and data
            if hasattr(self, '_bootstrap_calculator') and self._bootstrap_calculator is not None:
                self._bootstrap_calculator = None

            if hasattr(self, '_quantile_calculator') and self._quantile_calculator is not None:
                self._quantile_calculator = None

            if hasattr(self, '_ensemble_calculator') and self._ensemble_calculator is not None:
                self._ensemble_calculator = None

            if hasattr(self, '_adaptive_calculator') and self._adaptive_calculator is not None:
                self._adaptive_calculator = None

            # Clear reference data
            if hasattr(self, '_reference_data'):
                self._reference_data = None

            # Clear training data
            if hasattr(self, '_X_train'):
                self._X_train = None
            if hasattr(self, '_y_train'):
                self._y_train = None

            self.logger.info("✅ Confidence Interval Calculator cleanup completed")

        except Exception as e:
            self.logger.warning(f"⚠️ Error during CI calculator cleanup: {e}")

# Global confidence interval calculator instance
_global_ci_calculator: Optional[NBAConfidenceIntervalCalculator] = None

def get_ci_calculator(config: Optional[ConfidenceIntervalConfig] = None) -> NBAConfidenceIntervalCalculator:
    """Get global confidence interval calculator instance"""
    global _global_ci_calculator
    if _global_ci_calculator is None:
        _global_ci_calculator = NBAConfidenceIntervalCalculator(config)
    return _global_ci_calculator

def calculate_nba_confidence_intervals(prediction: float,
                                      features: Optional[pd.DataFrame] = None,
                                      confidence_levels: Optional[List[float]] = None) -> NBAConfidenceResult:
    """Calculate confidence intervals for NBA prediction"""

    calculator = get_ci_calculator()
    return calculator.calculate_intervals(
        prediction=prediction,
        features=features,
        confidence_levels=confidence_levels
    )