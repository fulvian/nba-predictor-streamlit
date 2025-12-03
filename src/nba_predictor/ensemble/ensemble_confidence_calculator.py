#!/usr/bin/env python3
"""
🎯 NBA Ensemble Confidence Interval Calculator - Task 2.2.2 Implementation

Sistema avanzato di calcolo confidence intervals specifico per NBA Ensemble Predictor.
Implementa metodi bayesian bootstrap, quantile ensemble, e prediction intervals robusti
per XGBoost + Neural Network ensemble.

Author: NBA Predictive Analytics System
Date: 2025-01-11
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import warnings
from abc import ABC, abstractmethod

# Statistical libraries
from scipy import stats
from scipy.stats import norm, t
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import joblib
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logger = logging.getLogger(__name__)

class EnsembleCIMethod(Enum):
    """Ensemble-specific confidence interval methods"""
    BAYESIAN_BOOTSTRAP = "bayesian_bootstrap"
    QUANTILE_ENSEMBLE = "quantile_ensemble"
    PREDICTION_INTERVAL_ENSEMBLE = "prediction_interval_ensemble"
    CONFORMAL_PREDICTION = "conformal_prediction"
    JACKKNIFE_PLUS = "jackknife_plus"
    DEEP_ENSEMBLE = "deep_ensemble"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"

@dataclass
class EnsembleCIConfig:
    """Configuration for ensemble confidence intervals"""

    # Method settings
    primary_method: EnsembleCIMethod = EnsembleCIMethod.ADAPTIVE_ENSEMBLE
    confidence_levels: List[float] = field(default_factory=lambda: [0.50, 0.80, 0.90, 0.95, 0.99])

    # Bayesian Bootstrap settings
    n_bootstrap_samples: int = 2000
    bootstrap_weights_prior: str = "dirichlet"  # dirichlet, uniform, exponential
    dirichlet_alpha: float = 1.0

    # Quantile Ensemble settings
    quantile_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    ensemble_quantile_method: str = "weighted"  # weighted, median, mean

    # Conformal Prediction settings
    conformal_alpha: float = 0.1
    n_calibration_samples: int = 1000
    conformal_method: str = "split"  # split, cv+, jackknife

    # Deep Ensemble settings
    n_deep_ensemble_models: int = 10
    deep_ensemble_dropout: float = 0.1
    monte_carlo_samples: int = 100

    # NBA-specific settings
    min_samples_for_ensemble_ci: int = 100
    ensemble_weight_smoothing: bool = True
    temporal_decay_factor: float = 0.95  # For time-weighted samples

    # Robustness settings
    outlier_detection_ensemble: bool = True
    ensemble_disagreement_threshold: float = 0.2
    use_model_disagreement: bool = True

@dataclass
class EnsemblePredictionInterval:
    """Enhanced prediction interval for ensemble predictions"""

    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    ensemble_mean: float
    ensemble_std: float
    model_disagreement: float
    effective_sample_size: int
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

    def calibration_score(self, actual: float) -> float:
        """Calculate calibration score against actual value"""
        if self.contains(actual):
            return 1.0
        else:
            # Penalize based on distance from interval
            distance = min(abs(actual - self.lower_bound), abs(actual - self.upper_bound))
            return max(0.0, 1.0 - distance / self.width)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level,
            'method': self.method,
            'ensemble_mean': self.ensemble_mean,
            'ensemble_std': self.ensemble_std,
            'model_disagreement': self.model_disagreement,
            'effective_sample_size': self.effective_sample_size,
            'width': self.width,
            'center': self.center
        }

class BayesianBootstrapEnsemble:
    """Bayesian Bootstrap for Ensemble Confidence Intervals"""

    def __init__(self, config: EnsembleCIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".BayesianBootstrapEnsemble")

    def calculate_ensemble_ci(self,
                              ensemble_predictions: np.ndarray,
                              ensemble_weights: Optional[np.ndarray] = None,
                              confidence_level: float = 0.95) -> EnsemblePredictionInterval:
        """
        Calculate confidence interval using Bayesian Bootstrap for ensemble predictions

        Args:
            ensemble_predictions: Array of predictions from ensemble models
            ensemble_weights: Optional weights for ensemble models
            confidence_level: Desired confidence level

        Returns:
            EnsemblePredictionInterval with bootstrap CI
        """
        try:
            n_models = len(ensemble_predictions)
            if n_models == 0:
                raise ValueError("No ensemble predictions provided")

            # Set default weights if not provided
            if ensemble_weights is None:
                ensemble_weights = np.ones(n_models) / n_models

            # Generate bootstrap weights using Dirichlet distribution
            bootstrap_weights = np.random.dirichlet(
                [self.config.dirichlet_alpha] * n_models,
                size=self.config.n_bootstrap_samples
            )

            # Calculate weighted bootstrap predictions
            if self.config.bootstrap_weights_prior == "dirichlet":
                # Use Dirichlet weights
                bootstrap_predictions = np.dot(bootstrap_weights, ensemble_predictions)
            elif self.config.bootstrap_weights_prior == "exponential":
                # Use exponential weights
                exp_weights = np.random.exponential(scale=1.0, size=(self.config.n_bootstrap_samples, n_models))
                exp_weights = exp_weights / exp_weights.sum(axis=1, keepdims=True)
                bootstrap_predictions = np.dot(exp_weights, ensemble_predictions)
            else:
                # Use uniform weights
                uniform_weights = np.ones((self.config.n_bootstrap_samples, n_models)) / n_models
                bootstrap_predictions = np.dot(uniform_weights, ensemble_predictions)

            # Calculate quantiles for confidence interval
            alpha = 1 - confidence_level
            lower_quantile = (alpha / 2) * 100
            upper_quantile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(bootstrap_predictions, lower_quantile)
            upper_bound = np.percentile(bootstrap_predictions, upper_quantile)

            # Calculate ensemble statistics
            ensemble_mean = np.average(ensemble_predictions, weights=ensemble_weights)
            ensemble_std = np.sqrt(np.average((ensemble_predictions - ensemble_mean)**2, weights=ensemble_weights))

            # Calculate model disagreement
            model_disagreement = np.std(ensemble_predictions)

            return EnsemblePredictionInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="bayesian_bootstrap_ensemble",
                ensemble_mean=ensemble_mean,
                ensemble_std=ensemble_std,
                model_disagreement=model_disagreement,
                effective_sample_size=n_models * self.config.n_bootstrap_samples
            )

        except Exception as e:
            self.logger.error(f"Bayesian bootstrap CI calculation failed: {e}")
            # Fallback to simple ensemble statistics
            return self._fallback_ensemble_ci(ensemble_predictions, confidence_level)

    def _fallback_ensemble_ci(self, predictions: np.ndarray, confidence_level: float) -> EnsemblePredictionInterval:
        """Fallback CI calculation using simple statistics"""
        mean = np.mean(predictions)
        std = np.std(predictions)

        # Use t-distribution for small samples
        n = len(predictions)
        if n < 30:
            t_critical = t.ppf((1 + confidence_level) / 2, df=n-1)
            margin = t_critical * (std / np.sqrt(n))
        else:
            z_critical = norm.ppf((1 + confidence_level) / 2)
            margin = z_critical * (std / np.sqrt(n))

        return EnsemblePredictionInterval(
            lower_bound=mean - margin,
            upper_bound=mean + margin,
            confidence_level=confidence_level,
            method="fallback_ensemble_statistics",
            ensemble_mean=mean,
            ensemble_std=std,
            model_disagreement=std,
            effective_sample_size=n
        )

class QuantileEnsembleCI:
    """Quantile-based Confidence Intervals for Ensemble"""

    def __init__(self, config: EnsembleCIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".QuantileEnsembleCI")

    def calculate_ensemble_ci(self,
                              ensemble_predictions: np.ndarray,
                              historical_predictions: Optional[np.ndarray] = None,
                              confidence_level: float = 0.95) -> EnsemblePredictionInterval:
        """
        Calculate confidence interval using quantile ensemble method

        Args:
            ensemble_predictions: Array of predictions from ensemble models
            historical_predictions: Historical predictions for calibration
            confidence_level: Desired confidence level

        Returns:
            EnsemblePredictionInterval with quantile CI
        """
        try:
            if historical_predictions is not None and len(historical_predictions) > 0:
                # Use historical predictions for better quantile estimation
                combined_predictions = np.concatenate([ensemble_predictions, historical_predictions])
            else:
                combined_predictions = ensemble_predictions

            # Calculate ensemble statistics
            ensemble_mean = np.mean(ensemble_predictions)
            ensemble_std = np.std(ensemble_predictions)
            model_disagreement = np.std(ensemble_predictions)

            # Calculate quantiles
            alpha = 1 - confidence_level
            lower_quantile = (alpha / 2) * 100
            upper_quantile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(combined_predictions, lower_quantile)
            upper_bound = np.percentile(combined_predictions, upper_quantile)

            # Apply ensemble weighting if configured
            if self.config.ensemble_quantile_method == "weighted":
                weights = self._calculate_quantile_weights(ensemble_predictions)
                weighted_mean = np.average(ensemble_predictions, weights=weights)
                # Adjust bounds based on weighting
                center_shift = weighted_mean - ensemble_mean
                lower_bound += center_shift
                upper_bound += center_shift
                ensemble_mean = weighted_mean

            return EnsemblePredictionInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="quantile_ensemble",
                ensemble_mean=ensemble_mean,
                ensemble_std=ensemble_std,
                model_disagreement=model_disagreement,
                effective_sample_size=len(combined_predictions)
            )

        except Exception as e:
            self.logger.error(f"Quantile ensemble CI calculation failed: {e}")
            # Fallback to simple quantiles
            return self._fallback_quantile_ci(ensemble_predictions, confidence_level)

    def _calculate_quantile_weights(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate weights for quantile ensemble based on prediction consistency"""
        # Simple weighting: more weight to predictions closer to median
        median = np.median(predictions)
        distances = np.abs(predictions - median)
        # Convert distances to weights (closer = higher weight)
        weights = 1.0 / (1.0 + distances)
        return weights / weights.sum()

    def _fallback_quantile_ci(self, predictions: np.ndarray, confidence_level: float) -> EnsemblePredictionInterval:
        """Fallback quantile CI calculation"""
        alpha = 1 - confidence_level
        lower_bound = np.percentile(predictions, (alpha / 2) * 100)
        upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100)
        mean = np.mean(predictions)

        return EnsemblePredictionInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="fallback_quantile",
            ensemble_mean=mean,
            ensemble_std=np.std(predictions),
            model_disagreement=np.std(predictions),
            effective_sample_size=len(predictions)
        )

class ConformalPredictionEnsemble:
    """Conformal Prediction for Ensemble Confidence Intervals"""

    def __init__(self, config: EnsembleCIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".ConformalPredictionEnsemble")
        self.calibration_scores = []
        self.is_calibrated = False

    def calibrate(self,
                  calibration_predictions: np.ndarray,
                  calibration_targets: np.ndarray) -> None:
        """
        Calibrate conformal prediction using calibration data

        Args:
            calibration_predictions: Ensemble predictions on calibration set
            calibration_targets: True target values
        """
        try:
            # Calculate nonconformity scores (absolute errors)
            if len(calibration_predictions.shape) > 1:
                # Ensemble predictions: take mean across models
                pred_means = np.mean(calibration_predictions, axis=1)
            else:
                pred_means = calibration_predictions

            self.calibration_scores = np.abs(calibration_targets - pred_means)
            self.is_calibrated = True

            self.logger.info(f"Calibrated conformal prediction with {len(self.calibration_scores)} samples")

        except Exception as e:
            self.logger.error(f"Conformal prediction calibration failed: {e}")
            self.is_calibrated = False

    def calculate_ensemble_ci(self,
                              ensemble_predictions: np.ndarray,
                              confidence_level: float = 0.95) -> EnsemblePredictionInterval:
        """
        Calculate confidence interval using conformal prediction

        Args:
            ensemble_predictions: Array of predictions from ensemble models
            confidence_level: Desired confidence level

        Returns:
            EnsemblePredictionInterval with conformal CI
        """
        try:
            if not self.is_calibrated or len(self.calibration_scores) == 0:
                return self._fallback_conformal_ci(ensemble_predictions, confidence_level)

            # Calculate ensemble prediction
            ensemble_mean = np.mean(ensemble_predictions)
            ensemble_std = np.std(ensemble_predictions)
            model_disagreement = np.std(ensemble_predictions)

            # Calculate quantile of calibration scores
            alpha = 1 - confidence_level
            n_calib = len(self.calibration_scores)

            if self.config.conformal_method == "split":
                # Split conformal prediction
                quantile_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
                q_hat = np.quantile(self.calibration_scores, quantile_level)
            else:
                # CV+ or other methods (simplified here)
                q_hat = np.quantile(self.calibration_scores, 1 - alpha)

            # Calculate prediction interval
            lower_bound = ensemble_mean - q_hat
            upper_bound = ensemble_mean + q_hat

            return EnsemblePredictionInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="conformal_prediction_ensemble",
                ensemble_mean=ensemble_mean,
                ensemble_std=ensemble_std,
                model_disagreement=model_disagreement,
                effective_sample_size=n_calib
            )

        except Exception as e:
            self.logger.error(f"Conformal prediction CI calculation failed: {e}")
            return self._fallback_conformal_ci(ensemble_predictions, confidence_level)

    def _fallback_conformal_ci(self, predictions: np.ndarray, confidence_level: float) -> EnsemblePredictionInterval:
        """Fallback conformal CI using simple statistics"""
        mean = np.mean(predictions)
        std = np.std(predictions)
        z = norm.ppf((1 + confidence_level) / 2)
        margin = z * std

        return EnsemblePredictionInterval(
            lower_bound=mean - margin,
            upper_bound=mean + margin,
            confidence_level=confidence_level,
            method="fallback_conformal",
            ensemble_mean=mean,
            ensemble_std=std,
            model_disagreement=std,
            effective_sample_size=len(predictions)
        )

class NBAEnsembleConfidenceCalculator:
    """
    Main confidence interval calculator for NBA Ensemble Predictor
    Integrates multiple CI methods specifically for XGBoost + Neural Network ensembles
    """

    def __init__(self, config: Optional[EnsembleCIConfig] = None):
        self.config = config or EnsembleCIConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize CI methods
        self.bayesian_bootstrap = BayesianBootstrapEnsemble(self.config)
        self.quantile_ensemble = QuantileEnsembleCI(self.config)
        self.conformal_prediction = ConformalPredictionEnsemble(self.config)

        # Statistics tracking
        self.ci_history = defaultdict(list)
        self.performance_metrics = {}

        self.logger.info("NBA Ensemble Confidence Calculator initialized with SuperPowered features")

    def calculate_ensemble_confidence_intervals(self,
                                               xgboost_predictions: np.ndarray,
                                               neural_network_predictions: np.ndarray,
                                               xgboost_weights: Optional[np.ndarray] = None,
                                               neural_weights: Optional[np.ndarray] = None,
                                               historical_data: Optional[Dict[str, np.ndarray]] = None,
                                               confidence_levels: Optional[List[float]] = None) -> Dict[str, EnsemblePredictionInterval]:
        """
        Calculate comprehensive confidence intervals for NBA ensemble predictions

        Args:
            xgboost_predictions: XGBoost model predictions
            neural_network_predictions: Neural network predictions
            xgboost_weights: Optional weights for XGBoost predictions
            neural_weights: Optional weights for neural network predictions
            historical_data: Optional historical predictions for calibration
            confidence_levels: List of confidence levels to calculate

        Returns:
            Dictionary mapping confidence levels to prediction intervals
        """
        try:
            confidence_levels = confidence_levels or self.config.confidence_levels

            # Combine predictions from both models
            all_predictions = np.concatenate([xgboost_predictions, neural_network_predictions])

            # Combine weights if provided
            if xgboost_weights is not None and neural_weights is not None:
                all_weights = np.concatenate([xgboost_weights, neural_weights])
            else:
                all_weights = None

            # Store historical data if provided
            if historical_data:
                self._update_historical_data(historical_data)

            # Calculate CIs using different methods
            ci_results = {}

            for conf_level in confidence_levels:
                # Choose method based on configuration and data availability
                method = self._select_optimal_method(all_predictions, conf_level)

                if method == EnsembleCIMethod.BAYESIAN_BOOTSTRAP:
                    ci = self.bayesian_bootstrap.calculate_ensemble_ci(
                        all_predictions, all_weights, conf_level
                    )
                elif method == EnsembleCIMethod.QUANTILE_ENSEMBLE:
                    hist_preds = historical_data.get('predictions') if historical_data else None
                    ci = self.quantile_ensemble.calculate_ensemble_ci(
                        all_predictions, hist_preds, conf_level
                    )
                elif method == EnsembleCIMethod.CONFORMAL_PREDICTION:
                    # Use conformal prediction if calibrated
                    if self.conformal_prediction.is_calibrated:
                        ci = self.conformal_prediction.calculate_ensemble_ci(
                            all_predictions, conf_level
                        )
                    else:
                        # Fallback to Bayesian bootstrap
                        ci = self.bayesian_bootstrap.calculate_ensemble_ci(
                            all_predictions, all_weights, conf_level
                        )
                else:
                    # Default to Bayesian bootstrap
                    ci = self.bayesian_bootstrap.calculate_ensemble_ci(
                        all_predictions, all_weights, conf_level
                    )

                ci_results[f"{conf_level:.2f}"] = ci

                # Store in history
                self.ci_history[f"{conf_level:.2f}"].append({
                    'timestamp': datetime.now(),
                    'interval': ci,
                    'method': method.value
                })

            self.logger.info(f"Calculated {len(ci_results)} confidence intervals for ensemble predictions")
            return ci_results

        except Exception as e:
            self.logger.error(f"Ensemble confidence interval calculation failed: {e}")
            return {}

    def calibrate_with_historical_data(self,
                                      xgboost_historical: np.ndarray,
                                      neural_historical: np.ndarray,
                                      targets: np.ndarray) -> None:
        """
        Calibrate confidence intervals using historical data

        Args:
            xgboost_historical: Historical XGBoost predictions
            neural_historical: Historical neural network predictions
            targets: True target values
        """
        try:
            # Combine historical predictions
            historical_predictions = np.concatenate([xgboost_historical, neural_historical])

            # Calibrate conformal prediction
            if len(historical_predictions.shape) > 1:
                # Take mean across models for calibration
                calib_predictions = np.mean(historical_predictions, axis=1)
            else:
                calib_predictions = historical_predictions

            self.conformal_prediction.calibrate(calib_predictions, targets)

            self.logger.info(f"Calibrated ensemble confidence intervals with {len(targets)} samples")

        except Exception as e:
            self.logger.error(f"Ensemble calibration failed: {e}")

    def _select_optimal_method(self, predictions: np.ndarray, confidence_level: float) -> EnsembleCIMethod:
        """Select optimal CI method based on data characteristics"""
        n_samples = len(predictions)

        if self.config.primary_method == EnsembleCIMethod.ADAPTIVE_ENSEMBLE:
            # Adaptive selection based on sample size and variance
            if n_samples < 20:
                return EnsembleCIMethod.QUANTILE_ENSEMBLE
            elif n_samples < 100:
                return EnsembleCIMethod.BAYESIAN_BOOTSTRAP
            else:
                return EnsembleCIMethod.CONFORMAL_PREDICTION if self.conformal_prediction.is_calibrated else EnsembleCIMethod.BAYESIAN_BOOTSTRAP
        else:
            return self.config.primary_method

    def _update_historical_data(self, historical_data: Dict[str, np.ndarray]) -> None:
        """Update historical data for CI calculations"""
        # Store historical data for future use
        for key, data in historical_data.items():
            if hasattr(self, f'historical_{key}'):
                current = getattr(self, f'historical_{key}')
                updated = np.concatenate([current, data])
                setattr(self, f'historical_{key}', updated[-10000:])  # Keep last 10k samples
            else:
                setattr(self, f'historical_{key}', data)

    def get_ensemble_disagreement_metrics(self,
                                          xgboost_predictions: np.ndarray,
                                          neural_network_predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate disagreement metrics between ensemble models

        Args:
            xgboost_predictions: XGBoost predictions
            neural_network_predictions: Neural network predictions

        Returns:
            Dictionary of disagreement metrics
        """
        try:
            # Calculate mean predictions
            xgb_mean = np.mean(xgboost_predictions)
            nn_mean = np.mean(neural_network_predictions)

            # Calculate disagreement metrics
            absolute_disagreement = abs(xgb_mean - nn_mean)
            relative_disagreement = absolute_disagreement / max(abs(xgb_mean), abs(nn_mean), 1e-6)

            # Calculate variance within each model
            xgb_variance = np.var(xgboost_predictions)
            nn_variance = np.var(neural_network_predictions)

            # Combined variance
            all_predictions = np.concatenate([xgboost_predictions, neural_network_predictions])
            total_variance = np.var(all_predictions)

            # Disagreement ratio (between-model vs within-model variance)
            within_model_variance = (xgb_variance + nn_variance) / 2
            disagreement_ratio = (total_variance - within_model_variance) / max(total_variance, 1e-6)

            return {
                'absolute_disagreement': absolute_disagreement,
                'relative_disagreement': relative_disagreement,
                'xgb_variance': xgb_variance,
                'nn_variance': nn_variance,
                'total_variance': total_variance,
                'disagreement_ratio': disagreement_ratio,
                'n_xgb_samples': len(xgboost_predictions),
                'n_nn_samples': len(neural_network_predictions)
            }

        except Exception as e:
            self.logger.error(f"Disagreement metrics calculation failed: {e}")
            return {}

    def get_confidence_interval_statistics(self) -> Dict[str, Any]:
        """Get statistics about calculated confidence intervals"""
        try:
            stats = {
                'total_calculations': sum(len(history) for history in self.ci_history.values()),
                'methods_used': set(),
                'average_widths': {},
                'calibration_status': self.conformal_prediction.is_calibrated
            }

            for conf_level, history in self.ci_history.items():
                if history:
                    widths = [item['interval'].width for item in history]
                    stats['average_widths'][conf_level] = np.mean(widths)
                    stats['methods_used'].update(item['method'] for item in history)

            stats['methods_used'] = list(stats['methods_used'])

            return stats

        except Exception as e:
            self.logger.error(f"CI statistics calculation failed: {e}")
            return {}

    def save_confidence_calculator(self, filepath: str) -> None:
        """Save confidence calculator state"""
        try:
            state = {
                'config': self.config,
                'calibration_scores': self.conformal_prediction.calibration_scores,
                'is_calibrated': self.conformal_prediction.is_calibrated,
                'ci_history': dict(self.ci_history),
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }

            joblib.dump(state, filepath)
            self.logger.info(f"Confidence calculator saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save confidence calculator: {e}")

    def load_confidence_calculator(self, filepath: str) -> None:
        """Load confidence calculator state"""
        try:
            state = joblib.load(filepath)

            self.config = state['config']
            self.conformal_prediction.calibration_scores = state['calibration_scores']
            self.conformal_prediction.is_calibrated = state['is_calibrated']
            self.ci_history = defaultdict(list, state['ci_history'])
            self.performance_metrics = state['performance_metrics']

            self.logger.info(f"Confidence calculator loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load confidence calculator: {e}")

    def get_available_methods(self) -> list:
        """
        Get list of available confidence interval calculation methods.

        Returns:
            list: Available CI methods
        """
        return [
            "bayesian_bootstrap",
            "quantile_ensemble",
            "conformal_prediction",
            "model_disagreement",
            "adaptive_combination"
        ]

    def get_calibration_report(self) -> dict:
        """
        Get calibration report for confidence intervals.

        Returns:
            dict: Calibration status and metrics
        """
        try:
            return {
                "status": "calibrated" if self.conformal_prediction.is_calibrated else "uncalibrated",
                "calibration_scores": len(self.conformal_prediction.calibration_scores),
                "ci_methods_available": self.get_available_methods(),
                "total_calculations": sum(len(history) for history in self.ci_history.values()),
                "supported_confidence_levels": [0.90, 0.95, 0.99],
                "ensemble_methods": {
                    "bayesian_bootstrap": True,
                    "quantile_ensemble": True,
                    "conformal_prediction": True,
                    "model_disagreement": True
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get calibration report: {e}")
            return {"error": str(e), "status": "error"}