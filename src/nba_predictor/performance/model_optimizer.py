"""Model Performance Optimization for NBA Predictor.

This module implements comprehensive model performance optimization including:
- Model caching for <20ms prediction time
- Ensemble weight optimization
- Performance monitoring and benchmarking
- Prediction batching for efficiency
"""

import logging
import time
import pickle
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import OptimizationError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model optimization."""

    prediction_time_ms: float
    cache_hit_rate: float
    memory_usage_mb: float
    model_accuracy: float
    ensemble_weight_efficiency: float
    batch_processing_efficiency: float
    optimization_score: float


@dataclass
class OptimizationResult:
    """Result of model performance optimization."""

    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_percentage: float
    optimization_techniques: List[str]
    recommendations: List[str]
    optimization_timestamp: datetime


class ModelCache:
    """Intelligent model caching system for performance optimization."""

    def __init__(self, cache_dir: str = ".model_cache/", max_cache_size: int = 100):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory for cached models
            max_cache_size: Maximum number of cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size

        # Cache metadata
        self.cache_metadata = {}
        self.cache_index_file = self.cache_dir / "cache_index.json"

        # Performance tracking
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_mb": 0.0,
        }

        self._load_cache_index()
        logger.info(f"🗄️ Model cache initialized: {cache_dir}")

    def _get_cache_key(self, model_config: Dict[str, Any]) -> str:
        """Generate cache key for model configuration."""
        try:
            # Create deterministic key from model configuration
            config_str = json.dumps(model_config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}")
            return str(hash(model_config))

    def _load_cache_index(self) -> None:
        """Load cache index from file."""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, "r") as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"Loaded {len(self.cache_metadata)} cached models")
        except Exception as e:
            logger.warning(f"Error loading cache index: {e}")
            self.cache_metadata = {}

    def _save_cache_index(self) -> None:
        """Save cache index to file."""
        try:
            with open(self.cache_index_file, "w") as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def get_cached_model(self, model_config: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached model if available.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Cached model or None if not available
        """
        try:
            cache_key = self._get_cache_key(model_config)

            if cache_key in self.cache_metadata:
                cache_info = self.cache_metadata[cache_key]
                cache_file = self.cache_dir / f"{cache_key}.pkl"

                if cache_file.exists():
                    # Check if cache is still valid (24 hour TTL)
                    cache_time = datetime.fromisoformat(cache_info["cached_at"])
                    if datetime.now() - cache_time < timedelta(hours=24):
                        with open(cache_file, "rb") as f:
                            model = pickle.load(f)

                        self.cache_stats["hits"] += 1
                        logger.debug(f"Cache hit for model: {cache_key}")
                        return model
                    else:
                        # Cache expired, remove it
                        self._remove_cached_model(cache_key)

            self.cache_stats["misses"] += 1
            logger.debug(f"Cache miss for model: {cache_key}")
            return None

        except Exception as e:
            logger.error(f"Error getting cached model: {e}")
            self.cache_stats["misses"] += 1
            return None

    def cache_model(
        self,
        model_config: Dict[str, Any],
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Cache a trained model.

        Args:
            model_config: Model configuration dictionary
            model: Trained model to cache
            metadata: Optional metadata for the model

        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(model_config)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # Save model
            with open(cache_file, "wb") as f:
                pickle.dump(model, f)

            # Update metadata
            file_size = cache_file.stat().st_size

            cache_info = {
                "cached_at": datetime.now().isoformat(),
                "file_size_bytes": file_size,
                "model_config": model_config,
                "metadata": metadata or {},
            }

            self.cache_metadata[cache_key] = cache_info

            # Check cache size limit
            self._enforce_cache_size_limit()

            # Save index
            self._save_cache_index()

            self.cache_stats["total_size_mb"] += file_size / (1024 * 1024)
            logger.info(f"Model cached: {cache_key} ({file_size / 1024:.1f} KB)")
            return True

        except Exception as e:
            logger.error(f"Error caching model: {e}")
            return False

    def _remove_cached_model(self, cache_key: str) -> None:
        """Remove cached model."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()

            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]
                self.cache_stats["evictions"] += 1

            self._save_cache_index()
            logger.debug(f"Cache model removed: {cache_key}")

        except Exception as e:
            logger.warning(f"Error removing cached model: {e}")

    def _enforce_cache_size_limit(self) -> None:
        """Enforce maximum cache size limit."""
        try:
            if len(self.cache_metadata) > self.max_cache_size:
                # Sort by last access time and remove oldest
                sorted_cache = sorted(
                    self.cache_metadata.items(),
                    key=lambda x: datetime.fromisoformat(x[1]["cached_at"]),
                )

                # Remove oldest entries
                num_to_remove = len(self.cache_metadata) - self.max_cache_size
                for i in range(num_to_remove):
                    cache_key = sorted_cache[i][0]
                    self._remove_cached_model(cache_key)

                logger.info(f"Evicted {num_to_remove} old cache entries")

        except Exception as e:
            logger.warning(f"Error enforcing cache size limit: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "cache_evictions": self.cache_stats["evictions"],
            "hit_rate": f"{hit_rate:.2%}",
            "total_size_mb": f"{self.cache_stats['total_size_mb']:.1f}",
            "cached_models": len(self.cache_metadata),
            "max_cache_size": self.max_cache_size,
        }

    def clear_cache(self) -> None:
        """Clear all cached models."""
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            # Clear metadata
            self.cache_metadata.clear()
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_size_mb": 0.0,
            }

            # Remove index file
            if self.cache_index_file.exists():
                self.cache_index_file.unlink()

            logger.info("Model cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class EnsembleWeightOptimizer:
    """Optimize ensemble model weights for better performance."""

    def __init__(self, optimization_method: str = "grid_search"):
        """
        Initialize ensemble weight optimizer.

        Args:
            optimization_method: Method for optimization ('grid_search', 'bayesian', 'genetic')
        """
        self.optimization_method = optimization_method
        self.weight_history = []

        logger.info(f"⚖️ Ensemble weight optimizer initialized: {optimization_method}")

    def optimize_ensemble_weights(
        self,
        ensemble_model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        weight_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize ensemble model weights.

        Args:
            ensemble_model: Trained ensemble model
            X_val: Validation features
            y_val: Validation targets
            weight_ranges: Optional weight ranges for optimization

        Returns:
            Optimized weights dictionary
        """
        try:
            logger.info("🎯 Optimizing ensemble weights")

            # Default weight ranges if not provided
            if weight_ranges is None:
                weight_ranges = {
                    "xgboost": (0.1, 0.5),
                    "lightgbm": (0.1, 0.5),
                    "random_forest": (0.05, 0.3),
                    "ridge": (0.05, 0.2),
                    "mlp_meta": (0.0, 0.1),
                }

            if self.optimization_method == "grid_search":
                return self._grid_search_optimization(
                    ensemble_model, X_val, y_val, weight_ranges
                )
            elif self.optimization_method == "bayesian":
                return self._bayesian_optimization(
                    ensemble_model, X_val, y_val, weight_ranges
                )
            else:
                logger.warning(
                    f"Unknown optimization method: {self.optimization_method}"
                )
                return self._get_default_weights()

        except Exception as e:
            logger.error(f"Error optimizing ensemble weights: {e}")
            raise OptimizationError(f"Failed to optimize ensemble weights: {e}") from e

    def _grid_search_optimization(
        self,
        ensemble_model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        weight_ranges: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """Perform grid search optimization."""
        try:
            best_weights = None
            best_score = float("inf")

            # Generate weight combinations
            import itertools

            model_names = list(weight_ranges.keys())
            weight_steps = 5  # Number of steps per weight

            for weight_combination in itertools.product(
                *[
                    np.linspace(min_w, max_w, weight_steps)
                    for min_w, max_w in weight_ranges.values()
                ]
            ):
                # Normalize weights to sum to 1
                weights = np.array(weight_combination)
                weights = weights / weights.sum()

                # Create weight dictionary
                weight_dict = dict(zip(model_names, weights))

                # Update ensemble weights
                if hasattr(ensemble_model, "set_weights"):
                    ensemble_model.set_weights(weight_dict)

                # Evaluate performance
                y_pred = ensemble_model.predict(X_val)
                score = self._calculate_performance_score(y_val, y_pred)

                if score < best_score:
                    best_score = score
                    best_weights = weight_dict.copy()

            logger.info(f"Grid search completed: best score = {best_score:.4f}")
            return best_weights or self._get_default_weights()

        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            return self._get_default_weights()

    def _bayesian_optimization(
        self,
        ensemble_model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        weight_ranges: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """Perform Bayesian optimization (simplified)."""
        try:
            # Simplified Bayesian optimization using random sampling
            best_weights = None
            best_score = float("inf")

            model_names = list(weight_ranges.keys())

            for iteration in range(50):  # 50 random samples
                # Generate random weights
                weights = np.random.random(len(model_names))
                weights = weights / weights.sum()  # Normalize

                weight_dict = dict(zip(model_names, weights))

                # Update ensemble weights
                if hasattr(ensemble_model, "set_weights"):
                    ensemble_model.set_weights(weight_dict)

                # Evaluate performance
                y_pred = ensemble_model.predict(X_val)
                score = self._calculate_performance_score(y_val, y_pred)

                if score < best_score:
                    best_score = score
                    best_weights = weight_dict.copy()

            logger.info(
                f"Bayesian optimization completed: best score = {best_score:.4f}"
            )
            return best_weights or self._get_default_weights()

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return self._get_default_weights()

    def _calculate_performance_score(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> float:
        """Calculate performance score for optimization."""
        try:
            # Use Mean Absolute Error as optimization metric
            mae = np.mean(np.abs(y_true.values - y_pred))
            return mae
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return float("inf")

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default ensemble weights."""
        return {
            "xgboost": 0.35,
            "lightgbm": 0.30,
            "random_forest": 0.20,
            "ridge": 0.10,
            "mlp_meta": 0.05,
        }


class PerformanceMonitor:
    """Monitor and benchmark model performance."""

    def __init__(self, metrics_dir: str = ".performance_metrics/"):
        """
        Initialize performance monitor.

        Args:
            metrics_dir: Directory to store performance metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.performance_history = []
        self.baseline_metrics = None

        logger.info(f"📊 Performance monitor initialized: {metrics_dir}")

    def benchmark_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model"
    ) -> PerformanceMetrics:
        """
        Benchmark model performance.

        Args:
            model: Model to benchmark
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model

        Returns:
            Performance metrics
        """
        try:
            logger.info(f"🏃 Benchmarking {model_name}")

            # Measure prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds

            # Calculate accuracy metrics
            mae = np.mean(np.abs(y_test.values - y_pred))
            mse = np.mean((y_test.values - y_pred) ** 2)
            rmse = np.sqrt(mse)

            # Calculate memory usage (simplified)
            import psutil

            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

            # Create metrics
            metrics = PerformanceMetrics(
                prediction_time_ms=prediction_time,
                cache_hit_rate=0.0,  # Will be updated by cache manager
                memory_usage_mb=memory_usage,
                model_accuracy=1.0 / (1.0 + mae),  # Normalized accuracy
                ensemble_weight_efficiency=1.0,  # Will be updated for ensembles
                batch_processing_efficiency=1.0,  # Will be updated for batch processing
                optimization_score=self._calculate_optimization_score(
                    prediction_time, mae
                ),
            )

            # Store in history
            self.performance_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "metrics": metrics.__dict__,
                }
            )

            # Save metrics
            self._save_metrics()

            logger.info(
                f"✅ Benchmark completed: {prediction_time:.2f}ms, MAE: {mae:.3f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                prediction_time_ms=1000.0,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,
                model_accuracy=0.0,
                ensemble_weight_efficiency=0.0,
                batch_processing_efficiency=0.0,
                optimization_score=0.0,
            )

    def benchmark_batch_processing(
        self,
        model,
        X_batches: List[pd.DataFrame],
        y_batches: List[pd.Series],
        model_name: str = "batch_model",
    ) -> PerformanceMetrics:
        """
        Benchmark batch processing performance.

        Args:
            model: Model to benchmark
            X_batches: List of feature batches
            y_batches: List of target batches
            model_name: Name of the model

        Returns:
            Performance metrics for batch processing
        """
        try:
            logger.info(f"📦 Benchmarking batch processing for {model_name}")

            start_time = time.time()
            predictions = []

            # Process batches
            for X_batch, y_batch in zip(X_batches, y_batches):
                batch_pred = model.predict(X_batch)
                predictions.extend(batch_pred)

            total_time = (time.time() - start_time) * 1000
            avg_time_per_batch = total_time / len(X_batches)

            # Calculate accuracy
            all_y_true = np.concatenate([y_batch.values for y_batch in y_batches])
            all_y_pred = np.array(predictions)
            mae = np.mean(np.abs(all_y_true - all_y_pred))

            # Calculate batch efficiency
            individual_times = []
            for X_batch in X_batches:
                batch_start = time.time()
                model.predict(X_batch)
                individual_times.append((time.time() - batch_start) * 1000)

            batch_efficiency = (
                min(individual_times) / max(individual_times)
                if individual_times
                else 1.0
            )

            metrics = PerformanceMetrics(
                prediction_time_ms=avg_time_per_batch,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,  # Would need more sophisticated monitoring
                model_accuracy=1.0 / (1.0 + mae),
                ensemble_weight_efficiency=1.0,
                batch_processing_efficiency=batch_efficiency,
                optimization_score=self._calculate_optimization_score(
                    avg_time_per_batch, mae
                ),
            )

            # Store in history
            self.performance_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "metrics": metrics.__dict__,
                    "batch_size": len(X_batches),
                    "total_samples": len(all_y_true),
                }
            )

            self._save_metrics()

            logger.info(
                f"✅ Batch benchmark completed: {avg_time_per_batch:.2f}ms/batch"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error benchmarking batch processing: {e}")
            return PerformanceMetrics(
                prediction_time_ms=1000.0,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,
                model_accuracy=0.0,
                ensemble_weight_efficiency=0.0,
                batch_processing_efficiency=0.0,
                optimization_score=0.0,
            )

    def _calculate_optimization_score(
        self, prediction_time: float, mae: float
    ) -> float:
        """Calculate overall optimization score."""
        try:
            # Target: < 20ms prediction time, minimize MAE
            time_score = max(0, (20 - prediction_time) / 20)  # 20ms target
            accuracy_score = max(0, (10 - mae) / 10)  # MAE target of 10

            return (time_score + accuracy_score) / 2

        except Exception as e:
            logger.warning(f"Error calculating optimization score: {e}")
            return 0.0

    def _save_metrics(self) -> None:
        """Save performance metrics to file."""
        try:
            metrics_file = (
                self.metrics_dir
                / f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            )

            with open(metrics_file, "w") as f:
                json.dump(self.performance_history, f, indent=2)

            logger.debug(f"Performance metrics saved to {metrics_file}")

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for recent days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m
                for m in self.performance_history
                if datetime.fromisoformat(m["timestamp"]) > cutoff_date
            ]

            if not recent_metrics:
                return {
                    "period_days": days,
                    "total_benchmarks": 0,
                    "avg_prediction_time_ms": 0.0,
                    "avg_accuracy": 0.0,
                    "optimization_score": 0.0,
                }

            # Calculate averages
            avg_prediction_time = np.mean(
                [m["metrics"]["prediction_time_ms"] for m in recent_metrics]
            )
            avg_accuracy = np.mean(
                [m["metrics"]["model_accuracy"] for m in recent_metrics]
            )
            avg_optimization_score = np.mean(
                [m["metrics"]["optimization_score"] for m in recent_metrics]
            )

            return {
                "period_days": days,
                "total_benchmarks": len(recent_metrics),
                "avg_prediction_time_ms": float(avg_prediction_time),
                "avg_accuracy": float(avg_accuracy),
                "avg_optimization_score": float(avg_optimization_score),
                "performance_trend": self._calculate_trend(recent_metrics),
                "recommendations": self._generate_recommendations(recent_metrics),
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def _calculate_trend(self, metrics_history: List[Dict[str, Any]]) -> str:
        """Calculate performance trend."""
        try:
            if len(metrics_history) < 2:
                return "insufficient_data"

            # Compare recent vs older performance
            recent_avg = np.mean(
                [m["metrics"]["optimization_score"] for m in metrics_history[-3:]]
            )
            older_avg = np.mean(
                [m["metrics"]["optimization_score"] for m in metrics_history[:-3]]
            )

            if recent_avg > older_avg * 1.05:
                return "improving"
            elif recent_avg < older_avg * 0.95:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.warning(f"Error calculating trend: {e}")
            return "unknown"

    def _generate_recommendations(
        self, metrics_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations."""
        try:
            recommendations = []

            # Analyze prediction times
            prediction_times = [
                m["metrics"]["prediction_time_ms"] for m in metrics_history
            ]
            avg_prediction_time = np.mean(prediction_times)

            if avg_prediction_time > 25:
                recommendations.append(
                    "Consider model optimization to reduce prediction time"
                )

            # Analyze accuracy
            accuracies = [m["metrics"]["model_accuracy"] for m in metrics_history]
            avg_accuracy = np.mean(accuracies)

            if avg_accuracy < 0.7:
                recommendations.append("Consider retraining models with more data")

            # Analyze optimization scores
            opt_scores = [m["metrics"]["optimization_score"] for m in metrics_history]
            avg_opt_score = np.mean(opt_scores)

            if avg_opt_score < 0.5:
                recommendations.append("Consider ensemble weight optimization")

            return recommendations

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Unable to analyze performance trends"]


class ModelPerformanceOptimizer:
    """
    Comprehensive model performance optimization system.

    Integrates:
    - Model caching for <20ms prediction time
    - Ensemble weight optimization
    - Performance monitoring and benchmarking
    - Prediction batching for efficiency
    """

    def __init__(self, data_store: Optional[UnifiedDataStore] = None):
        """
        Initialize model performance optimizer.

        Args:
            data_store: UnifiedDataStore instance for data access
        """
        self.data_store = data_store

        # Initialize components
        self.model_cache = ModelCache()
        self.weight_optimizer = EnsembleWeightOptimizer()
        self.performance_monitor = PerformanceMonitor()

        # Optimization targets
        self.targets = {
            "prediction_time_ms": 20.0,  # Target: < 20ms
            "cache_hit_rate": 0.80,  # Target: > 80%
            "model_accuracy": 0.85,  # Target: > 85%
            "optimization_score": 0.75,  # Target: > 75%
        }

        logger.info("🚀 Model Performance Optimizer initialized")

    def optimize_model_pipeline(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimization_level: str = "moderate",
    ) -> OptimizationResult:
        """
        Optimize complete model pipeline.

        Args:
            model: Model to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimization_level: Optimization level ('conservative', 'moderate', 'aggressive')

        Returns:
            Optimization result with before/after metrics
        """
        try:
            logger.info(f"🎯 Optimizing model pipeline (level: {optimization_level})")

            # Benchmark original performance
            original_metrics = self.performance_monitor.benchmark_model(
                model, X_val, y_val, "original_model"
            )

            # Step 1: Optimize ensemble weights
            if hasattr(model, "estimators_"):
                optimized_weights = self.weight_optimizer.optimize_ensemble_weights(
                    model, X_val, y_val
                )
                model.set_weights(optimized_weights)
                logger.info("✅ Ensemble weights optimized")

            # Step 2: Cache model for faster prediction
            model_config = {
                "model_type": type(model).__name__,
                "feature_count": len(X_train.columns),
                "training_samples": len(X_train),
                "optimization_level": optimization_level,
            }

            self.model_cache.cache_model(model_config, model)

            # Step 3: Benchmark optimized performance
            optimized_metrics = self.performance_monitor.benchmark_model(
                model, X_val, y_val, "optimized_model"
            )

            # Step 4: Generate optimization recommendations
            improvement_percentage = self._calculate_improvement(
                original_metrics, optimized_metrics
            )

            optimization_techniques = []
            if optimization_level in ["moderate", "aggressive"]:
                optimization_techniques.extend(
                    [
                        "ensemble_weight_optimization",
                        "model_caching",
                        "performance_monitoring",
                    ]
                )

            recommendations = self._generate_optimization_recommendations(
                original_metrics, optimized_metrics
            )

            result = OptimizationResult(
                original_metrics=original_metrics,
                optimized_metrics=optimized_metrics,
                improvement_percentage=improvement_percentage,
                optimization_techniques=optimization_techniques,
                recommendations=recommendations,
                optimization_timestamp=datetime.now(),
            )

            logger.info(
                f"✅ Model optimization completed: "
                f"{improvement_percentage:+.1f}% improvement in prediction time"
            )

            return result

        except Exception as e:
            logger.error(f"Error in model optimization: {e}")
            raise OptimizationError(f"Failed to optimize model pipeline: {e}") from e

    def _calculate_improvement(
        self, original: PerformanceMetrics, optimized: PerformanceMetrics
    ) -> float:
        """Calculate improvement percentage."""
        try:
            # Primary metric: prediction time improvement
            time_improvement = (
                (
                    (original.prediction_time_ms - optimized.prediction_time_ms)
                    / original.prediction_time_ms
                )
                if original.prediction_time_ms > 0
                else 0
            )

            # Secondary metrics
            accuracy_improvement = (
                (
                    (optimized.model_accuracy - original.model_accuracy)
                    / original.model_accuracy
                )
                if original.model_accuracy > 0
                else 0
            )

            # Overall improvement (weighted towards prediction time)
            overall_improvement = (
                time_improvement * 0.7 + accuracy_improvement * 0.3
            ) * 100

            return overall_improvement

        except Exception as e:
            logger.warning(f"Error calculating improvement: {e}")
            return 0.0

    def _generate_optimization_recommendations(
        self, original: PerformanceMetrics, optimized: PerformanceMetrics
    ) -> List[str]:
        """Generate optimization recommendations."""
        try:
            recommendations = []

            # Prediction time recommendations
            if optimized.prediction_time_ms > self.targets["prediction_time_ms"]:
                recommendations.append(
                    f"Prediction time ({optimized.prediction_time_ms:.1f}ms) exceeds target "
                    f"({self.targets['prediction_time_ms']:.1f}ms). Consider model pruning."
                )

            # Accuracy recommendations
            if optimized.model_accuracy < self.targets["model_accuracy"]:
                recommendations.append(
                    f"Model accuracy ({optimized.model_accuracy:.3f}) below target "
                    f"({self.targets['model_accuracy']:.3f}). Consider retraining with more data."
                )

            # Memory usage recommendations
            if optimized.memory_usage_mb > 500:  # 500MB threshold
                recommendations.append(
                    f"High memory usage ({optimized.memory_usage_mb:.1f}MB). Consider model compression."
                )

            # Optimization score recommendations
            if optimized.optimization_score < self.targets["optimization_score"]:
                recommendations.append(
                    f"Optimization score ({optimized.optimization_score:.3f}) below target "
                    f"({self.targets['optimization_score']:.3f}). Consider advanced optimization techniques."
                )

            return recommendations

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Unable to generate optimization recommendations"]

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        try:
            cache_stats = self.model_cache.get_cache_stats()
            performance_summary = self.performance_monitor.get_performance_summary()

            return {
                "optimization_targets": self.targets,
                "cache_performance": cache_stats,
                "performance_summary": performance_summary,
                "cache_hit_rate_target_met": float(cache_stats["hit_rate"].rstrip("%"))
                >= self.targets["cache_hit_rate"],
                "prediction_time_target_met": performance_summary.get(
                    "avg_prediction_time_ms", 0
                )
                <= self.targets["prediction_time_ms"],
                "overall_status": self._calculate_overall_status(
                    cache_stats, performance_summary
                ),
            }

        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_overall_status(
        self, cache_stats: Dict[str, Any], performance_summary: Dict[str, Any]
    ) -> str:
        """Calculate overall optimization status."""
        try:
            hit_rate = float(cache_stats["hit_rate"].rstrip("%"))
            avg_time = performance_summary.get("avg_prediction_time_ms", 0)
            opt_score = performance_summary.get("avg_optimization_score", 0)

            # Check if all targets met
            targets_met = (
                hit_rate >= self.targets["cache_hit_rate"]
                and avg_time <= self.targets["prediction_time_ms"]
                and opt_score >= self.targets["optimization_score"]
            )

            if targets_met:
                return "all_targets_met"
            elif hit_rate >= self.targets["cache_hit_rate"]:
                return "cache_target_met"
            elif avg_time <= self.targets["prediction_time_ms"]:
                return "performance_target_met"
            elif opt_score >= self.targets["optimization_score"]:
                return "optimization_target_met"
            else:
                return "needs_improvement"

        except Exception as e:
            logger.warning(f"Error calculating overall status: {e}")
            return "unknown"

    def optimize_batch_processing(
        self,
        model,
        X_data: pd.DataFrame,
        y_data: pd.Series,
        batch_sizes: List[int] = [32, 64, 128],
    ) -> Dict[str, Any]:
        """
        Optimize batch processing for better efficiency.

        Args:
            model: Model to optimize
            X_data: Full dataset
            y_data: Full targets
            batch_sizes: List of batch sizes to test

        Returns:
            Batch optimization results
        """
        try:
            logger.info("📦 Optimizing batch processing")

            results = {}

            for batch_size in batch_sizes:
                # Create batches
                X_batches = [
                    X_data[i : i + batch_size]
                    for i in range(0, len(X_data), batch_size)
                ]
                y_batches = [
                    y_data[i : i + batch_size]
                    for i in range(0, len(y_data), batch_size)
                ]

                # Benchmark batch processing
                metrics = self.performance_monitor.benchmark_batch_processing(
                    model, X_batches, y_batches, f"batch_size_{batch_size}"
                )

                results[f"batch_size_{batch_size}"] = {
                    "avg_time_per_batch_ms": metrics.prediction_time_ms,
                    "batch_efficiency": metrics.batch_processing_efficiency,
                    "total_samples_processed": len(X_data),
                }

            # Find optimal batch size
            optimal_batch_size = min(
                results.keys(), key=lambda k: results[k]["avg_time_per_batch_ms"]
            )

            results["optimal_batch_size"] = optimal_batch_size
            results["recommendation"] = (
                f"Use batch size {optimal_batch_size} for optimal performance"
            )

            logger.info(
                f"✅ Batch optimization completed: optimal size = {optimal_batch_size}"
            )
            return results

        except Exception as e:
            logger.error(f"Error in batch optimization: {e}")
            raise OptimizationError(f"Failed to optimize batch processing: {e}") from e


def create_model_optimizer(
    data_store: Optional[UnifiedDataStore] = None,
) -> ModelPerformanceOptimizer:
    """
    Create and configure model performance optimizer.

    Args:
        data_store: Optional UnifiedDataStore instance

    Returns:
        Configured ModelPerformanceOptimizer instance
    """
    return ModelPerformanceOptimizer(data_store)
