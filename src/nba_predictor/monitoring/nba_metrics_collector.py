#!/usr/bin/env python3
"""
🎯 NBA Prediction Metrics Collector for Prometheus
DevStream SuperPowered Implementation - Context7 Compliant

Real-time metrics collection for NBA prediction models with Prometheus integration.
Provides comprehensive monitoring of model performance, prediction accuracy, and system health.

Author: NBA Predictive Analytics System
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - using mock metrics")

from src.nba_predictor.streamlit.components.ml_integration_bridge import MLIntegrationBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction"""
    timestamp: datetime
    model_name: str
    prediction_type: str  # 'game_outcome', 'player_performance', 'betting_odds'
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    actual_outcome: Optional[Any] = None
    prediction_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model"""
    model_name: str
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    avg_confidence: float = 0.0
    avg_response_time_ms: float = 0.0
    accuracy: float = 0.0
    last_prediction_time: Optional[datetime] = None
    error_rate: float = 0.0
    predictions_last_hour: int = 0
    predictions_last_24h: int = 0


class NBAPredictionMetricsCollector:
    """
    Advanced metrics collector for NBA prediction models with Prometheus integration.

    Features:
    - Real-time prediction metrics collection
    - Model performance tracking
    - Error monitoring and alerting
    - Prometheus metrics export
    - Historical data analysis
    - Confidence calibration monitoring
    """

    def __init__(self,
                 bridge: Optional[MLIntegrationBridge] = None,
                 metrics_port: int = 8000,
                 max_history_size: int = 10000,
                 cleanup_interval_minutes: int = 60):
        """
        Initialize NBA Prediction Metrics Collector

        Args:
            bridge: ML Integration Bridge for model access
            metrics_port: Port for Prometheus metrics server
            max_history_size: Maximum number of predictions to keep in memory
            cleanup_interval_minutes: Interval for cleaning up old metrics
        """
        self.bridge = bridge or MLIntegrationBridge()
        self.metrics_port = metrics_port
        self.max_history_size = max_history_size
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        # Storage for metrics
        self.prediction_history: deque = deque(maxlen=max_history_size)
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.confidence_buckets: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        # Prometheus metrics (if available)
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()

        # Start Prometheus server
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.metrics_port, registry=self.registry)
                logger.info(f"🌐 Prometheus metrics server started on port {self.metrics_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")

        # Start background cleanup task
        self.last_cleanup = datetime.now()

        logger.info("🎯 NBA Prediction Metrics Collector initialized")
        logger.info(f"   - Prometheus integration: {'✅' if PROMETHEUS_AVAILABLE else '❌ (mock mode)'}")
        logger.info(f"   - Metrics port: {self.metrics_port}")
        logger.info(f"   - Max history size: {max_history_size}")
        logger.info(f"   - Cleanup interval: {cleanup_interval_minutes} minutes")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        # Prediction counters
        self.prediction_counter = Counter(
            'nba_predictions_total',
            'Total number of NBA predictions',
            ['model_name', 'prediction_type', 'status'],
            registry=self.registry
        )

        self.prediction_errors = Counter(
            'nba_prediction_errors_total',
            'Total number of NBA prediction errors',
            ['model_name', 'error_type'],
            registry=self.registry
        )

        # Response time histogram
        self.prediction_duration = Histogram(
            'nba_prediction_duration_seconds',
            'Time spent making NBA predictions',
            ['model_name', 'prediction_type'],
            registry=self.registry
        )

        # Confidence gauge
        self.prediction_confidence = Gauge(
            'nba_prediction_confidence',
            'Confidence scores for NBA predictions',
            ['model_name', 'prediction_type'],
            registry=self.registry
        )

        # Model performance gauges
        self.model_accuracy = Gauge(
            'nba_model_accuracy',
            'Accuracy of NBA prediction models',
            ['model_name'],
            registry=self.registry
        )

        self.model_error_rate = Gauge(
            'nba_model_error_rate',
            'Error rate of NBA prediction models',
            ['model_name'],
            registry=self.registry
        )

        # System health gauges
        self.active_models = Gauge(
            'nba_active_models_count',
            'Number of active NBA prediction models',
            registry=self.registry
        )

        self.predictions_per_minute = Gauge(
            'nba_predictions_per_minute',
            'NBA predictions per minute',
            registry=self.registry
        )

        # Feature drift metrics
        self.feature_drift_score = Gauge(
            'nba_feature_drift_score',
            'Drift score for input features',
            ['feature_name'],
            registry=self.registry
        )

        logger.info("✅ Prometheus metrics setup complete")

    def record_prediction(self,
                         model_name: str,
                         prediction_type: str,
                         input_features: Dict[str, Any],
                         prediction: Any,
                         confidence: float,
                         prediction_time_ms: float,
                         success: bool = True,
                         error_message: Optional[str] = None,
                         actual_outcome: Optional[Any] = None) -> str:
        """
        Record a prediction and its metrics

        Args:
            model_name: Name of the model that made the prediction
            prediction_type: Type of prediction (game_outcome, player_performance, etc.)
            input_features: Input features used for prediction
            prediction: The prediction result
            confidence: Confidence score (0-1)
            prediction_time_ms: Time taken to make prediction in milliseconds
            success: Whether the prediction was successful
            error_message: Error message if prediction failed
            actual_outcome: Actual outcome if available (for accuracy calculation)

        Returns:
            Unique prediction ID
        """
        prediction_id = f"{model_name}_{int(time.time() * 1000)}"

        # Create prediction metrics
        metrics = PredictionMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            prediction_type=prediction_type,
            input_features=input_features,
            prediction=prediction,
            confidence=confidence,
            actual_outcome=actual_outcome,
            prediction_time_ms=prediction_time_ms,
            success=success,
            error_message=error_message
        )

        # Store in history
        self.prediction_history.append(metrics)

        # Update model metrics
        self._update_model_metrics(model_name, metrics)

        # Update Prometheus metrics
        self._update_prometheus_metrics(model_name, prediction_type, metrics)

        # Cleanup old data if needed
        self._cleanup_old_metrics()

        return prediction_id

    def _update_model_metrics(self, model_name: str, metrics: PredictionMetrics):
        """Update model performance metrics"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelPerformanceMetrics(model_name=model_name)

        model_metrics = self.model_metrics[model_name]
        model_metrics.total_predictions += 1
        model_metrics.last_prediction_time = metrics.timestamp

        if metrics.success:
            model_metrics.successful_predictions += 1
        else:
            model_metrics.failed_predictions += 1
            self.error_counts[model_name] += 1

        # Update rolling averages
        self._update_averages(model_metrics, metrics)

        # Update confidence buckets for calibration
        self.confidence_buckets[model_name].append((metrics.confidence, metrics.success))

        # Calculate accuracy if we have actual outcomes
        if metrics.actual_outcome is not None:
            self._calculate_accuracy(model_name)

    def _update_averages(self, model_metrics: ModelPerformanceMetrics, metrics: PredictionMetrics):
        """Update rolling averages for model metrics"""
        alpha = 0.1  # Smoothing factor

        # Update average confidence
        if model_metrics.total_predictions == 1:
            model_metrics.avg_confidence = metrics.confidence
        else:
            model_metrics.avg_confidence = (
                alpha * metrics.confidence +
                (1 - alpha) * model_metrics.avg_confidence
            )

        # Update average response time
        if model_metrics.total_predictions == 1:
            model_metrics.avg_response_time_ms = metrics.prediction_time_ms
        else:
            model_metrics.avg_response_time_ms = (
                alpha * metrics.prediction_time_ms +
                (1 - alpha) * model_metrics.avg_response_time_ms
            )

        # Calculate error rate
        model_metrics.error_rate = (
            model_metrics.failed_predictions / model_metrics.total_predictions
        )

        # Update recent predictions counts
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        recent_predictions = [
            p for p in self.prediction_history
            if p.model_name == model_metrics.model_name and p.timestamp > one_hour_ago
        ]
        model_metrics.predictions_last_hour = len(recent_predictions)

        day_predictions = [
            p for p in self.prediction_history
            if p.model_name == model_metrics.model_name and p.timestamp > one_day_ago
        ]
        model_metrics.predictions_last_24h = len(day_predictions)

    def _update_prometheus_metrics(self, model_name: str, prediction_type: str, metrics: PredictionMetrics):
        """Update Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        # Update prediction counter
        status = 'success' if metrics.success else 'error'
        self.prediction_counter.labels(
            model_name=model_name,
            prediction_type=prediction_type,
            status=status
        ).inc()

        # Update error counter
        if not metrics.success and metrics.error_message:
            error_type = self._classify_error(metrics.error_message)
            self.prediction_errors.labels(
                model_name=model_name,
                error_type=error_type
            ).inc()

        # Update prediction duration
        duration_seconds = metrics.prediction_time_ms / 1000.0
        self.prediction_duration.labels(
            model_name=model_name,
            prediction_type=prediction_type
        ).observe(duration_seconds)

        # Update confidence gauge
        self.prediction_confidence.labels(
            model_name=model_name,
            prediction_type=prediction_type
        ).set(metrics.confidence)

        # Update model metrics
        if model_name in self.model_metrics:
            model_metrics = self.model_metrics[model_name]
            self.model_accuracy.labels(model_name=model_name).set(model_metrics.accuracy)
            self.model_error_rate.labels(model_name=model_name).set(model_metrics.error_rate)

        # Update system metrics
        self.active_models.set(len(self.model_metrics))
        self._update_predictions_per_minute()

        # Update feature drift metrics
        self._update_feature_drift_metrics(metrics.input_features)

    def _classify_error(self, error_message: str) -> str:
        """Classify error type for Prometheus metrics"""
        error_message = error_message.lower()

        if 'timeout' in error_message or 'time out' in error_message:
            return 'timeout'
        elif 'memory' in error_message or 'out of memory' in error_message:
            return 'memory'
        elif 'network' in error_message or 'connection' in error_message:
            return 'network'
        elif 'model' in error_message and 'load' in error_message:
            return 'model_load'
        elif 'feature' in error_message or 'validation' in error_message:
            return 'feature_validation'
        else:
            return 'unknown'

    def _update_predictions_per_minute(self):
        """Update predictions per minute metric"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_predictions = [
            p for p in self.prediction_history if p.timestamp > one_minute_ago
        ]

        self.predictions_per_minute.set(len(recent_predictions))

    def _update_feature_drift_metrics(self, input_features: Dict[str, Any]):
        """Update feature drift metrics"""
        # Simple drift detection based on feature value ranges
        # In a real implementation, this would use more sophisticated drift detection

        for feature_name, value in input_features.items():
            if isinstance(value, (int, float)):
                # Normalize feature value to 0-1 range for drift score
                # This is a simplified approach - real implementation would use historical baselines
                drift_score = abs(value) / (abs(value) + 1.0)  # Simple normalization

                self.feature_drift_score.labels(
                    feature_name=feature_name
                ).set(drift_score)

    def _calculate_accuracy(self, model_name: str):
        """Calculate model accuracy based on predictions with known outcomes"""
        model_predictions = [
            p for p in self.prediction_history
            if p.model_name == model_name and p.actual_outcome is not None
        ]

        if not model_predictions:
            return

        correct_predictions = sum(
            1 for p in model_predictions
            if self._is_prediction_correct(p)
        )

        accuracy = correct_predictions / len(model_predictions)
        self.model_metrics[model_name].accuracy = accuracy

    def _is_prediction_correct(self, metrics: PredictionMetrics) -> bool:
        """Determine if a prediction was correct"""
        # This would be implemented based on the specific prediction type
        # For now, we'll use a simple comparison
        if metrics.actual_outcome is None:
            return False

        # Simple implementation - would be more sophisticated for real predictions
        try:
            if isinstance(metrics.prediction, (int, float)) and isinstance(metrics.actual_outcome, (int, float)):
                return abs(metrics.prediction - metrics.actual_outcome) < 0.1
            else:
                return str(metrics.prediction) == str(metrics.actual_outcome)
        except Exception:
            return False

    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        now = datetime.now()

        # Check if it's time for cleanup
        if now - self.last_cleanup < self.cleanup_interval:
            return

        # Remove old confidence bucket data
        cutoff_time = now - timedelta(days=7)  # Keep 7 days of confidence data
        for model_name in self.confidence_buckets:
            self.confidence_buckets[model_name] = [
                (confidence, success) for confidence, success in self.confidence_buckets[model_name]
                if confidence > cutoff_time.timestamp()
            ]

        self.last_cleanup = now
        logger.debug("🧹 Old metrics data cleaned up")

    def get_model_metrics(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """Get performance metrics for a specific model"""
        return self.model_metrics.get(model_name)

    def get_all_model_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models"""
        return self.model_metrics.copy()

    def get_predictions_summary(self,
                              minutes: int = 60,
                              model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of predictions in the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        predictions = [
            p for p in self.prediction_history
            if p.timestamp > cutoff_time and (model_name is None or p.model_name == model_name)
        ]

        if not predictions:
            return {
                'total_predictions': 0,
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_response_time_ms': 0.0,
                'error_count': 0
            }

        successful_predictions = [p for p in predictions if p.success]
        total_confidence = sum(p.confidence for p in predictions)
        total_response_time = sum(p.prediction_time_ms for p in predictions)

        return {
            'total_predictions': len(predictions),
            'successful_predictions': len(successful_predictions),
            'success_rate': len(successful_predictions) / len(predictions),
            'avg_confidence': total_confidence / len(predictions),
            'avg_response_time_ms': total_response_time / len(predictions),
            'error_count': len(predictions) - len(successful_predictions),
            'predictions_by_model': {
                model: len([p for p in predictions if p.model_name == model])
                for model in set(p.model_name for p in predictions)
            }
        }

    def get_confidence_calibration(self, model_name: str, bins: int = 10) -> Dict[str, List]:
        """Get confidence calibration data for a model"""
        if model_name not in self.confidence_buckets:
            return {'bins': [], 'accuracy': [], 'count': []}

        confidence_data = self.confidence_buckets[model_name]

        # Create confidence bins
        bin_edges = [i/bins for i in range(bins + 1)]
        bin_accuracies = []
        bin_counts = []

        for i in range(bins):
            lower_bound = bin_edges[i]
            upper_bound = bin_edges[i + 1]

            bin_predictions = [
                (confidence, success) for confidence, success in confidence_data
                if lower_bound <= confidence < upper_bound
            ]

            if bin_predictions:
                accuracy = sum(success for _, success in bin_predictions) / len(bin_predictions)
                bin_accuracies.append(accuracy)
                bin_counts.append(len(bin_predictions))
            else:
                bin_accuracies.append(0.0)
                bin_counts.append(0)

        return {
            'bins': [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(bins)],
            'accuracy': bin_accuracies,
            'count': bin_counts
        }

    def export_metrics_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON"""
        return {
            'timestamp': datetime.now().isoformat(),
            'models': {
                name: {
                    'total_predictions': metrics.total_predictions,
                    'successful_predictions': metrics.successful_predictions,
                    'failed_predictions': metrics.failed_predictions,
                    'accuracy': metrics.accuracy,
                    'avg_confidence': metrics.avg_confidence,
                    'avg_response_time_ms': metrics.avg_response_time_ms,
                    'error_rate': metrics.error_rate,
                    'predictions_last_hour': metrics.predictions_last_hour,
                    'predictions_last_24h': metrics.predictions_last_24h,
                    'last_prediction_time': metrics.last_prediction_time.isoformat() if metrics.last_prediction_time else None
                }
                for name, metrics in self.model_metrics.items()
            },
            'summary_1h': self.get_predictions_summary(60),
            'summary_24h': self.get_predictions_summary(1440),
            'total_predictions': len(self.prediction_history),
            'prometheus_port': self.metrics_port if PROMETHEUS_AVAILABLE else None
        }

    def shutdown(self):
        """Cleanup and shutdown the metrics collector"""
        logger.info("🔄 Shutting down NBA Prediction Metrics Collector")

        # Export final metrics
        final_metrics = self.export_metrics_json()
        logger.info(f"📊 Final metrics summary: {json.dumps(final_metrics, indent=2)}")

        logger.info("✅ Metrics collector shutdown complete")


# Singleton instance for global access
_metrics_collector: Optional[NBAPredictionMetricsCollector] = None


def get_metrics_collector() -> NBAPredictionMetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = NBAPredictionMetricsCollector()
    return _metrics_collector


def create_metrics_collector(bridge: Optional[MLIntegrationBridge] = None,
                           metrics_port: int = 8000) -> NBAPredictionMetricsCollector:
    """Create a new metrics collector instance"""
    return NBAPredictionMetricsCollector(bridge=bridge, metrics_port=metrics_port)


if __name__ == "__main__":
    # Example usage
    collector = get_metrics_collector()

    # Record some example predictions
    for i in range(10):
        prediction_id = collector.record_prediction(
            model_name="nba_game_predictor",
            prediction_type="game_outcome",
            input_features={
                "home_team_momentum": 0.7 + i * 0.1,
                "away_team_momentum": -0.3 + i * 0.05,
                "home_team_rest_days": 2,
                "away_team_rest_days": 1
            },
            prediction="home_win" if i % 2 == 0 else "away_win",
            confidence=0.7 + (i % 3) * 0.1,
            prediction_time_ms=150 + i * 10,
            success=True
        )
        print(f"Recorded prediction: {prediction_id}")

    # Print metrics summary
    print("\n📊 Metrics Summary:")
    print(json.dumps(collector.export_metrics_json(), indent=2))

    # Keep the server running
    print(f"\n🌐 Prometheus metrics available at http://localhost:{collector.metrics_port}")
    try:
        while True:
            time.sleep(60)
            collector._cleanup_old_metrics()
    except KeyboardInterrupt:
        collector.shutdown()