#!/usr/bin/env python3
"""
📊 Model Performance Monitor - Drift Detection & Performance Tracking
Advanced monitoring system for NBA predictive models with drift detection and alerts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Single model performance metrics snapshot."""
    timestamp: datetime
    predictions_count: int
    mae: float
    rmse: float
    r2_score: float
    prediction_bias: float
    confidence_mean: float
    feature_drift_score: float
    accuracy_trend: str  # 'improving', 'declining', 'stable'

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection parameters."""
    performance_window: int = 50  # Number of recent predictions to evaluate
    drift_threshold: float = 0.15  # Performance degradation threshold
    feature_drift_threshold: float = 0.2  # Feature distribution drift threshold
    confidence_threshold: float = 0.1  # Minimum confidence for reliable predictions
    alert_threshold: int = 3  # Consecutive poor performances before alert

class ModelPerformanceMonitor:
    """
    Advanced model monitoring system with drift detection,
    performance tracking, and automatic alerts.
    """

    def __init__(self,
                 model_name: str,
                 config: DriftDetectionConfig = None,
                 monitoring_dir: str = "monitoring"):
        """
        Initialize model performance monitor.

        Args:
            model_name: Name of the model being monitored
            config: Drift detection configuration
            monitoring_dir: Directory to store monitoring data
        """
        self.model_name = model_name
        self.config = config or DriftDetectionConfig()
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.metrics_history: List[ModelMetrics] = []
        self.predictions_buffer = deque(maxlen=self.config.performance_window)
        self.feature_baseline: Optional[Dict] = None
        self.performance_baseline: Optional[Dict] = None

        # Alert system
        self.alert_history: List[Dict] = []
        self.consecutive_poor_performance = 0
        self.last_alert_time = None

        # File paths
        self.metrics_file = self.monitoring_dir / f"{model_name}_metrics.json"
        self.alerts_file = self.monitoring_dir / f"{model_name}_alerts.json"

        logger.info(f"📊 ModelPerformanceMonitor initialized for {model_name}")

    def record_prediction(self,
                         prediction: float,
                         actual: Optional[float] = None,
                         confidence: float = 0.0,
                         features: Optional[Dict[str, float]] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Record a new prediction and its outcome.

        Args:
            prediction: Model prediction
            actual: Actual value (if available)
            confidence: Prediction confidence score
            features: Feature values used for prediction
            metadata: Additional metadata

        Returns:
            Current performance status
        """
        timestamp = datetime.now()

        prediction_record = {
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'features': features or {},
            'metadata': metadata or {},
            'error': abs(prediction - actual) if actual is not None else None
        }

        # Add to buffer
        self.predictions_buffer.append(prediction_record)

        # Calculate metrics if we have enough data
        status = {
            'timestamp': timestamp,
            'predictions_count': len(self.predictions_buffer),
            'status': 'collecting_data'
        }

        if len(self.predictions_buffer) >= 10:  # Minimum for metrics calculation
            metrics = self._calculate_current_metrics()
            status.update({
                'status': 'active_monitoring',
                'current_metrics': metrics,
                'drift_detected': self._check_performance_drift(metrics),
                'feature_drift': self._check_feature_drift(features) if features else None
            })

            # Store metrics history
            self.metrics_history.append(ModelMetrics(
                timestamp=timestamp,
                predictions_count=len(self.predictions_buffer),
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                r2_score=metrics['r2_score'],
                prediction_bias=metrics['prediction_bias'],
                confidence_mean=metrics['confidence_mean'],
                feature_drift_score=metrics.get('feature_drift_score', 0.0),
                accuracy_trend=metrics['trend']
            ))

        return status

    def _calculate_current_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics from buffer."""
        if not self.predictions_buffer:
            return {}

        # Get completed predictions (with actual values)
        completed_predictions = [
            p for p in self.predictions_buffer
            if p['actual'] is not None and p['error'] is not None
        ]

        if len(completed_predictions) < 5:
            return {'status': 'insufficient_data'}

        # Calculate basic metrics
        predictions = [p['prediction'] for p in completed_predictions]
        actuals = [p['actual'] for p in completed_predictions]
        errors = [p['error'] for p in completed_predictions]
        confidences = [p['confidence'] for p in completed_predictions]

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        prediction_bias = np.mean(predictions) - np.mean(actuals)

        # Calculate R² score
        if len(actuals) > 1 and np.var(actuals) > 0:
            r2_score = 1 - (np.var([a - p for a, p in zip(actuals, predictions)]) / np.var(actuals))
        else:
            r2_score = 0.0

        # Calculate trend (compare with recent history)
        trend = self._calculate_performance_trend(errors)

        metrics = {
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'r2_score': round(max(0, r2_score), 3),
            'prediction_bias': round(prediction_bias, 3),
            'confidence_mean': round(np.mean(confidences), 3),
            'trend': trend,
            'sample_size': len(completed_predictions)
        }

        # Set baseline if this is first calculation
        if self.performance_baseline is None:
            self.performance_baseline = metrics.copy()
            logger.info(f"📊 Performance baseline established: MAE={mae:.3f}, R²={r2_score:.3f}")

        return metrics

    def _calculate_performance_trend(self, recent_errors: List[float]) -> str:
        """Calculate performance trend based on recent errors."""
        if len(self.metrics_history) < 2:
            return 'stable'

        recent_mae = np.mean(recent_errors)
        historical_maes = [m.mae for m in self.metrics_history[-5:]]  # Last 5 metrics

        if len(historical_maes) >= 3:
            historical_avg = np.mean(historical_maes[:-1])  # Exclude most recent
            change_ratio = (recent_mae - historical_avg) / historical_avg

            if change_ratio > 0.1:
                return 'declining'
            elif change_ratio < -0.1:
                return 'improving'
            else:
                return 'stable'

        return 'stable'

    def _check_performance_drift(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check for performance degradation compared to baseline."""
        if not self.performance_baseline:
            return {'drift_detected': False, 'reason': 'no_baseline'}

        drift_indicators = []

        # Check MAE degradation
        baseline_mae = self.performance_baseline['mae']
        current_mae = current_metrics['mae']
        mae_degradation = (current_mae - baseline_mae) / baseline_mae

        if mae_degradation > self.config.drift_threshold:
            drift_indicators.append(f"MAE degradation: {mae_degradation:.1%}")

        # Check R² degradation
        baseline_r2 = self.performance_baseline['r2_score']
        current_r2 = current_metrics['r2_score']
        r2_degradation = (baseline_r2 - current_r2) / max(baseline_r2, 0.1)

        if r2_degradation > self.config.drift_threshold:
            drift_indicators.append(f"R² degradation: {r2_degradation:.1%}")

        # Check prediction bias increase
        baseline_bias = abs(self.performance_baseline['prediction_bias'])
        current_bias = abs(current_metrics['prediction_bias'])
        bias_increase = (current_bias - baseline_bias) / max(baseline_bias, 0.1)

        if bias_increase > self.config.drift_threshold:
            drift_indicators.append(f"Bias increase: {bias_increase:.1%}")

        drift_detected = len(drift_indicators) > 0

        if drift_detected:
            self.consecutive_poor_performance += 1
            logger.warning(f"🚨 Performance drift detected: {', '.join(drift_indicators)}")

            # Trigger alert if threshold reached
            if self.consecutive_poor_performance >= self.config.alert_threshold:
                self._trigger_alert('performance_drift', drift_indicators)
        else:
            self.consecutive_poor_performance = 0

        return {
            'drift_detected': drift_detected,
            'indicators': drift_indicators,
            'consecutive_poor_performance': self.consecutive_poor_performance
        }

    def _check_feature_drift(self, current_features: Dict[str, float]) -> Dict[str, Any]:
        """Check for feature distribution drift."""
        if not current_features or not self.feature_baseline:
            return {'drift_detected': False, 'reason': 'no_baseline'}

        drift_scores = {}
        significant_drifts = []

        for feature_name, current_value in current_features.items():
            if feature_name in self.feature_baseline:
                baseline_stats = self.feature_baseline[feature_name]
                baseline_mean = baseline_stats['mean']
                baseline_std = baseline_stats['std']

                if baseline_std > 0:
                    z_score = abs(current_value - baseline_mean) / baseline_std
                    drift_scores[feature_name] = z_score

                    if z_score > 2.5:  # Significant drift threshold
                        significant_drifts.append(f"{feature_name}: {z_score:.1f}σ")

        drift_detected = len(significant_drifts) > 0

        if drift_detected:
            logger.warning(f"🚨 Feature drift detected: {', '.join(significant_drifts)}")
            self._trigger_alert('feature_drift', significant_drifts)

        return {
            'drift_detected': drift_detected,
            'drift_scores': drift_scores,
            'significant_features': significant_drifts
        }

    def establish_feature_baseline(self, feature_data: pd.DataFrame):
        """
        Establish baseline statistics for features.

        Args:
            feature_data: DataFrame with feature values
        """
        feature_stats = {}

        for column in feature_data.select_dtypes(include=[np.number]).columns:
            if feature_data[column].notna().sum() > 0:
                feature_stats[column] = {
                    'mean': float(feature_data[column].mean()),
                    'std': float(feature_data[column].std()),
                    'min': float(feature_data[column].min()),
                    'max': float(feature_data[column].max()),
                    'median': float(feature_data[column].median())
                }

        self.feature_baseline = feature_stats
        logger.info(f"📊 Feature baseline established for {len(feature_stats)} features")

    def _trigger_alert(self, alert_type: str, details: List[str]):
        """Trigger a monitoring alert."""
        alert = {
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'details': details,
            'model_name': self.model_name,
            'consecutive_count': self.consecutive_poor_performance,
            'current_performance': self._calculate_current_metrics() if self.predictions_buffer else None
        }

        self.alert_history.append(alert)
        self.last_alert_time = datetime.now()

        logger.error(f"🚨 ALERT [{alert_type.upper()}]: {', '.join(details)}")
        self._save_alerts()

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        if not self.metrics_history:
            return {
                'model_name': self.model_name,
                'status': 'no_data',
                'message': 'No performance data available'
            }

        # Calculate summary statistics
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        avg_mae = np.mean([m.mae for m in recent_metrics])
        avg_rmse = np.mean([m.rmse for m in recent_metrics])
        avg_r2 = np.mean([m.r2_score for m in recent_metrics])

        # Calculate trend
        if len(self.metrics_history) >= 5:
            recent_maes = [m.mae for m in self.metrics_history[-5:]]
            older_maes = [m.mae for m in self.metrics_history[-10:-5]] if len(self.metrics_history) >= 10 else recent_maes

            recent_avg = np.mean(recent_maes)
            older_avg = np.mean(older_maes)

            trend = 'improving' if recent_avg < older_avg else 'declining' if recent_avg > older_avg else 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'model_name': self.model_name,
            'status': 'active',
            'monitoring_period': {
                'start': self.metrics_history[0].timestamp,
                'end': self.metrics_history[-1].timestamp,
                'duration_days': (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp).days
            },
            'performance_summary': {
                'current_mae': round(self.metrics_history[-1].mae, 3),
                'average_mae': round(avg_mae, 3),
                'average_rmse': round(avg_rmse, 3),
                'average_r2': round(avg_r2, 3),
                'trend': trend
            },
            'drift_status': {
                'consecutive_poor_performance': self.consecutive_poor_performance,
                'total_alerts': len(self.alert_history),
                'last_alert': self.last_alert_time.isoformat() if self.last_alert_time else None
            },
            'data_quality': {
                'total_predictions': len(self.predictions_buffer),
                'completed_predictions': len([p for p in self.predictions_buffer if p['actual'] is not None]),
                'completion_rate': len([p for p in self.predictions_buffer if p['actual'] is not None]) / len(self.predictions_buffer) if self.predictions_buffer else 0
            }
        }

    def _save_alerts(self):
        """Save alerts to file."""
        alerts_data = [
            {
                'timestamp': alert['timestamp'].isoformat(),
                'alert_type': alert['alert_type'],
                'details': alert['details'],
                'consecutive_count': alert['consecutive_count']
            }
            for alert in self.alert_history
        ]

        with open(self.alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)

    def save_metrics(self):
        """Save performance metrics to file."""
        metrics_data = [
            {
                'timestamp': m.timestamp.isoformat(),
                'predictions_count': m.predictions_count,
                'mae': m.mae,
                'rmse': m.rmse,
                'r2_score': m.r2_score,
                'prediction_bias': m.prediction_bias,
                'confidence_mean': m.confidence_mean,
                'feature_drift_score': m.feature_drift_score,
                'accuracy_trend': m.accuracy_trend
            }
            for m in self.metrics_history
        ]

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    def generate_health_report(self) -> str:
        """Generate human-readable health report."""
        summary = self.get_monitoring_summary()

        if summary['status'] == 'no_data':
            return f"❌ {self.model_name}: No monitoring data available"

        health_status = "🟢 HEALTHY"
        if summary['drift_status']['consecutive_poor_performance'] > 0:
            health_status = "🟡 WARNING"
        if summary['drift_status']['consecutive_poor_performance'] >= self.config.alert_threshold:
            health_status = "🔴 CRITICAL"

        report = f"""
{health_status} - {self.model_name} Health Report
{'='*50}

📊 Performance (Recent):
   • MAE: {summary['performance_summary']['current_mae']:.3f} (avg: {summary['performance_summary']['average_mae']:.3f})
   • R² Score: {summary['performance_summary']['average_r2']:.3f}
   • Trend: {summary['performance_summary']['trend'].upper()}

🚨 Drift Status:
   • Consecutive Poor Performance: {summary['drift_status']['consecutive_poor_performance']}
   • Total Alerts: {summary['drift_status']['total_alerts']}
   • Last Alert: {summary['drift_status']['last_alert'] or 'None'}

📈 Data Quality:
   • Total Predictions: {summary['data_quality']['total_predictions']}
   • Completion Rate: {summary['data_quality']['completion_rate']:.1%}
   • Monitoring Period: {summary['monitoring_period']['duration_days']} days

💡 Recommendations:
{self._generate_recommendations(summary)}
        """

        return report.strip()

    def _generate_recommendations(self, summary: Dict[str, Any]) -> str:
        """Generate actionable recommendations based on current status."""
        recommendations = []

        if summary['drift_status']['consecutive_poor_performance'] >= self.config.alert_threshold:
            recommendations.append("• 🚨 URGENT: Model retraining required - significant performance degradation detected")

        elif summary['drift_status']['consecutive_poor_performance'] > 0:
            recommendations.append("• ⚠️ Monitor closely - recent performance decline detected")

        if summary['performance_summary']['trend'] == 'declining':
            recommendations.append("• 📉 Performance trending downward - consider investigating data quality")

        if summary['data_quality']['completion_rate'] < 0.8:
            recommendations.append("• 📊 Low completion rate - check data pipeline for missing actual values")

        if summary['performance_summary']['average_r2'] < 0.3:
            recommendations.append("• 🎯 Low predictive power - feature engineering may be needed")

        if not recommendations:
            recommendations.append("• ✅ Model performing within expected parameters")

        return "\n".join(recommendations)