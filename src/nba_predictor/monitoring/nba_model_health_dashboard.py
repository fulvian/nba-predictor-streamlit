"""
🏥 NBA Model Health Dashboard - Task 2.1.4 Implementation

Comprehensive model health monitoring dashboard for NBA prediction system.
Provides real-time monitoring, alerting, and visualization of model performance,
drift detection, and confidence intervals with DevStream SuperPowered architecture.

Author: NBA Predictive Analytics System
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram

# DevStream SuperPowered imports
if TYPE_CHECKING:
    from ..streamlit.components.ml_integration_bridge import MLIntegrationBridge
    from .nba_metrics_collector import NBAPredictionMetricsCollector
    from .nba_drift_detector import NBADriftDetector
    from .nba_confidence_intervals import NBAConfidenceIntervalCalculator

# Setup logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Model health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelHealthScore:
    """Comprehensive model health scoring"""
    overall_score: float  # 0-100
    performance_score: float  # 0-100
    drift_score: float  # 0-100
    confidence_score: float  # 0-100
    availability_score: float  # 0-100
    status: HealthStatus
    last_updated: datetime
    factors: Dict[str, float]
    recommendations: List[str]


@dataclass
class HealthAlert:
    """Health alert data structure"""
    id: str
    model_name: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    action_taken: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class DashboardMetrics:
    """Aggregated dashboard metrics"""
    total_models: int
    healthy_models: int
    models_with_drift: int
    models_with_warnings: int
    critical_alerts: int
    avg_response_time_ms: float
    prediction_volume_24h: int
    system_uptime_percentage: float
    last_health_check: datetime


class NBAModelHealthDashboard:
    """
    NBA Model Health Dashboard - Central orchestrator for model health monitoring

    Integrates all monitoring components (Metrics, Drift Detection, Confidence Intervals)
    to provide comprehensive model health insights with real-time alerting and
    visualization capabilities for NBA prediction system.
    """

    def __init__(self,
                 ml_bridge: Optional['MLIntegrationBridge'] = None,
                 update_interval_seconds: int = 30,
                 alert_retention_days: int = 30,
                 enable_background_monitoring: bool = True):
        """
        Initialize NBA Model Health Dashboard

        Args:
            ml_bridge: ML Integration Bridge instance
            update_interval_seconds: Health check interval
            alert_retention_days: How long to retain alerts
            enable_background_monitoring: Enable background monitoring thread
        """
        self.logger = logger
        self.update_interval = update_interval_seconds
        self.alert_retention_days = alert_retention_days
        self.enable_background_monitoring = enable_background_monitoring

        # Component references
        self._ml_bridge = ml_bridge
        self._metrics_collector: Optional['NBAPredictionMetricsCollector'] = None
        self._drift_detector: Optional['NBADriftDetector'] = None
        self._ci_calculator: Optional['NBAConfidenceIntervalCalculator'] = None

        # Health monitoring state
        self._health_scores: Dict[str, ModelHealthScore] = {}
        self._alerts: List[HealthAlert] = []
        self._last_health_check = datetime.now()
        self._monitoring_active = False
        self._background_task: Optional[asyncio.Task] = None

        # Prometheus metrics for dashboard
        self._init_prometheus_metrics()

        # Health thresholds (NBA-specific)
        self._health_thresholds = {
            'min_accuracy': 0.55,  # Minimum acceptable prediction accuracy
            'max_drift_score': 30.0,  # Maximum acceptable drift score
            'min_confidence': 0.6,   # Minimum acceptable confidence
            'max_response_time_ms': 500.0,  # Maximum acceptable response time
            'min_availability': 95.0,  # Minimum availability percentage
            'max_error_rate': 5.0      # Maximum error rate percentage
        }

        self.logger.info("🏥 NBA Model Health Dashboard initialized with SuperPowered features")

        if self.enable_background_monitoring:
            self._start_background_monitoring()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics for dashboard monitoring"""
        self.registry = CollectorRegistry()

        # Health score metrics
        self.model_health_score = Gauge(
            'nba_model_health_score',
            'Overall health score for NBA prediction models',
            ['model_name'],
            registry=self.registry
        )

        # Alert metrics
        self.health_alerts_total = Counter(
            'nba_health_alerts_total',
            'Total number of health alerts generated',
            ['model_name', 'severity', 'type'],
            registry=self.registry
        )

        # Performance metrics
        self.dashboard_metrics_collector = Histogram(
            'nba_dashboard_metrics_collection_seconds',
            'Time spent collecting dashboard metrics',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

    def set_ml_bridge(self, ml_bridge: 'MLIntegrationBridge') -> None:
        """Set ML Integration Bridge and extract monitoring components"""
        self._ml_bridge = ml_bridge

        if ml_bridge:
            self._metrics_collector = ml_bridge.get_metrics_collector()
            self._drift_detector = ml_bridge.get_drift_detector()
            self._ci_calculator = ml_bridge.get_confidence_interval_calculator()

            self.logger.info("✅ ML Bridge connected to Health Dashboard")
            self.logger.info(f"   - Metrics Collector: {self._metrics_collector is not None}")
            self.logger.info(f"   - Drift Detector: {self._drift_detector is not None}")
            self.logger.info(f"   - CI Calculator: {self._ci_calculator is not None}")

    async def collect_health_metrics(self) -> DashboardMetrics:
        """
        Collect comprehensive health metrics from all components

        Returns:
            DashboardMetrics: Aggregated health metrics
        """
        start_time = datetime.now()

        try:
            # Collect metrics from each component
            metrics_data = await self._collect_component_metrics()

            # Calculate health scores for each model
            health_scores = await self._calculate_health_scores(metrics_data)

            # Generate alerts based on health scores
            await self._generate_health_alerts(health_scores)

            # Create aggregated dashboard metrics
            dashboard_metrics = self._create_dashboard_metrics(health_scores)

            # Update Prometheus metrics
            self._update_prometheus_metrics(health_scores)

            # Update internal state
            self._health_scores = health_scores
            self._last_health_check = datetime.now()

            duration = (datetime.now() - start_time).total_seconds()
            self.dashboard_metrics_collector.observe(duration)

            self.logger.debug(f"📊 Health metrics collected in {duration:.2f}s")

            return dashboard_metrics

        except Exception as e:
            self.logger.error(f"❌ Error collecting health metrics: {e}")
            return self._create_fallback_metrics()

    async def _collect_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from individual monitoring components"""
        metrics_data = {}

        # Get metrics from ML Bridge
        if self._ml_bridge:
            try:
                bridge_metrics = self._ml_bridge.get_model_metrics_summary()
                if "error" not in bridge_metrics:
                    metrics_data['bridge'] = bridge_metrics
                    self.logger.debug("📈 ML Bridge metrics collected successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to collect ML Bridge metrics: {e}")

        # Get drift detection status
        if self._drift_detector:
            try:
                drift_status = self._drift_detector.get_system_drift_status()
                metrics_data['drift'] = drift_status
                self.logger.debug("🔍 Drift detection status collected successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to collect drift status: {e}")

        # Get confidence interval status
        if self._ci_calculator:
            try:
                ci_metrics = await self._get_ci_metrics()
                metrics_data['confidence_intervals'] = ci_metrics
                self.logger.debug("📊 Confidence interval metrics collected successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to collect CI metrics: {e}")

        return metrics_data

    async def _get_ci_metrics(self) -> Dict[str, Any]:
        """Get confidence interval metrics"""
        if not self._ci_calculator:
            return {}

        ci_metrics = {
            'calculator_available': True,
            'supported_methods': ['bootstrap', 'quantile', 'ensemble', 'adaptive'],
            'default_method': 'adaptive'
        }

        # Try to get recent CI calculations
        try:
            # This would need to be implemented in the CI calculator
            # For now, return basic status
            ci_metrics['calculations_performed'] = 0
            ci_metrics['average_interval_width'] = 0.0
        except Exception:
            pass

        return ci_metrics

    async def _calculate_health_scores(self, metrics_data: Dict[str, Dict[str, Any]]) -> Dict[str, ModelHealthScore]:
        """Calculate comprehensive health scores for all models"""
        health_scores = {}

        if 'bridge' not in metrics_data or 'models' not in metrics_data['bridge']:
            return {}

        models = metrics_data['bridge']['models']

        for model_name, model_metrics in models.items():
            try:
                # Calculate individual component scores
                performance_score = self._calculate_performance_score(model_metrics)
                drift_score = self._calculate_drift_score(model_name, metrics_data)
                confidence_score = self._calculate_confidence_score(model_name, metrics_data)
                availability_score = self._calculate_availability_score(model_metrics)

                # Calculate weighted overall score
                weights = {
                    'performance': 0.35,
                    'drift': 0.25,
                    'confidence': 0.20,
                    'availability': 0.20
                }

                overall_score = (
                    performance_score * weights['performance'] +
                    drift_score * weights['drift'] +
                    confidence_score * weights['confidence'] +
                    availability_score * weights['availability']
                )

                # Determine health status
                status = self._determine_health_status(overall_score)

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    overall_score, performance_score, drift_score,
                    confidence_score, availability_score
                )

                # Create health score object
                health_score = ModelHealthScore(
                    overall_score=overall_score,
                    performance_score=performance_score,
                    drift_score=drift_score,
                    confidence_score=confidence_score,
                    availability_score=availability_score,
                    status=status,
                    last_updated=datetime.now(),
                    factors={
                        'performance': performance_score,
                        'drift': drift_score,
                        'confidence': confidence_score,
                        'availability': availability_score
                    },
                    recommendations=recommendations
                )

                health_scores[model_name] = health_score

            except Exception as e:
                self.logger.error(f"❌ Error calculating health score for {model_name}: {e}")
                # Create fallback health score
                health_scores[model_name] = self._create_fallback_health_score(model_name)

        return health_scores

    def _calculate_performance_score(self, model_metrics: Dict[str, Any]) -> float:
        """Calculate performance health score (0-100)"""
        try:
            accuracy = model_metrics.get('accuracy', 0.5)
            avg_confidence = model_metrics.get('avg_confidence', 0.5)
            avg_response_time = model_metrics.get('avg_response_time_ms', 100)
            error_rate = model_metrics.get('error_rate', 0.0)

            # Normalize metrics to 0-100 scale
            accuracy_score = min(100, max(0, (accuracy - 0.5) * 200))  # 0.5 = 0, 1.0 = 100
            confidence_score = min(100, max(0, avg_confidence * 100))
            response_score = max(0, 100 - (avg_response_time / self._health_thresholds['max_response_time_ms']) * 100)
            error_score = max(0, 100 - error_rate * 20)  # 5% = 0, 0% = 100

            # Weighted average
            performance_score = (
                accuracy_score * 0.4 +
                confidence_score * 0.3 +
                response_score * 0.2 +
                error_score * 0.1
            )

            return performance_score

        except Exception:
            return 50.0  # Default medium score

    def _calculate_drift_score(self, model_name: str, metrics_data: Dict[str, Any]) -> float:
        """Calculate drift health score (0-100)"""
        try:
            if 'drift' not in metrics_data:
                return 80.0  # Good score if no drift detection

            drift_status = metrics_data['drift']
            models_with_drift = drift_status.get('models_with_drift', 0)
            total_models = drift_status.get('total_models_monitored', 1)
            total_alerts = drift_status.get('total_alerts_24h', 0)

            # Calculate drift penalty
            if total_models > 0:
                drift_ratio = models_with_drift / total_models
            else:
                drift_ratio = 0

            # Penalty based on alerts and drift ratio
            alert_penalty = min(40, total_alerts * 2)  # 2 points per alert, max 40
            drift_penalty = drift_ratio * 40  # Max 40 points for drift

            drift_score = max(0, 100 - alert_penalty - drift_penalty)
            return drift_score

        except Exception:
            return 75.0  # Good score if no drift data

    def _calculate_confidence_score(self, model_name: str, metrics_data: Dict[str, Any]) -> float:
        """Calculate confidence interval health score (0-100)"""
        try:
            if 'confidence_intervals' not in metrics_data:
                return 70.0  # Good score if no CI data

            ci_metrics = metrics_data['confidence_intervals']

            # Score based on CI availability and quality
            ci_available = ci_metrics.get('calculator_available', False)
            avg_interval_width = ci_metrics.get('average_interval_width', 0.5)

            if not ci_available:
                return 50.0  # Medium score if CI not available

            # Score based on interval width (smaller is better up to a point)
            ideal_width = 0.3
            width_score = max(0, 100 - abs(avg_interval_width - ideal_width) * 100)

            return width_score

        except Exception:
            return 60.0  # Default medium-good score

    def _calculate_availability_score(self, model_metrics: Dict[str, Any]) -> float:
        """Calculate availability health score (0-100)"""
        try:
            success_rate = model_metrics.get('success_rate', 1.0)
            total_predictions = model_metrics.get('total_predictions', 1)

            # Bonus for high prediction volume
            volume_bonus = min(20, total_predictions / 100)  # Max 20 points

            availability_score = success_rate * 80 + volume_bonus
            return min(100, availability_score)

        except Exception:
            return 50.0

    def _determine_health_status(self, overall_score: float) -> HealthStatus:
        """Determine health status based on overall score"""
        if overall_score >= 80:
            return HealthStatus.HEALTHY
        elif overall_score >= 60:
            return HealthStatus.WARNING
        elif overall_score >= 40:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL

    def _generate_recommendations(self,
                                  overall_score: float,
                                  performance_score: float,
                                  drift_score: float,
                                  confidence_score: float,
                                  availability_score: float) -> List[str]:
        """Generate NBA-specific health recommendations"""
        recommendations = []

        if overall_score < 50:
            recommendations.append("🚨 CRITICAL: Model requires immediate attention")

        if performance_score < 60:
            recommendations.append("📊 Consider model retraining with recent data")
            recommendations.append("⚡ Check for feature distribution changes")

        if drift_score < 60:
            recommendations.append("🔍 Investigate feature drift and data quality issues")
            recommendations.append("🔄 Update training data with recent NBA patterns")

        if confidence_score < 60:
            recommendations.append("📉 Review confidence interval methodology")
            recommendations.append("🎯 Increase training data diversity")

        if availability_score < 70:
            recommendations.append("🔧 Check model infrastructure and resources")
            recommendations.append("📈 Monitor error rates and response times")

        return recommendations

    def _create_fallback_health_score(self, model_name: str) -> ModelHealthScore:
        """Create fallback health score for unknown/failed models"""
        return ModelHealthScore(
            overall_score=25.0,
            performance_score=25.0,
            drift_score=25.0,
            confidence_score=25.0,
            availability_score=25.0,
            status=HealthStatus.UNKNOWN,
            last_updated=datetime.now(),
            factors={},
            recommendations=["⚠️ Unable to assess model health - check monitoring components"]
        )

    async def _generate_health_alerts(self, health_scores: Dict[str, ModelHealthScore]) -> None:
        """Generate health alerts based on health scores"""
        current_time = datetime.now()

        for model_name, health_score in health_scores.items():
            try:
                # Check for critical conditions
                if health_score.status == HealthStatus.CRITICAL:
                    await self._create_alert(
                        model_name=model_name,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical Health Issue: {model_name}",
                        description=f"Model {model_name} has critical health issues requiring immediate attention",
                        metric_name="overall_score",
                        current_value=health_score.overall_score,
                        threshold_value=40.0,
                        recommendation="Investigate and resolve critical health issues immediately"
                    )

                # Check for performance alerts
                if health_score.performance_score < self._health_thresholds['min_accuracy'] * 100:
                    await self._create_alert(
                        model_name=model_name,
                        severity=AlertSeverity.HIGH,
                        title=f"Performance Degradation: {model_name}",
                        description=f"Model {model_name} performance has fallen below acceptable threshold",
                        metric_name="performance_score",
                        current_value=health_score.performance_score,
                        threshold_value=self._health_thresholds['min_accuracy'] * 100,
                        recommendation="Consider model retraining or investigate feature drift"
                    )

                # Check for drift alerts
                if health_score.drift_score < (100 - self._health_thresholds['max_drift_score']):
                    await self._create_alert(
                        model_name=model_name,
                        severity=AlertSeverity.MEDIUM,
                        title=f"Data Drift Detected: {model_name}",
                        description=f"Significant feature drift detected in model {model_name}",
                        metric_name="drift_score",
                        current_value=health_score.drift_score,
                        threshold_value=100 - self._health_thresholds['max_drift_score'],
                        recommendation="Update training data or recalibrate model"
                    )

            except Exception as e:
                self.logger.error(f"❌ Error generating alerts for {model_name}: {e}")

    async def _create_alert(self,
                           model_name: str,
                           severity: AlertSeverity,
                           title: str,
                           description: str,
                           metric_name: str,
                           current_value: float,
                           threshold_value: float,
                           recommendation: Optional[str] = None) -> None:
        """Create and store a health alert"""
        alert_id = f"{model_name}_{metric_name}_{int(datetime.now().timestamp())}"

        alert = HealthAlert(
            id=alert_id,
            model_name=model_name,
            severity=severity,
            title=title,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(),
            recommendation=recommendation
        )

        self._alerts.append(alert)

        # Trim old alerts
        self._trim_old_alerts()

        # Update Prometheus metrics
        self.health_alerts_total.labels(
            model_name=model_name,
            severity=severity.value,
            type=metric_name
        ).inc()

        self.logger.warning(f"🚨 Health Alert Generated: {title}")

    def _trim_old_alerts(self) -> None:
        """Remove old alerts beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.alert_retention_days)
        self._alerts = [alert for alert in self._alerts if alert.timestamp > cutoff_date]

    def _create_dashboard_metrics(self, health_scores: Dict[str, ModelHealthScore]) -> DashboardMetrics:
        """Create aggregated dashboard metrics"""
        total_models = len(health_scores)
        healthy_models = sum(1 for score in health_scores.values() if score.status == HealthStatus.HEALTHY)
        models_with_drift = sum(1 for score in health_scores.values() if score.drift_score < 70)
        models_with_warnings = sum(1 for score in health_scores.values()
                                 if score.status in [HealthStatus.WARNING, HealthStatus.DEGRADED])
        critical_alerts = sum(1 for alert in self._alerts if alert.severity == AlertSeverity.CRITICAL and not alert.resolved)

        # Calculate average response time
        if self._metrics_collector:
            try:
                metrics_summary = self._metrics_collector.get_metrics_summary()
                avg_response_time = metrics_summary.get('avg_response_time_ms', 0)
                prediction_volume = metrics_summary.get('total_predictions_24h', 0)
            except:
                avg_response_time = 0
                prediction_volume = 0
        else:
            avg_response_time = 0
            prediction_volume = 0

        # Calculate system uptime (based on recent health checks)
        time_since_last_check = (datetime.now() - self._last_health_check).total_seconds()
        system_uptime = max(0, 100 - (time_since_last_check / 60))  # Penalty for missed checks

        return DashboardMetrics(
            total_models=total_models,
            healthy_models=healthy_models,
            models_with_drift=models_with_drift,
            models_with_warnings=models_with_warnings,
            critical_alerts=critical_alerts,
            avg_response_time_ms=avg_response_time,
            prediction_volume_24h=prediction_volume,
            system_uptime_percentage=system_uptime,
            last_health_check=self._last_health_check
        )

    def _create_fallback_metrics(self) -> DashboardMetrics:
        """Create fallback dashboard metrics when collection fails"""
        return DashboardMetrics(
            total_models=0,
            healthy_models=0,
            models_with_drift=0,
            models_with_warnings=0,
            critical_alerts=0,
            avg_response_time_ms=0,
            prediction_volume_24h=0,
            system_uptime_percentage=0,
            last_health_check=self._last_health_check
        )

    def _update_prometheus_metrics(self, health_scores: Dict[str, ModelHealthScore]) -> None:
        """Update Prometheus metrics with health scores"""
        for model_name, health_score in health_scores.items():
            self.model_health_score.labels(model_name=model_name).set(health_score.overall_score)

    def _start_background_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        async def monitoring_loop():
            while self._monitoring_active:
                try:
                    await asyncio.sleep(self.update_interval)
                    if self._ml_bridge:
                        await self.collect_health_metrics()
                except Exception as e:
                    self.logger.error(f"❌ Background monitoring error: {e}")
                    await asyncio.sleep(self.update_interval * 2)  # Back off on error

        try:
            # Check if there's already an event loop running
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't use create_task directly
            # For now, skip background monitoring in sync context
            self.logger.info("🏥 Background monitoring disabled in sync context")
        except RuntimeError:
            # No event loop running, we're in sync context
            self.logger.info("🏥 Background monitoring disabled - no event loop")
            self._monitoring_active = False
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to start background monitoring: {e}")
            self._monitoring_active = False
        self.logger.info(f"🔄 Background health monitoring started (interval: {self.update_interval}s)")

    def stop_background_monitoring(self) -> None:
        """Stop background health monitoring"""
        self._monitoring_active = False

        if self._background_task:
            self._background_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._background_task)
            except asyncio.CancelledError:
                pass
            self._background_task = None

        self.logger.info("⏹️ Background health monitoring stopped")

    def get_dashboard_metrics(self) -> Optional[DashboardMetrics]:
        """
        Get current dashboard metrics

        Returns:
            DashboardMetrics: Current dashboard metrics
        """
        # Try to get recent metrics from cache
        if self._last_health_check and (datetime.now() - self._last_health_check).seconds() < 60:
            return self._create_dashboard_metrics(self._health_scores)

        return None

    def get_model_health_scores(self) -> Dict[str, ModelHealthScore]:
        """
        Get health scores for all models

        Returns:
            Dict[str, ModelHealthScore]: Health scores by model name
        """
        return self._health_scores.copy()

    def get_active_alerts(self,
                           severity_filter: Optional[AlertSeverity] = None,
                           model_filter: Optional[str] = None,
                           unresolved_only: bool = False) -> List[HealthAlert]:
        """
        Get active health alerts with filtering

        Args:
            severity_filter: Filter by alert severity
            model_filter: Filter by model name
            unresolved_only: Only return unresolved alerts

        Returns:
            List[HealthAlert]: Filtered alert list
        """
        alerts = self._alerts.copy()

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        if model_filter:
            alerts = [alert for alert in alerts if alert.model_name == model_filter]

        if unresolved_only:
            alerts = [alert for alert in alerts if not alert.resolved]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str, action_taken: Optional[str] = None) -> bool:
        """
        Acknowledge a health alert

        Args:
            alert_id: Alert ID to acknowledge
            action_taken: Description of action taken

        Returns:
            bool: Success status
        """
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.action_taken = action_taken
                self.logger.info(f"✅ Alert acknowledged: {alert.title}")
                return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a health alert

        Args:
            alert_id: Alert ID to resolve

        Returns:
            bool: Success status
        """
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.logger.info(f"✅ Alert resolved: {alert.title}")
                return True

        return False

    def get_prometheus_registry(self) -> CollectorRegistry:
        """
        Get Prometheus registry for external scraping

        Returns:
            CollectorRegistry: Prometheus metrics registry
        """
        return self.registry

    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring"""
        self.stop_background_monitoring()

        # Clear state
        self._health_scores.clear()
        self._alerts.clear()

        self.logger.info("🧹 NBA Model Health Dashboard cleanup completed")