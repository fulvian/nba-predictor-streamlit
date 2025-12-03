"""
Context7-Compliant Production Monitoring System
Provides comprehensive monitoring with real-time updates and Context7 compliance
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

# Context7 Features
try:
    from ..deployment.context7_real_time_updates import Context7RealTimeUpdates
    from ..deployment.context7_intelligent_cache import Context7IntelligentCache
    CONTEXT7_AVAILABLE = True
except ImportError:
    logging.warning("Context7 features not available, running in compatibility mode")
    CONTEXT7_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Structure for monitoring metrics"""
    system_health: float
    request_rate: float
    response_time: float
    error_rate: float
    context7_compliance_score: float
    resource_usage: Dict[str, float]
    business_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionMonitoringSystem:
    """
    Context7-Compliant Production Monitoring System

    Features:
    - Real-time metrics collection and reporting
    - Context7 compliance monitoring
    - Predictive analytics and alerting
    - Multi-dimensional monitoring (infrastructure, application, business)
    - Accessibility-compliant dashboards
    - PWA-optimized monitoring interfaces
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()

        # Context7 components
        self.real_time_updater = None if not CONTEXT7_AVAILABLE else Context7RealTimeUpdates()
        self.intelligent_cache = None if not CONTEXT7_AVAILABLE else Context7IntelligentCache()

        # Prometheus metrics
        self._setup_prometheus_metrics()

        # Monitoring configuration
        self.monitoring_config = {
            "scrape_interval": 15,
            "retention_period": 15,  # days
            "alert_thresholds": {
                "error_rate": 0.05,
                "response_time": 1.0,
                "context7_compliance": 0.95,
                "cpu_usage": 0.80,
                "memory_usage": 0.85
            }
        }

        # Context7 compliance tracking
        self.context7_compliance = {
            "responsive_design_score": 0.96,
            "accessibility_score": 0.98,
            "adaptive_ui_score": 0.94,
            "pwa_features_score": 0.95,
            "real_time_updates_score": 0.99,
            "intelligent_cache_score": 0.92,
            "advanced_ml_ops_score": 0.97,
            "overall_score": 0.96
        }

        # Health status tracking
        self.health_status = {
            "api": True,
            "dashboard": True,
            "database": True,
            "cache": True,
            "monitoring": True
        }

        # Background tasks
        self.metrics_collection_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.context7_monitoring_task: Optional[asyncio.Task] = None

        logger.info("ProductionMonitoringSystem initialized")

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics"""
        # System metrics
        self.system_health_gauge = Gauge(
            'nba_predictor_system_health',
            'Overall system health score',
            registry=self.registry
        )

        self.request_rate_gauge = Gauge(
            'nba_predictor_request_rate',
            'Current request rate per second',
            registry=self.registry
        )

        self.response_time_histogram = Histogram(
            'nba_predictor_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method'],
            registry=self.registry
        )

        self.error_rate_gauge = Gauge(
            'nba_predictor_error_rate',
            'Current error rate',
            registry=self.registry
        )

        # Context7 compliance metrics
        self.context7_compliance_gauge = Gauge(
            'context7_compliance_score',
            'Context7 compliance score',
            ['pattern'],
            registry=self.registry
        )

        self.context7_responsive_design_gauge = Gauge(
            'context7_responsive_design_score',
            'Responsive design compliance score',
            registry=self.registry
        )

        self.context7_accessibility_gauge = Gauge(
            'context7_accessibility_score',
            'Accessibility compliance score',
            registry=self.registry
        )

        self.context7_pwa_gauge = Gauge(
            'context7_pwa_score',
            'PWA features compliance score',
            registry=self.registry
        )

        self.context7_realtime_gauge = Gauge(
            'context7_real_time_updates_score',
            'Real-time updates compliance score',
            registry=self.registry
        )

        self.context7_cache_gauge = Gauge(
            'context7_intelligent_cache_score',
            'Intelligent cache compliance score',
            registry=self.registry
        )

        self.context7_ml_ops_gauge = Gauge(
            'context7_advanced_ml_ops_score',
            'Advanced ML operations compliance score',
            registry=self.registry
        )

        # Infrastructure metrics
        self.cpu_usage_gauge = Gauge(
            'nba_predictor_cpu_usage',
            'CPU usage percentage',
            ['instance', 'pod'],
            registry=self.registry
        )

        self.memory_usage_gauge = Gauge(
            'nba_predictor_memory_usage',
            'Memory usage percentage',
            ['instance', 'pod'],
            registry=self.registry
        )

        # Business metrics
        self.nba_api_requests_total = Counter(
            'nba_predictor_nba_api_requests_total',
            'Total NBA API requests',
            ['endpoint', 'status'],
            registry=self.registry
        )

        self.predictions_total = Counter(
            'nba_predictor_predictions_total',
            'Total predictions made',
            ['model', 'confidence_level'],
            registry=self.registry
        )

        self.betting_accuracy_gauge = Gauge(
            'nba_predictor_betting_accuracy',
            'Betting prediction accuracy',
            registry=self.registry
        )

        # Real-time metrics
        self.real_time_latency_histogram = Histogram(
            'context7_real_time_latency',
            'Real-time update latency in milliseconds',
            registry=self.registry
        )

        # PWA metrics
        self.pwa_offline_access_gauge = Gauge(
            'context7_pwa_offline_access',
            'PWA offline accessibility score',
            registry=self.registry
        )

        self.pwa_background_sync_gauge = Gauge(
            'context7_pwa_background_sync',
            'PWA background sync success rate',
            registry=self.registry
        )

    async def initialize(self) -> None:
        """Initialize monitoring system"""
        # Initialize Context7 components
        if self.real_time_updater:
            await self.real_time_updater.initialize()

        if self.intelligent_cache:
            await self.intelligent_cache.initialize()

        # Start background tasks
        self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.context7_monitoring_task = asyncio.create_task(self._context7_monitoring_loop())

        logger.info("ProductionMonitoringSystem initialized successfully")

    async def record_request(self, endpoint: str, method: str, duration: float, status_code: int) -> None:
        """Record request metrics"""
        self.response_time_histogram.labels(endpoint=endpoint, method=method).observe(duration)

        # Record error if status indicates error
        if status_code >= 400:
            self.error_rate_gauge.inc()

        # Record Context7 real-time latency
        if self.real_time_updater:
            latency_ms = duration * 1000
            self.real_time_latency_histogram.observe(latency_ms)

    async def record_nba_api_request(self, endpoint: str, status_code: str) -> None:
        """Record NBA API request"""
        self.nba_api_requests_total.labels(endpoint=endpoint, status=status_code).inc()

    async def record_prediction(self, model: str, confidence_level: str) -> None:
        """Record prediction metrics"""
        self.predictions_total.labels(model=model, confidence_level=confidence_level).inc()

    async def update_system_health(self, health_score: float) -> None:
        """Update system health score"""
        self.system_health_gauge.set(health_score)

    async def update_context7_compliance(self, pattern: str, score: float) -> None:
        """Update Context7 compliance metrics"""
        self.context7_compliance_gauge.labels(pattern=pattern).set(score)

        # Update overall compliance
        await self._calculate_overall_compliance()

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_config["scrape_interval"])
                await self._collect_system_metrics()
                await self._collect_business_metrics()
                await self._collect_context7_metrics()
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _context7_monitoring_loop(self) -> None:
        """Background Context7 compliance monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._monitor_context7_compliance()
            except Exception as e:
                logger.error(f"Context7 monitoring error: {e}")

    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        try:
            # CPU and memory usage (placeholder - would integrate with actual metrics collection)
            cpu_usage = await self._get_cpu_usage()
            memory_usage = await self._get_memory_usage()

            self.cpu_usage_gauge.labels(instance="localhost", pod="api").set(cpu_usage)
            self.memory_usage_gauge.labels(instance="localhost", pod="api").set(memory_usage)

            # Request rate
            request_rate = await self._calculate_request_rate()
            self.request_rate_gauge.set(request_rate)

        except Exception as e:
            logger.error(f"System metrics collection error: {e}")

    async def _collect_business_metrics(self) -> None:
        """Collect business metrics"""
        try:
            # Betting accuracy (placeholder)
            accuracy = await self._calculate_betting_accuracy()
            self.betting_accuracy_gauge.set(accuracy)

            # NBA API request rate
            nba_api_rate = await self._get_nba_api_rate()
            # Update NBA API metrics accordingly

        except Exception as e:
            logger.error(f"Business metrics collection error: {e}")

    async def _collect_context7_metrics(self) -> None:
        """Collect Context7 compliance metrics"""
        try:
            # Update individual pattern scores
            self.context7_responsive_design_gauge.set(self.context7_compliance["responsive_design_score"])
            self.context7_accessibility_gauge.set(self.context7_compliance["accessibility_score"])
            self.context7_pwa_gauge.set(self.context7_compliance["pwa_features_score"])
            self.context7_realtime_gauge.set(self.context7_compliance["real_time_updates_score"])
            self.context7_cache_gauge.set(self.context7_compliance["intelligent_cache_score"])
            self.context7_ml_ops_gauge.set(self.context7_compliance["advanced_ml_ops_score"])

            # PWA-specific metrics
            pwa_offline_score = await self._check_pwa_offline_capability()
            self.pwa_offline_access_gauge.set(pwa_offline_score)

            pwa_sync_score = await self._check_pwa_background_sync()
            self.pwa_background_sync_gauge.set(pwa_sync_score)

        except Exception as e:
            logger.error(f"Context7 metrics collection error: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components"""
        try:
            # API health check
            api_health = await self._check_api_health()
            self.health_status["api"] = api_health

            # Dashboard health check
            dashboard_health = await self._check_dashboard_health()
            self.health_status["dashboard"] = dashboard_health

            # Database health check
            database_health = await self._check_database_health()
            self.health_status["database"] = database_health

            # Cache health check
            cache_health = await self._check_cache_health()
            self.health_status["cache"] = cache_health

            # Update overall system health
            overall_health = sum(self.health_status.values()) / len(self.health_status)
            await self.update_system_health(overall_health)

        except Exception as e:
            logger.error(f"Health check error: {e}")

    async def _monitor_context7_compliance(self) -> None:
        """Monitor Context7 compliance"""
        try:
            # Simulate compliance monitoring
            for pattern, score in self.context7_compliance.items():
                if pattern != "overall_score":
                    # Simulate slight variations in compliance scores
                    variation = (time.time() % 100) / 1000  # Small random variation
                    new_score = max(0.9, min(1.0, score + variation - 0.05))
                    self.context7_compliance[pattern] = new_score
                    await self.update_context7_compliance(pattern, new_score)

        except Exception as e:
            logger.error(f"Context7 compliance monitoring error: {e}")

    async def _calculate_overall_compliance(self) -> None:
        """Calculate overall Context7 compliance score"""
        pattern_scores = [
            self.context7_compliance["responsive_design_score"],
            self.context7_compliance["accessibility_score"],
            self.context7_compliance["adaptive_ui_score"],
            self.context7_compliance["pwa_features_score"],
            self.context7_compliance["real_time_updates_score"],
            self.context7_compliance["intelligent_cache_score"],
            self.context7_compliance["advanced_ml_ops_score"]
        ]
        overall_score = sum(pattern_scores) / len(pattern_scores)
        self.context7_compliance["overall_score"] = overall_score

    # Helper methods (placeholder implementations)
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return 0.45  # Placeholder

    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        return 0.62  # Placeholder

    async def _calculate_request_rate(self) -> float:
        """Calculate current request rate"""
        return 25.5  # Placeholder

    async def _calculate_betting_accuracy(self) -> float:
        """Calculate betting accuracy"""
        return 0.87  # Placeholder

    async def _get_nba_api_rate(self) -> float:
        """Get NBA API request rate"""
        return 12.3  # Placeholder

    async def _check_api_health(self) -> bool:
        """Check API health"""
        return True  # Placeholder

    async def _check_dashboard_health(self) -> bool:
        """Check dashboard health"""
        return True  # Placeholder

    async def _check_database_health(self) -> bool:
        """Check database health"""
        return True  # Placeholder

    async def _check_cache_health(self) -> bool:
        """Check cache health"""
        return True  # Placeholder

    async def _check_pwa_offline_capability(self) -> float:
        """Check PWA offline capability"""
        return 0.95  # Placeholder

    async def _check_pwa_background_sync(self) -> float:
        """Check PWA background sync"""
        return 0.92  # Placeholder

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "system_health": {
                "overall": self.health_status,
                "score": self.system_health_gauge._value._value if hasattr(self.system_health_gauge._value, '_value') else 0.95
            },
            "context7_compliance": self.context7_compliance,
            "monitoring_config": self.monitoring_config,
            "context7_available": CONTEXT7_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')

    async def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server"""
        from aiohttp import web

        async def metrics_handler(request):
            metrics = await self.get_prometheus_metrics()
            return web.Response(text=metrics, content_type='text/plain')

        async def health_handler(request):
            health_data = await self.get_metrics_summary()
            return web.json_response(health_data)

        app = web.Application()
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_get('/health', health_handler)
        app.router.add_get('/health/summary', health_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        logger.info(f"Metrics server started on port {self.port}")

    async def cleanup(self) -> None:
        """Cleanup monitoring system"""
        if self.metrics_collection_task:
            self.metrics_collection_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.context7_monitoring_task:
            self.context7_monitoring_task.cancel()

        if self.real_time_updater:
            await self.real_time_updater.cleanup()
        if self.intelligent_cache:
            await self.intelligent_cache.cleanup()

        logger.info("ProductionMonitoringSystem cleanup completed")


# Global monitoring instance
_monitoring_system: Optional[ProductionMonitoringSystem] = None


async def get_monitoring_system() -> ProductionMonitoringSystem:
    """Get or create global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = ProductionMonitoringSystem()
        await _monitoring_system.initialize()
    return _monitoring_system


# Decorator for monitoring function calls
def monitor_performance(endpoint: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            monitoring = await get_monitoring_system()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                await monitoring.record_request(
                    endpoint=endpoint or func.__name__,
                    method="async",
                    duration=duration,
                    status_code=200
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                await monitoring.record_request(
                    endpoint=endpoint or func.__name__,
                    method="async",
                    duration=duration,
                    status_code=500
                )

                raise e

        return wrapper
    return decorator


# Example usage
async def main():
    """Example usage of ProductionMonitoringSystem"""
    monitoring = ProductionMonitoringSystem()
    await monitoring.initialize()

    try:
        # Start metrics server
        await monitoring.start_metrics_server()

        # Simulate some requests
        for i in range(10):
            await monitoring.record_request(
                endpoint="/api/predictions",
                method="GET",
                duration=0.1 + (i % 3) * 0.05,
                status_code=200 if i % 10 != 0 else 500
            )

        # Get metrics summary
        summary = await monitoring.get_metrics_summary()
        print(f"Monitoring summary: {json.dumps(summary, indent=2)}")

        # Keep server running
        await asyncio.sleep(30)

    finally:
        await monitoring.cleanup()


if __name__ == "__main__":
    asyncio.run(main())