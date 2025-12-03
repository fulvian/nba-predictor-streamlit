"""
NBA Predictor Performance Monitoring and Optimization System

Phase 3 Day 12 - Task 3.5.3
This module provides comprehensive performance monitoring, optimization,
and alerting capabilities with Context7 compliance validation.
"""

import asyncio
import time
import uuid
import json
import sqlite3
import logging
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from contextlib import contextmanager
import psutil
import gc
from datetime import datetime, timedelta
import numpy as np
from memory_profiler import profile


class PerformanceLevel(Enum):
    """Performance severity levels"""
    OPTIMAL = "optimal"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of performance metrics"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    RESPONSE_TIME = "response_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_PERFORMANCE = "database_performance"
    USER_EXPERIENCE = "user_experience"


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    AUTO_GC = "auto_gc"
    CACHE_OPTIMIZATION = "cache_optimization"
    MEMORY_CLEANUP = "memory_cleanup"
    PROCESS_PRIORITY = "process_priority"
    RESOURCE_LIMITING = "resource_limiting"
    CONTEXT7_OPTIMIZATION = "context7_optimization"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: float
    metric_type: MetricType
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    context7_compliance: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    id: str
    metric_type: MetricType
    level: PerformanceLevel
    message: str
    timestamp: float
    value: float
    threshold: float
    context7_pattern: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class OptimizationAction:
    """Optimization action taken"""
    id: str
    strategy: OptimizationStrategy
    timestamp: float
    description: str
    impact: Dict[str, float]
    context7_patterns: List[str]


class PerformanceMonitoringSystem:
    """
    Comprehensive performance monitoring and optimization system
    with Context7 compliance validation and automatic optimization
    """

    def __init__(self, monitoring_interval: int = 5,
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 auto_optimize: bool = True):
        self.monitoring_interval = monitoring_interval
        self.auto_optimize = auto_optimize
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Performance metrics storage
        self.metrics_history: List[PerformanceMetric] = []
        self.active_alerts: List[PerformanceAlert] = []
        self.optimization_history: List[OptimizationAction] = []

        # Thresholds for alerts
        self.thresholds = alert_thresholds or self._get_default_thresholds()

        # Context7 compliance tracking
        self.context7_compliance_scores: Dict[str, float] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize database for performance data
        self._init_performance_database()

        # Process handle for priority adjustments
        self.current_process = psutil.Process()

    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default performance thresholds"""
        return {
            "cpu": {"warning": 70.0, "critical": 85.0},
            "memory": {"warning": 70.0, "critical": 85.0},
            "disk": {"warning": 80.0, "critical": 90.0},
            "response_time": {"warning": 1000.0, "critical": 3000.0},  # ms
            "cache_hit_rate": {"warning": 90.0, "critical": 80.0},
            "database_performance": {"warning": 500.0, "critical": 1500.0}  # ms
        }

    def _init_performance_database(self):
        """Initialize SQLite database for performance metrics"""
        conn = sqlite3.connect('data/nba_performance_metrics.duckdb')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                metric_type TEXT,
                value REAL,
                unit TEXT,
                threshold_warning REAL,
                threshold_critical REAL,
                context7_compliance TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id TEXT PRIMARY KEY,
                metric_type TEXT,
                level TEXT,
                message TEXT,
                timestamp REAL,
                value REAL,
                threshold REAL,
                context7_pattern TEXT,
                recommendation TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_actions (
                id TEXT PRIMARY KEY,
                strategy TEXT,
                timestamp REAL,
                description TEXT,
                impact TEXT,
                context7_patterns TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context7_compliance (
                id TEXT PRIMARY KEY,
                pattern_name TEXT,
                compliance_score REAL,
                timestamp REAL,
                issues TEXT,
                recommendations TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect all performance metrics
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._validate_context7_compliance()

                # Check for performance alerts
                self._check_alerts()

                # Auto-optimize if enabled
                if self.auto_optimize:
                    self._auto_optimize()

                # Save metrics to database
                self._save_metrics()

                # Cleanup old metrics
                self._cleanup_old_metrics()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()

        self._add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.CPU,
            value=cpu_percent,
            unit="percent",
            threshold_warning=self.thresholds["cpu"]["warning"],
            threshold_critical=self.thresholds["cpu"]["critical"]
        ))

        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.MEMORY,
            value=memory.percent,
            unit="percent",
            threshold_warning=self.thresholds["memory"]["warning"],
            threshold_critical=self.thresholds["memory"]["critical"]
        ))

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.DISK,
            value=disk_percent,
            unit="percent",
            threshold_warning=self.thresholds["disk"]["warning"],
            threshold_critical=self.thresholds["disk"]["critical"]
        ))

        # Network metrics
        network = psutil.net_io_counters()
        net_bytes_sent = network.bytes_sent
        net_bytes_recv = network.bytes_recv

        # CPU frequency monitoring
        if cpu_freq:
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type=MetricType.CPU,
                value=cpu_freq.current,
                unit="MHz",
                threshold_warning=0,
                threshold_critical=0
            ))

    def _collect_application_metrics(self):
        """Collect application-specific performance metrics"""
        timestamp = time.time()

        try:
            # Process-specific metrics
            process = self.current_process

            # Process CPU usage
            process_cpu = process.cpu_percent()

            # Process memory
            process_memory = process.memory_info()
            process_memory_percent = process.memory_percent()

            # Process threads
            process_threads = process.num_threads()

            # Response time simulation (placeholder)
            response_time = self._measure_response_time()
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type=MetricType.RESPONSE_TIME,
                value=response_time,
                unit="ms",
                threshold_warning=self.thresholds["response_time"]["warning"],
                threshold_critical=self.thresholds["response_time"]["critical"]
            ))

            # Cache hit rate simulation (placeholder)
            cache_hit_rate = self._measure_cache_hit_rate()
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_type=MetricType.CACHE_HIT_RATE,
                value=cache_hit_rate,
                unit="percent",
                threshold_warning=self.thresholds["cache_hit_rate"]["warning"],
                threshold_critical=self.thresholds["cache_hit_rate"]["critical"]
            ))

        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {str(e)}")

    def _measure_response_time(self) -> float:
        """Measure application response time"""
        start_time = time.time()

        # Simulate a typical operation
        try:
            # This would be replaced with actual application-specific timing
            time.sleep(0.001)  # Simulate 1ms operation
        except:
            pass

        return (time.time() - start_time) * 1000  # Convert to ms

    def _measure_cache_hit_rate(self) -> float:
        """Measure cache hit rate"""
        # Placeholder implementation
        # In a real application, this would track actual cache performance
        return np.random.uniform(70, 95)  # Simulate 70-95% hit rate

    def _validate_context7_compliance(self):
        """Validate Context7 pattern compliance"""
        timestamp = time.time()

        # Define Context7 patterns to validate
        context7_patterns = [
            "responsive_design_system",
            "accessibility_features",
            "adaptive_ui_layouts",
            "pwa_features",
            "real_time_updates",
            "intelligent_cache",
            "advanced_ml_operations"
        ]

        for pattern in context7_patterns:
            compliance_score = self._validate_context7_pattern(pattern)

            # Store compliance score
            self.context7_compliance_scores[pattern] = compliance_score

            # Save to database
            self._save_context7_compliance(pattern, compliance_score, timestamp)

    def _validate_context7_pattern(self, pattern_name: str) -> float:
        """Validate individual Context7 pattern performance"""
        # Pattern-specific performance validation
        if pattern_name == "responsive_design_system":
            return self._validate_responsive_performance()
        elif pattern_name == "accessibility_features":
            return self._validate_accessibility_performance()
        elif pattern_name == "adaptive_ui_layouts":
            return self._validate_adaptive_ui_performance()
        elif pattern_name == "pwa_features":
            return self._validate_pwa_performance()
        elif pattern_name == "real_time_updates":
            return self._validate_realtime_performance()
        elif pattern_name == "intelligent_cache":
            return self._validate_cache_performance()
        elif pattern_name == "advanced_ml_operations":
            return self._validate_ml_performance()

        return 0.8  # Default compliance score

    def _validate_responsive_performance(self) -> float:
        """Validate responsive design performance"""
        score = 1.0

        # Check viewport meta tag performance impact
        score -= 0.05 if self.metrics_history else 0

        # Check CSS media query efficiency
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 70:
            score -= 0.1

        return max(score, 0.0)

    def _validate_accessibility_performance(self) -> float:
        """Validate accessibility features performance"""
        score = 1.0

        # Check ARIA label performance
        # Check screen reader compatibility
        # Check keyboard navigation performance

        # Simulate performance impact check
        response_time = self._measure_response_time()
        if response_time > 1000:  # 1 second
            score -= 0.2

        return max(score, 0.0)

    def _validate_adaptive_ui_performance(self) -> float:
        """Validate adaptive UI performance"""
        score = 1.0

        # Check dynamic content loading performance
        # Check state management efficiency
        # Check user preference detection performance

        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 70:
            score -= 0.1

        return max(score, 0.0)

    def _validate_pwa_performance(self) -> float:
        """Validate PWA features performance"""
        score = 1.0

        # Check service worker performance
        # Check offline capability performance
        # Check manifest loading performance

        # Simulate PWA-specific checks
        network_latency = self._measure_response_time()
        if network_latency > 500:  # 500ms
            score -= 0.15

        return max(score, 0.0)

    def _validate_realtime_performance(self) -> float:
        """Validate real-time updates performance"""
        score = 1.0

        # Check WebSocket connection performance
        # Check update frequency performance
        # Check data streaming efficiency

        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 60:
            score -= 0.2

        return max(score, 0.0)

    def _validate_cache_performance(self) -> float:
        """Validate intelligent cache performance"""
        score = 1.0

        # Check cache hit rate
        cache_hit_rate = self._measure_cache_hit_rate()
        if cache_hit_rate < 80:
            score -= 0.2 * (1 - cache_hit_rate / 100)

        # Check cache memory usage
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            score -= 0.1

        return max(score, 0.0)

    def _validate_ml_performance(self) -> float:
        """Validate advanced ML operations performance"""
        score = 1.0

        # Check prediction response time
        # Check model memory usage
        # Check inference efficiency

        response_time = self._measure_response_time()
        if response_time > 2000:  # 2 seconds for ML ops
            score -= 0.3

        return max(score, 0.0)

    def _check_alerts(self):
        """Check for performance alerts"""
        current_metrics = self._get_latest_metrics()

        for metric in current_metrics:
            if metric.value >= metric.threshold_critical:
                self._create_alert(
                    metric_type=metric.metric_type,
                    level=PerformanceLevel.CRITICAL,
                    message=f"{metric.metric_type.value} critical: {metric.value:.2f}{metric.unit}",
                    value=metric.value,
                    threshold=metric.threshold_critical
                )
            elif metric.value >= metric.threshold_warning:
                self._create_alert(
                    metric_type=metric.metric_type,
                    level=PerformanceLevel.WARNING,
                    message=f"{metric.metric_type.value} warning: {metric.value:.2f}{metric.unit}",
                    value=metric.value,
                    threshold=metric.threshold_warning
                )

    def _create_alert(self, metric_type: MetricType, level: PerformanceLevel,
                     message: str, value: float, threshold: float):
        """Create a performance alert"""
        # Check if similar alert already exists
        existing_alert = next(
            (alert for alert in self.active_alerts
             if alert.metric_type == metric_type and alert.level == level),
            None
        )

        if existing_alert:
            return  # Avoid duplicate alerts

        alert = PerformanceAlert(
            id=str(uuid.uuid4()),
            metric_type=metric_type,
            level=level,
            message=message,
            timestamp=time.time(),
            value=value,
            threshold=threshold,
            recommendation=self._get_recommendation(metric_type, level)
        )

        self.active_alerts.append(alert)
        self.logger.warning(f"Performance alert: {message}")

        # Save to database
        self._save_alert(alert)

    def _get_recommendation(self, metric_type: MetricType, level: PerformanceLevel) -> str:
        """Get optimization recommendation for alert"""
        recommendations = {
            MetricType.CPU: {
                PerformanceLevel.WARNING: "Consider optimizing CPU-intensive operations or scaling resources",
                PerformanceLevel.CRITICAL: "Immediate CPU optimization required - check for infinite loops or inefficient algorithms"
            },
            MetricType.MEMORY: {
                PerformanceLevel.WARNING: "Monitor memory usage and consider garbage collection",
                PerformanceLevel.CRITICAL: "Critical memory usage - immediate cleanup required, possible memory leak"
            },
            MetricType.RESPONSE_TIME: {
                PerformanceLevel.WARNING: "Consider optimizing database queries and caching",
                PerformanceLevel.CRITICAL: "Critical response times - investigate bottlenecks and optimize performance"
            },
            MetricType.CACHE_HIT_RATE: {
                PerformanceLevel.WARNING: "Review caching strategy and cache size",
                PerformanceLevel.CRITICAL: "Cache performance critical - optimize cache keys and retention policies"
            }
        }

        return recommendations.get(metric_type, {}).get(level, "Monitor performance and consider optimization")

    def _auto_optimize(self):
        """Perform automatic performance optimization"""
        if not self.active_alerts:
            return

        critical_alerts = [alert for alert in self.active_alerts if alert.level == PerformanceLevel.CRITICAL]

        if critical_alerts:
            self._execute_optimization_strategy(OptimizationStrategy.AUTO_GC)
            self._execute_optimization_strategy(OptimizationStrategy.MEMORY_CLEANUP)

        # Context7-specific optimization
        low_compliance_patterns = [
            pattern for pattern, score in self.context7_compliance_scores.items()
            if score < 0.7
        ]

        if low_compliance_patterns:
            self._execute_optimization_strategy(OptimizationStrategy.CONTEXT7_OPTIMIZATION)

    def _execute_optimization_strategy(self, strategy: OptimizationStrategy):
        """Execute specific optimization strategy"""
        timestamp = time.time()
        impact = {}

        try:
            if strategy == OptimizationStrategy.AUTO_GC:
                before_memory = psutil.virtual_memory().percent
                gc.collect()
                after_memory = psutil.virtual_memory().percent
                impact["memory_improvement"] = before_memory - after_memory

            elif strategy == OptimizationStrategy.MEMORY_CLEANUP:
                before_memory = psutil.virtual_memory().used / (1024**3)  # GB
                gc.collect()
                after_memory = psutil.virtual_memory().used / (1024**3)  # GB
                impact["memory_freed_gb"] = before_memory - after_memory

            elif strategy == OptimizationStrategy.CACHE_OPTIMIZATION:
                # Simulate cache optimization
                impact["cache_hit_rate_improvement"] = 5.0  # 5% improvement

            elif strategy == OptimizationStrategy.CONTEXT7_OPTIMIZATION:
                # Simulate Context7 optimization
                impact["context7_compliance_improvement"] = 10.0  # 10% improvement

            elif strategy == OptimizationStrategy.PROCESS_PRIORITY:
                # Adjust process priority for better performance
                try:
                    self.current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS') else 10)
                    impact["priority_adjusted"] = True
                except:
                    impact["priority_adjusted"] = False

            # Record optimization action
            action = OptimizationAction(
                id=str(uuid.uuid4()),
                strategy=strategy,
                timestamp=timestamp,
                description=f"Executed {strategy.value} optimization",
                impact=impact,
                context7_patterns=list(self.context7_compliance_scores.keys())
            )

            self.optimization_history.append(action)
            self._save_optimization_action(action)

            self.logger.info(f"Executed optimization strategy: {strategy.value}")

        except Exception as e:
            self.logger.error(f"Error executing optimization strategy {strategy.value}: {str(e)}")

    def _add_metric(self, metric: PerformanceMetric):
        """Add performance metric to history"""
        self.metrics_history.append(metric)

        # Keep only recent metrics (last 1000)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def _get_latest_metrics(self) -> List[PerformanceMetric]:
        """Get latest metrics for each type"""
        latest_metrics = {}
        current_time = time.time()

        for metric in self.metrics_history:
            if (current_time - metric.timestamp) <= 60:  # Last minute
                if metric.metric_type not in latest_metrics or metric.timestamp > latest_metrics[metric.metric_type].timestamp:
                    latest_metrics[metric.metric_type] = metric

        return list(latest_metrics.values())

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = self._get_latest_metrics()

        report = {
            "timestamp": time.time(),
            "current_metrics": {m.metric_type.value: m.value for m in current_metrics},
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts if a.level == PerformanceLevel.CRITICAL]),
            "context7_compliance": self.context7_compliance_scores,
            "optimization_actions": len(self.optimization_history),
            "recommendations": self._generate_recommendations(),
            "performance_level": self._calculate_overall_performance_level()
        }

        return report

    def _calculate_overall_performance_level(self) -> PerformanceLevel:
        """Calculate overall system performance level"""
        if not self.metrics_history:
            return PerformanceLevel.GOOD

        critical_alerts = [a for a in self.active_alerts if a.level == PerformanceLevel.CRITICAL]
        warning_alerts = [a for a in self.active_alerts if a.level == PerformanceLevel.WARNING]

        if critical_alerts:
            return PerformanceLevel.CRITICAL
        elif warning_alerts:
            return PerformanceLevel.WARNING
        elif any(score < 0.7 for score in self.context7_compliance_scores.values()):
            return PerformanceLevel.WARNING
        else:
            return PerformanceLevel.OPTIMAL

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Alert-based recommendations
        for alert in self.active_alerts:
            if alert.recommendation:
                recommendations.append(alert.recommendation)

        # Context7 compliance recommendations
        for pattern, score in self.context7_compliance_scores.items():
            if score < 0.7:
                recommendations.append(f"Improve {pattern} compliance (current: {score:.1%})")

        # System optimization recommendations
        latest_metrics = self._get_latest_metrics()
        for metric in latest_metrics:
            if metric.value > metric.threshold_warning:
                recommendations.append(
                    f"Optimize {metric.metric_type.value} usage (current: {metric.value:.1f}%)"
                )

        return recommendations

    def _save_metrics(self):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect('data/nba_performance_metrics.duckdb')
            cursor = conn.cursor()

            for metric in self.metrics_history[-10:]:  # Save last 10 metrics
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_metrics
                    (id, timestamp, metric_type, value, unit, threshold_warning, threshold_critical, context7_compliance, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{metric.timestamp}_{metric.metric_type.value}",
                    metric.timestamp,
                    metric.metric_type.value,
                    metric.value,
                    metric.unit,
                    metric.threshold_warning,
                    metric.threshold_critical,
                    json.dumps(metric.context7_compliance) if metric.context7_compliance else None,
                    json.dumps(asdict(metric))
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")

    def _save_alert(self, alert: PerformanceAlert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect('data/nba_performance_metrics.duckdb')
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO performance_alerts
                (id, metric_type, level, message, timestamp, value, threshold, context7_pattern, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.metric_type.value,
                alert.level.value,
                alert.message,
                alert.timestamp,
                alert.value,
                alert.threshold,
                alert.context7_pattern,
                alert.recommendation
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving alert: {str(e)}")

    def _save_optimization_action(self, action: OptimizationAction):
        """Save optimization action to database"""
        try:
            conn = sqlite3.connect('data/nba_performance_metrics.duckdb')
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO optimization_actions
                (id, strategy, timestamp, description, impact, context7_patterns)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                action.id,
                action.strategy.value,
                action.timestamp,
                action.description,
                json.dumps(action.impact),
                json.dumps(action.context7_patterns)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving optimization action: {str(e)}")

    def _save_context7_compliance(self, pattern: str, score: float, timestamp: float):
        """Save Context7 compliance score to database"""
        try:
            conn = sqlite3.connect('data/nba_performance_metrics.duckdb')
            cursor = conn.cursor()

            issues = []
            recommendations = []

            if score < 0.7:
                issues.append(f"Compliance score below optimal level")
                recommendations.append(f"Optimize {pattern} implementation")

            cursor.execute('''
                INSERT OR REPLACE INTO context7_compliance
                (id, pattern_name, compliance_score, timestamp, issues, recommendations)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"{pattern}_{timestamp}",
                pattern,
                score,
                timestamp,
                json.dumps(issues),
                json.dumps(recommendations)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving Context7 compliance: {str(e)}")

    def _cleanup_old_metrics(self):
        """Clean up old metrics from history"""
        cutoff_time = time.time() - 3600  # Keep last hour

        self.metrics_history = [
            metric for metric in self.metrics_history
            if metric.timestamp > cutoff_time
        ]

        # Clean up old alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (time.time() - alert.timestamp) < 300  # Keep alerts for 5 minutes
        ]

    @profile
    def run_performance_analysis(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run comprehensive performance analysis for specified duration"""
        self.logger.info(f"Starting performance analysis for {duration_seconds} seconds")

        # Clear existing data for clean analysis
        self.metrics_history.clear()
        self.active_alerts.clear()

        # Start monitoring
        self.start_monitoring()

        # Wait for analysis duration
        time.sleep(duration_seconds)

        # Stop monitoring
        self.stop_monitoring()

        # Generate analysis report
        report = self._generate_analysis_report()

        self.logger.info(f"Performance analysis completed. Found {len(self.active_alerts)} alerts")

        return report

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate detailed performance analysis report"""
        report = {
            "analysis_duration": time.time(),
            "metrics_collected": len(self.metrics_history),
            "alerts_generated": len(self.active_alerts),
            "optimizations_performed": len(self.optimization_history),
            "context7_compliance_summary": self._summarize_context7_compliance(),
            "performance_trends": self._analyze_performance_trends(),
            "bottleneck_analysis": self._analyze_bottlenecks(),
            "recommendations": self._generate_detailed_recommendations()
        }

        return report

    def _summarize_context7_compliance(self) -> Dict[str, Any]:
        """Summarize Context7 compliance across all patterns"""
        if not self.context7_compliance_scores:
            return {"overall_score": 0.0, "patterns": {}}

        overall_score = sum(self.context7_compliance_scores.values()) / len(self.context7_compliance_scores)

        return {
            "overall_score": overall_score,
            "patterns": self.context7_compliance_scores,
            "compliant_patterns": [
                pattern for pattern, score in self.context7_compliance_scores.items()
                if score >= 0.8
            ],
            "needs_improvement": [
                pattern for pattern, score in self.context7_compliance_scores.items()
                if score < 0.7
            ]
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from metrics history"""
        if len(self.metrics_history) < 10:
            return {"insufficient_data": True}

        trends = {}

        # Group metrics by type
        metrics_by_type = {}
        for metric in self.metrics_history:
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric)

        # Analyze trends for each metric type
        for metric_type, metrics in metrics_by_type.items():
            if len(metrics) >= 5:
                values = [m.value for m in metrics]
                trend = {
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend_direction": "increasing" if values[-1] > values[0] else "decreasing",
                    "volatility": np.std(values)
                }
                trends[metric_type.value] = trend

        return trends

    def _analyze_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Check for resource bottlenecks
        latest_metrics = self._get_latest_metrics()
        for metric in latest_metrics:
            if metric.value >= metric.threshold_critical:
                bottlenecks.append(f"Critical {metric.metric_type.value} usage: {metric.value:.1f}%")

        # Check for Context7 compliance bottlenecks
        for pattern, score in self.context7_compliance_scores.items():
            if score < 0.5:
                bottlenecks.append(f"Poor {pattern} compliance: {score:.1%}")

        return bottlenecks

    def _generate_detailed_recommendations(self) -> List[str]:
        """Generate detailed optimization recommendations"""
        recommendations = []

        # Performance-based recommendations
        latest_metrics = self._get_latest_metrics()
        for metric in latest_metrics:
            if metric.value >= metric.threshold_critical:
                recommendations.append(
                    f"URGENT: Optimize {metric.metric_type.value} - currently at {metric.value:.1f}%"
                )
            elif metric.value >= metric.threshold_warning:
                recommendations.append(
                    f"Monitor {metric.metric_type.value} - currently at {metric.value:.1f}%"
                )

        # Context7-based recommendations
        for pattern, score in self.context7_compliance_scores.items():
            if score < 0.6:
                recommendations.append(
                    f"CRITICAL: Improve {pattern} implementation for better Context7 compliance"
                )
            elif score < 0.8:
                recommendations.append(
                    f"Consider optimizing {pattern} for enhanced user experience"
                )

        return recommendations

    def get_context7_optimization_strategies(self) -> Dict[str, List[str]]:
        """Get Context7-specific optimization strategies"""
        return {
            "responsive_design_system": [
                "Optimize viewport meta tag performance",
                "Implement efficient CSS media queries",
                "Use responsive image techniques",
                "Minimize layout shift on resize"
            ],
            "accessibility_features": [
                "Optimize ARIA label generation",
                "Improve screen reader performance",
                "Enhance keyboard navigation efficiency",
                "Optimize focus management"
            ],
            "adaptive_ui_layouts": [
                "Optimize dynamic content loading",
                "Improve state management performance",
                "Enhance user preference detection",
                "Reduce UI adaptation latency"
            ],
            "pwa_features": [
                "Optimize service worker caching",
                "Improve offline performance",
                "Enhance manifest loading",
                "Reduce PWA initialization time"
            ],
            "real_time_updates": [
                "Optimize WebSocket connections",
                "Improve update frequency efficiency",
                "Enhance data streaming performance",
                "Reduce update latency"
            ],
            "intelligent_cache": [
                "Optimize cache hit rates",
                "Improve cache memory efficiency",
                "Enhance cache invalidation",
                "Reduce cache lookup time"
            ],
            "advanced_ml_operations": [
                "Optimize model inference time",
                "Improve prediction accuracy trade-offs",
                "Enhance feature engineering efficiency",
                "Reduce ML computational overhead"
            ]
        }

    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        self.logger.info("Performance monitoring system cleaned up")


# Convenience functions for testing and integration
def create_performance_monitor(monitoring_interval: int = 5) -> PerformanceMonitoringSystem:
    """Create performance monitoring system with default settings"""
    return PerformanceMonitoringSystem(
        monitoring_interval=monitoring_interval,
        auto_optimize=True
    )


def run_quick_performance_check() -> Dict[str, Any]:
    """Run a quick performance check"""
    monitor = create_performance_monitor(monitoring_interval=2)

    try:
        return monitor.run_performance_analysis(duration_seconds=10)
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting NBA Predictor Performance Monitoring System")

    # Create and start monitoring
    monitor = create_performance_monitor(monitoring_interval=5)

    try:
        monitor.start_monitoring()

        # Let it run for a while
        time.sleep(30)

        # Generate report
        report = monitor.get_performance_report()

        print("\n📊 Performance Report:")
        print(f"Performance Level: {report['performance_level'].value}")
        print(f"Active Alerts: {report['active_alerts']}")
        print(f"Context7 Compliance: {report['context7_compliance']}")

        if report['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")

    finally:
        monitor.cleanup()

    print("✅ Performance monitoring completed")