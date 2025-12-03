"""
Test suite for Performance Monitoring System
Phase 3 Day 12 - Task 3.5.3 Performance Optimization and Monitoring
"""

import pytest
import time
import psutil
from src.nba_predictor.streamlit.components.performance_monitoring_system import (
    PerformanceMonitoringSystem, PerformanceLevel, MetricType, OptimizationStrategy,
    create_performance_monitor, run_quick_performance_check
)


def test_performance_monitoring_initialization():
    """Test performance monitoring system initialization"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    assert monitor.monitoring_interval == 1
    assert monitor.auto_optimize == False
    assert monitor.is_monitoring == False
    assert monitor.monitoring_thread is None
    assert len(monitor.metrics_history) == 0
    assert len(monitor.active_alerts) == 0
    assert len(monitor.optimization_history) == 0

    # Check default thresholds
    assert "cpu" in monitor.thresholds
    assert "memory" in monitor.thresholds
    assert monitor.thresholds["cpu"]["warning"] == 70.0
    assert monitor.thresholds["cpu"]["critical"] == 85.0

    monitor.cleanup()


def test_default_thresholds_configuration():
    """Test default performance thresholds"""
    monitor = PerformanceMonitoringSystem()
    thresholds = monitor._get_default_thresholds()

    expected_thresholds = ["cpu", "memory", "disk", "response_time", "cache_hit_rate", "database_performance"]
    for threshold in expected_thresholds:
        assert threshold in thresholds
        assert "warning" in thresholds[threshold]
        assert "critical" in thresholds[threshold]
        assert thresholds[threshold]["warning"] < thresholds[threshold]["critical"]

    monitor.cleanup()


def test_performance_metric_creation():
    """Test performance metric creation and validation"""
    from src.nba_predictor.streamlit.components.performance_monitoring_system import PerformanceMetric

    metric = PerformanceMetric(
        timestamp=time.time(),
        metric_type=MetricType.CPU,
        value=75.5,
        unit="percent",
        threshold_warning=70.0,
        threshold_critical=85.0
    )

    assert metric.metric_type == MetricType.CPU
    assert metric.value == 75.5
    assert metric.unit == "percent"
    assert metric.threshold_warning == 70.0
    assert metric.threshold_critical == 85.0


def test_performance_alert_creation():
    """Test performance alert creation"""
    from src.nba_predictor.streamlit.components.performance_monitoring_system import PerformanceAlert

    alert = PerformanceAlert(
        id="test-alert-123",
        metric_type=MetricType.CPU,
        level=PerformanceLevel.WARNING,
        message="CPU usage warning: 75.5%",
        timestamp=time.time(),
        value=75.5,
        threshold=70.0,
        recommendation="Consider optimizing CPU-intensive operations"
    )

    assert alert.metric_type == MetricType.CPU
    assert alert.level == PerformanceLevel.WARNING
    assert "CPU usage warning" in alert.message
    assert alert.recommendation is not None


def test_optimization_action_creation():
    """Test optimization action creation"""
    from src.nba_predictor.streamlit.components.performance_monitoring_system import OptimizationAction

    action = OptimizationAction(
        id="opt-action-123",
        strategy=OptimizationStrategy.AUTO_GC,
        timestamp=time.time(),
        description="Executed auto_gc optimization",
        impact={"memory_improvement": 15.2},
        context7_patterns=["intelligent_cache", "adaptive_ui_layouts"]
    )

    assert action.strategy == OptimizationStrategy.AUTO_GC
    assert "auto_gc" in action.description
    assert "memory_improvement" in action.impact
    assert len(action.context7_patterns) == 2


def test_metric_collection():
    """Test system metric collection functionality"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    # Collect metrics
    monitor._collect_system_metrics()

    # Should have collected CPU, memory, and disk metrics
    metric_types = {metric.metric_type for metric in monitor.metrics_history}
    assert MetricType.CPU in metric_types
    assert MetricType.MEMORY in metric_types
    assert MetricType.DISK in metric_types

    # Verify metric values are reasonable
    cpu_metrics = [m for m in monitor.metrics_history if m.metric_type == MetricType.CPU]
    assert len(cpu_metrics) > 0
    assert 0 <= cpu_metrics[0].value <= 100

    monitor.cleanup()


def test_application_metrics_collection():
    """Test application-specific metric collection"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    # Collect application metrics
    monitor._collect_application_metrics()

    # Should have collected response time and cache hit rate metrics
    metric_types = {metric.metric_type for metric in monitor.metrics_history}
    assert MetricType.RESPONSE_TIME in metric_types
    assert MetricType.CACHE_HIT_RATE in metric_types

    # Verify response time is reasonable
    response_metrics = [m for m in monitor.metrics_history if m.metric_type == MetricType.RESPONSE_TIME]
    assert len(response_metrics) > 0
    assert response_metrics[0].value >= 0

    monitor.cleanup()


def test_context7_compliance_validation():
    """Test Context7 compliance validation"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    # Validate Context7 compliance
    monitor._validate_context7_compliance()

    # Should have compliance scores for all patterns
    expected_patterns = [
        "responsive_design_system",
        "accessibility_features",
        "adaptive_ui_layouts",
        "pwa_features",
        "real_time_updates",
        "intelligent_cache",
        "advanced_ml_operations"
    ]

    for pattern in expected_patterns:
        assert pattern in monitor.context7_compliance_scores
        assert 0 <= monitor.context7_compliance_scores[pattern] <= 1

    monitor.cleanup()


def test_individual_context7_pattern_validation():
    """Test individual Context7 pattern validation"""
    monitor = PerformanceMonitoringSystem()

    # Test each pattern validation
    patterns = [
        "responsive_design_system",
        "accessibility_features",
        "adaptive_ui_layouts",
        "pwa_features",
        "real_time_updates",
        "intelligent_cache",
        "advanced_ml_operations"
    ]

    for pattern in patterns:
        score = monitor._validate_context7_pattern(pattern)
        assert 0 <= score <= 1, f"Pattern {pattern} should return score between 0 and 1"

    monitor.cleanup()


def test_alert_generation():
    """Test performance alert generation"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    # Add some high-value metrics to trigger alerts
    high_cpu_metric = {
        "timestamp": time.time(),
        "metric_type": MetricType.CPU,
        "value": 90.0,
        "unit": "percent",
        "threshold_warning": 70.0,
        "threshold_critical": 85.0
    }

    monitor.metrics_history.append(PerformanceMetric(**high_cpu_metric))

    # Check for alerts
    monitor._check_alerts()

    # Should have generated a critical alert
    critical_alerts = [a for a in monitor.active_alerts if a.level == PerformanceLevel.CRITICAL]
    assert len(critical_alerts) > 0

    cpu_alerts = [a for a in monitor.active_alerts if a.metric_type == MetricType.CPU]
    assert len(cpu_alerts) > 0

    monitor.cleanup()


def test_recommendation_generation():
    """Test optimization recommendation generation"""
    monitor = PerformanceMonitoringSystem()

    # Add a critical alert
    alert = {
        "id": "test-alert",
        "metric_type": MetricType.CPU,
        "level": PerformanceLevel.CRITICAL,
        "message": "CPU critical: 90%",
        "timestamp": time.time(),
        "value": 90.0,
        "threshold": 85.0
    }

    monitor.active_alerts.append(PerformanceAlert(**alert))

    # Generate recommendations
    recommendations = monitor._generate_recommendations()

    assert len(recommendations) > 0
    assert any("CPU" in rec for rec in recommendations)

    monitor.cleanup()


def test_optimization_strategy_execution():
    """Test optimization strategy execution"""
    monitor = PerformanceMonitoringSystem()

    # Test garbage collection optimization
    monitor._execute_optimization_strategy(OptimizationStrategy.AUTO_GC)

    # Should have recorded optimization action
    gc_actions = [a for a in monitor.optimization_history if a.strategy == OptimizationStrategy.AUTO_GC]
    assert len(gc_actions) > 0
    assert "memory_improvement" in gc_actions[-1].impact

    # Test memory cleanup optimization
    monitor._execute_optimization_strategy(OptimizationStrategy.MEMORY_CLEANUP)

    cleanup_actions = [a for a in monitor.optimization_history if a.strategy == OptimizationStrategy.MEMORY_CLEANUP]
    assert len(cleanup_actions) > 0
    assert "memory_freed_gb" in cleanup_actions[-1].impact

    monitor.cleanup()


def test_auto_optimization():
    """Test automatic optimization functionality"""
    monitor = PerformanceMonitoringSystem(auto_optimize=True)

    # Add critical alerts to trigger auto-optimization
    critical_alert = {
        "id": "critical-alert",
        "metric_type": MetricType.CPU,
        "level": PerformanceLevel.CRITICAL,
        "message": "Critical CPU usage",
        "timestamp": time.time(),
        "value": 95.0,
        "threshold": 85.0
    }

    monitor.active_alerts.append(PerformanceAlert(**critical_alert))

    # Trigger auto-optimization
    monitor._auto_optimize()

    # Should have executed optimization strategies
    assert len(monitor.optimization_history) > 0

    # Check for expected optimization types
    gc_executed = any(a.strategy == OptimizationStrategy.AUTO_GC for a in monitor.optimization_history)
    memory_executed = any(a.strategy == OptimizationStrategy.MEMORY_CLEANUP for a in monitor.optimization_history)

    assert gc_executed or memory_executed

    monitor.cleanup()


def test_performance_report_generation():
    """Test performance report generation"""
    monitor = PerformanceMonitoringSystem()

    # Add some sample data
    monitor._collect_system_metrics()
    monitor._validate_context7_compliance()

    # Generate report
    report = monitor.get_performance_report()

    # Verify report structure
    required_fields = [
        "timestamp",
        "current_metrics",
        "active_alerts",
        "critical_alerts",
        "context7_compliance",
        "optimization_actions",
        "recommendations",
        "performance_level"
    ]

    for field in required_fields:
        assert field in report

    # Verify performance level is valid
    assert report["performance_level"] in PerformanceLevel

    monitor.cleanup()


def test_overall_performance_level_calculation():
    """Test overall performance level calculation"""
    monitor = PerformanceMonitoringSystem()

    # Test with no alerts (should be optimal)
    level = monitor._calculate_overall_performance_level()
    assert level == PerformanceLevel.OPTIMAL

    # Add warning alert
    warning_alert = {
        "id": "warning",
        "metric_type": MetricType.CPU,
        "level": PerformanceLevel.WARNING,
        "message": "CPU warning",
        "timestamp": time.time(),
        "value": 75.0,
        "threshold": 70.0
    }

    monitor.active_alerts.append(PerformanceAlert(**warning_alert))
    level = monitor._calculate_overall_performance_level()
    assert level == PerformanceLevel.WARNING

    # Add critical alert
    critical_alert = {
        "id": "critical",
        "metric_type": MetricType.MEMORY,
        "level": PerformanceLevel.CRITICAL,
        "message": "Memory critical",
        "timestamp": time.time(),
        "value": 90.0,
        "threshold": 85.0
    }

    monitor.active_alerts.append(PerformanceAlert(**critical_alert))
    level = monitor._calculate_overall_performance_level()
    assert level == PerformanceLevel.CRITICAL

    monitor.cleanup()


def test_context7_optimization_strategies():
    """Test Context7-specific optimization strategies"""
    monitor = PerformanceMonitoringSystem()

    strategies = monitor.get_context7_optimization_strategies()

    # Verify all expected patterns are present
    expected_patterns = [
        "responsive_design_system",
        "accessibility_features",
        "adaptive_ui_layouts",
        "pwa_features",
        "real_time_updates",
        "intelligent_cache",
        "advanced_ml_operations"
    ]

    for pattern in expected_patterns:
        assert pattern in strategies
        assert len(strategies[pattern]) > 0
        assert all(isinstance(item, str) for item in strategies[pattern])

    monitor.cleanup()


def test_monitoring_start_stop():
    """Test monitoring start and stop functionality"""
    monitor = PerformanceMonitoringSystem(monitoring_interval=1, auto_optimize=False)

    # Test starting monitoring
    assert not monitor.is_monitoring
    monitor.start_monitoring()
    assert monitor.is_monitoring
    assert monitor.monitoring_thread is not None

    # Let it run briefly
    time.sleep(2)

    # Test stopping monitoring
    monitor.stop_monitoring()
    assert not monitor.is_monitoring

    monitor.cleanup()


def test_convenience_functions():
    """Test convenience functions"""
    # Test create_performance_monitor
    monitor = create_performance_monitor()
    assert monitor.monitoring_interval == 5
    assert monitor.auto_optimize == True
    monitor.cleanup()

    # Test run_quick_performance_check
    report = run_quick_performance_check()

    # Verify report structure
    required_fields = [
        "analysis_duration",
        "metrics_collected",
        "alerts_generated",
        "optimizations_performed",
        "context7_compliance_summary",
        "performance_trends",
        "bottleneck_analysis",
        "recommendations"
    ]

    for field in required_fields:
        assert field in report


def test_database_initialization():
    """Test performance database initialization"""
    monitor = PerformanceMonitoringSystem()

    # Database should be initialized automatically
    # Verify by trying to save some data
    test_metric = PerformanceMetric(
        timestamp=time.time(),
        metric_type=MetricType.CPU,
        value=50.0,
        unit="percent",
        threshold_warning=70.0,
        threshold_critical=85.0
    )

    monitor._add_metric(test_metric)
    monitor._save_metrics()

    # If no exceptions occurred, database initialization was successful
    assert True

    monitor.cleanup()


def test_metrics_cleanup():
    """Test old metrics cleanup functionality"""
    monitor = PerformanceMonitoringSystem()

    # Add old metrics (more than an hour old)
    old_timestamp = time.time() - 3700  # More than 1 hour
    old_metric = PerformanceMetric(
        timestamp=old_timestamp,
        metric_type=MetricType.CPU,
        value=50.0,
        unit="percent",
        threshold_warning=70.0,
        threshold_critical=85.0
    )

    monitor.metrics_history.append(old_metric)

    # Add recent metrics
    monitor._collect_system_metrics()

    # Run cleanup
    monitor._cleanup_old_metrics()

    # Should only have recent metrics
    assert all(m.timestamp > (time.time() - 3600) for m in monitor.metrics_history)

    monitor.cleanup()


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])