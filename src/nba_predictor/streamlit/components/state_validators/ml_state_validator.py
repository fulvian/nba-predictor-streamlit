"""
🎯 PHASE 3 DAY 8: ML State Validator
=====================================

X7 Compliant ML State Validation System for NBA Predictor Dashboard.

This module provides comprehensive validation for ML component states:
- Component-specific validation rules
- Performance threshold validation
- Data integrity checks
- Business logic validation
- Error recovery recommendations

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum

from ..state_manager import MLComponentState, ComponentState, StateValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """X7 Compliant validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """X7 Compliant validation result data structure."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    component_id: str
    rule_id: str
    timestamp: datetime
    details: Dict[str, Any] = None
    recommendations: List[str] = None

    def __post_init__(self):
        """X7 Compliant post-initialization."""
        if self.details is None:
            self.details = {}
        if self.recommendations is None:
            self.recommendations = []


class MLStateValidator:
    """
    X7 Compliant ML State Validation System.

    Provides comprehensive validation framework for all ML system components
    with configurable rules, performance thresholds, and business logic validation.
    """

    def __init__(self):
        """Initialize X7 Compliant state validator."""
        logger.info("Initializing X7 Compliant ML State Validator")

        # Validation rules registry
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}
        self.business_rules: Dict[str, List[Callable]] = {}

        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.component_scores: Dict[str, float] = {}

        # Configuration
        self._config = {
            'max_history_size': 1000,
            'score_decay_hours': 24,
            'validation_timeout': 5.0,
            'default_performance_thresholds': {
                'response_time_ms': 1000.0,
                'memory_usage_mb': 512.0,
                'cpu_usage_percent': 80.0,
                'error_rate_percent': 5.0
            }
        }

        # Initialize validation rules
        self._initialize_validation_rules()
        self._initialize_performance_thresholds()
        self._initialize_business_rules()

        logger.info("X7 Compliant ML State Validator initialized successfully")

    def _initialize_validation_rules(self):
        """Initialize component-specific validation rules."""
        self.validation_rules = {
            'data_pipeline': [
                self._validate_data_pipeline_state,
                self._validate_data_quality,
                self._validate_data_freshness
            ],
            'ml_models': [
                self._validate_ml_model_state,
                self._validate_model_accuracy,
                self._validate_model_freshness
            ],
            'model_monitoring': [
                self._validate_monitoring_state,
                self._validate_drift_detection,
                self._validate_alert_thresholds
            ],
            'predictions_engine': [
                self._validate_prediction_engine_state,
                self._validate_prediction_quality,
                self._validate_prediction_latency
            ],
            'betting_system': [
                self._validate_betting_system_state,
                self._validate_bet_processing,
                self._validate_risk_management
            ],
            'user_interface': [
                self._validate_ui_state,
                self._validate_ui_responsiveness,
                self._validate_user_experience
            ],
            'analytics': [
                self._validate_analytics_state,
                self._validate_data_aggregation,
                self._validate_report_generation
            ]
        }

        logger.info(f"Initialized validation rules for {len(self.validation_rules)} components")

    def _initialize_performance_thresholds(self):
        """Initialize performance thresholds for components."""
        self.performance_thresholds = {
            'data_pipeline': {
                'max_processing_time_ms': 5000.0,
                'max_memory_usage_mb': 1024.0,
                'min_data_quality_score': 0.8,
                'max_data_staleness_minutes': 30.0
            },
            'ml_models': {
                'max_inference_time_ms': 100.0,
                'min_accuracy_score': 0.7,
                'max_model_age_days': 30.0,
                'max_memory_usage_mb': 512.0
            },
            'model_monitoring': {
                'max_monitoring_latency_ms': 1000.0,
                'min_drift_detection_sensitivity': 0.1,
                'max_false_positive_rate': 0.05
            },
            'predictions_engine': {
                'max_prediction_time_ms': 50.0,
                'min_confidence_score': 0.6,
                'max_batch_processing_time_ms': 5000.0,
                'max_concurrent_predictions': 1000
            },
            'betting_system': {
                'max_bet_processing_time_ms': 200.0,
                'min_settlement_accuracy': 0.99,
                'max_risk_exposure_percent': 5.0,
                'max_concurrent_bets': 500
            },
            'user_interface': {
                'max_page_load_time_ms': 3000.0,
                'max_interaction_response_ms': 500.0,
                'min_accessibility_score': 0.9,
                'max_error_rate_percent': 1.0
            },
            'analytics': {
                'max_query_time_ms': 2000.0,
                'min_report_accuracy': 0.95,
                'max_aggregation_latency_ms': 10000.0
            }
        }

        # Apply default thresholds to all components
        for component_id in self.validation_rules.keys():
            if component_id not in self.performance_thresholds:
                self.performance_thresholds[component_id] = self._config['default_performance_thresholds'].copy()

    def _initialize_business_rules(self):
        """Initialize business logic validation rules."""
        self.business_rules = {
            'data_pipeline': [
                self._validate_business_data_freshness,
                self._validate_business_data_completeness
            ],
            'ml_models': [
                self._validate_business_model_accuracy,
                self._validate_business_model_fairness
            ],
            'predictions_engine': [
                self._validate_business_prediction_confidence,
                self._validate_business_prediction_distribution
            ],
            'betting_system': [
                self._validate_business_bet_limits,
                self._validate_business_risk_compliance
            ]
        }

    def validate_component_state(self, component_state: MLComponentState) -> List[ValidationResult]:
        """
        Validate component state with X7 Compliant comprehensive checks.

        Args:
            component_state: Component state to validate

        Returns:
            List of validation results
        """
        component_id = component_state.component_id
        validation_results = []

        try:
            # Component-specific validation rules
            if component_id in self.validation_rules:
                for rule_func in self.validation_rules[component_id]:
                    try:
                        result = rule_func(component_state)
                        if result:
                            validation_results.append(result)
                    except Exception as e:
                        logger.error(f"Validation rule failed for {component_id}: {e}")
                        validation_results.append(self._create_error_result(
                            component_id, f"validation_rule_error", str(e)
                        ))

            # Performance threshold validation
            performance_results = self._validate_performance_thresholds(component_state)
            validation_results.extend(performance_results)

            # Business logic validation
            if component_id in self.business_rules:
                for business_rule in self.business_rules[component_id]:
                    try:
                        result = business_rule(component_state)
                        if result:
                            validation_results.append(result)
                    except Exception as e:
                        logger.error(f"Business rule failed for {component_id}: {e}")
                        validation_results.append(self._create_error_result(
                            component_id, f"business_rule_error", str(e)
                        ))

            # Calculate component validation score
            self._update_component_score(component_id, validation_results)

            # Store validation results
            self._store_validation_results(validation_results)

            logger.debug(f"Validation completed for {component_id}: {len(validation_results)} results")

        except Exception as e:
            logger.error(f"Validation failed for {component_id}: {e}")
            validation_results.append(self._create_error_result(
                component_id, "validation_error", str(e)
            ))

        return validation_results

    def validate_system_consistency(self, all_states: Dict[str, MLComponentState]) -> List[ValidationResult]:
        """
        Validate system-wide consistency across all components.

        Args:
            all_states: Dictionary of all component states

        Returns:
            List of consistency validation results
        """
        consistency_results = []

        try:
            # Cross-component dependency validation
            dependency_results = self._validate_component_dependencies(all_states)
            consistency_results.extend(dependency_results)

            # System-wide performance validation
            performance_results = self._validate_system_performance(all_states)
            consistency_results.extend(performance_results)

            # Business flow validation
            flow_results = self._validate_business_flow(all_states)
            consistency_results.extend(flow_results)

            # Data consistency validation
            data_results = self._validate_data_consistency(all_states)
            consistency_results.extend(data_results)

            logger.debug(f"System consistency validation completed: {len(consistency_results)} results")

        except Exception as e:
            logger.error(f"System consistency validation failed: {e}")
            consistency_results.append(self._create_error_result(
                "system", "consistency_validation_error", str(e)
            ))

        return consistency_results

    def _validate_performance_thresholds(self, component_state: MLComponentState) -> List[ValidationResult]:
        """Validate component against performance thresholds."""
        results = []
        component_id = component_state.component_id

        if component_id not in self.performance_thresholds:
            return results

        thresholds = self.performance_thresholds[component_id]
        performance_metrics = component_state.performance_metrics

        for metric_name, threshold_value in thresholds.items():
            current_value = performance_metrics.get(metric_name)

            if current_value is None:
                # Missing metric - generate warning
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing performance metric: {metric_name}",
                    component_id=component_id,
                    rule_id="missing_performance_metric",
                    timestamp=datetime.now(),
                    details={
                        'metric_name': metric_name,
                        'threshold': threshold_value
                    },
                    recommendations=[
                        f"Ensure {metric_name} is being tracked and reported",
                        "Check performance monitoring configuration"
                    ]
                ))
                continue

            # Determine if metric should be below or above threshold
            is_max_metric = any(keyword in metric_name.lower()
                               for keyword in ['max', 'time', 'latency', 'usage', 'rate'])

            if is_max_metric:
                # Should be below threshold
                if current_value > threshold_value:
                    severity = self._determine_performance_severity(
                        current_value, threshold_value, is_violation=True
                    )
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=severity,
                        message=f"Performance threshold exceeded: {metric_name}",
                        component_id=component_id,
                        rule_id="performance_threshold_exceeded",
                        timestamp=datetime.now(),
                        details={
                            'metric_name': metric_name,
                            'current_value': current_value,
                            'threshold': threshold_value,
                            'violation_percentage': ((current_value - threshold_value) / threshold_value) * 100
                        },
                        recommendations=self._get_performance_recommendations(metric_name, current_value, threshold_value)
                    ))
            else:
                # Should be above threshold
                if current_value < threshold_value:
                    severity = self._determine_performance_severity(
                        current_value, threshold_value, is_violation=True
                    )
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=severity,
                        message=f"Performance threshold not met: {metric_name}",
                        component_id=component_id,
                        rule_id="performance_threshold_not_met",
                        timestamp=datetime.now(),
                        details={
                            'metric_name': metric_name,
                            'current_value': current_value,
                            'threshold': threshold_value,
                            'shortfall_percentage': ((threshold_value - current_value) / threshold_value) * 100
                        },
                        recommendations=self._get_performance_recommendations(metric_name, current_value, threshold_value)
                    ))

        return results

    def _determine_performance_severity(self, current_value: float, threshold: float, is_violation: bool) -> ValidationSeverity:
        """Determine validation severity based on performance violation magnitude."""
        if is_violation:
            violation_ratio = current_value / threshold
        else:
            violation_ratio = threshold / current_value

        if violation_ratio > 2.0:
            return ValidationSeverity.CRITICAL
        elif violation_ratio > 1.5:
            return ValidationSeverity.ERROR
        elif violation_ratio > 1.1:
            return ValidationSeverity.WARNING
        else:
            return ValidationSeverity.INFO

    def _get_performance_recommendations(self, metric_name: str, current_value: float, threshold: float) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []

        if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
            recommendations.extend([
                "Optimize algorithm efficiency",
                "Consider caching frequently accessed data",
                "Review database query performance",
                "Scale horizontally if needed"
            ])
        elif 'memory' in metric_name.lower():
            recommendations.extend([
                "Check for memory leaks",
                "Optimize data structures",
                "Implement memory-efficient algorithms",
                "Consider increasing memory allocation"
            ])
        elif 'accuracy' in metric_name.lower():
            recommendations.extend([
                "Retrain model with recent data",
                "Feature engineering improvements",
                "Hyperparameter tuning",
                "Consider ensemble methods"
            ])
        elif 'error' in metric_name.lower():
            recommendations.extend([
                "Review error logs for patterns",
                "Implement better error handling",
                "Add input validation",
                "Monitor system health more closely"
            ])
        else:
            recommendations.extend([
                "Review component configuration",
                "Check system resources",
                "Monitor performance trends",
                "Consider capacity planning"
            ])

        return recommendations

    # Component-specific validation methods

    def _validate_data_pipeline_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate data pipeline component state."""
        if state.status == ComponentState.ERROR:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Data pipeline is in ERROR state",
                component_id=state.component_id,
                rule_id="data_pipeline_error",
                timestamp=datetime.now(),
                details={'error_info': state.error_info},
                recommendations=[
                    "Check data source connections",
                    "Review data processing logs",
                    "Verify data format specifications",
                    "Check for data corruption"
                ]
            )

        # Check required data fields
        required_fields = ['last_fetch_time', 'data_quality_score', 'record_count']
        missing_fields = [field for field in required_fields if field not in state.data]

        if missing_fields:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Missing required data fields: {', '.join(missing_fields)}",
                component_id=state.component_id,
                rule_id="missing_data_fields",
                timestamp=datetime.now(),
                details={'missing_fields': missing_fields}
            )

        return None

    def _validate_data_quality(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate data quality metrics."""
        quality_score = state.data.get('data_quality_score', 0.0)

        if quality_score < 0.5:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Very low data quality score: {quality_score:.2f}",
                component_id=state.component_id,
                rule_id="low_data_quality",
                timestamp=datetime.now(),
                details={'quality_score': quality_score},
                recommendations=[
                    "Check data source integrity",
                    "Review data cleaning processes",
                    "Implement data validation rules",
                    "Consider data source replacement"
                ]
            )
        elif quality_score < 0.8:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Low data quality score: {quality_score:.2f}",
                component_id=state.component_id,
                rule_id="moderate_data_quality",
                timestamp=datetime.now(),
                details={'quality_score': quality_score},
                recommendations=[
                    "Monitor data quality trends",
                    "Review data processing pipeline",
                    "Consider additional data cleaning"
                ]
            )

        return None

    def _validate_data_freshness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate data freshness."""
        last_fetch = state.data.get('last_fetch_time')
        if not last_fetch:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="No last fetch time available",
                component_id=state.component_id,
                rule_id="missing_fetch_time",
                timestamp=datetime.now(),
                recommendations=["Ensure data fetch timestamp is recorded"]
            )

        try:
            if isinstance(last_fetch, str):
                last_fetch = datetime.fromisoformat(last_fetch)

            age_minutes = (datetime.now() - last_fetch).total_seconds() / 60
            max_age = self.performance_thresholds.get('data_pipeline', {}).get('max_data_staleness_minutes', 30.0)

            if age_minutes > max_age:
                severity = ValidationSeverity.CRITICAL if age_minutes > max_age * 2 else ValidationSeverity.ERROR
                return ValidationResult(
                    is_valid=False,
                    severity=severity,
                    message=f"Data is stale: {age_minutes:.1f} minutes old",
                    component_id=state.component_id,
                    rule_id="stale_data",
                    timestamp=datetime.now(),
                    details={
                        'age_minutes': age_minutes,
                        'max_age_minutes': max_age
                    },
                    recommendations=[
                        "Check data refresh schedule",
                        "Verify data source connectivity",
                        "Review data pipeline performance"
                    ]
                )
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid last fetch time format: {e}",
                component_id=state.component_id,
                rule_id="invalid_fetch_time",
                timestamp=datetime.now()
            )

        return None

    def _validate_ml_model_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate ML model component state."""
        if state.status == ComponentState.ERROR:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="ML model is in ERROR state",
                component_id=state.component_id,
                rule_id="ml_model_error",
                timestamp=datetime.now(),
                details={'error_info': state.error_info},
                recommendations=[
                    "Check model loading process",
                    "Verify model file integrity",
                    "Review model configuration",
                    "Check system resources"
                ]
            )

        return None

    def _validate_model_accuracy(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate model accuracy metrics."""
        accuracy = state.data.get('accuracy_score')
        if accuracy is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No accuracy score available",
                component_id=state.component_id,
                rule_id="missing_accuracy",
                timestamp=datetime.now()
            )

        min_accuracy = self.performance_thresholds.get('ml_models', {}).get('min_accuracy_score', 0.7)

        if accuracy < min_accuracy - 0.2:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Very low model accuracy: {accuracy:.3f}",
                component_id=state.component_id,
                rule_id="very_low_accuracy",
                timestamp=datetime.now(),
                details={'accuracy': accuracy, 'min_accuracy': min_accuracy},
                recommendations=[
                    "Retrain model with fresh data",
                    "Review feature engineering",
                    "Check for data drift",
                    "Consider model architecture changes"
                ]
            )
        elif accuracy < min_accuracy:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Low model accuracy: {accuracy:.3f}",
                component_id=state.component_id,
                rule_id="low_accuracy",
                timestamp=datetime.now(),
                details={'accuracy': accuracy, 'min_accuracy': min_accuracy},
                recommendations=[
                    "Monitor accuracy trends",
                    "Schedule model retraining",
                    "Review training data quality"
                ]
            )

        return None

    def _validate_model_freshness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate model freshness."""
        trained_date = state.data.get('model_trained_date')
        if not trained_date:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No model training date available",
                component_id=state.component_id,
                rule_id="missing_training_date",
                timestamp=datetime.now()
            )

        try:
            if isinstance(trained_date, str):
                trained_date = datetime.fromisoformat(trained_date)

            age_days = (datetime.now() - trained_date).days
            max_age = self.performance_thresholds.get('ml_models', {}).get('max_model_age_days', 30.0)

            if age_days > max_age:
                severity = ValidationSeverity.CRITICAL if age_days > max_age * 2 else ValidationSeverity.WARNING
                return ValidationResult(
                    is_valid=False,
                    severity=severity,
                    message=f"Model is outdated: {age_days} days old",
                    component_id=state.component_id,
                    rule_id="outdated_model",
                    timestamp=datetime.now(),
                    details={
                        'age_days': age_days,
                        'max_age_days': max_age
                    },
                    recommendations=[
                        "Retrain model with recent data",
                        "Update model training pipeline",
                        "Consider automated retraining"
                    ]
                )
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid training date format: {e}",
                component_id=state.component_id,
                rule_id="invalid_training_date",
                timestamp=datetime.now()
            )

        return None

    def _validate_monitoring_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate model monitoring component state."""
        if state.status == ComponentState.OFFLINE:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Model monitoring is OFFLINE",
                component_id=state.component_id,
                rule_id="monitoring_offline",
                timestamp=datetime.now(),
                recommendations=[
                    "Restart monitoring service",
                    "Check monitoring configuration",
                    "Verify alert system connectivity"
                ]
            )

        return None

    def _validate_drift_detection(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate drift detection functionality."""
        last_drift_check = state.data.get('last_drift_check')
        if not last_drift_check:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No recent drift check performed",
                component_id=state.component_id,
                rule_id="missing_drift_check",
                timestamp=datetime.now(),
                recommendations=["Ensure drift detection is running regularly"]
            )

        return None

    def _validate_alert_thresholds(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate alert threshold configuration."""
        alert_config = state.data.get('alert_thresholds', {})
        if not alert_config:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No alert thresholds configured",
                component_id=state.component_id,
                rule_id="missing_alert_thresholds",
                timestamp=datetime.now(),
                recommendations=["Configure alert thresholds for proactive monitoring"]
            )

        return None

    def _validate_prediction_engine_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate prediction engine component state."""
        if state.status == ComponentState.ERROR:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Prediction engine is in ERROR state",
                component_id=state.component_id,
                rule_id="prediction_engine_error",
                timestamp=datetime.now(),
                details={'error_info': state.error_info},
                recommendations=[
                    "Check model loading",
                    "Verify prediction input format",
                    "Review prediction pipeline"
                ]
            )

        return None

    def _validate_prediction_quality(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate prediction quality metrics."""
        confidence_avg = state.data.get('average_confidence')
        if confidence_avg is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No confidence metrics available",
                component_id=state.component_id,
                rule_id="missing_confidence_metrics",
                timestamp=datetime.now()
            )

        min_confidence = self.performance_thresholds.get('predictions_engine', {}).get('min_confidence_score', 0.6)

        if confidence_avg < min_confidence:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Low prediction confidence: {confidence_avg:.3f}",
                component_id=state.component_id,
                rule_id="low_prediction_confidence",
                timestamp=datetime.now(),
                details={'confidence_avg': confidence_avg, 'min_confidence': min_confidence},
                recommendations=[
                    "Review model calibration",
                    "Check prediction input quality",
                    "Consider confidence threshold adjustment"
                ]
            )

        return None

    def _validate_prediction_latency(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate prediction latency metrics."""
        latency_ms = state.performance_metrics.get('prediction_time_ms')
        if latency_ms is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.INFO,
                message="No latency metrics available",
                component_id=state.component_id,
                rule_id="missing_latency_metrics",
                timestamp=datetime.now()
            )

        max_latency = self.performance_thresholds.get('predictions_engine', {}).get('max_prediction_time_ms', 50.0)

        if latency_ms > max_latency:
            severity = ValidationSeverity.CRITICAL if latency_ms > max_latency * 3 else ValidationSeverity.WARNING
            return ValidationResult(
                is_valid=False,
                severity=severity,
                message=f"High prediction latency: {latency_ms:.1f}ms",
                component_id=state.component_id,
                rule_id="high_prediction_latency",
                timestamp=datetime.now(),
                details={'latency_ms': latency_ms, 'max_latency_ms': max_latency},
                recommendations=[
                    "Optimize prediction pipeline",
                    "Consider model optimization",
                    "Implement prediction caching"
                ]
            )

        return None

    # Additional validation methods for other components...

    def _validate_betting_system_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate betting system component state."""
        if state.status == ComponentState.ERROR:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Betting system is in ERROR state",
                component_id=state.component_id,
                rule_id="betting_system_error",
                timestamp=datetime.now(),
                details={'error_info': state.error_info},
                recommendations=[
                    "Check betting system logs",
                    "Verify database connectivity",
                    "Review bet processing pipeline"
                ]
            )
        return None

    def _validate_bet_processing(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate bet processing functionality."""
        processing_time = state.performance_metrics.get('bet_processing_time_ms')
        if processing_time and processing_time > 200.0:  # 200ms threshold
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Slow bet processing: {processing_time:.1f}ms",
                component_id=state.component_id,
                rule_id="slow_bet_processing",
                timestamp=datetime.now(),
                details={'processing_time_ms': processing_time}
            )
        return None

    def _validate_risk_management(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate risk management controls."""
        risk_exposure = state.data.get('risk_exposure_percent')
        if risk_exposure and risk_exposure > 5.0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"High risk exposure: {risk_exposure:.1f}%",
                component_id=state.component_id,
                rule_id="high_risk_exposure",
                timestamp=datetime.now(),
                details={'risk_exposure_percent': risk_exposure},
                recommendations=[
                    "Review bet limits",
                    "Implement position limits",
                    "Enhance risk monitoring"
                ]
            )
        return None

    def _validate_ui_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate user interface component state."""
        if state.status == ComponentState.ERROR:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="UI is in ERROR state",
                component_id=state.component_id,
                rule_id="ui_error",
                timestamp=datetime.now(),
                recommendations=["Check UI error logs", "Verify frontend configuration"]
            )
        return None

    def _validate_ui_responsiveness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate UI responsiveness metrics."""
        response_time = state.performance_metrics.get('interaction_response_ms')
        if response_time and response_time > 500.0:  # 500ms threshold
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Slow UI response: {response_time:.1f}ms",
                component_id=state.component_id,
                rule_id="slow_ui_response",
                timestamp=datetime.now(),
                details={'response_time_ms': response_time}
            )
        return None

    def _validate_user_experience(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate user experience metrics."""
        error_rate = state.data.get('error_rate_percent')
        if error_rate and error_rate > 1.0:  # 1% threshold
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"High UI error rate: {error_rate:.1f}%",
                component_id=state.component_id,
                rule_id="high_ui_error_rate",
                timestamp=datetime.now(),
                details={'error_rate_percent': error_rate}
            )
        return None

    def _validate_analytics_state(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate analytics component state."""
        return None  # Implementation placeholder

    def _validate_data_aggregation(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate data aggregation functionality."""
        return None  # Implementation placeholder

    def _validate_report_generation(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate report generation functionality."""
        return None  # Implementation placeholder

    # Business rule validation methods

    def _validate_business_data_freshness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business data freshness requirements."""
        # Implementation placeholder
        return None

    def _validate_business_data_completeness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business data completeness requirements."""
        # Implementation placeholder
        return None

    def _validate_business_model_accuracy(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business model accuracy requirements."""
        # Implementation placeholder
        return None

    def _validate_business_model_fairness(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business model fairness requirements."""
        # Implementation placeholder
        return None

    def _validate_business_prediction_confidence(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business prediction confidence requirements."""
        # Implementation placeholder
        return None

    def _validate_business_prediction_distribution(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business prediction distribution requirements."""
        # Implementation placeholder
        return None

    def _validate_business_bet_limits(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business betting limits compliance."""
        # Implementation placeholder
        return None

    def _validate_business_risk_compliance(self, state: MLComponentState) -> Optional[ValidationResult]:
        """Validate business risk management compliance."""
        # Implementation placeholder
        return None

    # System consistency validation methods

    def _validate_component_dependencies(self, all_states: Dict[str, MLComponentState]) -> List[ValidationResult]:
        """Validate cross-component dependencies."""
        results = []

        # Check data pipeline -> ML models dependency
        data_state = all_states.get('data_pipeline')
        model_state = all_states.get('ml_models')

        if data_state and model_state:
            if data_state.status == ComponentState.ERROR and model_state.status != ComponentState.ERROR:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="ML models running despite data pipeline errors",
                    component_id="system",
                    rule_id="data_model_dependency",
                    timestamp=datetime.now(),
                    details={
                        'data_pipeline_status': data_state.status.value,
                        'ml_models_status': model_state.status.value
                    }
                ))

        return results

    def _validate_system_performance(self, all_states: Dict[str, MLComponentState]) -> List[ValidationResult]:
        """Validate system-wide performance metrics."""
        results = []

        # Calculate system-wide response times
        total_response_time = 0
        component_count = 0

        for component_id, state in all_states.items():
            response_time = state.performance_metrics.get('response_time_ms')
            if response_time:
                total_response_time += response_time
                component_count += 1

        if component_count > 0:
            avg_response_time = total_response_time / component_count
            if avg_response_time > 2000.0:  # 2 seconds
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"High average system response time: {avg_response_time:.1f}ms",
                    component_id="system",
                    rule_id="high_system_response_time",
                    timestamp=datetime.now(),
                    details={'avg_response_time_ms': avg_response_time}
                ))

        return results

    def _validate_business_flow(self, all_states: Dict[str, MLComponentState]) -> List[ValidationResult]:
        """Validate business flow consistency."""
        results = []

        # Check betting flow dependencies
        prediction_state = all_states.get('predictions_engine')
        betting_state = all_states.get('betting_system')

        if prediction_state and betting_state:
            if prediction_state.status == ComponentState.ERROR and betting_state.status == ComponentState.HEALTHY:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Betting system healthy despite prediction engine errors",
                    component_id="system",
                    rule_id="prediction_betting_flow",
                    timestamp=datetime.now(),
                    details={
                        'predictions_status': prediction_state.status.value,
                        'betting_status': betting_state.status.value
                    }
                ))

        return results

    def _validate_data_consistency(self, all_states: Dict[str, MLComponentState]) -> List[ValidationResult]:
        """Validate data consistency across components."""
        results = []

        # Check for consistent timestamps
        timestamps = []
        for component_id, state in all_states.items():
            if state.last_updated:
                timestamps.append((component_id, state.last_updated))

        if timestamps:
            latest_time = max(ts for _, ts in timestamps)
            earliest_time = min(ts for _, ts in timestamps)
            time_diff = (latest_time - earliest_time).total_seconds()

            if time_diff > 300:  # 5 minutes
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"High timestamp variance across components: {time_diff:.1f}s",
                    component_id="system",
                    rule_id="timestamp_inconsistency",
                    timestamp=datetime.now(),
                    details={'time_variance_seconds': time_diff}
                ))

        return results

    def _create_error_result(self, component_id: str, rule_id: str, error_message: str) -> ValidationResult:
        """Create error validation result."""
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            message=f"Validation error: {error_message}",
            component_id=component_id,
            rule_id=rule_id,
            timestamp=datetime.now(),
            details={'error_message': error_message}
        )

    def _update_component_score(self, component_id: str, validation_results: List[ValidationResult]):
        """Update component validation score."""
        if not validation_results:
            self.component_scores[component_id] = 1.0
            return

        # Calculate weighted score based on severity
        total_weight = 0
        weighted_sum = 0

        severity_weights = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.6,
            ValidationSeverity.CRITICAL: 1.0
        }

        for result in validation_results:
            weight = severity_weights.get(result.severity, 0.5)
            total_weight += weight
            weighted_sum += weight * (0 if result.is_valid else 1)

        # Score from 0 (worst) to 1 (best)
        score = 1.0 - (weighted_sum / max(total_weight, 1))
        self.component_scores[component_id] = max(0.0, min(1.0, score))

    def _store_validation_results(self, validation_results: List[ValidationResult]):
        """Store validation results in history."""
        self.validation_history.extend(validation_results)

        # Maintain history size limit
        if len(self.validation_history) > self._config['max_history_size']:
            self.validation_history = self.validation_history[-self._config['max_history_size'] // 2:]

    def get_validation_summary(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get validation summary for component or entire system.

        Args:
            component_id: Optional component to filter by

        Returns:
            Validation summary statistics
        """
        filtered_results = self.validation_history

        if component_id:
            filtered_results = [
                result for result in filtered_results
                if result.component_id == component_id
            ]

        if not filtered_results:
            return {
                'total_validations': 0,
                'valid_count': 0,
                'invalid_count': 0,
                'severity_breakdown': {},
                'score': 1.0
            }

        # Count by severity
        severity_counts = {}
        invalid_count = 0

        for result in filtered_results:
            severity = result.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            if not result.is_valid:
                invalid_count += 1

        # Calculate score
        if component_id:
            score = self.component_scores.get(component_id, 1.0)
        else:
            # System-wide score as average of component scores
            score = sum(self.component_scores.values()) / max(len(self.component_scores), 1)

        return {
            'total_validations': len(filtered_results),
            'valid_count': len(filtered_results) - invalid_count,
            'invalid_count': invalid_count,
            'severity_breakdown': severity_counts,
            'score': round(score, 3),
            'recent_validations': len([r for r in filtered_results
                                     if (datetime.now() - r.timestamp).total_seconds() < 3600])
        }

    def get_component_score(self, component_id: str) -> float:
        """Get validation score for specific component."""
        return self.component_scores.get(component_id, 1.0)

    def clear_validation_history(self, component_id: Optional[str] = None):
        """Clear validation history for component or all components."""
        if component_id:
            self.validation_history = [
                result for result in self.validation_history
                if result.component_id != component_id
            ]
            if component_id in self.component_scores:
                del self.component_scores[component_id]
        else:
            self.validation_history.clear()
            self.component_scores.clear()

        logger.info(f"Cleared validation history for {component_id or 'all components'}")


# X7 Compliant global validator instance
_validator_instance = None

def get_ml_state_validator() -> MLStateValidator:
    """Get global ML state validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MLStateValidator()
    return _validator_instance


if __name__ == "__main__":
    # X7 Compliant self-test when run directly
    logger.info("Running X7 Compliant ML State Validator self-test")

    # Create test validator
    validator = MLStateValidator()

    # Test component validation
    from ..state_manager import MLComponentState, ComponentState
    test_state = MLComponentState(
        component_id="test_component",
        status=ComponentState.HEALTHY,
        last_updated=datetime.now(),
        data={"test_data": "value"},
        performance_metrics={"response_time_ms": 50.0}
    )

    results = validator.validate_component_state(test_state)
    print(f"✅ Test validation: {len(results)} results")

    # Test validation summary
    summary = validator.get_validation_summary()
    print(f"✅ Validation summary: {summary}")

    print("✅ X7 Compliant ML State Validator self-test completed successfully")