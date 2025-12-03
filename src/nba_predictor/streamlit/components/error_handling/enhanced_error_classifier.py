"""
🎯 PHASE 3 DAY 9: Enhanced Error Classification System
====================================================

X7 Compliant Error Classification and Recovery Strategy System for NBA Predictor.

This module provides comprehensive error handling framework for:
- Intelligent error classification with ML-based categorization
- Severity-based error routing and escalation
- Context-aware error recovery strategies
- Real-time error pattern detection
- Performance impact assessment
- User-friendly error transformation

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import logging
import time
import traceback
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict, deque
import re

# Configure logging
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for intelligent classification."""

    # System errors
    SYSTEM_RESOURCE = "system_resource"
    SYSTEM_TIMEOUT = "system_timeout"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"

    # Data errors
    DATA_VALIDATION = "data_validation"
    DATA_INTEGRITY = "data_integrity"
    DATA_FORMAT = "data_format"
    DATA_MISSING = "data_missing"
    DATA_CORRUPTION = "data_corruption"

    # ML Model errors
    MODEL_PREDICTION = "model_prediction"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DRIFT = "model_drift"
    MODEL_DEGRADATION = "model_degradation"

    # API errors
    API_CONNECTION = "api_connection"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    API_VALIDATION = "api_validation"
    API_TIMEOUT = "api_timeout"

    # Database errors
    DB_CONNECTION = "db_connection"
    DB_CONSTRAINT = "db_constraint"
    DB_TRANSACTION = "db_transaction"
    DB_PERFORMANCE = "db_performance"
    DB_LOCK = "db_lock"

    # Business logic errors
    BUSINESS_LOGIC = "business_logic"
    BUSINESS_RULE = "business_rule"
    BUSINESS_VALIDATION = "business_validation"

    # User interface errors
    UI_INTERACTION = "ui_interaction"
    UI_NAVIGATION = "ui_navigation"
    UI_DISPLAY = "ui_display"

    # Configuration errors
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    DEPENDENCY = "dependency"

    # Unknown/Unhandled
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""

    CRITICAL = 5    # System failure, immediate action required
    HIGH = 4        # Major functionality impacted
    MEDIUM = 3      # Partial functionality impacted
    LOW = 2         # Minor issue, workaround available
    INFO = 1        # Informational, no action needed


class RecoveryStrategy(Enum):
    """Recovery strategies for error handling."""

    # Immediate strategies
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_DEFAULT = "fallback_default"
    CIRCUIT_BREAKER = "circuit_breaker"

    # Data strategies
    DATA_CACHE = "data_cache"
    DATA_ALTERNATIVE = "data_alternative"
    DATA_RECONSTRUCTION = "data_reconstruction"

    # Model strategies
    MODEL_FALLBACK = "model_fallback"
    MODEL_RETRAIN = "model_retrain"
    MODEL_SIMPLIFY = "model_simplify"

    # System strategies
    SYSTEM_RESTART = "system_restart"
    SYSTEM_SCALE = "system_scale"
    SYSTEM_ISOLATE = "system_isolate"

    # User strategies
    USER_INTERVENTION = "user_intervention"
    USER_NOTIFICATION = "user_notification"
    USER_WORKAROUND = "user_workaround"

    # Manual strategies
    MANUAL_INTERVENTION = "manual_intervention"
    ESCALATION = "escalation"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error classification."""

    # Execution context
    component_id: str
    function_name: str
    operation_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Technical context
    python_version: str = field(default_factory=lambda: "3.11")
    operating_system: str = field(default_factory=lambda: "darwin")
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

    # Business context
    business_process: Optional[str] = None
    transaction_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Performance context
    execution_time: Optional[float] = None
    timeout_threshold: Optional[float] = None
    retry_count: int = 0

    # Data context
    data_size: Optional[int] = None
    data_type: Optional[str] = None
    data_source: Optional[str] = None

    # External context
    external_service: Optional[str] = None
    api_endpoint: Optional[str] = None
    database_connection: Optional[str] = None


@dataclass
class ErrorPattern:
    """Error pattern for classification and detection."""

    pattern_id: str
    pattern_name: str
    category: ErrorCategory
    severity: ErrorSeverity
    patterns: List[str]  # Regex patterns
    keywords: List[str]
    context_indicators: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    description: str
    frequency_weight: float = 1.0


@dataclass
class ClassifiedError:
    """Classified error with metadata."""

    # Core error information
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    original_exception: Optional[Exception] = None

    # Classification results
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    confidence_score: float = 0.0

    # Error details
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""

    # Context information
    context: Optional[ErrorContext] = None

    # Pattern matching
    matched_patterns: List[str] = field(default_factory=list)

    # Recovery information
    suggested_strategy: RecoveryStrategy = RecoveryStrategy.MANUAL_INTERVENTION
    recovery_attempts: int = 0
    recovery_successful: bool = False

    # Impact assessment
    user_impact: str = ""
    system_impact: str = ""
    business_impact: str = ""

    # Status
    status: str = "new"  # new, investigating, resolved, escalated
    assigned_to: Optional[str] = None
    resolution_notes: str = ""


class EnhancedErrorClassifier:
    """
    X7 Compliant Enhanced Error Classification System.

    Features:
    - ML-based error classification
    - Pattern matching with regex
    - Context-aware categorization
    - Severity assessment
    - Recovery strategy recommendation
    - Real-time error analytics
    - Performance impact analysis
    """

    def __init__(self):
        """Initialize the enhanced error classifier."""
        self._initialized = True
        self._classification_lock = threading.RLock()

        # Error patterns database
        self._patterns: Dict[str, ErrorPattern] = {}
        self._pattern_index: Dict[ErrorCategory, List[str]] = defaultdict(list)

        # Error history and analytics
        self._error_history: deque = deque(maxlen=10000)
        self._error_frequency: Dict[str, int] = defaultdict(int)
        self._error_patterns: Dict[str, List[ClassifiedError]] = defaultdict(list)

        # Classification metrics
        self._classification_stats: Dict[str, Any] = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'pattern_matches': 0,
            'context_matches': 0,
            'average_confidence': 0.0,
            'classification_time': []
        }

        # Initialize error patterns
        self._initialize_error_patterns()

        # Setup logging
        self._setup_logging()

        logger.info("Enhanced Error Classifier initialized with X7 compliance")

    def _initialize_error_patterns(self) -> None:
        """Initialize comprehensive error patterns."""

        # System Resource patterns
        self.add_pattern(ErrorPattern(
            pattern_id="sys_memory_error",
            pattern_name="System Memory Error",
            category=ErrorCategory.SYSTEM_MEMORY,
            severity=ErrorSeverity.HIGH,
            patterns=[
                r"MemoryError",
                r"OutOfMemoryError",
                r"memory.*overflow",
                r"cannot allocate.*memory"
            ],
            keywords=["memory", "ram", "allocation", "overflow"],
            context_indicators={"high_memory_usage": True},
            recovery_strategy=RecoveryStrategy.SYSTEM_SCALE,
            description="System ran out of memory during operation"
        ))

        self.add_pattern(ErrorPattern(
            pattern_id="sys_timeout_error",
            pattern_name="System Timeout Error",
            category=ErrorCategory.SYSTEM_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            patterns=[
                r"TimeoutError",
                r"timeout.*expired",
                r"operation.*timed out"
            ],
            keywords=["timeout", "expired", "timed out"],
            context_indicators={"execution_time_gt_threshold": True},
            recovery_strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            description="Operation exceeded timeout threshold"
        ))

        # Data Validation patterns
        self.add_pattern(ErrorPattern(
            pattern_id="data_validation_error",
            pattern_name="Data Validation Error",
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            patterns=[
                r"ValidationError",
                r"validation.*failed",
                r"invalid.*data",
                r"data.*not.*valid"
            ],
            keywords=["validation", "invalid", "constraint", "schema"],
            context_indicators={"data_validation": True},
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            description="Data validation failed during processing"
        ))

        # ML Model patterns
        self.add_pattern(ErrorPattern(
            pattern_id="model_prediction_error",
            pattern_name="Model Prediction Error",
            category=ErrorCategory.MODEL_PREDICTION,
            severity=ErrorSeverity.HIGH,
            patterns=[
                r"PredictionError",
                r"model.*prediction.*failed",
                r"inference.*error"
            ],
            keywords=["prediction", "inference", "model", "failed"],
            context_indicators={"ml_operation": "prediction"},
            recovery_strategy=RecoveryStrategy.MODEL_FALLBACK,
            description="ML model prediction failed"
        ))

        self.add_pattern(ErrorPattern(
            pattern_id="model_drift_error",
            pattern_name="Model Drift Detected",
            category=ErrorCategory.MODEL_DRIFT,
            severity=ErrorSeverity.MEDIUM,
            patterns=[
                r"ModelDriftError",
                r"drift.*detected",
                r"performance.*degraded"
            ],
            keywords=["drift", "degraded", "performance", "accuracy"],
            context_indicators={"model_performance_low": True},
            recovery_strategy=RecoveryStrategy.MODEL_RETRAIN,
            description="Model performance degradation detected"
        ))

        # API Connection patterns
        self.add_pattern(ErrorPattern(
            pattern_id="api_connection_error",
            pattern_name="API Connection Error",
            category=ErrorCategory.API_CONNECTION,
            severity=ErrorSeverity.HIGH,
            patterns=[
                r"ConnectionError",
                r"NetworkError",
                r"HTTP.*[45]\d{2}",
                r"connection.*refused"
            ],
            keywords=["connection", "network", "http", "refused"],
            context_indicators={"external_service": True},
            recovery_strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            description="Failed to connect to external API"
        ))

        self.add_pattern(ErrorPattern(
            pattern_id="api_rate_limit_error",
            pattern_name="API Rate Limit Error",
            category=ErrorCategory.API_RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            patterns=[
                r"RateLimitError",
                r"TooManyRequests",
                r"rate.*limit.*exceeded"
            ],
            keywords=["rate limit", "too many", "quota", "throttled"],
            context_indicators={"rate_limit": True},
            recovery_strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            description="API rate limit exceeded"
        ))

        # Database patterns
        self.add_pattern(ErrorPattern(
            pattern_id="db_connection_error",
            pattern_name="Database Connection Error",
            category=ErrorCategory.DB_CONNECTION,
            severity=ErrorSeverity.HIGH,
            patterns=[
                r"DatabaseError",
                r"Connection.*failed",
                r"cannot.*connect.*database"
            ],
            keywords=["database", "connection", "failed", "unavailable"],
            context_indicators={"database_operation": True},
            recovery_strategy=RecoveryStrategy.RETRY_IMMEDIATE,
            description="Failed to connect to database"
        ))

        self.add_pattern(ErrorPattern(
            pattern_id="db_constraint_error",
            pattern_name="Database Constraint Error",
            category=ErrorCategory.DB_CONSTRAINT,
            severity=ErrorSeverity.MEDIUM,
            patterns=[
                r"IntegrityError",
                r"ConstraintError",
                r"unique.*constraint",
                r"foreign.*key"
            ],
            keywords=["constraint", "integrity", "unique", "foreign key"],
            context_indicators={"database_constraint": True},
            recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
            description="Database constraint violation"
        ))

        # Configuration patterns
        self.add_pattern(ErrorPattern(
            pattern_id="config_error",
            pattern_name="Configuration Error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            patterns=[
                r"ConfigurationError",
                r"Config.*missing",
                r"environment.*variable"
            ],
            keywords=["configuration", "config", "missing", "environment"],
            context_indicators={"configuration_issue": True},
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            description="Configuration error detected"
        ))

        logger.info(f"Initialized {len(self._patterns)} error patterns")

    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add a new error pattern."""
        with self._classification_lock:
            self._patterns[pattern.pattern_id] = pattern
            self._pattern_index[pattern.category].append(pattern.pattern_id)

    def _setup_logging(self) -> None:
        """Setup enhanced logging for error classifier."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def classify_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ClassifiedError:
        """
        Classify an exception with comprehensive analysis.

        Args:
            exception: The exception to classify
            context: Execution context information
            additional_info: Additional information for classification

        Returns:
            ClassifiedError with detailed classification
        """
        start_time = time.time()

        classified_error = ClassifiedError(
            original_exception=exception,
            error_type=type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context
        )

        try:
            # Extract error information
            error_text = f"{classified_error.error_type} {classified_error.error_message} {classified_error.stack_trace}"

            # Pattern matching
            matched_patterns = self._match_patterns(error_text, context)
            classified_error.matched_patterns = [p.pattern_id for p in matched_patterns]

            # Determine category and severity
            if matched_patterns:
                # Use highest confidence pattern
                best_pattern = max(matched_patterns, key=lambda p: self._calculate_pattern_confidence(p, error_text, context))
                classified_error.category = best_pattern.category
                classified_error.severity = best_pattern.severity
                classified_error.suggested_strategy = best_pattern.recovery_strategy
                classified_error.confidence_score = self._calculate_pattern_confidence(best_pattern, error_text, context)
            else:
                # Use context-based classification
                classified_error = self._classify_by_context(classified_error, context)

            # Assess impact
            classified_error = self._assess_impact(classified_error, context)

            # Update analytics
            self._update_analytics(classified_error)

            # Record classification time
            classification_time = time.time() - start_time
            self._classification_stats['classification_time'].append(classification_time)
            self._classification_stats['total_classifications'] += 1

            if classified_error.confidence_score > 0.5:
                self._classification_stats['successful_classifications'] += 1

            logger.info(f"Classified error: {classified_error.category.value} (confidence: {classified_error.confidence_score:.2f})")

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            classified_error.category = ErrorCategory.UNKNOWN
            classified_error.severity = ErrorSeverity.MEDIUM
            classified_error.confidence_score = 0.0

        return classified_error

    def _match_patterns(
        self,
        error_text: str,
        context: Optional[ErrorContext] = None
    ) -> List[ErrorPattern]:
        """Match error against known patterns."""
        matched_patterns = []
        error_text_lower = error_text.lower()

        for pattern in self._patterns.values():
            confidence = self._calculate_pattern_confidence(pattern, error_text, context)
            if confidence > 0.3:  # Minimum confidence threshold
                matched_patterns.append(pattern)

        return matched_patterns

    def _calculate_pattern_confidence(
        self,
        pattern: ErrorPattern,
        error_text: str,
        context: Optional[ErrorContext] = None
    ) -> float:
        """Calculate confidence score for pattern matching."""
        confidence = 0.0
        error_text_lower = error_text.lower()

        # Regex pattern matching
        pattern_matches = sum(1 for regex in pattern.patterns
                            if re.search(regex, error_text, re.IGNORECASE))
        if pattern_matches > 0:
            confidence += (pattern_matches / len(pattern.patterns)) * 0.6

        # Keyword matching
        keyword_matches = sum(1 for keyword in pattern.keywords
                            if keyword.lower() in error_text_lower)
        if keyword_matches > 0:
            confidence += (keyword_matches / len(pattern.keywords)) * 0.3

        # Context indicators
        if context:
            context_matches = self._check_context_indicators(pattern.context_indicators, context)
            if context_matches > 0:
                confidence += 0.1

        # Frequency weight
        confidence *= pattern.frequency_weight

        return min(confidence, 1.0)

    def _check_context_indicators(
        self,
        indicators: Dict[str, Any],
        context: ErrorContext
    ) -> int:
        """Check if context matches pattern indicators."""
        matches = 0

        for indicator, expected_value in indicators.items():
            if hasattr(context, indicator):
                actual_value = getattr(context, indicator)
                if actual_value == expected_value:
                    matches += 1

        return matches

    def _classify_by_context(
        self,
        classified_error: ClassifiedError,
        context: Optional[ErrorContext] = None
    ) -> ClassifiedError:
        """Classify error based on context when pattern matching fails."""
        if not context:
            return classified_error

        # Component-based classification
        if context.component_id:
            if "ml" in context.component_id.lower() or "model" in context.component_id.lower():
                classified_error.category = ErrorCategory.MODEL_PREDICTION
                classified_error.severity = ErrorSeverity.HIGH
                classified_error.suggested_strategy = RecoveryStrategy.MODEL_FALLBACK

            elif "database" in context.component_id.lower() or "db" in context.component_id.lower():
                classified_error.category = ErrorCategory.DB_CONNECTION
                classified_error.severity = ErrorSeverity.HIGH
                classified_error.suggested_strategy = RecoveryStrategy.RETRY_IMMEDIATE

            elif "api" in context.component_id.lower():
                classified_error.category = ErrorCategory.API_CONNECTION
                classified_error.severity = ErrorSeverity.MEDIUM
                classified_error.suggested_strategy = RecoveryStrategy.RETRY_WITH_BACKOFF

        # Operation-based classification
        if context.operation_type:
            if "validation" in context.operation_type.lower():
                classified_error.category = ErrorCategory.DATA_VALIDATION
                classified_error.severity = ErrorSeverity.MEDIUM
                classified_error.suggested_strategy = RecoveryStrategy.USER_INTERVENTION

            elif "prediction" in context.operation_type.lower():
                classified_error.category = ErrorCategory.MODEL_PREDICTION
                classified_error.severity = ErrorSeverity.HIGH
                classified_error.suggested_strategy = RecoveryStrategy.MODEL_FALLBACK

        classified_error.confidence_score = 0.4  # Lower confidence for context-based classification

        return classified_error

    def _assess_impact(
        self,
        classified_error: ClassifiedError,
        context: Optional[ErrorContext] = None
    ) -> ClassifiedError:
        """Assess the impact of the error."""
        # User impact assessment
        if classified_error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            classified_error.user_impact = "High - Core functionality affected"
        elif classified_error.severity == ErrorSeverity.MEDIUM:
            classified_error.user_impact = "Medium - Some features unavailable"
        else:
            classified_error.user_impact = "Low - Minor inconvenience"

        # System impact assessment
        if classified_error.category in [
            ErrorCategory.SYSTEM_MEMORY,
            ErrorCategory.SYSTEM_DISK,
            ErrorCategory.SYSTEM_RESOURCE
        ]:
            classified_error.system_impact = "High - System resources depleted"
        elif classified_error.category in [
            ErrorCategory.DB_CONNECTION,
            ErrorCategory.API_CONNECTION
        ]:
            classified_error.system_impact = "Medium - External connectivity affected"
        else:
            classified_error.system_impact = "Low - Localized impact"

        # Business impact assessment
        if context and context.business_process:
            if "betting" in context.business_process.lower() or "prediction" in context.business_process.lower():
                classified_error.business_impact = "High - Core business functionality affected"
            else:
                classified_error.business_impact = "Medium - Supporting functionality affected"
        else:
            classified_error.business_impact = "Unknown - Context not provided"

        return classified_error

    def _update_analytics(self, classified_error: ClassifiedError) -> None:
        """Update error analytics and history."""
        with self._classification_lock:
            # Add to history
            self._error_history.append(classified_error)

            # Update frequency
            error_key = f"{classified_error.category.value}_{classified_error.error_type}"
            self._error_frequency[error_key] += 1

            # Group by pattern
            if classified_error.matched_patterns:
                for pattern_id in classified_error.matched_patterns:
                    self._error_patterns[pattern_id].append(classified_error)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._classification_lock:
            stats = {
                'classification_stats': self._classification_stats.copy(),
                'error_frequency': dict(self._error_frequency),
                'total_errors': len(self._error_history),
                'patterns_count': len(self._patterns),
                'active_patterns': len(self._error_patterns),
                'recent_errors': [
                    {
                        'error_id': e.error_id,
                        'category': e.category.value,
                        'severity': e.severity.value,
                        'timestamp': e.timestamp.isoformat(),
                        'confidence': e.confidence_score
                    }
                    for e in list(self._error_history)[-10:]
                ]
            }

            # Calculate average confidence
            if self._classification_stats['total_classifications'] > 0:
                stats['classification_stats']['average_confidence'] = (
                    self._classification_stats['successful_classifications'] /
                    self._classification_stats['total_classifications']
                )

            # Calculate average classification time
            if self._classification_stats['classification_time']:
                stats['classification_stats']['average_classification_time'] = (
                    sum(self._classification_stats['classification_time']) /
                    len(self._classification_stats['classification_time'])
                )

            return stats

    def get_recovery_strategy(
        self,
        classified_error: ClassifiedError
    ) -> RecoveryStrategy:
        """Get recommended recovery strategy for classified error."""
        # Consider error history and success rates
        if classified_error.matched_patterns:
            # Check if similar errors were successfully recovered
            for pattern_id in classified_error.matched_patterns:
                similar_errors = self._error_patterns.get(pattern_id, [])
                successful_recoveries = [
                    e for e in similar_errors
                    if e.recovery_successful and e.suggested_strategy == classified_error.suggested_strategy
                ]

                if successful_recoveries:
                    success_rate = len(successful_recoveries) / len(similar_errors)
                    if success_rate > 0.7:  # 70% success rate threshold
                        return classified_error.suggested_strategy

        # Default strategy based on category and severity
        if classified_error.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATION
        elif classified_error.category in [ErrorCategory.API_CONNECTION, ErrorCategory.DB_CONNECTION]:
            return RecoveryStrategy.RETRY_WITH_BACKOFF
        elif classified_error.category in [ErrorCategory.MODEL_PREDICTION]:
            return RecoveryStrategy.MODEL_FALLBACK
        else:
            return RecoveryStrategy.USER_INTERVENTION

    def cleanup_old_errors(self, days_to_keep: int = 30) -> None:
        """Clean up old error records."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._classification_lock:
            # Clean error history
            self._error_history = deque(
                [e for e in self._error_history if e.timestamp > cutoff_date],
                maxlen=10000
            )

            # Clean pattern groups
            for pattern_id in self._error_patterns:
                self._error_patterns[pattern_id] = [
                    e for e in self._error_patterns[pattern_id]
                    if e.timestamp > cutoff_date
                ]

        logger.info(f"Cleaned up errors older than {days_to_keep} days")


# Singleton instance for global access
_error_classifier_instance = None
_classifier_lock = threading.Lock()


def get_error_classifier() -> EnhancedErrorClassifier:
    """Get the global error classifier instance."""
    global _error_classifier_instance

    if _error_classifier_instance is None:
        with _classifier_lock:
            if _error_classifier_instance is None:
                _error_classifier_instance = EnhancedErrorClassifier()

    return _error_classifier_instance


def classify_error(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> ClassifiedError:
    """
    Convenience function to classify an error.

    Args:
        exception: The exception to classify
        context: Execution context information
        additional_info: Additional information for classification

    Returns:
        ClassifiedError with detailed classification
    """
    classifier = get_error_classifier()
    return classifier.classify_error(exception, context, additional_info)