"""
🎯 PHASE 3 DAY 9: Error Handling Module
========================================

X7 Compliant Error Handling and Recovery System for NBA Predictor.

This module provides comprehensive error handling frameworks for:
- Enhanced error classification with ML-based categorization
- Intelligent retry logic with exponential backoff
- User-friendly error message transformation
- Error reporting and analytics
- Recovery strategy automation

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

from .enhanced_error_classifier import (
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    ErrorPattern,
    ClassifiedError,
    EnhancedErrorClassifier,
    get_error_classifier,
    classify_error
)

from .retry_manager import (
    BackoffStrategy,
    RetryDecision,
    RetryPolicy,
    RetryAttempt,
    RetrySession,
    CircuitBreakerState,
    RetryManager,
    get_retry_manager,
    retry
)

from .error_message_formatter import (
    MessageTone,
    MessageComplexity,
    AudienceType,
    MessageTemplate,
    FormattedErrorMessage,
    ErrorMessageFormatter,
    get_error_message_formatter,
    format_error_message
)

from .error_reporter import (
    ReportingPeriod,
    AlertSeverity,
    ErrorEvent,
    ErrorAggregation,
    Alert,
    ErrorReporter,
    get_error_reporter,
    report_error
)

from .state_integration import (
    StateErrorSyncStatus,
    StateErrorContext,
    ErrorStateRecoveryPlan,
    StateAwareErrorHandler,
    get_state_aware_error_handler,
    handle_error_with_state,
    execute_with_state_retry
)

__all__ = [
    'ErrorCategory',
    'ErrorSeverity',
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorPattern',
    'ClassifiedError',
    'EnhancedErrorClassifier',
    'get_error_classifier',
    'classify_error',
    'BackoffStrategy',
    'RetryDecision',
    'RetryPolicy',
    'RetryAttempt',
    'RetrySession',
    'CircuitBreakerState',
    'RetryManager',
    'get_retry_manager',
    'retry',
    'MessageTone',
    'MessageComplexity',
    'AudienceType',
    'MessageTemplate',
    'FormattedErrorMessage',
    'ErrorMessageFormatter',
    'get_error_message_formatter',
    'format_error_message',
    'ReportingPeriod',
    'AlertSeverity',
    'ErrorEvent',
    'ErrorAggregation',
    'Alert',
    'ErrorReporter',
    'get_error_reporter',
    'report_error',
    'StateErrorSyncStatus',
    'StateErrorContext',
    'ErrorStateRecoveryPlan',
    'StateAwareErrorHandler',
    'get_state_aware_error_handler',
    'handle_error_with_state',
    'execute_with_state_retry'
]

__version__ = "1.0.0"
__author__ = "DevStream SuperPowered"