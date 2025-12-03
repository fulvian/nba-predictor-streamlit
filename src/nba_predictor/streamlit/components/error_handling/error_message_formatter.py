"""
🎯 PHASE 3 DAY 9: User-Friendly Error Messages
==============================================

X7 Compliant Error Message Transformation System for NBA Predictor.

This module provides comprehensive error message formatting for:
- User-friendly error message transformation
- Contextual message adaptation
- Multi-language support infrastructure
- Actionable error guidance
- Progressive disclosure of technical details
- Brand-consistent error communication

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import logging
import re
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading

# Enhanced Error Classifier imports
from .enhanced_error_classifier import (
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    ClassifiedError,
    get_error_classifier
)

# Configure logging
logger = logging.getLogger(__name__)


class MessageTone(Enum):
    """Error message tone levels."""

    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    URGENT = "urgent"
    REASSURING = "reassuring"


class MessageComplexity(Enum):
    """Error message complexity levels."""

    SIMPLE = "simple"  # Basic explanation, no technical details
    STANDARD = "standard"  # Clear explanation with some context
    DETAILED = "detailed"  # Full explanation with technical context
    COMPREHENSIVE = "comprehensive"  # Everything including debugging info


class AudienceType(Enum):
    """Target audience for error messages."""

    END_USER = "end_user"  # Non-technical users
    BUSINESS_USER = "business_user"  # Domain experts, non-technical
    TECHNICAL_USER = "technical_user"  # Developers, IT staff
    SYSTEM_ADMIN = "system_admin"  # Operations, DevOps
    SUPPORT_STAFF = "support_staff"  # Customer support


@dataclass
class MessageTemplate:
    """Template for error message formatting."""

    template_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    audience: AudienceType
    tone: MessageTone
    complexity: MessageComplexity

    # Message components
    title_template: str
    message_template: str
    action_template: Optional[str] = None
    technical_template: Optional[str] = None

    # Localization support
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Metadata
    variables: List[str] = field(default_factory=list)
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=list)
    requires_user_action: bool = False


@dataclass
class FormattedErrorMessage:
    """Complete formatted error message."""

    # Core message
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    action: Optional[str] = None
    technical_details: Optional[str] = None

    # Metadata
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    audience: AudienceType = AudienceType.END_USER
    tone: MessageTone = MessageTone.FRIENDLY
    complexity: MessageComplexity = MessageComplexity.STANDARD

    # Formatting information
    template_id: Optional[str] = None
    variables_used: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    language: str = "en"

    # Interactive elements
    suggested_actions: List[str] = field(default_factory=list)
    recovery_options: List[str] = field(default_factory=list)
    help_links: List[Dict[str, str]] = field(default_factory=list)

    # Progress tracking
    can_retry: bool = False
    can_continue: bool = False
    requires_support: bool = False


class ErrorMessageFormatter:
    """
    X7 Compliant User-Friendly Error Message Formatter.

    Features:
    - Intelligent error message transformation
    - Contextual message adaptation
    - Multi-audience targeting
    - Progressive disclosure of technical details
    - Actionable guidance and next steps
    - Brand-consistent communication
    - Localization-ready architecture
    """

    def __init__(self):
        """Initialize the error message formatter."""
        self._initialized = True
        self._formatting_lock = threading.RLock()

        # Message templates
        self._templates: Dict[str, MessageTemplate] = {}
        self._category_templates: Dict[ErrorCategory, List[str]] = {}

        # Message formatting history and analytics
        self._formatting_history: List[FormattedErrorMessage] = []
        self._template_usage: Dict[str, int] = {}
        self._audience_stats: Dict[AudienceType, int] = {}

        # Caching for performance
        self._template_cache: Dict[str, MessageTemplate] = {}

        # Initialize default templates
        self._initialize_default_templates()

        # Setup logging
        self._setup_logging()

        logger.info("Error Message Formatter initialized with X7 compliance")

    def _initialize_default_templates(self) -> None:
        """Initialize comprehensive set of default message templates."""

        # System Error Templates
        self.add_template(MessageTemplate(
            template_id="system_memory_friendly",
            category=ErrorCategory.SYSTEM_MEMORY,
            severity=ErrorSeverity.HIGH,
            audience=AudienceType.END_USER,
            tone=MessageTone.FRIENDLY,
            complexity=MessageComplexity.SIMPLE,
            title_template="⚠️ System Resources Low",
            message_template="The system is running low on memory and needs a moment to catch up. Please try your action again in a few moments.",
            action_template="🔄 Try your action again",
            variables=["memory_usage"],
            recovery_strategies=[RecoveryStrategy.SYSTEM_SCALE],
            requires_user_action=True
        ))

        self.add_template(MessageTemplate(
            template_id="system_timeout_technical",
            category=ErrorCategory.SYSTEM_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.TECHNICAL_USER,
            tone=MessageTone.PROFESSIONAL,
            complexity=MessageComplexity.DETAILED,
            title_template="⏱️ Operation Timeout",
            message_template="The operation exceeded the timeout threshold of {timeout_threshold}s. Execution time: {execution_time}s.",
            action_template="🔧 Check system resources and retry",
            technical_template="Stack trace: {stack_trace}\nTimeout settings: {timeout_config}",
            variables=["timeout_threshold", "execution_time", "stack_trace", "timeout_config"],
            recovery_strategies=[RecoveryStrategy.RETRY_WITH_BACKOFF],
            requires_user_action=True
        ))

        # API Error Templates
        self.add_template(MessageTemplate(
            template_id="api_connection_user",
            category=ErrorCategory.API_CONNECTION,
            severity=ErrorSeverity.HIGH,
            audience=AudienceType.END_USER,
            tone=MessageTone.REASSURING,
            complexity=MessageComplexity.SIMPLE,
            title_template="🌐 Connection Issues",
            message_template="We're having trouble connecting to our services. Don't worry - your data is safe and we're working on it!",
            action_template="🔄 Try again in a moment",
            variables=["service_name"],
            recovery_strategies=[RecoveryStrategy.RETRY_WITH_BACKOFF],
            requires_user_action=True
        ))

        self.add_template(MessageTemplate(
            template_id="api_rate_limit_business",
            category=ErrorCategory.API_RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.BUSINESS_USER,
            tone=MessageTone.PROFESSIONAL,
            complexity=MessageComplexity.STANDARD,
            title_template="📊 Rate Limit Reached",
            message_template="You've reached the maximum number of requests for {service_name}. This helps us maintain performance for all users.",
            action_template="⏰ Wait {retry_after} seconds and try again",
            technical_template="Rate limit: {requests_per_minute} requests per minute. Reset at: {reset_time}",
            variables=["service_name", "retry_after", "requests_per_minute", "reset_time"],
            recovery_strategies=[RecoveryStrategy.RETRY_WITH_BACKOFF],
            requires_user_action=True
        ))

        # Database Error Templates
        self.add_template(MessageTemplate(
            template_id="db_connection_admin",
            category=ErrorCategory.DB_CONNECTION,
            severity=ErrorSeverity.HIGH,
            audience=AudienceType.SYSTEM_ADMIN,
            tone=MessageTone.URGENT,
            complexity=MessageComplexity.COMPREHENSIVE,
            title_template="🚨 Database Connection Failure",
            message_template="Critical database connection issue detected. Multiple connection attempts failed. System health: {system_health}%.",
            action_template="🔧 Immediate investigation required",
            technical_template="Connection string: {connection_string}\nPool status: {pool_status}\nLast error: {last_error}",
            variables=["connection_string", "pool_status", "last_error", "system_health"],
            recovery_strategies=[RecoveryStrategy.ESCALATION],
            requires_user_action=True
        ))

        self.add_template(MessageTemplate(
            template_id="db_constraint_user",
            category=ErrorCategory.DB_CONSTRAINT,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.END_USER,
            tone=MessageTone.FRIENDLY,
            complexity=MessageComplexity.STANDARD,
            title_template="📋 Data Validation",
            message_template="We couldn't save your information because some details don't match our requirements. Please review and try again.",
            action_template="✏️ Check your input and submit again",
            variables=["field_name"],
            recovery_strategies=[RecoveryStrategy.USER_INTERVENTION],
            requires_user_action=True
        ))

        # ML Model Error Templates
        self.add_template(MessageTemplate(
            template_id="ml_prediction_business",
            category=ErrorCategory.MODEL_PREDICTION,
            severity=ErrorSeverity.HIGH,
            audience=AudienceType.BUSINESS_USER,
            tone=MessageTone.REASSURING,
            complexity=MessageComplexity.STANDARD,
            title_template="🤖 Prediction Service Unavailable",
            message_template="Our prediction system is temporarily unavailable. We're using fallback analysis to maintain service continuity.",
            action_template="📊 View alternative predictions",
            variables=["model_name", "fallback_status"],
            recovery_strategies=[RecoveryStrategy.MODEL_FALLBACK]
        ))

        self.add_template(MessageTemplate(
            template_id="ml_drift_technical",
            category=ErrorCategory.MODEL_DRIFT,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.TECHNICAL_USER,
            tone=MessageTone.PROFESSIONAL,
            complexity=MessageComplexity.DETAILED,
            title_template="📈 Model Performance Drift",
            message_template="Model {model_name} shows performance degradation (accuracy: {current_accuracy}% vs baseline: {baseline_accuracy}%). Drift analysis in progress.",
            action_template="🔧 Model retraining recommended",
            technical_template="Drift metrics: {drift_metrics}\nRetraining threshold: {retrain_threshold}",
            variables=["model_name", "current_accuracy", "baseline_accuracy", "drift_metrics", "retrain_threshold"],
            recovery_strategies=[RecoveryStrategy.MODEL_RETRAIN],
            requires_user_action=True
        ))

        # Data Error Templates
        self.add_template(MessageTemplate(
            template_id="data_validation_user",
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.END_USER,
            tone=MessageTone.FRIENDLY,
            complexity=MessageComplexity.SIMPLE,
            title_template="✅ Data Check Needed",
            message_template="We need to verify some information before proceeding. Please check your input and try again.",
            action_template="📝 Review and correct the highlighted fields",
            variables=["validation_errors"],
            recovery_strategies=[RecoveryStrategy.USER_INTERVENTION],
            requires_user_action=True
        ))

        # Business Logic Error Templates
        self.add_template(MessageTemplate(
            template_id="business_logic_user",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.BUSINESS_USER,
            tone=MessageTone.PROFESSIONAL,
            complexity=MessageComplexity.STANDARD,
            title_template="📊 Business Rule Validation",
            message_template="This action doesn't comply with current business rules: {rule_description}. Please adjust your approach.",
            action_template="🔄 Review business requirements",
            variables=["rule_name", "rule_description"],
            recovery_strategies=[RecoveryStrategy.USER_WORKAROUND],
            requires_user_action=True
        ))

        # Configuration Error Templates
        self.add_template(MessageTemplate(
            template_id="config_system_admin",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            audience=AudienceType.SYSTEM_ADMIN,
            tone=MessageTone.URGENT,
            complexity=MessageComplexity.COMPREHENSIVE,
            title_template="⚙️ Configuration Error",
            message_template="Critical configuration issue detected in {config_section}. System may not function correctly until resolved.",
            action_template="🔧 Review configuration settings",
            technical_template="Config file: {config_file}\nInvalid setting: {invalid_setting}\nCurrent value: {current_value}",
            variables=["config_section", "config_file", "invalid_setting", "current_value"],
            recovery_strategies=[RecoveryStrategy.MANUAL_INTERVENTION],
            requires_user_action=True
        ))

        # Default fallback template
        self.add_template(MessageTemplate(
            template_id="default_friendly",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            audience=AudienceType.END_USER,
            tone=MessageTone.FRIENDLY,
            complexity=MessageComplexity.STANDARD,
            title_template="🤔 Something Unexpected Happened",
            message_template="We encountered an unexpected issue. Our team has been notified and we're working on a resolution.",
            action_template="🔄 Try your action again or contact support",
            recovery_strategies=[RecoveryStrategy.USER_WORKAROUND]
        ))

        logger.info(f"Initialized {len(self._templates)} error message templates")

    def add_template(self, template: MessageTemplate) -> None:
        """Add a message template."""
        with self._formatting_lock:
            self._templates[template.template_id] = template

            # Update category index
            if template.category not in self._category_templates:
                self._category_templates[template.category] = []
            self._category_templates[template.category].append(template.template_id)

    def _setup_logging(self) -> None:
        """Setup enhanced logging for message formatter."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def format_error(
        self,
        classified_error: ClassifiedError,
        audience: AudienceType = AudienceType.END_USER,
        tone: MessageTone = MessageTone.FRIENDLY,
        complexity: MessageComplexity = MessageComplexity.STANDARD,
        language: str = "en",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> FormattedErrorMessage:
        """
        Format classified error into user-friendly message.

        Args:
            classified_error: Classified error from Enhanced Error Classifier
            audience: Target audience for the message
            tone: Desired message tone
            complexity: Message complexity level
            language: Language code (future localization)
            additional_context: Additional context for formatting

        Returns:
            FormattedErrorMessage with user-friendly content
        """
        start_time = datetime.now()

        # Find best matching template
        template = self._find_best_template(
            classified_error.category,
            classified_error.severity,
            audience,
            tone,
            complexity
        )

        # Prepare variables for template substitution
        variables = self._prepare_variables(classified_error, additional_context)

        # Format message components
        title = self._format_template(template.title_template, variables)
        message = self._format_template(template.message_template, variables)
        action = None
        technical_details = None

        if template.action_template:
            action = self._format_template(template.action_template, variables)

        if template.technical_template and complexity in [MessageComplexity.DETAILED, MessageComplexity.COMPREHENSIVE]:
            technical_details = self._format_template(template.technical_template, variables)

        # Create formatted message
        formatted_message = FormattedErrorMessage(
            title=title,
            message=message,
            action=action,
            technical_details=technical_details,
            category=classified_error.category,
            severity=classified_error.severity,
            audience=audience,
            tone=tone,
            complexity=complexity,
            template_id=template.template_id,
            variables_used=variables,
            language=language,
            generated_at=start_time
        )

        # Add interactive elements
        formatted_message = self._add_interactive_elements(formatted_message, template, classified_error)

        # Update analytics
        self._update_analytics(formatted_message)

        logger.info(f"Formatted error message for {audience.value}: {template.template_id}")
        return formatted_message

    def _find_best_template(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        audience: AudienceType,
        tone: MessageTone,
        complexity: MessageComplexity
    ) -> MessageTemplate:
        """Find best matching template for error characteristics."""
        # Direct match search
        best_template = None
        best_score = 0

        for template in self._templates.values():
            score = 0

            # Category match (highest weight)
            if template.category == category:
                score += 10

            # Severity match
            if template.severity == severity:
                score += 5

            # Audience match
            if template.audience == audience:
                score += 8

            # Tone match
            if template.tone == tone:
                score += 3

            # Complexity match
            if template.complexity == complexity:
                score += 2

            if score > best_score:
                best_score = score
                best_template = template

        # Fallback to default template if no match found
        if best_template is None:
            best_template = self._templates.get("default_friendly")
            if best_template is None:
                # Create emergency fallback template
                best_template = MessageTemplate(
                    template_id="emergency_fallback",
                    category=ErrorCategory.UNKNOWN,
                    severity=ErrorSeverity.MEDIUM,
                    audience=audience,
                    tone=tone,
                    complexity=complexity,
                    title_template="⚠️ Error",
                    message_template="An error occurred. Please try again.",
                    action_template="🔄 Retry action"
                )

        return best_template

    def _prepare_variables(
        self,
        classified_error: ClassifiedError,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare variables for template substitution."""
        variables = {}

        # Core error information
        variables.update({
            'error_type': classified_error.error_type,
            'error_message': classified_error.error_message,
            'category': classified_error.category.value,
            'severity': classified_error.severity.value,
            'confidence': f"{classified_error.confidence_score:.1%}"
        })

        # Context information
        if classified_error.context:
            ctx = classified_error.context
            variables.update({
                'component_id': ctx.component_id,
                'function_name': ctx.function_name,
                'operation_type': ctx.operation_type,
                'execution_time': ctx.execution_time,
                'retry_count': ctx.retry_count,
                'timeout_threshold': ctx.timeout_threshold,
                'business_process': ctx.business_process,
                'user_id': ctx.user_id,
                'data_source': ctx.data_source,
                'external_service': ctx.external_service
            })

        # Additional context
        if additional_context:
            variables.update(additional_context)

        # Sanitize variables for template usage
        return {k: self._sanitize_variable(v) for k, v in variables.items()}

    def _sanitize_variable(self, value: Any) -> str:
        """Sanitize variable value for template usage."""
        if value is None:
            return ""
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Sanitize string values
            str_value = str(value)
            # Remove potentially harmful characters
            str_value = re.sub(r'[<>"\']', '', str_value)
            # Limit length
            return str_value[:500]

    def _format_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Format template string with variables."""
        try:
            return template.format(**variables)
        except KeyError as e:
            # Missing variable - use placeholder
            return template.replace(f"{{{e.args[0]}}}", "[UNAVAILABLE]")
        except Exception as e:
            logger.warning(f"Template formatting error: {e}")
            return template

    def _add_interactive_elements(
        self,
        formatted_message: FormattedErrorMessage,
        template: MessageTemplate,
        classified_error: ClassifiedError
    ) -> FormattedErrorMessage:
        """Add interactive elements to formatted message."""

        # Recovery actions based on template strategies
        for strategy in template.recovery_strategies:
            if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
                formatted_message.suggested_actions.append("Try again immediately")
                formatted_message.can_retry = True
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                formatted_message.suggested_actions.append("Wait and try again")
                formatted_message.can_retry = True
            elif strategy == RecoveryStrategy.USER_INTERVENTION:
                formatted_message.suggested_actions.append("Review and correct input")
                formatted_message.requires_user_action = True
            elif strategy == RecoveryStrategy.USER_WORKAROUND:
                formatted_message.suggested_actions.append("Try alternative approach")
                formatted_message.can_continue = True
            elif strategy == RecoveryStrategy.ESCALATION:
                formatted_message.suggested_actions.append("Contact support team")
                formatted_message.requires_support = True

        # Help links based on category and severity
        if classified_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            formatted_message.help_links.append({
                "title": "Support Center",
                "url": "/support",
                "description": "Get help from our support team"
            })

        if classified_error.category == ErrorCategory.DATA_VALIDATION:
            formatted_message.help_links.append({
                "title": "Data Guidelines",
                "url": "/help/data-format",
                "description": "Learn about proper data formatting"
            })

        return formatted_message

    def _update_analytics(self, formatted_message: FormattedErrorMessage) -> None:
        """Update formatting analytics."""
        with self._formatting_lock:
            # Store message in history
            self._formatting_history.append(formatted_message)

            # Update template usage
            if formatted_message.template_id:
                self._template_usage[formatted_message.template_id] = \
                    self._template_usage.get(formatted_message.template_id, 0) + 1

            # Update audience statistics
            self._audience_stats[formatted_message.audience] = \
                self._audience_stats.get(formatted_message.audience, 0) + 1

            # Keep history manageable
            if len(self._formatting_history) > 10000:
                self._formatting_history = self._formatting_history[-5000:]

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get comprehensive template usage statistics."""
        with self._formatting_lock:
            stats = {
                'total_templates': len(self._templates),
                'total_formatted_messages': len(self._formatting_history),
                'template_usage': self._template_usage.copy(),
                'audience_usage': {aud.value: count for aud, count in self._audience_stats.items()},
                'category_distribution': {},
                'severity_distribution': {},
                'popular_templates': sorted(
                    self._template_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

            # Calculate category and severity distributions
            for message in self._formatting_history:
                stats['category_distribution'][message.category.value] = \
                    stats['category_distribution'].get(message.category.value, 0) + 1
                stats['severity_distribution'][message.severity.value] = \
                    stats['severity_distribution'].get(message.severity.value, 0) + 1

            return stats

    def cleanup_old_messages(self, days_to_keep: int = 30) -> None:
        """Clean up old formatted messages."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._formatting_lock:
            # Clean formatting history
            self._formatting_history = [
                msg for msg in self._formatting_history
                if msg.generated_at > cutoff_date
            ]

        logger.info(f"Cleaned up messages older than {days_to_keep} days")


# Singleton instance for global access
_message_formatter_instance = None
_formatter_lock = threading.Lock()


def get_error_message_formatter() -> ErrorMessageFormatter:
    """Get the global error message formatter instance."""
    global _message_formatter_instance

    if _message_formatter_instance is None:
        with _formatter_lock:
            if _message_formatter_instance is None:
                _message_formatter_instance = ErrorMessageFormatter()

    return _message_formatter_instance


def format_error_message(
    classified_error: ClassifiedError,
    audience: AudienceType = AudienceType.END_USER,
    tone: MessageTone = MessageTone.FRIENDLY,
    complexity: MessageComplexity = MessageComplexity.STANDARD,
    language: str = "en",
    additional_context: Optional[Dict[str, Any]] = None
) -> FormattedErrorMessage:
    """
    Convenience function to format an error message.

    Args:
        classified_error: Classified error to format
        audience: Target audience
        tone: Message tone
        complexity: Message complexity
        language: Language code
        additional_context: Additional context

    Returns:
        FormattedErrorMessage with user-friendly content
    """
    formatter = get_error_message_formatter()
    return formatter.format_error(
        classified_error, audience, tone, complexity, language, additional_context
    )