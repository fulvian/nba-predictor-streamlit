"""
🎯 PHASE 3 DAY 9: Error Handling State Manager Integration
========================================================

X7 Compliant Error Handling Integration with ML State Manager.

This module provides seamless integration between the error handling system
and the ML state manager, enabling:
- Error-aware state management
- State-driven error handling policies
- Recovery automation through state validation
- Comprehensive error tracking in state context
- Automatic retry policies based on state health

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

# Import from our error handling system
from .enhanced_error_classifier import (
    ErrorCategory, ErrorSeverity, RecoveryStrategy,
    ClassifiedError, get_error_classifier
)
from .retry_manager import (
    RetryPolicy, BackoffStrategy, get_retry_manager, retry
)
from .error_message_formatter import (
    MessageTone, AudienceType, get_error_message_formatter
)
from .error_reporter import (
    ErrorEvent, AlertSeverity, get_error_reporter
)

# Import ML state manager
from ..state_manager import get_state_manager

logger = logging.getLogger(__name__)


class StateErrorSyncStatus(Enum):
    """Status of error-state synchronization"""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class StateErrorContext:
    """Context for errors within state management"""
    component_id: str
    operation: str
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    previous_state: Dict[str, Any] = field(default_factory=dict)
    state_health_score: float = 0.0
    error_frequency: int = 0
    last_error_time: Optional[datetime] = None
    recovery_actions_taken: List[str] = field(default_factory=list)
    sync_status: StateErrorSyncStatus = StateErrorSyncStatus.PENDING


@dataclass
class ErrorStateRecoveryPlan:
    """Recovery plan based on state and error context"""
    plan_id: str
    error_context: StateErrorContext
    recovery_steps: List[Dict[str, Any]]
    estimated_success_rate: float
    requires_state_reset: bool = False
    rollback_available: bool = True
    priority: int = 1  # 1 = highest priority


class StateAwareErrorHandler:
    """
    🎯 STATE-AWARE ERROR HANDLER

    Advanced error handler that integrates with ML state manager
    for intelligent error processing and recovery.
    """

    def __init__(self):
        self.error_classifier = get_error_classifier()
        self.retry_manager = get_retry_manager()
        self.message_formatter = get_error_message_formatter()
        self.error_reporter = get_error_reporter()
        self.ml_state_manager = get_state_manager()

        # State tracking
        self.active_error_contexts: Dict[str, StateErrorContext] = {}
        self.recovery_plans: Dict[str, ErrorStateRecoveryPlan] = {}
        self.error_state_history: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Configuration
        self.config = {
            'max_error_contexts': 1000,
            'state_sync_timeout': 30.0,
            'auto_recovery_enabled': True,
            'state_health_threshold': 0.7,
            'error_frequency_threshold': 5,
            'recovery_plan_timeout': 300.0  # 5 minutes
        }

        # Start background tasks
        self._start_background_tasks()

        logger.info("🚀 StateAwareErrorHandler initialized")
        logger.info(f"🔧 Auto-recovery: {self.config['auto_recovery_enabled']}")
        logger.info(f"🎯 State health threshold: {self.config['state_health_threshold']}")

    def handle_error_with_state_context(
        self,
        error: Exception,
        component_id: str,
        operation: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ClassifiedError, StateErrorContext]:
        """
        Handle error with full state context awareness.

        Args:
            error: The exception that occurred
            component_id: ID of the component where error occurred
            operation: Operation being performed when error occurred
            additional_context: Additional context information

        Returns:
            Tuple of (classified error, state error context)
        """
        try:
            # Classify the error
            classified_error = self.error_classifier.classify_error(
                error=error,
                context=additional_context or {}
            )

            # Get current component state
            current_state = self._get_component_state(component_id)
            previous_state = self._get_previous_state(component_id)

            # Calculate state health score
            health_score = self._calculate_state_health_score(
                component_id, current_state, classified_error
            )

            # Create state error context
            error_context = StateErrorContext(
                component_id=component_id,
                operation=operation,
                state_snapshot=current_state,
                previous_state=previous_state,
                state_health_score=health_score,
                error_frequency=self._get_error_frequency(component_id),
                last_error_time=datetime.now(timezone.utc)
            )

            # Determine synchronization status
            error_context.sync_status = self._determine_sync_status(
                classified_error, error_context
            )

            # Store error context
            with self._lock:
                self.active_error_contexts[f"{component_id}_{operation}"] = error_context
                self._cleanup_old_contexts()

            # Report error with state context
            self._report_error_with_state(classified_error, error_context)

            # Attempt automatic recovery if enabled
            if self.config['auto_recovery_enabled']:
                recovery_result = self._attempt_automatic_recovery(
                    classified_error, error_context
                )
                if recovery_result:
                    error_context.recovery_actions_taken.append(recovery_result)

            logger.info(
                f"✅ Error handled with state context: {component_id}/{operation} | "
                f"Health: {health_score:.2f} | Sync: {error_context.sync_status.value}"
            )

            return classified_error, error_context

        except Exception as e:
            logger.error(f"❌ Error in state-aware error handling: {e}")
            # Fallback to basic error classification
            fallback_error = self.error_classifier.classify_error(
                error=error,
                context={"component_id": component_id, "operation": operation}
            )
            return fallback_error, StateErrorContext(
                component_id=component_id,
                operation=operation,
                sync_status=StateErrorSyncStatus.ERROR
            )

    def execute_with_state_aware_retry(
        self,
        operation_func,
        component_id: str,
        operation_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute operation with state-aware retry logic.

        Args:
            operation_func: Function to execute
            component_id: ID of the component
            operation_name: Name of the operation
            retry_policy: Optional custom retry policy
            operation_context: Additional operation context

        Returns:
            Result of operation function

        Raises:
            Last exception if all retries exhausted
        """
        # Determine appropriate retry policy based on state health
        if retry_policy is None:
            retry_policy = self._determine_retry_policy(component_id, operation_name)

        # Wrap operation with state monitoring
        def state_monitored_operation():
            try:
                # Pre-execution state check
                pre_state = self._get_component_state(component_id)
                health_score = self._calculate_state_health_score(
                    component_id, pre_state, None
                )

                # Execute the operation
                result = operation_func()

                # Post-execution state update
                self._update_component_state_after_success(
                    component_id, operation_name, pre_state
                )

                return result

            except Exception as e:
                # Handle error with state context
                classified_error, error_context = self.handle_error_with_state_context(
                    error=e,
                    component_id=component_id,
                    operation=operation_name,
                    additional_context=operation_context
                )

                # Update state for error
                self._update_component_state_after_error(
                    component_id, operation_name, classified_error
                )

                raise e

        # Execute with retry logic
        return self.retry_manager.execute_with_retry(
            operation=state_monitored_operation,
            policy=retry_policy
        )

    def create_recovery_plan(
        self,
        classified_error: ClassifiedError,
        error_context: StateErrorContext
    ) -> ErrorStateRecoveryPlan:
        """
        Create comprehensive recovery plan based on error and state.

        Args:
            classified_error: The classified error
            error_context: State error context

        Returns:
            Recovery plan with actionable steps
        """
        plan_id = str(uuid.uuid4())

        # Determine recovery steps based on error category and state health
        recovery_steps = []

        # Common recovery steps
        if classified_error.category == ErrorCategory.NETWORK:
            recovery_steps.extend([
                {
                    'step': 'validate_network_connectivity',
                    'description': 'Check network connectivity and DNS resolution',
                    'action': 'self._validate_network_connectivity()',
                    'timeout': 30
                },
                {
                    'step': 'reset_connection_pool',
                    'description': 'Reset connection pools and retry connections',
                    'action': 'self.ml_state_manager.reset_component_network_state()',
                    'timeout': 10
                }
            ])

        elif classified_error.category == ErrorCategory.DB_TIMEOUT:
            recovery_steps.extend([
                {
                    'step': 'check_database_health',
                    'description': 'Verify database connectivity and performance',
                    'action': 'self.ml_state_manager.validate_database_health()',
                    'timeout': 60
                },
                {
                    'step': 'optimize_query_performance',
                    'description': 'Optimize slow queries and check indexes',
                    'action': 'self._optimize_database_queries()',
                    'timeout': 120
                }
            ])

        elif classified_error.category == ErrorCategory.MEMORY:
            recovery_steps.extend([
                {
                    'step': 'garbage_collect_memory',
                    'description': 'Force garbage collection and memory cleanup',
                    'action': 'self._perform_memory_cleanup()',
                    'timeout': 30
                },
                {
                    'step': 'reset_component_state',
                    'description': 'Reset component state to free memory',
                    'action': 'self.ml_state_manager.reset_component_state()',
                    'timeout': 15
                }
            ])

        # State-specific recovery steps
        if error_context.state_health_score < self.config['state_health_threshold']:
            recovery_steps.append({
                'step': 'restore_previous_state',
                'description': 'Restore component to previous healthy state',
                'action': 'self._restore_component_state()',
                'timeout': 45
            })

        # High frequency error recovery
        if error_context.error_frequency > self.config['error_frequency_threshold']:
            recovery_steps.append({
                'step': 'implement_circuit_breaker',
                'description': 'Temporarily disable component to prevent cascade failures',
                'action': 'self._enable_circuit_breaker()',
                'timeout': 10
            })

        # Calculate estimated success rate
        estimated_success_rate = self._calculate_recovery_success_rate(
            classified_error, error_context, recovery_steps
        )

        recovery_plan = ErrorStateRecoveryPlan(
            plan_id=plan_id,
            error_context=error_context,
            recovery_steps=recovery_steps,
            estimated_success_rate=estimated_success_rate,
            priority=self._calculate_recovery_priority(classified_error, error_context)
        )

        # Store recovery plan
        with self._lock:
            self.recovery_plans[plan_id] = recovery_plan

        logger.info(
            f"📋 Recovery plan created: {plan_id} | "
            f"Success rate: {estimated_success_rate:.1%} | "
            f"Priority: {recovery_plan.priority}"
        )

        return recovery_plan

    def get_error_state_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of error state across all components.

        Returns:
            Dictionary with error state summary and analytics
        """
        with self._lock:
            active_contexts = list(self.active_error_contexts.values())
            recovery_plans = list(self.recovery_plans.values())

        # Component error summary
        component_summary = {}
        for context in active_contexts:
            component_id = context.component_id
            if component_id not in component_summary:
                component_summary[component_id] = {
                    'error_count': 0,
                    'operations_with_errors': set(),
                    'avg_health_score': 0,
                    'critical_errors': 0,
                    'recovery_actions': []
                }

            comp_summary = component_summary[component_id]
            comp_summary['error_count'] += 1
            comp_summary['operations_with_errors'].add(context.operation)
            comp_summary['avg_health_score'] += context.state_health_score
            comp_summary['recovery_actions'].extend(context.recovery_actions_taken)

            if context.sync_status == StateErrorSyncStatus.ERROR:
                comp_summary['critical_errors'] += 1

        # Calculate averages
        for component_id, summary in component_summary.items():
            if summary['error_count'] > 0:
                summary['avg_health_score'] /= summary['error_count']
            summary['operations_with_errors'] = list(summary['operations_with_errors'])

        # Recovery plan summary
        recovery_summary = {
            'total_plans': len(recovery_plans),
            'high_success_rate_plans': len([p for p in recovery_plans if p.estimated_success_rate > 0.8]),
            'urgent_plans': len([p for p in recovery_plans if p.priority <= 2]),
            'plans_requiring_reset': len([p for p in recovery_plans if p.requires_state_reset])
        }

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_error_contexts': len(active_contexts),
            'components_with_errors': len(component_summary),
            'component_summary': component_summary,
            'recovery_plans': recovery_summary,
            'overall_health_score': self._calculate_overall_health_score(),
            'sync_status_distribution': self._get_sync_status_distribution(active_contexts)
        }

    def _get_component_state(self, component_id: str) -> Dict[str, Any]:
        """Get current state of component from ML state manager."""
        try:
            return self.ml_state_manager.get_component_state(component_id)
        except Exception as e:
            logger.warning(f"Could not get state for {component_id}: {e}")
            return {'error': str(e)}

    def _get_previous_state(self, component_id: str) -> Dict[str, Any]:
        """Get previous state of component."""
        try:
            # Implementation depends on ML state manager capabilities
            return self.ml_state_manager.get_component_previous_state(component_id)
        except Exception:
            return {}

    def _calculate_state_health_score(
        self,
        component_id: str,
        state: Dict[str, Any],
        error: Optional[ClassifiedError]
    ) -> float:
        """Calculate health score for component state."""
        base_score = 1.0

        # Deductions for errors
        if error:
            severity_deductions = {
                ErrorSeverity.LOW: 0.1,
                ErrorSeverity.MEDIUM: 0.3,
                ErrorSeverity.HIGH: 0.5,
                ErrorSeverity.CRITICAL: 0.8
            }
            base_score -= severity_deductions.get(error.severity, 0.3)

        # Deductions for state issues
        if isinstance(state, dict):
            if 'error' in state:
                base_score -= 0.4
            if 'warnings' in state and len(state['warnings']) > 0:
                base_score -= min(0.2, len(state['warnings']) * 0.05)
            if 'performance_metrics' in state:
                perf = state['performance_metrics']
                if isinstance(perf, dict):
                    if perf.get('cpu_usage', 0) > 0.8:
                        base_score -= 0.2
                    if perf.get('memory_usage', 0) > 0.8:
                        base_score -= 0.2
                    if perf.get('error_rate', 0) > 0.1:
                        base_score -= 0.3

        return max(0.0, base_score)

    def _get_error_frequency(self, component_id: str) -> int:
        """Get error frequency for component."""
        with self._lock:
            return sum(
                1 for context in self.active_error_contexts.values()
                if context.component_id == component_id
            )

    def _determine_sync_status(
        self,
        error: ClassifiedError,
        context: StateErrorContext
    ) -> StateErrorSyncStatus:
        """Determine synchronization status between error and state."""
        if context.state_health_score > 0.8:
            return StateErrorSyncStatus.SYNCED
        elif context.state_health_score > 0.5:
            return StateErrorSyncStatus.PENDING
        elif error.severity == ErrorSeverity.CRITICAL:
            return StateErrorSyncStatus.ERROR
        else:
            return StateErrorSyncStatus.CONFLICT

    def _report_error_with_state(
        self,
        error: ClassifiedError,
        context: StateErrorContext
    ) -> None:
        """Report error with enhanced state context."""
        try:
            # Create enhanced error event
            error_event = ErrorEvent(
                event_id=str(uuid.uuid4()),
                error_id=error.error_id,
                category=error.category,
                severity=error.severity,
                message=error.message,
                component_id=context.component_id,
                operation=context.operation,
                additional_context={
                    'state_health_score': context.state_health_score,
                    'sync_status': context.sync_status.value,
                    'error_frequency': context.error_frequency,
                    'state_snapshot': context.state_snapshot,
                    'recovery_strategy': error.recovery_strategy.value if error.recovery_strategy else None
                }
            )

            self.error_reporter.record_error_event(error_event)

            # Create alert for critical errors or poor state health
            if (error.severity == ErrorSeverity.CRITICAL or
                context.state_health_score < self.config['state_health_threshold']):

                alert = self.error_reporter.create_alert(
                    error_event=error_event,
                    severity=AlertSeverity.CRITICAL if error.severity == ErrorSeverity.CRITICAL else AlertSeverity.HIGH,
                    message=f"Critical error in {context.component_id}: {error.message} (Health: {context.state_health_score:.2f})"
                )

                logger.warning(f"🚨 Alert created: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to report error with state: {e}")

    def _attempt_automatic_recovery(
        self,
        error: ClassifiedError,
        context: StateErrorContext
    ) -> Optional[str]:
        """Attempt automatic recovery based on error and state context."""
        if context.state_health_score < 0.3:
            return "auto_recovery_disabled_low_health"

        # Simple automatic recovery strategies
        if error.category == ErrorCategory.NETWORK:
            try:
                self.ml_state_manager.reset_component_network_state(context.component_id)
                return "network_state_reset"
            except Exception:
                pass

        elif error.category == ErrorCategory.MEMORY:
            try:
                self.ml_state_manager.clear_component_cache(context.component_id)
                return "cache_cleared"
            except Exception:
                pass

        return None

    def _determine_retry_policy(
        self,
        component_id: str,
        operation_name: str
    ) -> RetryPolicy:
        """Determine appropriate retry policy based on component state."""
        state = self._get_component_state(component_id)
        health_score = self._calculate_state_health_score(component_id, state, None)

        if health_score > 0.8:
            # Healthy state - standard retry policy
            return self.retry_manager.get_policy("database_operations")
        elif health_score > 0.5:
            # Degraded state - more conservative policy
            return self.retry_manager.get_policy("api_calls")
        else:
            # Poor state - very conservative policy
            return RetryPolicy(
                name="degraded_component_retry",
                max_attempts=2,
                base_delay=5.0,
                max_delay=30.0,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                jitter=True
            )

    def _update_component_state_after_success(
        self,
        component_id: str,
        operation: str,
        previous_state: Dict[str, Any]
    ) -> None:
        """Update component state after successful operation."""
        try:
            # Clear error contexts for successful operations
            context_key = f"{component_id}_{operation}"
            with self._lock:
                if context_key in self.active_error_contexts:
                    del self.active_error_contexts[context_key]

            # Update ML state manager
            self.ml_state_manager.update_component_state(
                component_id=component_id,
                updates={"last_successful_operation": operation},
                metadata={"operation_success": True}
            )

        except Exception as e:
            logger.error(f"Failed to update state after success: {e}")

    def _update_component_state_after_error(
        self,
        component_id: str,
        operation: str,
        error: ClassifiedError
    ) -> None:
        """Update component state after error."""
        try:
            self.ml_state_manager.update_component_state(
                component_id=component_id,
                updates={
                    "last_error": error.message,
                    "last_error_time": datetime.now(timezone.utc).isoformat(),
                    "error_category": error.category.value,
                    "error_severity": error.severity.value
                },
                metadata={"operation_error": True}
            )
        except Exception as e:
            logger.error(f"Failed to update state after error: {e}")

    def _cleanup_old_contexts(self) -> None:
        """Clean up old error contexts to prevent memory leaks."""
        if len(self.active_error_contexts) <= self.config['max_error_contexts']:
            return

        # Sort by last error time and keep only the most recent
        sorted_contexts = sorted(
            self.active_error_contexts.items(),
            key=lambda x: x[1].last_error_time or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )

        # Keep only the most recent contexts
        contexts_to_keep = dict(sorted_contexts[:self.config['max_error_contexts']])
        self.active_error_contexts = contexts_to_keep

    def _calculate_recovery_success_rate(
        self,
        error: ClassifiedError,
        context: StateErrorContext,
        recovery_steps: List[Dict[str, Any]]
    ) -> float:
        """Calculate estimated success rate for recovery plan."""
        base_rate = 0.7  # Base 70% success rate

        # Adjust based on state health
        base_rate += context.state_health_score * 0.2

        # Adjust based on error severity
        severity_adjustments = {
            ErrorSeverity.LOW: 0.2,
            ErrorSeverity.MEDIUM: 0.0,
            ErrorSeverity.HIGH: -0.2,
            ErrorSeverity.CRITICAL: -0.4
        }
        base_rate += severity_adjustments.get(error.severity, 0)

        # Adjust based on recovery step quality
        if len(recovery_steps) > 0:
            step_quality = min(1.0, len(recovery_steps) * 0.1)
            base_rate += step_quality * 0.1

        return max(0.0, min(1.0, base_rate))

    def _calculate_recovery_priority(
        self,
        error: ClassifiedError,
        context: StateErrorContext
    ) -> int:
        """Calculate recovery priority (1=highest, 5=lowest)."""
        if error.severity == ErrorSeverity.CRITICAL:
            return 1
        elif context.state_health_score < 0.3:
            return 1
        elif error.severity == ErrorSeverity.HIGH:
            return 2
        elif context.error_frequency > self.config['error_frequency_threshold']:
            return 2
        elif error.severity == ErrorSeverity.MEDIUM:
            return 3
        else:
            return 4

    def _calculate_overall_health_score(self) -> float:
        """Calculate overall health score across all components."""
        with self._lock:
            contexts = list(self.active_error_contexts.values())

        if not contexts:
            return 1.0

        return sum(context.state_health_score for context in contexts) / len(contexts)

    def _get_sync_status_distribution(self, contexts: List[StateErrorContext]) -> Dict[str, int]:
        """Get distribution of sync statuses."""
        distribution = {status.value: 0 for status in StateErrorSyncStatus}
        for context in contexts:
            distribution[context.sync_status.value] += 1
        return distribution

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        def maintenance_task():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_old_contexts()

                    # Clean up old recovery plans
                    with self._lock:
                        current_time = time.time()
                        expired_plans = [
                            plan_id for plan_id, plan in self.recovery_plans.items()
                            if (current_time - plan.error_context.last_error_time.timestamp()
                                > self.config['recovery_plan_timeout'])
                        ]
                        for plan_id in expired_plans:
                            del self.recovery_plans[plan_id]

                        if expired_plans:
                            logger.info(f"🧹 Cleaned up {len(expired_plans)} expired recovery plans")

                except Exception as e:
                    logger.error(f"Error in maintenance task: {e}")

        maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
        maintenance_thread.start()


# Global instance
_state_aware_handler = None

def get_state_aware_error_handler() -> StateAwareErrorHandler:
    """Get the singleton StateAwareErrorHandler instance."""
    global _state_aware_handler
    if _state_aware_handler is None:
        _state_aware_handler = StateAwareErrorHandler()
    return _state_aware_handler


def handle_error_with_state(
    error: Exception,
    component_id: str,
    operation: str,
    additional_context: Optional[Dict[str, Any]] = None
) -> Tuple[ClassifiedError, StateErrorContext]:
    """
    Convenience function to handle error with state context.
    """
    handler = get_state_aware_error_handler()
    return handler.handle_error_with_state_context(
        error, component_id, operation, additional_context
    )


def execute_with_state_retry(
    operation_func,
    component_id: str,
    operation_name: str,
    retry_policy: Optional[RetryPolicy] = None,
    operation_context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Convenience function to execute operation with state-aware retry.
    """
    handler = get_state_aware_error_handler()
    return handler.execute_with_state_aware_retry(
        operation_func, component_id, operation_name, retry_policy, operation_context
    )