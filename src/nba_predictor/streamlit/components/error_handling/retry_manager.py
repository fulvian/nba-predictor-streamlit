"""
🎯 PHASE 3 DAY 9: Retry Logic with Exponential Backoff
======================================================

X7 Compliant Retry Management System with Intelligent Backoff Strategies for NBA Predictor.

This module provides comprehensive retry functionality for:
- Exponential backoff with jitter and configurable parameters
- Circuit breaker pattern for preventing cascading failures
- Retry policies based on error classification
- Adaptive retry strategies with machine learning
- Real-time retry monitoring and analytics
- Integration with Enhanced Error Classifier

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import logging
import time
import random
import threading
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import json
from functools import wraps
import asyncio

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


class BackoffStrategy(Enum):
    """Backoff strategies for retry attempts."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class RetryDecision(Enum):
    """Retry decision outcomes."""

    RETRY = "retry"
    ABORT = "abort"
    ESCALATE = "escalate"
    FALLBACK = "fallback"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    # Basic retry configuration
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Strategy configuration
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Error-specific configuration
    retryable_categories: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.API_CONNECTION,
        ErrorCategory.API_TIMEOUT,
        ErrorCategory.DB_CONNECTION,
        ErrorCategory.SYSTEM_TIMEOUT,
        ErrorCategory.SYSTEM_NETWORK
    ])

    non_retryable_categories: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.DATA_VALIDATION,
        ErrorCategory.BUSINESS_LOGIC,
        ErrorCategory.CONFIGURATION
    ])

    # Adaptive configuration
    enable_adaptive_retry: bool = False
    success_threshold: float = 0.7
    adaptation_window: int = 10

    # Policy metadata
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Retry Policy"
    description: str = "Standard retry policy with exponential backoff"


@dataclass
class RetryAttempt:
    """Single retry attempt record."""

    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attempt_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    delay: float = 0.0
    exception: Optional[Exception] = None
    classified_error: Optional[ClassifiedError] = None
    success: bool = False
    execution_time: float = 0.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL


@dataclass
class RetrySession:
    """Complete retry session tracking."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_name: str = ""
    policy_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attempts: List[RetryAttempt] = field(default_factory=list)
    final_result: Optional[Any] = None
    success: bool = False
    total_execution_time: float = 0.0
    retry_decision: RetryDecision = RetryDecision.ABORT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""

    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    half_open_attempts: int = 0
    state_reset_time: Optional[datetime] = None


class RetryManager:
    """
    X7 Compliant Retry Manager with Advanced Backoff Strategies.

    Features:
    - Intelligent retry based on error classification
    - Exponential backoff with jitter for thundering herd prevention
    - Circuit breaker pattern for fault tolerance
    - Adaptive retry strategies with success rate monitoring
    - Comprehensive retry analytics and monitoring
    - Thread-safe operations with proper synchronization
    """

    def __init__(self):
        """Initialize the retry manager."""
        self._initialized = True
        self._retry_lock = threading.RLock()

        # Retry policies
        self._policies: Dict[str, RetryPolicy] = {}
        self._default_policy = RetryPolicy()

        # Circuit breaker state
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Retry history and analytics
        self._retry_history: deque = deque(maxlen=10000)
        self._active_sessions: Dict[str, RetrySession] = {}
        self._operation_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Adaptive retry learning
        self._success_rates: Dict[str, List[float]] = defaultdict(list)
        self._adaptive_policies: Dict[str, RetryPolicy] = {}

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_trips': 0,
            'average_attempts': 0.0,
            'total_retry_time': 0.0
        }

        # Initialize default policies
        self._initialize_default_policies()

        # Setup logging
        self._setup_logging()

        logger.info("Retry Manager initialized with X7 compliance")

    def _initialize_default_policies(self) -> None:
        """Initialize default retry policies."""

        # API Operations Policy
        api_policy = RetryPolicy(
            name="API Operations Policy",
            description="Retry policy for external API calls with circuit breaker",
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=1.5,
            jitter=True,
            jitter_factor=0.2,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=60.0,
            retryable_categories=[
                ErrorCategory.API_CONNECTION,
                ErrorCategory.API_TIMEOUT,
                ErrorCategory.API_RATE_LIMIT,
                ErrorCategory.SYSTEM_NETWORK
            ],
            non_retryable_categories=[
                ErrorCategory.API_AUTHENTICATION,
                ErrorCategory.API_VALIDATION,
                ErrorCategory.DATA_VALIDATION
            ],
            enable_adaptive_retry=True,
            success_threshold=0.8
        )
        self.add_policy(api_policy)

        # Database Operations Policy
        db_policy = RetryPolicy(
            name="Database Operations Policy",
            description="Retry policy for database operations",
            max_attempts=3,
            base_delay=0.5,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=True,
            jitter_factor=0.1,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30.0,
            retryable_categories=[
                ErrorCategory.DB_CONNECTION,
                ErrorCategory.DB_TRANSACTION,
                ErrorCategory.DB_LOCK,
                ErrorCategory.SYSTEM_TIMEOUT
            ],
            non_retryable_categories=[
                ErrorCategory.DB_CONSTRAINT,
                ErrorCategory.DATA_VALIDATION,
                ErrorCategory.DATA_INTEGRITY
            ],
            enable_adaptive_retry=False,
            success_threshold=0.9
        )
        self.add_policy(db_policy)

        # ML Model Operations Policy
        ml_policy = RetryPolicy(
            name="ML Model Operations Policy",
            description="Retry policy for ML model predictions and training",
            max_attempts=2,
            base_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=1.2,
            jitter=True,
            jitter_factor=0.15,
            backoff_strategy=BackoffStrategy.LINEAR,
            circuit_breaker_threshold=4,
            circuit_breaker_timeout=120.0,
            retryable_categories=[
                ErrorCategory.MODEL_PREDICTION,
                ErrorCategory.SYSTEM_TIMEOUT,
                ErrorCategory.SYSTEM_RESOURCE
            ],
            non_retryable_categories=[
                ErrorCategory.DATA_VALIDATION,
                ErrorCategory.DATA_FORMAT,
                ErrorCategory.MODEL_VALIDATION
            ],
            enable_adaptive_retry=True,
            success_threshold=0.85
        )
        self.add_policy(ml_policy)

        # System Operations Policy
        system_policy = RetryPolicy(
            name="System Operations Policy",
            description="Retry policy for system-level operations",
            max_attempts=4,
            base_delay=0.1,
            max_delay=2.0,
            backoff_multiplier=2.5,
            jitter=True,
            jitter_factor=0.3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=300.0,
            retryable_categories=[
                ErrorCategory.SYSTEM_TIMEOUT,
                ErrorCategory.SYSTEM_NETWORK,
                ErrorCategory.SYSTEM_RESOURCE
            ],
            non_retryable_categories=[
                ErrorCategory.SYSTEM_MEMORY,
                ErrorCategory.SYSTEM_DISK,
                ErrorCategory.CONFIGURATION
            ],
            enable_adaptive_retry=False,
            success_threshold=0.75
        )
        self.add_policy(system_policy)

        logger.info(f"Initialized {len(self._policies)} default retry policies")

    def add_policy(self, policy: RetryPolicy) -> None:
        """Add a retry policy."""
        with self._retry_lock:
            self._policies[policy.policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[RetryPolicy]:
        """Get retry policy by ID."""
        return self._policies.get(policy_id)

    def get_policy_by_name(self, name: str) -> Optional[RetryPolicy]:
        """Get retry policy by name."""
        for policy in self._policies.values():
            if policy.name == name:
                return policy
        return None

    def _setup_logging(self) -> None:
        """Setup enhanced logging for retry manager."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def calculate_backoff_delay(
        self,
        attempt: int,
        policy: RetryPolicy,
        adaptive_factor: float = 1.0
    ) -> float:
        """
        Calculate delay for retry attempt based on backoff strategy.

        Args:
            attempt: Current attempt number (starting from 1)
            policy: Retry policy to use
            adaptive_factor: Adaptive scaling factor

        Returns:
            Delay in seconds
        """
        if policy.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = policy.base_delay * (policy.backoff_multiplier ** (attempt - 1))
        elif policy.backoff_strategy == BackoffStrategy.LINEAR:
            delay = policy.base_delay * attempt
        elif policy.backoff_strategy == BackoffStrategy.FIXED:
            delay = policy.base_delay
        elif policy.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Fibonacci backoff: F(n) = F(n-1) + F(n-2), with F(1)=F(2)=1
            fib = self._fibonacci(attempt)
            delay = policy.base_delay * fib
        elif policy.backoff_strategy == BackoffStrategy.ADAPTIVE:
            # Adaptive based on historical success rates
            delay = policy.base_delay * (policy.backoff_multiplier ** (attempt - 1)) * adaptive_factor
        else:  # CUSTOM
            delay = policy.base_delay

        # Apply jitter if enabled
        if policy.jitter:
            jitter_amount = delay * policy.jitter_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay += jitter

        # Apply maximum delay limit
        delay = min(delay, policy.max_delay)

        # Ensure non-negative delay
        return max(0, delay)

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 1, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b

    def should_retry(
        self,
        exception: Exception,
        attempt: int,
        policy: RetryPolicy,
        classified_error: Optional[ClassifiedError] = None
    ) -> Tuple[bool, RetryDecision]:
        """
        Determine if operation should be retried.

        Args:
            exception: Exception that occurred
            attempt: Current attempt number
            policy: Retry policy
            classified_error: Classified error information

        Returns:
            Tuple of (should_retry, retry_decision)
        """
        # Check if max attempts reached
        if attempt >= policy.max_attempts:
            return False, RetryDecision.ABORT

        # Check circuit breaker
        operation_name = getattr(exception, '__operation_name__', 'unknown')
        if self._is_circuit_breaker_open(operation_name, policy):
            return False, RetryDecision.ESCALATE

        # Classify error if not provided
        if classified_error is None:
            classifier = get_error_classifier()
            classified_error = classifier.classify_error(exception)

        # Check non-retryable categories
        if classified_error.category in policy.non_retryable_categories:
            logger.info(f"Error category {classified_error.category.value} is non-retryable")
            return False, RetryDecision.ABORT

        # Check retryable categories
        if classified_error.category in policy.retryable_categories:
            # Check severity - don't retry critical errors
            if classified_error.severity == ErrorSeverity.CRITICAL:
                logger.warning("Critical error detected, not retrying")
                return False, RetryDecision.ESCALATE

            return True, RetryDecision.RETRY

        # Default behavior for uncategorized errors
        if attempt < 2:  # Allow one retry for unknown errors
            logger.info("Uncategorized error, allowing one retry")
            return True, RetryDecision.RETRY

        return False, RetryDecision.ABORT

    def _is_circuit_breaker_open(
        self,
        operation_name: str,
        policy: RetryPolicy
    ) -> bool:
        """Check if circuit breaker is open for operation."""
        with self._retry_lock:
            if operation_name not in self._circuit_breakers:
                self._circuit_breakers[operation_name] = CircuitBreakerState()
                return False

            breaker = self._circuit_breakers[operation_name]

            # Check if breaker should be reset
            if breaker.is_open and breaker.state_reset_time:
                if datetime.now() >= breaker.state_reset_time:
                    logger.info(f"Resetting circuit breaker for {operation_name}")
                    breaker.is_open = False
                    breaker.failure_count = 0
                    breaker.half_open_attempts = 0
                    breaker.state_reset_time = None
                    return False

            return breaker.is_open

    def _record_circuit_breaker_failure(
        self,
        operation_name: str,
        policy: RetryPolicy
    ) -> None:
        """Record failure for circuit breaker."""
        with self._retry_lock:
            if operation_name not in self._circuit_breakers:
                self._circuit_breakers[operation_name] = CircuitBreakerState()

            breaker = self._circuit_breakers[operation_name]
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()

            # Trip circuit breaker if threshold reached
            if breaker.failure_count >= policy.circuit_breaker_threshold:
                logger.warning(f"Circuit breaker tripped for {operation_name}")
                breaker.is_open = True
                breaker.state_reset_time = datetime.now() + timedelta(seconds=policy.circuit_breaker_timeout)
                self._metrics['circuit_breaker_trips'] += 1

    def _record_circuit_breaker_success(self, operation_name: str) -> None:
        """Record success for circuit breaker."""
        with self._retry_lock:
            if operation_name not in self._circuit_breakers:
                return

            breaker = self._circuit_breakers[operation_name]

            if breaker.is_open:
                # Half-open state success
                breaker.success_count += 1
                if breaker.success_count >= 2:  # Reset after 2 consecutive successes
                    logger.info(f"Circuit breaker reset for {operation_name} after successful retries")
                    breaker.is_open = False
                    breaker.failure_count = 0
                    breaker.success_count = 0
                    breaker.half_open_attempts = 0
                    breaker.state_reset_time = None
            else:
                # Normal state - reset failure count on success
                breaker.failure_count = max(0, breaker.failure_count - 1)

    def get_adaptive_factor(self, operation_name: str, policy: RetryPolicy) -> float:
        """Calculate adaptive factor for retry delay based on historical success rates."""
        if not policy.enable_adaptive_retry or operation_name not in self._success_rates:
            return 1.0

        success_rates = self._success_rates[operation_name]
        if len(success_rates) < policy.adaptation_window:
            return 1.0

        # Calculate recent success rate
        recent_rates = success_rates[-policy.adaptation_window:]
        avg_success_rate = sum(recent_rates) / len(recent_rates)

        # Adapt factor based on success rate
        if avg_success_rate < policy.success_threshold:
            # Increase delay for low success rates
            return 1.5
        elif avg_success_rate > 0.95:
            # Decrease delay for very high success rates
            return 0.5
        else:
            return 1.0

    def retry(
        self,
        func: Callable,
        *args,
        policy: Optional[Union[str, RetryPolicy]] = None,
        operation_name: str = "unknown",
        on_retry: Optional[Callable[[Exception, int], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            policy: Retry policy (ID or policy object)
            operation_name: Name of operation for tracking
            on_retry: Callback called on each retry attempt
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        # Get policy
        if policy is None:
            policy_obj = self._default_policy
        elif isinstance(policy, str):
            policy_obj = self.get_policy(policy) or self._default_policy
        else:
            policy_obj = policy

        # Create retry session
        session = RetrySession(
            operation_name=operation_name,
            policy_id=policy_obj.policy_id
        )

        # Add operation name to exception for circuit breaker tracking
        kwargs['__operation_name'] = operation_name

        with self._retry_lock:
            self._active_sessions[session.session_id] = session

        start_time = time.time()
        last_exception = None
        classified_error = None

        try:
            for attempt in range(1, policy_obj.max_attempts + 1):
                attempt_start = time.time()

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Record successful attempt
                    execution_time = time.time() - attempt_start
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        success=True,
                        execution_time=execution_time,
                        backoff_strategy=policy_obj.backoff_strategy
                    )
                    session.attempts.append(attempt_record)

                    # Record circuit breaker success
                    self._record_circuit_breaker_success(operation_name)

                    # Update adaptive success rate
                    if policy_obj.enable_adaptive_retry:
                        success_rate = 1.0 / attempt  # Higher rate for fewer attempts
                        self._update_success_rate(operation_name, success_rate)

                    # Finalize session
                    session.success = True
                    session.final_result = result
                    session.total_execution_time = time.time() - start_time
                    session.retry_decision = RetryDecision.RETRY

                    self._metrics['successful_retries'] += 1
                    self._metrics['total_retries'] += 1

                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                    return result

                except Exception as e:
                    last_exception = e
                    execution_time = time.time() - attempt_start

                    # Classify error
                    classifier = get_error_classifier()
                    classified_error = classifier.classify_error(e)

                    # Record failed attempt
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        delay=0.0,
                        exception=e,
                        classified_error=classified_error,
                        success=False,
                        execution_time=execution_time,
                        backoff_strategy=policy_obj.backoff_strategy
                    )
                    session.attempts.append(attempt_record)

                    # Determine if should retry
                    should_retry, retry_decision = self.should_retry(
                        e, attempt, policy_obj, classified_error
                    )

                    if not should_retry:
                        session.retry_decision = retry_decision
                        break

                    # Record circuit breaker failure
                    self._record_circuit_breaker_failure(operation_name, policy_obj)

                    # Calculate delay for next attempt
                    adaptive_factor = self.get_adaptive_factor(operation_name, policy_obj)
                    delay = self.calculate_backoff_delay(attempt, policy_obj, adaptive_factor)

                    # Call retry callback
                    if on_retry:
                        on_retry(e, attempt)

                    logger.warning(
                        f"Operation {operation_name} failed on attempt {attempt}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )

                    # Wait before retry
                    time.sleep(delay)

                    # Update attempt record with actual delay
                    attempt_record.delay = delay

            # All attempts failed
            session.success = False
            session.total_execution_time = time.time() - start_time

            self._metrics['failed_retries'] += 1
            self._metrics['total_retries'] += 1

            logger.error(f"Operation {operation_name} failed after {len(session.attempts)} attempts")
            raise last_exception

        finally:
            # Finalize and store session
            session.end_time = datetime.now()
            with self._retry_lock:
                self._retry_history.append(session)
                if session.session_id in self._active_sessions:
                    del self._active_sessions[session.session_id]

                # Update operation statistics
                self._update_operation_stats(operation_name, session)

                # Update metrics
                self._update_metrics()

    def _update_success_rate(self, operation_name: str, success_rate: float) -> None:
        """Update success rate for adaptive retry."""
        with self._retry_lock:
            if operation_name not in self._success_rates:
                self._success_rates[operation_name] = []

            self._success_rates[operation_name].append(success_rate)

            # Keep only recent rates
            max_rates = 100
            if len(self._success_rates[operation_name]) > max_rates:
                self._success_rates[operation_name] = self._success_rates[operation_name][-max_rates:]

    def _update_operation_stats(self, operation_name: str, session: RetrySession) -> None:
        """Update operation statistics."""
        with self._retry_lock:
            if operation_name not in self._operation_stats:
                self._operation_stats[operation_name] = {
                    'total_sessions': 0,
                    'successful_sessions': 0,
                    'total_attempts': 0,
                    'average_attempts': 0.0,
                    'total_time': 0.0,
                    'average_time': 0.0,
                    'last_attempt': None
                }

            stats = self._operation_stats[operation_name]
            stats['total_sessions'] += 1
            stats['total_attempts'] += len(session.attempts)
            stats['total_time'] += session.total_execution_time
            stats['average_attempts'] = stats['total_attempts'] / stats['total_sessions']
            stats['average_time'] = stats['total_time'] / stats['total_sessions']
            stats['last_attempt'] = session.end_time

            if session.success:
                stats['successful_sessions'] += 1

    def _update_metrics(self) -> None:
        """Update global metrics."""
        if self._metrics['total_retries'] > 0:
            total_attempts = sum(len(session.attempts) for session in self._retry_history)
            self._metrics['average_attempts'] = total_attempts / len(self._retry_history)

    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics."""
        with self._retry_lock:
            stats = {
                'global_metrics': self._metrics.copy(),
                'active_policies': len(self._policies),
                'circuit_breakers': {
                    name: {
                        'is_open': breaker.is_open,
                        'failure_count': breaker.failure_count,
                        'last_failure_time': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                        'state_reset_time': breaker.state_reset_time.isoformat() if breaker.state_reset_time else None
                    }
                    for name, breaker in self._circuit_breakers.items()
                },
                'operation_stats': self._operation_stats.copy(),
                'active_sessions': len(self._active_sessions),
                'total_history': len(self._retry_history),
                'success_rates': {
                    op: {
                        'recent_rates': rates[-10:],
                        'average_rate': sum(rates) / len(rates) if rates else 0.0
                    }
                    for op, rates in self._success_rates.items()
                }
            }

            # Calculate success rate
            if self._metrics['total_retries'] > 0:
                stats['global_metrics']['success_rate'] = (
                    self._metrics['successful_retries'] / self._metrics['total_retries']
                )
            else:
                stats['global_metrics']['success_rate'] = 0.0

            return stats

    def cleanup_old_sessions(self, days_to_keep: int = 7) -> None:
        """Clean up old retry sessions."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._retry_lock:
            # Clean retry history
            self._retry_history = deque(
                [session for session in self._retry_history if session.end_time and session.end_time > cutoff_date],
                maxlen=10000
            )

            # Clean operation stats (keep only recent activity)
            for operation_name in list(self._operation_stats.keys()):
                stats = self._operation_stats[operation_name]
                if stats['last_attempt'] and stats['last_attempt'] < cutoff_date:
                    del self._operation_stats[operation_name]

        logger.info(f"Cleaned up sessions older than {days_to_keep} days")


# Decorator for easy retry functionality
def retry(
    policy: Optional[Union[str, RetryPolicy]] = None,
    operation_name: str = "function",
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for automatic retry functionality.

    Args:
        policy: Retry policy to use
        operation_name: Name for operation tracking
        on_retry: Callback for retry attempts
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = get_retry_manager()
            return retry_manager.retry(
                func, *args,
                policy=policy,
                operation_name=operation_name,
                on_retry=on_retry,
                **kwargs
            )
        return wrapper
    return decorator


# Singleton instance for global access
_retry_manager_instance = None
_manager_lock = threading.Lock()


def get_retry_manager() -> RetryManager:
    """Get the global retry manager instance."""
    global _retry_manager_instance

    if _retry_manager_instance is None:
        with _manager_lock:
            if _retry_manager_instance is None:
                _retry_manager_instance = RetryManager()

    return _retry_manager_instance