"""
🎯 COMPREHENSIVE TEST: Error Handling State Integration
====================================================

Test suite for validating the complete integration between
Error Handling System and ML State Manager.

Tests all Phase 3 Day 9 components:
1. Enhanced Error Classification System
2. Retry Logic with Exponential Backoff
3. User-Friendly Error Messages
4. Error Reporting and Analytics
5. State Manager Integration

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import traceback
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all error handling components
from src.nba_predictor.streamlit.components.error_handling import (
    # Enhanced Error Classification
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    get_error_classifier,
    classify_error,

    # Retry Manager
    BackoffStrategy,
    RetryPolicy,
    get_retry_manager,
    retry,

    # Error Message Formatter
    MessageTone,
    MessageComplexity,
    AudienceType,
    get_error_message_formatter,
    format_error_message,

    # Error Reporter
    ReportingPeriod,
    AlertSeverity,
    get_error_reporter,
    report_error,

    # State Integration
    get_state_aware_error_handler,
    handle_error_with_state,
    execute_with_state_retry
)

# Import ML State Manager
from src.nba_predictor.streamlit.components.state_manager import get_state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandlingStateIntegrationTest:
    """
    🎯 COMPREHENSIVE INTEGRATION TEST SUITE

    Tests the complete error handling system with state manager integration.
    """

    def __init__(self):
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }

        self.test_components = {}
        self.initialized = False

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite for error handling state integration.

        Returns:
            Dictionary with complete test results and analysis
        """
        logger.info("🚀 Starting Comprehensive Error Handling State Integration Test Suite")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Initialize all components
            self._initialize_components()

            # Run test categories
            self._test_enhanced_error_classification()
            self._test_retry_logic_system()
            self._test_error_message_formatting()
            self._test_error_reporting_analytics()
            self._test_state_manager_integration()
            self._test_cross_component_integration()
            self._test_performance_scenarios()

        except Exception as e:
            logger.error(f"❌ Critical error in test suite: {e}")
            traceback.print_exc()

        # Calculate final results
        test_duration = time.time() - start_time
        success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests'] * 100) if self.test_results['total_tests'] > 0 else 0

        final_results = {
            'test_summary': {
                'total_tests': self.test_results['total_tests'],
                'passed_tests': self.test_results['passed_tests'],
                'failed_tests': self.test_results['failed_tests'],
                'success_rate': f"{success_rate:.1f}%",
                'duration_seconds': round(test_duration, 2)
            },
            'component_status': self._get_component_status(),
            'test_details': self.test_results['test_details'],
            'integration_analysis': self._analyze_integration_health(),
            'recommendations': self._generate_recommendations()
        }

        logger.info("=" * 80)
        logger.info("🎯 TEST SUITE COMPLETED")
        logger.info(f"📊 Results: {self.test_results['passed_tests']}/{self.test_results['total_tests']} passed ({success_rate:.1f}%)")
        logger.info(f"⏱️ Duration: {test_duration:.2f} seconds")

        return final_results

    def _initialize_components(self) -> None:
        """Initialize all error handling components."""
        logger.info("🔧 Initializing Error Handling Components...")

        try:
            # Enhanced Error Classifier
            self.test_components['error_classifier'] = get_error_classifier()
            logger.info("✅ Error Classifier initialized")

            # Retry Manager
            self.test_components['retry_manager'] = get_retry_manager()
            logger.info("✅ Retry Manager initialized")

            # Error Message Formatter
            self.test_components['message_formatter'] = get_error_message_formatter()
            logger.info("✅ Message Formatter initialized")

            # Error Reporter
            self.test_components['error_reporter'] = get_error_reporter()
            logger.info("✅ Error Reporter initialized")

            # State-Aware Error Handler
            self.test_components['state_handler'] = get_state_aware_error_handler()
            logger.info("✅ State-Aware Error Handler initialized")

            # ML State Manager
            self.test_components['ml_state_manager'] = get_state_manager()
            logger.info("✅ ML State Manager initialized")

            self.initialized = True
            logger.info("🎉 All components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _test_enhanced_error_classification(self) -> None:
        """Test Enhanced Error Classification System."""
        logger.info("🧪 Testing Enhanced Error Classification System...")

        test_cases = [
            {
                'name': 'Network Error Classification',
                'error': ConnectionError("Network timeout occurred"),
                'expected_category': ErrorCategory.NETWORK,
                'expected_min_severity': ErrorSeverity.MEDIUM
            },
            {
                'name': 'Database Timeout Classification',
                'error': TimeoutError("Database query timeout"),
                'expected_category': ErrorCategory.DB_TIMEOUT,
                'expected_min_severity': ErrorSeverity.HIGH
            },
            {
                'name': 'Memory Error Classification',
                'error': MemoryError("Out of memory"),
                'expected_category': ErrorCategory.MEMORY,
                'expected_min_severity': ErrorSeverity.CRITICAL
            },
            {
                'name': 'Validation Error Classification',
                'error': ValueError("Invalid input parameter"),
                'expected_category': ErrorCategory.VALIDATION,
                'expected_min_severity': ErrorSeverity.LOW
            }
        ]

        for test_case in test_cases:
            try:
                # Test error classification
                classified_error = self.test_components['error_classifier'].classify_error(
                    error=test_case['error'],
                    context={'test_case': test_case['name']}
                )

                # Validate classification
                assert classified_error.category == test_case['expected_category'], \
                    f"Expected category {test_case['expected_category']}, got {classified_error.category}"

                assert classified_error.severity.value >= test_case['expected_min_severity'].value, \
                    f"Severity {classified_error.severity} lower than expected minimum {test_case['expected_min_severity']}"

                # Test error classification function
                classified_error_func = classify_error(
                    test_case['error'],
                    {'function_test': True}
                )
                assert classified_error_func.error_id != classified_error.error_id, \
                    "Function should create different error instances"

                self._record_test_result(
                    test_name=f"Error Classification: {test_case['name']}",
                    passed=True,
                    details=f"Category: {classified_error.category.value}, Severity: {classified_error.severity.value}"
                )

            except Exception as e:
                self._record_test_result(
                    test_name=f"Error Classification: {test_case['name']}",
                    passed=False,
                    details=str(e)
                )

    def _test_retry_logic_system(self) -> None:
        """Test Retry Logic with Exponential Backoff."""
        logger.info("🔄 Testing Retry Logic System...")

        # Test 1: Successful retry operation
        try:
            attempt_count = 0

            def flaky_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ConnectionError("Temporary failure")
                return "success_after_retries"

            result = self.test_components['retry_manager'].execute_with_retry(
                operation=flaky_operation,
                policy=self.test_components['retry_manager'].get_policy("api_calls")
            )

            assert result == "success_after_retries", f"Expected success, got {result}"
            assert attempt_count == 3, f"Expected 3 attempts, got {attempt_count}"

            self._record_test_result(
                test_name="Retry Logic: Successful Retry",
                passed=True,
                details=f"Operation succeeded after {attempt_count} attempts"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Retry Logic: Successful Retry",
                passed=False,
                details=str(e)
            )

        # Test 2: Custom retry policy
        try:
            custom_policy = RetryPolicy(
                name="test_policy",
                max_attempts=2,
                base_delay=0.1,
                max_delay=1.0,
                backoff_strategy=BackoffStrategy.LINEAR,
                jitter=False
            )

            attempt_count = 0

            def failing_operation():
                nonlocal attempt_count
                attempt_count += 1
                raise ValueError("Always fails")

            start_time = time.time()
            try:
                self.test_components['retry_manager'].execute_with_retry(
                    operation=failing_operation,
                    policy=custom_policy
                )
                assert False, "Should have raised an exception"
            except ValueError:
                pass  # Expected

            duration = time.time() - start_time
            assert attempt_count == 2, f"Expected 2 attempts, got {attempt_count}"
            assert duration < 2.0, f"Duration too long: {duration:.2f}s"

            self._record_test_result(
                test_name="Retry Logic: Custom Policy",
                passed=True,
                details=f"Policy executed correctly: {attempt_count} attempts in {duration:.2f}s"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Retry Logic: Custom Policy",
                passed=False,
                details=str(e)
            )

    def _test_error_message_formatting(self) -> None:
        """Test User-Friendly Error Message Formatting."""
        logger.info("💬 Testing Error Message Formatting...")

        test_cases = [
            {
                'name': 'End User Simple Message',
                'audience': AudienceType.END_USER,
                'tone': MessageTone.FRIENDLY,
                'complexity': MessageComplexity.SIMPLE,
                'expected_elements': ['descrizione', 'cosa puoi fare']
            },
            {
                'name': 'Technical User Detailed Message',
                'audience': AudienceType.TECHNICAL_USER,
                'tone': MessageTone.PROFESSIONAL,
                'complexity': MessageComplexity.DETAILED,
                'expected_elements': ['timestamp', 'sistema']
            },
            {
                'name': 'System Admin Urgent Message',
                'audience': AudienceType.SYSTEM_ADMIN,
                'tone': MessageTone.URGENT,
                'complexity': MessageComplexity.TECHNICAL,
                'expected_elements': ['immediato', 'sistema']
            }
        ]

        # Create a test error
        test_error = ConnectionError("Database connection failed")
        classified_error = self.test_components['error_classifier'].classify_error(test_error)

        for test_case in test_cases:
            try:
                # Format error message
                formatted_message = self.test_components['message_formatter'].format_error_message(
                    classified_error=classified_error,
                    audience=test_case['audience'],
                    tone=test_case['tone'],
                    complexity=test_case['complexity']
                )

                # Validate message content
                message_text = formatted_message.get_message()
                assert len(message_text) > 50, f"Message too short: {len(message_text)} characters"

                for element in test_case['expected_elements']:
                    assert element.lower() in message_text.lower(), \
                        f"Expected element '{element}' not found in message"

                # Test function interface
                func_message = format_error_message(
                    classified_error,
                    test_case['audience'],
                    test_case['tone'],
                    test_case['complexity']
                )
                assert isinstance(func_message, str), "Function should return string"

                self._record_test_result(
                    test_name=f"Message Formatting: {test_case['name']}",
                    passed=True,
                    details=f"Message length: {len(message_text)}, Audience: {test_case['audience'].value}"
                )

            except Exception as e:
                self._record_test_result(
                    test_name=f"Message Formatting: {test_case['name']}",
                    passed=False,
                    details=str(e)
                )

    def _test_error_reporting_analytics(self) -> None:
        """Test Error Reporting and Analytics."""
        logger.info("📊 Testing Error Reporting and Analytics...")

        try:
            # Create test error events
            test_events = []
            for i in range(5):
                error = ValueError(f"Test error {i}")
                classified_error = self.test_components['error_classifier'].classify_error(error)

                event = self.test_components['error_reporter'].record_error_event(
                    classified_error=classified_error,
                    component_id=f"test_component_{i % 2}",
                    operation=f"test_operation_{i % 3}"
                )
                test_events.append(event)

            # Test aggregation retrieval
            aggregation = self.test_components['error_reporter.get_error_aggregation()'
                                            ](ReportingPeriod.LAST_24HOURS)

            assert aggregation.total_errors >= 5, f"Expected at least 5 errors, got {aggregation.total_errors}"
            assert len(aggregation.category_breakdown) > 0, "Category breakdown should not be empty"

            # Test alert generation
            high_severity_event = test_events[0]
            alert = self.test_components['error_reporter'].create_alert(
                error_event=high_severity_event,
                severity=AlertSeverity.HIGH,
                message="Test alert for high severity error"
            )

            assert alert.alert_id is not None, "Alert ID should be generated"
            assert alert.severity == AlertSeverity.HIGH, "Alert severity mismatch"

            # Test analytics summary
            analytics = self.test_components['error_reporter'].get_analytics_summary(
                ReportingPeriod.LAST_24HOURS
            )

            assert analytics['period'] == ReportingPeriod.LAST_24HOURS.value, "Period mismatch"
            assert 'error_trends' in analytics, "Error trends should be included"
            assert 'sla_compliance' in analytics, "SLA compliance should be included"

            # Test function interface
            func_result = report_error(
                ValueError("Function test error"),
                "function_test_component",
                "function_test_operation"
            )
            assert func_result is not None, "Function should return result"

            self._record_test_result(
                test_name="Error Reporting: Analytics",
                passed=True,
                details=f"Recorded {len(test_events)} events, created alert {alert.alert_id}"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Error Reporting: Analytics",
                passed=False,
                details=str(e)
            )

    def _test_state_manager_integration(self) -> None:
        """Test State Manager Integration."""
        logger.info("🔗 Testing State Manager Integration...")

        test_cases = [
            {
                'component_id': 'ml_predictor',
                'operation': 'prediction_request',
                'error': ConnectionError("API timeout")
            },
            {
                'component_id': 'database_manager',
                'operation': 'data_query',
                'error': TimeoutError("Query timeout")
            }
        ]

        for test_case in test_cases:
            try:
                # Test error handling with state context
                classified_error, state_context = self.test_components['state_handler'].handle_error_with_state_context(
                    error=test_case['error'],
                    component_id=test_case['component_id'],
                    operation=test_case['operation'],
                    additional_context={'test_integration': True}
                )

                # Validate state context
                assert state_context.component_id == test_case['component_id'], "Component ID mismatch"
                assert state_context.operation == test_case['operation'], "Operation mismatch"
                assert 0.0 <= state_context.state_health_score <= 1.0, "Invalid health score"
                assert state_context.last_error_time is not None, "Last error time should be set"

                # Test recovery plan creation
                recovery_plan = self.test_components['state_handler'].create_recovery_plan(
                    classified_error, state_context
                )

                assert recovery_plan.plan_id is not None, "Recovery plan ID should be generated"
                assert len(recovery_plan.recovery_steps) > 0, "Recovery steps should be generated"
                assert 0.0 <= recovery_plan.estimated_success_rate <= 1.0, "Invalid success rate"

                # Test state-aware retry execution
                operation_success = False

                def test_operation():
                    nonlocal operation_success
                    if not operation_success:
                        operation_success = True
                        raise test_case['error']  # Fail first time
                    return "operation_successful"

                result = self.test_components['state_handler'].execute_with_state_aware_retry(
                    operation_func=test_operation,
                    component_id=test_case['component_id'],
                    operation_name=test_case['operation']
                )

                assert result == "operation_successful", f"Expected success, got {result}"

                self._record_test_result(
                    test_name=f"State Integration: {test_case['component_id']}",
                    passed=True,
                    details=f"Health: {state_context.state_health_score:.2f}, Recovery steps: {len(recovery_plan.recovery_steps)}"
                )

            except Exception as e:
                self._record_test_result(
                    test_name=f"State Integration: {test_case['component_id']}",
                    passed=False,
                    details=str(e)
                )

    def _test_cross_component_integration(self) -> None:
        """Test integration between different error handling components."""
        logger.info("🔀 Testing Cross-Component Integration...")

        try:
            # Create a test error
            test_error = TimeoutError("Database connection timeout")
            component_id = "integration_test_component"
            operation = "integration_test_operation"

            # Step 1: Handle error with state context
            classified_error, state_context = handle_error_with_state(
                error=test_error,
                component_id=component_id,
                operation=operation
            )

            # Step 2: Create recovery plan
            handler = get_state_aware_error_handler()
            recovery_plan = handler.create_recovery_plan(classified_error, state_context)

            # Step 3: Format user-friendly message
            user_message = format_error_message(
                classified_error=classified_error,
                audience=AudienceType.END_USER,
                tone=MessageTone.FRIENDLY,
                complexity=MessageComplexity.SIMPLE
            )

            # Step 4: Report error with analytics
            report_result = report_error(
                error=test_error,
                component_id=component_id,
                operation=operation,
                additional_context={
                    'recovery_plan_id': recovery_plan.plan_id,
                    'state_health': state_context.state_health_score
                }
            )

            # Validate integration results
            assert classified_error.error_id is not None, "Error ID should be generated"
            assert state_context.sync_status is not None, "Sync status should be set"
            assert recovery_plan.plan_id is not None, "Recovery plan ID should be generated"
            assert len(user_message) > 50, "User message should be substantial"
            assert report_result is not None, "Error reporting should return result"

            # Test error state summary
            summary = handler.get_error_state_summary()
            assert 'active_error_contexts' in summary, "Summary should include active contexts"
            assert 'components_with_errors' in summary, "Summary should include component count"
            assert 'overall_health_score' in summary, "Summary should include health score"

            self._record_test_result(
                test_name="Cross-Component Integration",
                passed=True,
                details=f"Error ID: {classified_error.error_id[:8]}..., Health: {state_context.state_health_score:.2f}"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Cross-Component Integration",
                passed=False,
                details=str(e)
            )

    def _test_performance_scenarios(self) -> None:
        """Test performance under various scenarios."""
        logger.info("⚡ Testing Performance Scenarios...")

        # Test 1: High-frequency error handling
        try:
            start_time = time.time()
            error_count = 50

            for i in range(error_count):
                error = ValueError(f"Performance test error {i}")
                classified_error = self.test_components['error_classifier'].classify_error(error)

                # Quick state-aware handling
                handler = get_state_aware_error_handler()
                handler.handle_error_with_state_context(
                    error=error,
                    component_id="perf_test_component",
                    operation="performance_test_operation",
                    additional_context={'perf_test': True}
                )

            duration = time.time() - start_time
            errors_per_second = error_count / duration

            # Performance assertion (should handle at least 10 errors/second)
            assert errors_per_second >= 10, f"Performance too slow: {errors_per_second:.1f} errors/sec"

            self._record_test_result(
                test_name="Performance: High-Frequency Errors",
                passed=True,
                details=f"Handled {error_count} errors in {duration:.2f}s ({errors_per_second:.1f} errors/sec)"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Performance: High-Frequency Errors",
                passed=False,
                details=str(e)
            )

        # Test 2: Concurrent operations
        try:
            start_time = time.time()
            concurrent_count = 10

            async def concurrent_error_handling():
                handler = get_state_aware_error_handler()
                error = ConnectionError("Concurrent test error")

                classified_error, state_context = handler.handle_error_with_state_context(
                    error=error,
                    component_id="concurrent_test_component",
                    operation="concurrent_test_operation"
                )

                return classified_error.error_id

            # Run concurrent operations
            tasks = [concurrent_error_handling() for _ in range(concurrent_count)]
            results = asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

            # Validate results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == concurrent_count, \
                f"Expected {concurrent_count} successful results, got {len(successful_results)}"

            assert len(set(successful_results)) == concurrent_count, \
                "All error IDs should be unique"

            duration = time.time() - start_time
            operations_per_second = concurrent_count / duration

            self._record_test_result(
                test_name="Performance: Concurrent Operations",
                passed=True,
                details=f"Handled {concurrent_count} concurrent ops in {duration:.2f}s ({operations_per_second:.1f} ops/sec)"
            )

        except Exception as e:
            self._record_test_result(
                test_name="Performance: Concurrent Operations",
                passed=False,
                details=str(e)
            )

    def _record_test_result(self, test_name: str, passed: bool, details: str) -> None:
        """Record test result."""
        self.test_results['total_tests'] += 1

        if passed:
            self.test_results['passed_tests'] += 1
            status = "✅ PASS"
        else:
            self.test_results['failed_tests'] += 1
            status = "❌ FAIL"

        logger.info(f"  {status} {test_name}: {details}")

        self.test_results['test_details'].append({
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def _get_component_status(self) -> Dict[str, Any]:
        """Get status of all initialized components."""
        status = {}

        for component_name, component in self.test_components.items():
            try:
                # Basic health check for each component
                if hasattr(component, 'get_stats'):
                    stats = component.get_stats()
                    status[component_name] = {
                        'status': 'healthy',
                        'stats': stats
                    }
                elif hasattr(component, 'get_analytics_summary'):
                    analytics = component.get_analytics_summary(ReportingPeriod.LAST_24HOURS)
                    status[component_name] = {
                        'status': 'healthy',
                        'analytics': analytics
                    }
                else:
                    status[component_name] = {
                        'status': 'healthy',
                        'type': type(component).__name__
                    }
            except Exception as e:
                status[component_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        return status

    def _analyze_integration_health(self) -> Dict[str, Any]:
        """Analyze overall integration health."""
        handler = get_state_aware_error_handler()
        error_summary = handler.get_error_state_summary()

        return {
            'overall_health_score': error_summary.get('overall_health_score', 0.0),
            'active_error_contexts': error_summary.get('active_error_contexts', 0),
            'components_with_errors': error_summary.get('components_with_errors', 0),
            'recovery_plans': error_summary.get('recovery_plans', {}),
            'integration_quality': 'excellent' if error_summary.get('overall_health_score', 0) > 0.8 else 'good'
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests'] * 100) if self.test_results['total_tests'] > 0 else 0

        if success_rate < 80:
            recommendations.append("Some components may need additional configuration or debugging")

        if self.test_results['failed_tests'] > 0:
            failed_tests = [t for t in self.test_results['test_details'] if not t['passed']]
            recommendations.append(f"Review {len(failed_tests)} failed tests: {[t['test_name'] for t in failed_tests]}")

        handler = get_state_aware_error_handler()
        summary = handler.get_error_state_summary()

        if summary.get('overall_health_score', 0) < 0.7:
            recommendations.append("Monitor system health and consider preventive maintenance")

        if success_rate >= 90:
            recommendations.append("System is performing excellently - ready for production deployment")

        return recommendations


def main():
    """Main function to run the comprehensive test suite."""
    logger.info("🎯 NBA Predictor Error Handling State Integration Test Suite")
    logger.info("==============================================================")

    # Create and run test suite
    test_suite = ErrorHandlingStateIntegrationTest()
    results = test_suite.run_all_tests()

    # Display results
    print("\n" + "=" * 80)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 80)

    summary = results['test_summary']
    print(f"📊 Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")
    print(f"📈 Success Rate: {summary['success_rate']}")
    print(f"⏱️ Duration: {summary['duration_seconds']} seconds")

    # Component status
    print("\n🔧 COMPONENT STATUS:")
    for component, status in results['component_status'].items():
        status_icon = "✅" if status['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {component}: {status['status']}")

    # Integration analysis
    print("\n🔗 INTEGRATION ANALYSIS:")
    analysis = results['integration_analysis']
    print(f"  🏥 Overall Health: {analysis['overall_health_score']:.2f}")
    print(f"  📊 Active Contexts: {analysis['active_error_contexts']}")
    print(f"  🔧 Components with Errors: {analysis['components_with_errors']}")
    print(f"  ⭐ Integration Quality: {analysis['integration_quality'].upper()}")

    # Recommendations
    if results['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")

    # Final verdict
    success_rate = float(results['test_summary']['success_rate'].rstrip('%'))
    if success_rate >= 90:
        verdict = "🎉 EXCELLENT - System ready for production"
    elif success_rate >= 75:
        verdict = "✅ GOOD - System mostly functional"
    elif success_rate >= 50:
        verdict = "⚠️ MARGINAL - System needs attention"
    else:
        verdict = "❌ POOR - System requires significant work"

    print(f"\n{verdict}")
    print("=" * 80)

    # Save detailed results to file
    results_file = "error_handling_integration_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()