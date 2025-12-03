#!/usr/bin/env python3
"""
🧪 Phase 1 Implementation Test Suite

Tests the complete implementation of Phase 1: Real Data Foundations.

Author: NBA Predictive Analytics System
Task ID: phase1-test-validation
Date: 2025-01-10
"""

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import components to test
from nba_predictive_system.unified_nba_data_pipeline import (
    NBAAPIClient,
    MultiEndpointNBADataFetcher,
    UnifiedNBADataPipeline
)
from nba_predictive_system.data_validator import NBADataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase1ImplementationTest:
    """Test suite for Phase 1 implementation validation."""

    def __init__(self):
        """Initialize test suite."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.test_results = []

    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.logger.info(f"{status} {test_name}: {message}")
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })

    def test_nba_api_client_initialization(self) -> bool:
        """Test NBAAPIClient initialization."""
        try:
            client = NBAAPIClient()
            assert client.max_retries == 3
            assert client.base_delay == 1.0
            assert 'total_calls' in client.call_stats
            self.log_test_result("NBA API Client Initialization", True, "Client initialized successfully")
            return True
        except Exception as e:
            self.log_test_result("NBA API Client Initialization", False, str(e))
            return False

    def test_nba_api_client_error_classification(self) -> bool:
        """Test NBAAPIClient error classification."""
        try:
            client = NBAAPIClient()

            # Test temporary error classification
            temp_error = TimeoutError("Request timeout")
            assert client.classify_error(temp_error) == 'temporary'

            # Test rate limit error classification
            rate_error = Exception("429 Too Many Requests")
            assert client.classify_error(rate_error) == 'rate_limit'

            # Test permanent error classification
            perm_error = Exception("404 Not Found")
            assert client.classify_error(perm_error) == 'permanent'

            self.log_test_result("NBA API Client Error Classification", True, "All error types classified correctly")
            return True
        except Exception as e:
            self.log_test_result("NBA API Client Error Classification", False, str(e))
            return False

    def test_multi_endpoint_fetcher_initialization(self) -> bool:
        """Test MultiEndpointNBADataFetcher initialization."""
        try:
            fetcher = MultiEndpointNBADataFetcher()
            assert len(fetcher.endpoints) == 3
            assert all('name' in endpoint for endpoint in fetcher.endpoints)
            assert fetcher.endpoints[0]['priority'] == 1  # stats.nba.com should be first
            self.log_test_result("Multi-Endpoint Fetcher Initialization", True, f"Initialized with {len(fetcher.endpoints)} endpoints")
            return True
        except Exception as e:
            self.log_test_result("Multi-Endpoint Fetcher Initialization", False, str(e))
            return False

    def test_multi_endpoint_health_check(self) -> bool:
        """Test MultiEndpointNBADataFetcher health checks."""
        try:
            fetcher = MultiEndpointNBADataFetcher()

            # All endpoints should be healthy initially
            healthy_endpoints = fetcher.get_healthy_endpoints()
            assert len(healthy_endpoints) == 3

            # Test endpoint statistics
            stats = fetcher.get_endpoint_statistics()
            assert len(stats) == 3
            assert 'stats.nba.com' in stats

            self.log_test_result("Multi-Endpoint Health Check", True, "All endpoints healthy initially")
            return True
        except Exception as e:
            self.log_test_result("Multi-Endpoint Health Check", False, str(e))
            return False

    def test_data_validator_initialization(self) -> bool:
        """Test NBADataValidator initialization."""
        try:
            validator = NBADataValidator()
            assert len(validator.required_fields) > 0
            assert 'games' in validator.required_fields
            assert len(validator.data_types) > 0
            assert 'game_id' in validator.data_types
            self.log_test_result("Data Validator Initialization", True, "Validator initialized with required fields and types")
            return True
        except Exception as e:
            self.log_test_result("Data Validator Initialization", False, str(e))
            return False

    def test_data_validator_games_validation(self) -> bool:
        """Test NBADataValidator games data validation."""
        try:
            validator = NBADataValidator()

            # Create test games data
            test_games = pd.DataFrame({
                'game_id': ['0012400001', '0012400002'],
                'game_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
                'home_team': ['Lakers', 'Celtics'],
                'away_team': ['Celtics', 'Lakers'],
                'home_score': [120, 115],
                'away_score': [115, 120]
            })

            report = validator.validate_games_data(test_games)
            assert report.is_valid == True
            assert report.quality_score > 0.9
            assert report.total_rows == 2

            self.log_test_result("Data Validator Games Validation", True, f"Quality score: {report.quality_score:.2f}")
            return True
        except Exception as e:
            self.log_test_result("Data Validator Games Validation", False, str(e))
            return False

    def test_data_validator_error_detection(self) -> bool:
        """Test NBADataValidator error detection capabilities."""
        try:
            validator = NBADataValidator()

            # Create test data with errors
            bad_games = pd.DataFrame({
                'game_id': ['0012400001', '0012400002'],
                'game_date': ['2024-01-01', 'invalid_date'],  # One valid, one invalid date
                'home_team': ['Lakers', 'Lakers'],  # Same team playing itself
                'away_team': ['Lakers', 'Celtics'],
                'home_score': [-10, 115],  # Negative score
                'away_score': [115, 200]  # Valid but high score
            })

            report = validator.validate_games_data(bad_games)
            # Should detect issues (either marked as invalid OR have quality score < 1.0)
            has_issues = not report.is_valid or report.quality_score < 1.0
            assert has_issues  # Should detect issues
            assert report.errors_count > 0  # Should detect errors

            self.log_test_result("Data Validator Error Detection", True, f"Detected {report.errors_count} issues, quality score: {report.quality_score:.2f}")
            return True
        except Exception as e:
            self.log_test_result("Data Validator Error Detection", False, f"Error: {type(e).__name__}: {str(e)}")
            return False

    def test_data_sanitization(self) -> bool:
        """Test data sanitization functionality."""
        try:
            validator = NBADataValidator()

            # Create data with issues
            dirty_data = pd.DataFrame({
                'game_id': ['0012400001', None, '0012400002', '0012400001'],  # Duplicate and missing
                'game_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01']),
                'home_team': ['Lakers', 'Celtics', None, 'Lakers'],  # Missing value
                'away_team': ['Celtics', 'Lakers', 'Lakers', 'Celtics'],
                'home_score': [120, 115, None, 120],  # Missing value
                'away_score': [115, 120, 100, 115]
            })

            cleaned_data = validator.sanitize_data(dirty_data, 'games')

            # Should have fewer rows after cleaning
            assert len(cleaned_data) <= len(dirty_data)
            assert cleaned_data['game_id'].isna().sum() == 0  # No missing game_ids
            assert len(cleaned_data) == len(cleaned_data['game_id'].unique())  # No duplicates

            self.log_test_result("Data Sanitization", True, f"Cleaned from {len(dirty_data)} to {len(cleaned_data)} rows")
            return True
        except Exception as e:
            self.log_test_result("Data Sanitization", False, str(e))
            return False

    def test_unified_pipeline_initialization(self) -> bool:
        """Test UnifiedNBADataPipeline initialization."""
        try:
            pipeline = UnifiedNBADataPipeline()
            assert pipeline.data_validator is not None
            assert hasattr(pipeline, '_fetch_boxscores_data')
            self.log_test_result("Unified Pipeline Initialization", True, "Pipeline initialized with all components")
            return True
        except Exception as e:
            self.log_test_result("Unified Pipeline Initialization", False, str(e))
            return False

    def test_integration_components(self) -> bool:
        """Test integration between components."""
        try:
            # Create components
            api_client = NBAAPIClient()
            multi_fetcher = MultiEndpointNBADataFetcher()
            validator = NBADataValidator()
            pipeline = UnifiedNBADataPipeline()

            # Verify they can work together
            assert hasattr(multi_fetcher, 'api_clients')
            assert len(multi_fetcher.api_clients) == 3

            # Verify validator is properly integrated
            assert isinstance(pipeline.data_validator, NBADataValidator)

            self.log_test_result("Component Integration", True, "All components properly integrated")
            return True
        except Exception as e:
            self.log_test_result("Component Integration", False, str(e))
            return False

    def run_all_tests(self) -> dict:
        """Run all Phase 1 implementation tests."""
        self.logger.info("🚀 Starting Phase 1 Implementation Test Suite")
        self.logger.info("=" * 60)

        tests = [
            self.test_nba_api_client_initialization,
            self.test_nba_api_client_error_classification,
            self.test_multi_endpoint_fetcher_initialization,
            self.test_multi_endpoint_health_check,
            self.test_data_validator_initialization,
            self.test_data_validator_games_validation,
            self.test_data_validator_error_detection,
            self.test_data_sanitization,
            self.test_unified_pipeline_initialization,
            self.test_integration_components
        ]

        for test in tests:
            test()

        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100

        self.logger.info("=" * 60)
        self.logger.info(f"🏁 Test Suite Completed")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests} ✅")
        self.logger.info(f"Failed: {failed_tests} ❌")
        self.logger.info(f"Success Rate: {success_rate:.1f}%")

        if failed_tests > 0:
            self.logger.warning("Failed tests:")
            for result in self.test_results:
                if not result['passed']:
                    self.logger.warning(f"  - {result['test']}: {result['message']}")

        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': success_rate,
            'results': self.test_results
        }


def main():
    """Main test execution function."""
    print("🧪 Phase 1 Implementation Test Suite")
    print("=" * 50)
    print("Testing NBA Data Pipeline Real Data Foundations")
    print("=" * 50)

    # Create and run tests
    test_suite = Phase1ImplementationTest()
    results = test_suite.run_all_tests()

    # Exit with appropriate code
    if results['failed'] == 0:
        print("\n🎉 All tests passed! Phase 1 implementation is ready.")
        sys.exit(0)
    else:
        print(f"\n⚠️ {results['failed']} test(s) failed. Please review implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()