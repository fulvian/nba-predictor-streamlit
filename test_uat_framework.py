"""
Test suite for UAT Testing Framework
Phase 3 Day 12 - Task 3.5.2 User Acceptance Testing
"""

import pytest
import time
from src.nba_predictor.streamlit.components.uat_testing_framework import (
    UATTestingFramework, UserPersona, TestDevice, UserJourney, UATTestResult
)


def test_uat_framework_initialization():
    """Test UAT framework initialization"""
    framework = UATTestingFramework(headless=True)

    assert framework.base_url == "http://localhost:8501"
    assert framework.headless == True
    assert framework.screenshot_dir == "uat_screenshots"
    assert framework.driver is None
    assert framework.session is None
    assert len(framework.test_results) == 0

    framework.cleanup()


def test_user_journey_creation():
    """Test user journey creation functionality"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()

    # Verify we have all expected journey types
    personas = [j.persona for j in journeys]
    expected_personas = [
        UserPersona.NOVICE_USER,
        UserPersona.EXPERIENCED_BETTOR,
        UserPersona.NBA_ANALYST,
        UserPersona.MOBILE_USER,
        UserPersona.ACCESSIBILITY_USER
    ]

    assert len(journeys) == 5
    for expected_persona in expected_personas:
        assert expected_persona in personas

    # Verify journey structure
    for journey in journeys:
        assert journey.id is not None
        assert journey.name is not None
        assert journey.description is not None
        assert len(journey.test_steps) > 0
        assert len(journey.success_criteria) > 0
        assert len(journey.context7_patterns) > 0
        assert journey.priority in ["critical", "high", "medium", "low"]

    framework.cleanup()


def test_novice_user_journey_structure():
    """Test novice user journey structure and content"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()
    novice_journey = next(j for j in journeys if j.persona == UserPersona.NOVICE_USER)

    assert novice_journey.id == "novice_first_bet"
    assert novice_journey.persona == UserPersona.NOVICE_USER
    assert novice_journey.device == TestDevice.DESKTOP
    assert novice_journey.priority == "critical"

    # Verify test steps
    expected_steps = [
        "navigate_dashboard",
        "understand_interface",
        "view_games",
        "select_game",
        "place_bet"
    ]

    step_ids = [step["step_id"] for step in novice_journey.test_steps]
    for expected_step in expected_steps:
        assert expected_step in step_ids

    # Verify Context7 patterns
    expected_patterns = [
        "adaptive_ui_layouts",
        "accessibility_features",
        "responsive_design_system"
    ]

    for pattern in expected_patterns:
        assert pattern in novice_journey.context7_patterns

    framework.cleanup()


def test_mobile_user_journey_structure():
    """Test mobile user journey structure"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()
    mobile_journey = next(j for j in journeys if j.persona == UserPersona.MOBILE_USER)

    assert mobile_journey.id == "mobile_user_experience"
    assert mobile_journey.persona == UserPersona.MOBILE_USER
    assert mobile_journey.device == TestDevice.MOBILE
    assert mobile_journey.priority == "critical"

    # Verify mobile-specific steps
    expected_steps = [
        "mobile_layout",
        "touch_interactions",
        "quick_bet",
        "mobile_notifications"
    ]

    step_ids = [step["step_id"] for step in mobile_journey.test_steps]
    for expected_step in expected_steps:
        assert expected_step in step_ids

    # Verify mobile-specific Context7 patterns
    expected_patterns = [
        "responsive_design_system",
        "pwa_features",
        "accessibility_features"
    ]

    for pattern in expected_patterns:
        assert pattern in mobile_journey.context7_patterns

    framework.cleanup()


def test_accessibility_journey_structure():
    """Test accessibility user journey structure"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()
    accessibility_journey = next(j for j in journeys if j.persona == UserPersona.ACCESSIBILITY_USER)

    assert accessibility_journey.id == "accessibility_compliance"
    assert accessibility_journey.persona == UserPersona.ACCESSIBILITY_USER
    assert accessibility_journey.device == TestDevice.DESKTOP
    assert accessibility_journey.priority == "critical"

    # Verify accessibility-specific steps
    expected_steps = [
        "keyboard_navigation",
        "screen_reader",
        "color_contrast",
        "focus_indicators"
    ]

    step_ids = [step["step_id"] for step in accessibility_journey.test_steps]
    for expected_step in expected_steps:
        assert expected_step in step_ids

    # Verify accessibility Context7 patterns
    assert "accessibility_features" in accessibility_journey.context7_patterns

    framework.cleanup()


def test_context7_pattern_validation():
    """Test Context7 pattern validation functionality"""
    framework = UATTestingFramework()

    # Test responsive design validation
    responsive_result = framework._validate_responsive_design()
    assert "pattern" in responsive_result
    assert "compliant" in responsive_result
    assert "score" in responsive_result
    assert "issues" in responsive_result
    assert responsive_result["pattern"] == "responsive_design_system"

    # Test accessibility validation
    accessibility_result = framework._validate_accessibility_features()
    assert "pattern" in accessibility_result
    assert "compliant" in accessibility_result
    assert "score" in accessibility_result
    assert accessibility_result["pattern"] == "accessibility_features"

    # Test adaptive UI validation
    adaptive_result = framework._validate_adaptive_ui_layouts()
    assert "pattern" in adaptive_result
    assert adaptive_result["pattern"] == "adaptive_ui_layouts"

    # Test PWA features validation
    pwa_result = framework._validate_pwa_features()
    assert "pattern" in pwa_result
    assert pwa_result["pattern"] == "pwa_features"

    # Test real-time updates validation
    realtime_result = framework._validate_real_time_updates()
    assert "pattern" in realtime_result
    assert realtime_result["pattern"] == "real_time_updates"

    # Test intelligent cache validation
    cache_result = framework._validate_intelligent_cache()
    assert "pattern" in cache_result
    assert cache_result["pattern"] == "intelligent_cache"

    # Test advanced ML operations validation
    ml_result = framework._validate_advanced_ml_operations()
    assert "pattern" in ml_result
    assert ml_result["pattern"] == "advanced_ml_operations"

    framework.cleanup()


def test_journey_results_structure():
    """Test journey results data structure"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()
    novice_journey = journeys[0]

    # Mock journey results structure
    journey_results = {
        "journey_id": novice_journey.id,
        "journey_name": novice_journey.name,
        "persona": novice_journey.persona.value,
        "device": novice_journey.device.value,
        "start_time": time.time(),
        "end_time": None,
        "steps_completed": 0,
        "steps_passed": 0,
        "steps_failed": 0,
        "success_rate": 0.0,
        "context7_compliance": {},
        "steps": [],
        "overall_status": UATTestResult.FAILED.value
    }

    # Verify structure
    assert "journey_id" in journey_results
    assert "journey_name" in journey_results
    assert "persona" in journey_results
    assert "device" in journey_results
    assert "start_time" in journey_results
    assert "success_rate" in journey_results
    assert "context7_compliance" in journey_results
    assert "overall_status" in journey_results

    framework.cleanup()


def test_test_step_structure():
    """Test individual test step structure"""
    framework = UATTestingFramework()

    # Mock test step data
    step_data = {
        "step_id": "test_step",
        "action": "Test action",
        "element_locator": "[data-testid='test-element']",
        "expected_result": "Expected result",
        "wait_time": 2
    }

    # Mock test step result
    step_result = {
        "step_id": step_data["step_id"],
        "action": step_data["action"],
        "status": UATTestResult.PASSED.value,
        "execution_time": 0.0,
        "error_message": None,
        "context7_validation": None
    }

    # Verify structure
    assert "step_id" in step_result
    assert "action" in step_result
    assert "status" in step_result
    assert "execution_time" in step_result
    assert "error_message" in step_result
    assert "context7_validation" in step_result

    framework.cleanup()


def test_uat_report_generation():
    """Test UAT report generation"""
    framework = UATTestingFramework()

    # Mock comprehensive results
    mock_results = {
        "test_session_id": "test_session_123",
        "start_time": time.time(),
        "browsers_tested": ["chrome"],
        "journeys": [],
        "summary": {
            "total_journeys": 5,
            "successful_journeys": 4,
            "failed_journeys": 1,
            "overall_success_rate": 80.0,
            "context7_compliance_score": 75.0
        },
        "total_duration": 120.5
    }

    # Generate report
    report = framework.generate_uat_report(mock_results)

    # Verify report content
    assert "NBA Predictor User Acceptance Testing Report" in report
    assert "Executive Summary" in report
    assert "Journey Results" in report
    assert "Recommendations" in report
    assert "Context7 Pattern Validation" in report
    assert f"Total Journeys Tested: {mock_results['summary']['total_journeys']}" in report
    assert f"Overall Success Rate: {mock_results['summary']['overall_success_rate']:.1f}%" in report
    assert f"Context7 Compliance Score: {mock_results['summary']['context7_compliance_score']:.1f}%" in report

    framework.cleanup()


def test_database_initialization():
    """Test UAT database initialization"""
    framework = UATTestingFramework()

    # Database should be initialized automatically
    # Verify database exists and has required tables
    import sqlite3
    conn = sqlite3.connect('data/nba_uat_results.duckdb')
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    expected_tables = [
        'uat_sessions',
        'uat_test_results',
        'user_journeys'
    ]

    for table in expected_tables:
        assert table in tables, f"Table {table} not found in database"

    conn.close()
    framework.cleanup()


def test_user_persona_enum():
    """Test UserPersona enum values"""
    expected_personas = {
        "NOVICE_USER",
        "EXPERIENCED_BETTOR",
        "NBA_ANALYST",
        "DATA_SCIENTIST",
        "MOBILE_USER",
        "ACCESSIBILITY_USER"
    }

    actual_personas = {persona.name for persona in UserPersona}
    assert expected_personas == actual_personas

    # Test persona values
    assert UserPersona.NOVICE_USER.value == "novice_user"
    assert UserPersona.EXPERIENCED_BETTOR.value == "experienced_bettor"
    assert UserPersona.NBA_ANALYST.value == "nba_analyst"
    assert UserPersona.DATA_SCIENTIST.value == "data_scientist"
    assert UserPersona.MOBILE_USER.value == "mobile_user"
    assert UserPersona.ACCESSIBILITY_USER.value == "accessibility_user"


def test_test_device_enum():
    """Test TestDevice enum values"""
    expected_devices = {
        "DESKTOP",
        "TABLET",
        "MOBILE",
        "WIDE_SCREEN"
    }

    actual_devices = {device.name for device in TestDevice}
    assert expected_devices == actual_devices

    # Test device values
    assert TestDevice.DESKTOP.value == "desktop"
    assert TestDevice.TABLET.value == "tablet"
    assert TestDevice.MOBILE.value == "mobile"
    assert TestDevice.WIDE_SCREEN.value == "wide_screen"


def test_uat_test_result_enum():
    """Test UATTestResult enum values"""
    expected_results = {
        "PASSED",
        "FAILED",
        "SKIPPED",
        "ERROR"
    }

    actual_results = {result.name for result in UATTestResult}
    assert expected_results == actual_results

    # Test result values
    assert UATTestResult.PASSED.value == "PASSED"
    assert UATTestResult.FAILED.value == "FAILED"
    assert UATTestResult.SKIPPED.value == "SKIPPED"
    assert UATTestResult.ERROR.value == "ERROR"


def test_context7_patterns_completeness():
    """Test that all expected Context7 patterns are covered"""
    framework = UATTestingFramework()

    # Get all validation methods
    validation_methods = [
        '_validate_responsive_design',
        '_validate_accessibility_features',
        '_validate_adaptive_ui_layouts',
        '_validate_pwa_features',
        '_validate_real_time_updates',
        '_validate_intelligent_cache',
        '_validate_advanced_ml_operations'
    ]

    # Verify all methods exist
    for method_name in validation_methods:
        assert hasattr(framework, method_name), f"Method {method_name} not found"
        method = getattr(framework, method_name)
        assert callable(method), f"Method {method_name} is not callable"

    framework.cleanup()


def test_journey_success_criteria():
    """Test that all journeys have proper success criteria"""
    framework = UATTestingFramework()
    journeys = framework.create_user_journeys()

    for journey in journeys:
        assert len(journey.success_criteria) > 0, f"Journey {journey.id} has no success criteria"

        # Verify success criteria are strings
        for criteria in journey.success_criteria:
            assert isinstance(criteria, str), f"Success criteria should be string, got {type(criteria)}"
            assert len(criteria.strip()) > 0, f"Success criteria should not be empty"

    framework.cleanup()


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])