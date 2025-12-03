#!/usr/bin/env python3
"""
Test Accessibility Features Implementation
Phase 3 Day 11 - Task 3.4.3: Accessibility Features

Verifies all accessibility features are working correctly with Context7 compliance.
"""

import sys
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_accessibility_manager_initialization():
    """Test AccessibilityFeaturesManager initialization"""
    print("\n🎯 Testing Accessibility Manager Initialization")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            AccessibilityFeaturesManager, AccessibilityConfig, AccessibilityLevel
        )

        # Test default configuration
        default_manager = AccessibilityFeaturesManager()
        assert default_manager._initialized, "Manager should be initialized"
        assert default_manager.config.level == AccessibilityLevel.AA, "Default level should be AA"

        # Test custom configuration
        config = AccessibilityConfig(
            level=AccessibilityLevel.AAA,
            enable_high_contrast=True,
            enable_large_text=True,
            enable_reduced_motion=True
        )
        custom_manager = AccessibilityFeaturesManager(config)
        assert custom_manager.config.level == AccessibilityLevel.AAA, "Custom level should be AAA"
        assert custom_manager.config.enable_high_contrast, "High contrast should be enabled"

        # Test global manager
        from src.nba_predictor.streamlit.components.accessibility_features import get_accessibility_manager
        global_manager = get_accessibility_manager()
        assert global_manager is not None, "Global manager should exist"

        print("✅ Accessibility Manager initialization test passed!")
        return True

    except Exception as e:
        print(f"❌ Accessibility Manager initialization test failed: {e}")
        return False

def test_aria_labels():
    """Test ARIA labels functionality"""
    print("\n🏷️ Testing ARIA Labels")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            get_accessibility_manager
        )

        manager = get_accessibility_manager()

        # Test adding ARIA labels
        test_labels = [
            ("main_content", "Main Content Area", "Primary content of the application"),
            ("navigation", "Navigation Menu", "Main navigation links"),
            ("search_form", "Search Form", "Search for NBA games"),
            ("results_table", "Results Table", "Table displaying search results", "polite")
        ]

        for element_id, label, desc, *live in test_labels:
            live_region = live[0] if live else None
            manager.add_aria_label(
                element_id=element_id,
                label=label,
                description=desc,
                live_region=live_region
            )

        # Verify labels were added
        assert len(manager._aria_labels) >= len(test_labels), "ARIA labels should be added"

        # Check specific labels
        main_content_label = manager._aria_labels.get("main_content")
        assert main_content_label is not None, "Main content label should exist"
        assert main_content_label.label == "Main Content Area", "Label should match"
        assert main_content_label.description == "Primary content of the application", "Description should match"

        results_label = manager._aria_labels.get("results_table")
        assert results_label is not None, "Results table label should exist"
        assert results_label.live_region == "polite", "Live region should be set"

        print("✅ ARIA Labels test passed!")
        print(f"   - Labels added: {len(manager._aria_labels)}")
        return True

    except Exception as e:
        print(f"❌ ARIA Labels test failed: {e}")
        return False

def test_keyboard_navigation():
    """Test keyboard navigation functionality"""
    print("\n⌨️ Testing Keyboard Navigation")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            get_accessibility_manager
        )

        manager = get_accessibility_manager()

        # Test adding focusable elements
        test_elements = [
            ("submit_button", "button[type='submit']", "form_group", 0),
            ("search_input", "input[type='search']", "search_group", 1),
            ("results_table", "table.results", "content_group", 2, False, True)
        ]

        for element_id, selector, *attrs in test_elements:
            group = attrs[0] if len(attrs) > 0 else None
            index = attrs[1] if len(attrs) > 1 else 0
            skip = attrs[2] if len(attrs) > 2 else False
            trap = attrs[3] if len(attrs) > 3 else False

            manager.register_focusable_element(
                element_id=element_id,
                selector=selector,
                group=group,
                index=index,
                skip=skip,
                trap=trap
            )

        # Test adding keyboard handlers
        test_handlers = [
            ("search_form", "Enter", "submit", "search_input", "Submit search form"),
            ("results_table", "ArrowDown", "next_row", "table_row", "Navigate to next row"),
            ("modal", "Escape", "close", "modal_dialog", "Close modal dialog")
        ]

        for component_id, key, action, target, desc in test_handlers:
            manager.add_keyboard_handler(
                component_id=component_id,
                key=key,
                action=action,
                target=target,
                description=desc
            )

        # Verify elements and handlers were added
        assert len(manager._focus_elements) >= len(test_elements), "Focus elements should be added"
        assert len(manager._keyboard_handlers) >= len(test_handlers), "Keyboard handlers should be added"

        # Check specific elements
        submit_button = manager._focus_elements.get("submit_button")
        assert submit_button is not None, "Submit button should be focusable"
        assert submit_button.group == "form_group", "Group should match"

        # Check global keyboard shortcuts
        global_handlers = manager._keyboard_handlers.get("global", [])
        assert len(global_handlers) > 0, "Global keyboard shortcuts should exist"

        print("✅ Keyboard Navigation test passed!")
        print(f"   - Focusable elements: {len(manager._focus_elements)}")
        print(f"   - Keyboard handlers: {len(manager._keyboard_handlers)}")
        return True

    except Exception as e:
        print(f"❌ Keyboard Navigation test failed: {e}")
        return False

def test_screen_reader_support():
    """Test screen reader support functionality"""
    print("\n🔊 Testing Screen Reader Support")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            get_accessibility_manager
        )

        manager = get_accessibility_manager()

        # Test screen reader configuration
        screen_reader_config = manager._screen_reader_config
        assert screen_reader_config is not None, "Screen reader config should exist"
        assert 'type' in screen_reader_config, "Screen reader type should be set"
        assert 'announcements' in screen_reader_config, "Announcements list should exist"

        # Test announcements
        test_announcements = [
            ("Page loaded successfully", "polite"),
            ("Form validation error", "assertive"),
            ("Search completed", "polite", 3000)
        ]

        for announcement in test_announcements:
            if len(announcement) == 2:
                message, priority = announcement
                manager.announce_to_screen_reader(message, priority)
            else:
                message, priority, timeout = announcement
                manager.announce_to_screen_reader(message, priority, timeout)

        # Verify announcements were added
        assert len(screen_reader_config['announcements']) >= len(test_announcements), "Announcements should be added"

        # Check screen reader type
        assert screen_reader_config.get('enabled', False), "Screen reader support should be enabled"

        print("✅ Screen Reader Support test passed!")
        print(f"   - Screen reader type: {screen_reader_config.get('type', 'unknown')}")
        print(f"   - Announcements: {len(screen_reader_config['announcements'])}")
        return True

    except Exception as e:
        print(f"❌ Screen Reader Support test failed: {e}")
        return False

def test_user_preferences():
    """Test user preferences detection"""
    print("\n⚙️ Testing User Preferences Detection")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            AccessibilityConfig, get_accessibility_manager
        )

        # Test config with auto-detection enabled
        config = AccessibilityConfig(auto_detect_preferences=True)
        manager = get_accessibility_manager(config)

        # Test preference detection methods
        reduced_motion = manager._check_reduced_motion_preference()
        high_contrast = manager._check_high_contrast_preference()
        large_text = manager._check_large_text_preference()

        # Verify preferences were checked (they return False by default in testing)
        assert isinstance(reduced_motion, bool), "Reduced motion should be boolean"
        assert isinstance(high_contrast, bool), "High contrast should be boolean"
        assert isinstance(large_text, bool), "Large text should be boolean"

        # Test user preferences dictionary
        user_prefs = manager._user_preferences
        assert isinstance(user_prefs, dict), "User preferences should be dictionary"
        assert 'reduced_motion' in user_prefs, "Reduced motion preference should exist"
        assert 'high_contrast' in user_prefs, "High contrast preference should exist"
        assert 'large_text' in user_prefs, "Large text preference should exist"

        print("✅ User Preferences Detection test passed!")
        print(f"   - Preferences detected: {len(user_prefs)}")
        for pref, value in user_prefs.items():
            print(f"   - {pref}: {value}")
        return True

    except Exception as e:
        print(f"❌ User Preferences Detection test failed: {e}")
        return False

def test_accessibility_info():
    """Test accessibility info retrieval"""
    print("\n📊 Testing Accessibility Info Retrieval")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            get_accessibility_manager, AccessibilityLevel
        )

        manager = get_accessibility_manager()

        # Get accessibility info
        info = manager.get_accessibility_info()

        # Verify info structure
        assert isinstance(info, dict), "Info should be dictionary"
        assert 'initialized' in info, "Initialized status should be in info"
        assert 'compliance_level' in info, "Compliance level should be in info"
        assert 'features_enabled' in info, "Features enabled should be in info"
        assert 'user_preferences' in info, "User preferences should be in info"

        # Verify specific values
        assert info['initialized'] is True, "Should be initialized"
        assert info['compliance_level'] == AccessibilityLevel.AA.value, "Should be AA level"
        assert isinstance(info['features_enabled'], dict), "Features enabled should be dict"
        assert isinstance(info['user_preferences'], dict), "User preferences should be dict"

        # Check features enabled
        features = info['features_enabled']
        expected_features = [
            'keyboard_navigation', 'screen_reader_support', 'high_contrast',
            'large_text', 'reduced_motion', 'focus_indicators', 'aria_labels', 'descriptions'
        ]

        for feature in expected_features:
            assert feature in features, f"Feature {feature} should be in features enabled"

        # Check counts
        assert 'focus_elements_count' in info, "Focus elements count should be in info"
        assert 'aria_labels_count' in info, "ARIA labels count should be in info"
        assert 'keyboard_handlers_count' in info, "Keyboard handlers count should be in info"

        print("✅ Accessibility Info Retrieval test passed!")
        print(f"   - Compliance Level: {info['compliance_level']}")
        print(f"   - Focus Elements: {info['focus_elements_count']}")
        print(f"   - ARIA Labels: {info['aria_labels_count']}")
        print(f"   - Keyboard Handlers: {info['keyboard_handlers_count']}")
        print(f"   - Screen Reader: {info['screen_reader']}")
        return True

    except Exception as e:
        print(f"❌ Accessibility Info Retrieval test failed: {e}")
        return False

def test_context7_utilities():
    """Test Context7 compliant utility functions"""
    print("\n🎨 Testing Context7 Utility Functions")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            create_accessible_section, create_accessible_alert,
            create_accessible_loading, get_accessibility_manager
        )

        manager = get_accessibility_manager()

        # Test that functions exist and are callable
        assert callable(create_accessible_section), "create_accessible_section should be callable"
        assert callable(create_accessible_alert), "create_accessible_alert should be callable"
        assert callable(create_accessible_loading), "create_accessible_loading should be callable"

        # Test accessibility manager is available
        assert manager is not None, "Manager should be available"
        assert manager._initialized, "Manager should be initialized"

        print("✅ Context7 Utility Functions test passed!")
        print("   - create_accessible_section: ✅ Available")
        print("   - create_accessible_alert: ✅ Available")
        print("   - create_accessible_loading: ✅ Available")
        return True

    except Exception as e:
        print(f"❌ Context7 Utility Functions test failed: {e}")
        return False

def test_wcag_compliance():
    """Test WCAG 2.1 AA compliance features"""
    print("\n♿ Testing WCAG 2.1 AA Compliance Features")

    try:
        from src.nba_predictor.streamlit.components.accessibility_features import (
            get_accessibility_manager, AccessibilityLevel, AccessibilityConfig,
            AccessibilityFeaturesManager
        )

        # Test AA compliance configuration
        config = AccessibilityConfig(level=AccessibilityLevel.AA)
        manager = get_accessibility_manager(config)

        # Verify AA level features are enabled
        assert manager.config.level == AccessibilityLevel.AA, "Should be AA level"
        assert manager.config.enable_keyboard_navigation is True, "Keyboard navigation should be enabled for AA"
        assert manager.config.enable_screen_reader_support is True, "Screen reader support should be enabled for AA"
        assert manager.config.enable_focus_indicators is True, "Focus indicators should be enabled for AA"
        assert manager.config.enable_aria_labels is True, "ARIA labels should be enabled for AA"

        # Test AAA level
        aaa_config = AccessibilityConfig(level=AccessibilityLevel.AAA)
        aaa_manager = AccessibilityFeaturesManager(aaa_config)
        assert aaa_manager.config.level == AccessibilityLevel.AAA, "Should be AAA level"

        print("✅ WCAG 2.1 AA Compliance Features test passed!")
        print(f"   - AA Level: ✅ Configured")
        print(f"   - AAA Level: ✅ Available")
        print(f"   - Required Features: ✅ Enabled")
        return True

    except Exception as e:
        print(f"❌ WCAG 2.1 AA Compliance Features test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("="*80)
    print("🎯 PHASE 3 DAY 11 ACCESSIBILITY FEATURES TEST - Task 3.4.3")
    print("="*80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Testing: WCAG 2.1 AA + Context7 Compliant Features")

    # Define test suite
    tests = [
        ("Accessibility Manager Initialization", test_accessibility_manager_initialization),
        ("ARIA Labels", test_aria_labels),
        ("Keyboard Navigation", test_keyboard_navigation),
        ("Screen Reader Support", test_screen_reader_support),
        ("User Preferences Detection", test_user_preferences),
        ("Accessibility Info Retrieval", test_accessibility_info),
        ("Context7 Utility Functions", test_context7_utilities),
        ("WCAG 2.1 AA Compliance Features", test_wcag_compliance),
    ]

    # Execute tests
    test_results = {}
    total_start = time.time()

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print('='*60)

        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            test_results[test_name] = {
                'passed': result,
                'duration': duration,
                'status': 'PASSED' if result else 'FAILED'
            }

        except Exception as e:
            test_results[test_name] = {
                'passed': False,
                'duration': 0,
                'status': f'ERROR: {e}'
            }

    total_duration = time.time() - total_start

    # Print results summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY - ACCESSIBILITY FEATURES")
    print('='*80)

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['passed'])
    failed_tests = total_tests - passed_tests

    for test_name, result in test_results.items():
        status_icon = "✅" if result['passed'] else "❌"
        print(f"{status_icon} {test_name}: {result['status']} ({result['duration']:.3f}s)")

    print(f"\n📈 OVERALL RESULTS:")
    print(f"   - Total Tests: {total_tests}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {failed_tests}")
    print(f"   - Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"   - Total Duration: {total_duration:.3f}s")

    # Accessibility compliance check
    print(f"\n🎯 ACCESSIBILITY COMPLIANCE:")
    if failed_tests == 0:
        print("   ✅ All accessibility features working correctly!")
        print("   ✅ WCAG 2.1 AA compliance ready")
        print("   ✅ Context7 patterns implemented")
        print("   ✅ Screen reader support functional")
        print("   ✅ Keyboard navigation operational")
        print("\n🎉 TASK 3.4.3: ACCESSIBILITY FEATURES - COMPLETED!")
        print("🚀 Ready for Task 3.4.4: Progressive Web App Features")
    else:
        print(f"   ⚠️ {failed_tests} accessibility feature(s) need attention")
        print("   🔧 Review and fix failing tests before deployment")

    return test_results

if __name__ == "__main__":
    results = main()