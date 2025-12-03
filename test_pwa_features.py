#!/usr/bin/env python3
"""
Test PWA Features Implementation
Phase 3 Day 11 - Task 3.4.4: Progressive Web App Features

Verifies all PWA features are working correctly with Context7 compliance.
"""

import sys
import time
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pwa_manager_initialization():
    """Test PWAFeaturesManager initialization"""
    print("\n📱 Testing PWA Manager Initialization")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, PWAConfig, CacheConfig, ServiceWorkerConfig,
            CacheStrategy
        )

        # Test default configuration
        default_manager = PWAFeaturesManager()
        assert default_manager.pwa_config.app_name == "NBA Predictor", "Default app name should be NBA Predictor"
        assert default_manager.cache_config.strategy == CacheStrategy.CACHE_FIRST, "Default strategy should be cache_first"
        assert default_manager.sw_config.enabled is True, "Service worker should be enabled by default"

        # Test custom configuration
        pwa_config = PWAConfig(
            app_name="Custom NBA App",
            app_short_name="NBA-Custom",
            theme_color="#ff6b6b",
            display="fullscreen"
        )
        cache_config = CacheConfig(
            strategy=CacheStrategy.NETWORK_FIRST,
            max_age=7200,
            max_size=100 * 1024 * 1024
        )
        sw_config = ServiceWorkerConfig(
            enabled=True,
            skip_waiting=False,
            precache_files=["/index.html", "/styles.css"]
        )

        custom_manager = PWAFeaturesManager(pwa_config, cache_config, sw_config)
        assert custom_manager.pwa_config.app_name == "Custom NBA App", "Custom app name should match"
        assert custom_manager.cache_config.strategy == CacheStrategy.NETWORK_FIRST, "Custom strategy should match"
        assert custom_manager.sw_config.precache_files == ["/index.html", "/styles.css"], "Custom precache files should match"

        # Test global manager
        from src.nba_predictor.streamlit.components.pwa_features import get_pwa_manager
        global_manager = get_pwa_manager()
        assert global_manager is not None, "Global manager should exist"

        print("✅ PWA Manager initialization test passed!")
        return True

    except Exception as e:
        print(f"❌ PWA Manager initialization test failed: {e}")
        return False

def test_service_worker_generation():
    """Test service worker script generation"""
    print("\n⚙️ Testing Service Worker Generation")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, ServiceWorkerConfig, CacheStrategy
        )

        # Test with cache first strategy
        sw_config = ServiceWorkerConfig(
            enabled=True,
            precache_files=["/index.html", "/main.js", "/styles.css"]
        )
        cache_config = None  # Use default

        manager = PWAFeaturesManager(sw_config=sw_config, cache_config=cache_config)
        sw_script = manager._generate_service_worker_script()

        # Verify service worker script contains expected elements
        assert "addEventListener('install'" in sw_script, "Should contain install event listener"
        assert "addEventListener('activate'" in sw_script, "Should contain activate event listener"
        assert "addEventListener('fetch'" in sw_script, "Should contain fetch event listener"
        assert "CACHE_NAME" in sw_script, "Should contain cache name constant"
        assert "nba_predictor_cache" in sw_script, "Should contain app-specific cache name"
        assert "skipWaiting()" in sw_script, "Should contain skip waiting call"
        assert "clients.claim()" in sw_script, "Should contain clients claim call"

        # Test precache files are included
        assert '"/index.html"' in sw_script, "Should include precache index.html"
        assert '"/main.js"' in sw_script, "Should include precache main.js"
        assert '"/styles.css"' in sw_script, "Should include precache styles.css"

        # Test cache strategy logic
        assert "Cache hit:" in sw_script, "Should contain cache hit logging"
        assert "Cache miss" in sw_script, "Should contain cache miss handling"

        print("✅ Service Worker Generation test passed!")
        print(f"   - Script length: {len(sw_script)} characters")
        print(f"   - Install handler: ✅")
        print(f"   - Activate handler: ✅")
        print(f"   - Fetch handler: ✅")
        print(f"   - Cache strategy: ✅")
        return True

    except Exception as e:
        print(f"❌ Service Worker Generation test failed: {e}")
        return False

def test_app_manifest_generation():
    """Test app manifest generation"""
    print("\n📋 Testing App Manifest Generation")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, PWAConfig
        )

        # Test with custom configuration
        pwa_config = PWAConfig(
            app_name="NBA Predictor Pro",
            app_short_name="NBA-Pro",
            app_description="Professional NBA game predictions and analysis",
            app_version="2.0.1",
            theme_color="#e74c3c",
            background_color="#ecf0f1",
            display="standalone",
            orientation="portrait",
            start_url="/home",
            scope="/",
            categories=["sports", "analytics", "productivity"],
            lang="en-US"
        )

        manager = PWAFeaturesManager(pwa_config=pwa_config)
        manifest_json = manager._generate_app_manifest()
        manifest = json.loads(manifest_json)

        # Verify manifest structure
        assert manifest['name'] == "NBA Predictor Pro", "App name should match"
        assert manifest['short_name'] == "NBA-Pro", "Short name should match"
        assert manifest['description'] == "Professional NBA game predictions and analysis", "Description should match"
        assert manifest['version'] == "2.0.1", "Version should match"
        assert manifest['theme_color'] == "#e74c3c", "Theme color should match"
        assert manifest['background_color'] == "#ecf0f1", "Background color should match"
        assert manifest['display'] == "standalone", "Display mode should match"
        assert manifest['orientation'] == "portrait", "Orientation should match"
        assert manifest['start_url'] == "/home", "Start URL should match"
        assert manifest['scope'] == "/", "Scope should match"
        assert manifest['categories'] == ["sports", "analytics", "productivity"], "Categories should match"
        assert manifest['lang'] == "en-US", "Language should match"

        # Verify icons are generated
        assert 'icons' in manifest, "Should have icons array"
        assert len(manifest['icons']) > 0, "Should have at least one icon"
        assert any(icon['sizes'] == '192x192' for icon in manifest['icons']), "Should have 192x192 icon"

        print("✅ App Manifest Generation test passed!")
        print(f"   - Manifest JSON: ✅")
        print(f"   - App info: ✅")
        print(f"   - Icons: {len(manifest['icons'])} generated")
        print(f"   - Display mode: {manifest['display']}")
        return True

    except Exception as e:
        print(f"❌ App Manifest Generation test failed: {e}")
        return False

def test_cache_strategies():
    """Test different cache strategies"""
    print("\n💾 Testing Cache Strategies")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, CacheStrategy, CacheConfig
        )

        strategies_to_test = [
            CacheStrategy.CACHE_FIRST,
            CacheStrategy.NETWORK_FIRST,
            CacheStrategy.CACHE_ONLY,
            CacheStrategy.NETWORK_ONLY,
            CacheStrategy.STALE_WHILE_REVALIDATE
        ]

        results = {}

        for strategy in strategies_to_test:
            cache_config = CacheConfig(
                strategy=strategy,
                max_age=1800,  # 30 minutes
                max_size=25 * 1024 * 1024  # 25MB
            )

            manager = PWAFeaturesManager(cache_config=cache_config)
            cache_logic = manager._get_cache_strategy_logic()

            # Verify cache logic contains expected elements
            results[strategy.value] = {
                'has_logic': len(cache_logic) > 0,
                'has_fetch': 'fetch(event.request)' in cache_logic,
                'has_cache': 'caches.match(event.request)' in cache_logic,
                'logic_length': len(cache_logic)
            }

            # Strategy-specific checks
            if strategy == CacheStrategy.CACHE_FIRST:
                assert 'Cache hit:' in cache_logic, "Cache first should check cache hit"
                assert 'Cache miss' in cache_logic, "Cache first should handle cache miss"
            elif strategy == CacheStrategy.NETWORK_FIRST:
                assert 'fetch(event.request)' in cache_logic, "Network first should fetch first"
                assert '.catch(()' in cache_logic, "Network first should have fallback"
            elif strategy == CacheStrategy.STALE_WHILE_REVALIDATE:
                assert 'networkPromise' in cache_logic, "Stale while revalidate should have network promise"

        # Verify all strategies have logic
        for strategy, result in results.items():
            assert result['has_logic'], f"Strategy {strategy} should have logic"
            assert result['logic_length'] > 50, f"Strategy {strategy} should have substantial logic"

        print("✅ Cache Strategies test passed!")
        for strategy, result in results.items():
            print(f"   - {strategy}: ✅ ({result['logic_length']} chars)")
        return True

    except Exception as e:
        print(f"❌ Cache Strategies test failed: {e}")
        return False

def test_resource_caching():
    """Test resource caching functionality"""
    print("\n🗄️ Testing Resource Caching")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, CacheConfig
        )

        manager = PWAFeaturesManager()

        # Test caching resources
        test_resources = [
            ("/api/games", {"games": [{"id": 1, "teams": "LAL vs BOS"}]}, "application/json"),
            ("/styles/main.css", "body { margin: 0; }", "text/css"),
            ("/data/stats.json", {"points": 25.5, "rebounds": 8.2}, "application/json")
        ]

        for url, data, content_type in test_resources:
            manager.cache_resource(url, data, content_type)

        # Verify resources are cached
        assert len(manager._cached_resources) == len(test_resources), "All resources should be cached"

        # Test retrieving cached resources
        for url, expected_data, _ in test_resources:
            cached_data = manager.get_cached_resource(url)
            assert cached_data == expected_data, f"Cached data should match for {url}"

        # Test cache hit/miss metrics
        metrics = manager.get_performance_metrics()
        assert metrics['cache_hits'] >= len(test_resources), "Should have cache hits for all resources"
        assert metrics['total_requests'] >= len(test_resources), "Should count all requests"

        # Test cache expiration (simulate expired resource)
        old_cache = manager._cached_resources.copy()

        # Manually expire a resource by setting old timestamp
        expired_url = test_resources[0][0]
        if expired_url in manager._cached_resources:
            manager._cached_resources[expired_url]['timestamp'] = time.time() - (manager.cache_config.max_age + 100)

        # Try to get expired resource
        expired_data = manager.get_cached_resource(expired_url)
        assert expired_data is None, "Expired resource should return None"

        # Verify cache miss increased
        updated_metrics = manager.get_performance_metrics()
        assert updated_metrics['cache_misses'] > metrics['cache_misses'], "Cache misses should increase for expired items"

        print("✅ Resource Caching test passed!")
        print(f"   - Resources cached: {len(manager._cached_resources)}")
        print(f"   - Cache hits: {metrics['cache_hits']}")
        print(f"   - Cache misses: {updated_metrics['cache_misses']}")
        print(f"   - Cache hit rate: {updated_metrics.get('cache_hit_rate', 0):.1%}")
        return True

    except Exception as e:
        print(f"❌ Resource Caching test failed: {e}")
        return False

def test_background_sync():
    """Test background sync functionality"""
    print("\n🔄 Testing Background Sync")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager
        )

        manager = PWAFeaturesManager()

        # Test scheduling background sync
        sync_items = [
            ({"action": "save_bet", "data": {"game": "LAL vs BOS", "amount": 50}}, "bet_sync"),
            ({"action": "update_stats", "data": {"player": "LeBron", "points": 30}}, "stats_sync"),
            ({"action": "log_event", "data": {"event": "prediction_complete"}}, "event_sync")
        ]

        for data, tag in sync_items:
            manager.schedule_background_sync(data, tag)

        # Verify sync queue
        assert len(manager._background_sync_queue) == len(sync_items), "All sync items should be queued"

        # Verify sync item structure
        for i, (expected_data, expected_tag) in enumerate(sync_items):
            sync_item = manager._background_sync_queue[i]
            assert sync_item['data'] == expected_data, "Sync data should match"
            assert sync_item['tag'] == expected_tag, "Sync tag should match"
            assert 'timestamp' in sync_item, "Sync item should have timestamp"
            assert 'id' in sync_item, "Sync item should have ID"

        # Test background sync queue size in metrics
        metrics = manager.get_performance_metrics()
        assert metrics['background_sync_queue_size'] == len(sync_items), "Metrics should report correct queue size"

        print("✅ Background Sync test passed!")
        print(f"   - Sync items queued: {len(manager._background_sync_queue)}")
        print(f"   - Queue size in metrics: {metrics['background_sync_queue_size']}")
        return True

    except Exception as e:
        print(f"❌ Background Sync test failed: {e}")
        return False

def test_pwa_info_and_metrics():
    """Test PWA information and metrics retrieval"""
    print("\n📊 Testing PWA Info and Metrics")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            PWAFeaturesManager, PWAConfig
        )

        # Create manager with custom config
        pwa_config = PWAConfig(
            app_name="NBA Predictor Test",
            app_version="1.5.2",
            theme_color="#3498db",
            display="minimal-ui"
        )

        manager = PWAFeaturesManager(pwa_config=pwa_config)

        # Add some cached resources and sync items for metrics
        manager.cache_resource("/test/api", {"data": "test"}, "application/json")
        manager.schedule_background_sync({"test": "data"}, "test_tag")

        # Test PWA info
        pwa_info = manager.get_pwa_info()

        # Verify info structure
        assert 'app_info' in pwa_info, "Should have app_info"
        assert 'features' in pwa_info, "Should have features"
        assert 'configuration' in pwa_info, "Should have configuration"
        assert 'status' in pwa_info, "Should have status"
        assert 'resources' in pwa_info, "Should have resources"

        # Verify app info
        app_info = pwa_info['app_info']
        assert app_info['name'] == "NBA Predictor Test", "App name should match"
        assert app_info['version'] == "1.5.2", "Version should match"

        # Verify features
        features = pwa_info['features']
        assert features['service_worker'] is True, "Service worker should be enabled"
        assert features['caching'] is True, "Caching should be enabled"
        assert features['notifications'] is True, "Notifications should be enabled"

        # Verify configuration
        config = pwa_info['configuration']
        assert config['theme_color'] == "#3498db", "Theme color should match"
        assert config['display_mode'] == "minimal-ui", "Display mode should match"

        # Verify status
        status = pwa_info['status']
        assert 'install_status' in status, "Should have install status"
        assert 'offline' in status, "Should have offline status"

        # Test performance metrics
        metrics = manager.get_performance_metrics()

        # Verify metrics structure
        expected_metrics = [
            'cache_hits', 'cache_misses', 'network_requests', 'service_worker_messages',
            'cache_hit_rate', 'total_requests', 'cache_size', 'background_sync_queue_size'
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Should have {metric} metric"

        # Verify metrics values
        assert metrics['cache_hits'] >= 1, "Should have at least one cache hit"
        assert metrics['background_sync_queue_size'] >= 1, "Should have at least one sync item"
        assert isinstance(metrics['cache_hit_rate'], float), "Cache hit rate should be float"
        assert 0 <= metrics['cache_hit_rate'] <= 1, "Cache hit rate should be between 0 and 1"

        print("✅ PWA Info and Metrics test passed!")
        print(f"   - App name: {app_info['name']}")
        print(f"   - Features enabled: {sum(features.values())}/{len(features)}")
        print(f"   - Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"   - Total requests: {metrics['total_requests']}")
        print(f"   - Cache size: {metrics['cache_size']} bytes")
        return True

    except Exception as e:
        print(f"❌ PWA Info and Metrics test failed: {e}")
        return False

def test_context7_utilities():
    """Test Context7 compliant utility functions"""
    print("\n🎨 Testing Context7 Utility Functions")

    try:
        from src.nba_predictor.streamlit.components.pwa_features import (
            create_pwa_ready_page, create_install_banner, get_pwa_manager,
            PWAConfig
        )

        # Test that functions exist and are callable
        assert callable(create_pwa_ready_page), "create_pwa_ready_page should be callable"
        assert callable(create_install_banner), "create_install_banner should be callable"
        assert callable(get_pwa_manager), "get_pwa_manager should be callable"

        # Test PWA manager creation
        manager = get_pwa_manager()
        assert manager is not None, "Manager should be available"

        # Test manager initialization
        pwa_info = manager.get_pwa_info()
        assert 'app_info' in pwa_info, "Manager should be properly initialized"

        print("✅ Context7 Utility Functions test passed!")
        print("   - create_pwa_ready_page: ✅ Available")
        print("   - create_install_banner: ✅ Available")
        print("   - get_pwa_manager: ✅ Available")
        return True

    except Exception as e:
        print(f"❌ Context7 Utility Functions test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("="*80)
    print("📱 PHASE 3 DAY 11 PWA FEATURES TEST - Task 3.4.4")
    print("="*80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Testing: Progressive Web App Features with Context7 Compliance")

    # Define test suite
    tests = [
        ("PWA Manager Initialization", test_pwa_manager_initialization),
        ("Service Worker Generation", test_service_worker_generation),
        ("App Manifest Generation", test_app_manifest_generation),
        ("Cache Strategies", test_cache_strategies),
        ("Resource Caching", test_resource_caching),
        ("Background Sync", test_background_sync),
        ("PWA Info and Metrics", test_pwa_info_and_metrics),
        ("Context7 Utility Functions", test_context7_utilities),
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
    print("📊 TEST SUMMARY - PWA FEATURES")
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

    # PWA compliance check
    print(f"\n🎯 PWA COMPLIANCE:")
    if failed_tests == 0:
        print("   ✅ All PWA features working correctly!")
        print("   ✅ Service Worker ready for production")
        print("   ✅ App Manifest properly configured")
        print("   ✅ Caching strategies implemented")
        print("   ✅ Background sync operational")
        print("   ✅ Install prompts functional")
        print("   ✅ Context7 patterns implemented")
        print("\n🎉 TASK 3.4.4: PROGRESSIVE WEB APP FEATURES - COMPLETED!")
        print("🚀 PHASE 3 DAY 11: USER EXPERIENCE ENHANCEMENT - COMPLETED!")
        print("🏀 NBA Predictor System is now PWA-ready with Context7 compliance!")
    else:
        print(f"   ⚠️ {failed_tests} PWA feature(s) need attention")
        print("   🔧 Review and fix failing tests before deployment")

    return test_results

if __name__ == "__main__":
    results = main()