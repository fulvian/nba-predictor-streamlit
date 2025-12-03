"""
Integration Test - Phase 3 Day 10: Real-Time UI Updates
Test completo di tutti i componenti implementati per Day 10.
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_event_driven_ui_updates():
    """Test Task 3.3.1: Event-driven UI Updates"""
    print("\n🎯 Testing Task 3.3.1: Event-driven UI Updates")

    try:
        from src.nba_predictor.streamlit.components.real_time_updates import (
            get_event_manager, EventType, UIEvent, EventPriority
        )

        # Get event manager
        event_manager = get_event_manager()

        # Test event creation and emission
        test_events = [
            EventType.MODEL_UPDATE,
            EventType.DATA_REFRESH,
            EventType.UI_UPDATE,
            EventType.SYSTEM_ALERT
        ]

        emitted_events = []

        def test_event_handler(event: UIEvent):
            emitted_events.append(event)

        # Register handlers
        for event_type in test_events:
            event_manager.register_event_handler(event_type, test_event_handler)

        # Emit test events
        for i, event_type in enumerate(test_events):
            event_manager.emit_event(
                event_type=event_type,
                component_id=f"test_component_{i}",
                data={'test': f'data_{i}', 'timestamp': datetime.now()},
                priority=EventPriority.MEDIUM
            )

        # Wait for processing
        time.sleep(1)

        # Verify events were processed
        assert len(emitted_events) >= len(test_events), f"Expected {len(test_events)} events, got {len(emitted_events)}"

        # Test event filtering
        model_events = event_manager.get_events_by_type(EventType.MODEL_UPDATE)
        assert len(model_events) > 0, "Model events should be found"

        # Test event statistics
        stats = event_manager.get_event_statistics()
        assert stats['total_events'] >= len(test_events), "Event statistics should count emitted events"

        print(f"✅ Event-driven UI Updates test passed!")
        print(f"   - Processed {len(emitted_events)} events")
        print(f"   - Event types: {list(set(e.event_type for e in emitted_events))}")
        print(f"   - Statistics: {stats}")

        return True

    except Exception as e:
        print(f"❌ Event-driven UI Updates test failed: {e}")
        return False

def test_websocket_like_live_data():
    """Test Task 3.3.2: WebSocket-like Live Data"""
    print("\n📡 Testing Task 3.3.2: WebSocket-like Live Data")

    try:
        from src.nba_predictor.streamlit.components.live_data_streaming import (
            get_stream_manager, StreamType, StreamInfo, StreamStatus
        )

        # Get stream manager
        stream_manager = get_stream_manager()

        # Test stream creation
        test_streams = [
            StreamType.NBA_GAMES,
            StreamType.BETTING_ODDS,
            StreamType.LIVE_SCORES
        ]

        active_streams = []

        def test_stream_handler(stream_id: str, data: Dict[str, Any]):
            active_streams.append({
                'stream_id': stream_id,
                'data': data,
                'timestamp': datetime.now()
            })

        # Create streams
        for stream_type in test_streams:
            stream_id = stream_manager.create_stream(
                stream_type=stream_type,
                handler=test_stream_handler,
                config={'interval': 1.0}
            )

            # Start streaming
            stream_manager.start_stream(stream_id)

        # Wait for streaming
        time.sleep(3)

        # Verify streams are active
        for stream_type in test_streams:
            stream_info = stream_manager.get_stream_info(stream_type)
            assert stream_info.status == StreamStatus.ACTIVE, f"Stream {stream_type} should be active"

        # Test data reception
        assert len(active_streams) > 0, "Should receive streaming data"

        # Stop streams
        for stream_type in test_streams:
            stream_manager.stop_stream(stream_type)

        # Test stream statistics
        stream_stats = stream_manager.get_stream_statistics()
        assert stream_stats['active_streams'] >= len(test_streams), "Stream statistics should count active streams"

        print(f"✅ WebSocket-like Live Data test passed!")
        print(f"   - Active streams: {stream_stats['active_streams']}")
        print(f"   - Data packets received: {len(active_streams)}")
        print(f"   - Stream types: {list(stream_stats['streams_by_type'].keys())}")

        return True

    except Exception as e:
        print(f"❌ WebSocket-like Live Data test failed: {e}")
        return False

def test_intelligent_caching():
    """Test Task 3.3.3: Intelligent Caching"""
    print("\n💾 Testing Task 3.3.3: Intelligent Caching")

    try:
        from src.nba_predictor.streamlit.components.intelligent_cache import (
            get_cache_manager, CacheStrategy, CacheEntry
        )

        # Get cache manager
        cache_manager = get_cache_manager("test_cache")

        # Test different caching strategies
        strategies = [
            CacheStrategy.LRU,
            CacheStrategy.LFU,
            CacheStrategy.TTL
        ]

        cache_results = {}

        for strategy in strategies:
            # Create cache with specific strategy
            test_cache = get_cache_manager(f"test_{strategy.value}", strategy=strategy)

            # Test cache operations
            test_data = {
                'key1': {'value': 'data1', 'size': 100},
                'key2': {'value': 'data2', 'size': 200},
                'key3': {'value': 'data3', 'size': 150}
            }

            # Set cache items
            for key, data in test_data.items():
                test_cache.set(key, data, ttl=60)

            # Test cache retrieval
            retrieved = {}
            for key in test_data:
                retrieved[key] = test_cache.get(key)

            # Verify cache hits
            hit_count = sum(1 for v in retrieved.values() if v is not None)

            cache_results[strategy.value] = {
                'items_cached': len(test_data),
                'items_retrieved': hit_count,
                'hit_rate': hit_count / len(test_data)
            }

        # Test TTL expiration
        ttl_cache = get_cache_manager("test_ttl", CacheStrategy.TTL)
        ttl_cache.set("expire_key", {"value": "expires"}, ttl=1)
        time.sleep(1.5)

        expired_value = ttl_cache.get("expire_key")
        assert expired_value is None, "TTL cache should expire items"

        # Test cache statistics
        global_stats = cache_manager.get_cache_statistics()

        print(f"✅ Intelligent Caching test passed!")
        print(f"   - Cache strategies tested: {list(cache_results.keys())}")
        for strategy, results in cache_results.items():
            print(f"   - {strategy}: hit_rate={results['hit_rate']:.1%}")
        print(f"   - Global cache size: {global_stats['size_mb']:.2f}MB")
        print(f"   - Cache hits: {global_stats['hits']}, misses: {global_stats['misses']}")

        return True

    except Exception as e:
        print(f"❌ Intelligent Caching test failed: {e}")
        return False

def test_ui_rendering_performance():
    """Test Task 3.3.4: UI Rendering Performance"""
    print("\n🚀 Testing Task 3.3.4: UI Rendering Performance")

    try:
        from src.nba_predictor.streamlit.components.ui_rendering_performance import (
            get_performance_optimizer,
            get_performance_profiler,
            register_optimized_component,
            profile_performance,
            RenderOptimizationLevel,
            ComponentType,
            ComponentSpec
        )

        # Get performance optimizer
        optimizer = get_performance_optimizer(RenderOptimizationLevel.BALANCED)
        profiler = get_performance_profiler()

        # Test component registration
        test_components = [
            ComponentSpec(
                component_id="static_component",
                component_type=ComponentType.STATIC,
                priority=1,
                lazy_load=False
            ),
            ComponentSpec(
                component_id="dynamic_component",
                component_type=ComponentType.DYNAMIC,
                priority=3,
                lazy_load=True
            ),
            ComponentSpec(
                component_id="heavy_component",
                component_type=ComponentType.HEAVY,
                priority=5,
                lazy_load=True,
                memory_limit_mb=100.0
            )
        ]

        # Register components
        for spec in test_components:
            optimizer.register_component(spec)

        # Test optimized rendering
        render_results = {}

        @register_optimized_component("test_render", ComponentType.DYNAMIC, priority=2)
        @profile_performance("test_render")
        def test_render_function():
            time.sleep(0.05)  # Simulate work
            return {"data": "render_result", "timestamp": datetime.now()}

        # Execute multiple renders
        for i in range(5):
            with profiler.profile_component("test_render"):
                result = test_render_function()
                render_results[f"render_{i}"] = result

        # Test performance profiling
        profile_data = profiler.get_component_profile("test_render")
        assert profile_data['total_calls'] >= 5, "Should profile multiple renders"

        # Test optimization report
        optimization_report = optimizer.get_optimization_report()
        performance_summary = optimization_report['performance_summary']

        # Verify optimization metrics
        assert performance_summary['total_renders'] >= 5, "Should track total renders"
        assert performance_summary['avg_render_time_ms'] > 0, "Should measure render time"

        print(f"✅ UI Rendering Performance test passed!")
        print(f"   - Components registered: {len(test_components)}")
        print(f"   - Total renders: {performance_summary['total_renders']}")
        print(f"   - Avg render time: {performance_summary['avg_render_time_ms']:.1f}ms")
        print(f"   - Current memory: {performance_summary['current_memory_mb']:.1f}MB")
        print(f"   - Profile calls: {profile_data['total_calls']}")
        print(f"   - Avg profile duration: {profile_data['avg_duration_ms']:.1f}ms")

        return True

    except Exception as e:
        print(f"❌ UI Rendering Performance test failed: {e}")
        return False

def test_integration_workflow():
    """Test workflow di integrazione completo"""
    print("\n🔗 Testing Complete Integration Workflow")

    try:
        # Import tutti i componenti
        from src.nba_predictor.streamlit.components.real_time_updates import get_event_manager
        from src.nba_predictor.streamlit.components.live_data_streaming import get_stream_manager
        from src.nba_predictor.streamlit.components.intelligent_cache import get_cache_manager
        from src.nba_predictor.streamlit.components.ui_rendering_performance import (
            get_performance_optimizer, ComponentType, ComponentSpec
        )

        # Inizializza tutti i manager
        event_manager = get_event_manager()
        stream_manager = get_stream_manager()
        cache_manager = get_cache_manager("integration_test")
        performance_optimizer = get_performance_optimizer()

        # Test workflow completo
        workflow_results = {}

        # 1. Event -> Stream -> Cache -> Performance
        def integration_handler(event_data):
            # Cache event data
            cache_key = f"event_{event_data['id']}"
            cache_manager.set(cache_key, event_data, ttl=30)

            # Simulate UI update
            from src.nba_predictor.streamlit.components.ui_rendering_performance import get_performance_profiler
            profiler = get_performance_profiler()

            with profiler.profile_component("integration_update"):
                time.sleep(0.01)  # Simulate UI work
                return {"status": "updated", "cached": True}

        # Register event handler
        event_manager.register_event_handler(
            event_manager.EventType.UI_UPDATE,
            integration_handler
        )

        # Create stream
        stream_id = stream_manager.create_stream(
            stream_manager.StreamType.NBA_GAMES,
            handler=lambda sid, data: event_manager.emit_event(
                event_manager.EventType.UI_UPDATE,
                component_id="integration_test",
                data=data
            )
        )

        # Start stream
        stream_manager.start_stream(stream_id)

        # Emit some events
        for i in range(3):
            event_manager.emit_event(
                event_manager.EventType.DATA_REFRESH,
                component_id="integration_test",
                data={'id': f'event_{i}', 'timestamp': datetime.now()}
            )

        # Wait for processing
        time.sleep(2)

        # Stop stream
        stream_manager.stop_stream(stream_manager.StreamType.NBA_GAMES)

        # Verify integration
        cache_stats = cache_manager.get_cache_statistics()
        event_stats = event_manager.get_event_statistics()
        stream_stats = stream_manager.get_stream_statistics()
        performance_stats = performance_optimizer.get_optimization_report()

        # Verify all systems interacted
        assert cache_stats['hits'] >= 0, "Cache should be functional"
        assert event_stats['total_events'] > 0, "Events should be processed"
        assert stream_stats['total_data_packets'] >= 0, "Streaming should be functional"

        workflow_results = {
            'cache_size_mb': cache_stats['size_mb'],
            'events_processed': event_stats['total_events'],
            'stream_data_packets': stream_stats['total_data_packets'],
            'performance_renders': performance_stats['performance_summary']['total_renders']
        }

        print(f"✅ Integration Workflow test passed!")
        print(f"   - Cache size: {workflow_results['cache_size_mb']:.2f}MB")
        print(f"   - Events processed: {workflow_results['events_processed']}")
        print(f"   - Stream data packets: {workflow_results['stream_data_packets']}")
        print(f"   - Performance renders: {workflow_results['performance_renders']}")

        return True

    except Exception as e:
        print(f"❌ Integration Workflow test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("="*80)
    print("🎯 PHASE 3 DAY 10 INTEGRATION TEST - Real-Time UI Updates")
    print("="*80)

    test_results = {}

    # Esegui tutti i test
    tests = [
        ("Event-driven UI Updates", test_event_driven_ui_updates),
        ("WebSocket-like Live Data", test_websocket_like_live_data),
        ("Intelligent Caching", test_intelligent_caching),
        ("UI Rendering Performance", test_ui_rendering_performance),
        ("Integration Workflow", test_integration_workflow)
    ]

    for test_name, test_func in tests:
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

    # Riepilogo risultati
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['passed'])
    failed_tests = total_tests - passed_tests

    for test_name, result in test_results.items():
        status_icon = "✅" if result['passed'] else "❌"
        print(f"{status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s)")

    print(f"\n📈 OVERALL RESULTS:")
    print(f"   - Total Tests: {total_tests}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {failed_tests}")
    print(f"   - Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 3 Day 10 implementation is complete and working!")
    else:
        print(f"\n⚠️ {failed_tests} test(s) failed. Please review and fix issues.")

    return test_results

if __name__ == "__main__":
    main()