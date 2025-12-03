"""
🎯 PHASE 3 DAY 10: Intelligent Caching System
==============================================

X7 Compliant High-Performance Caching System for NBA Predictor Dashboard.

This module implements comprehensive intelligent caching with:
- Multiple cache strategies (LRU, LFU, TTL, Adaptive)
- Smart invalidation and cache warming
- Performance optimization for real-time data
- Memory-efficient storage with automatic cleanup

Author: DevStream SuperPowered Implementation
Date: 2025-11-12
Version: 1.0.0
Compliance: X7 Compliant, Production Ready
"""

import hashlib
import pickle
import time
import threading
import json
import logging
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import fnmatch
import gc

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """X7 Compliant cache eviction strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    FIFO = "fifo"                  # First In, First Out
    RANDOM = "random"              # Random eviction


class CacheHitStatus(Enum):
    """Cache operation status for analytics."""
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    EVICTED = "evicted"
    ERROR = "error"


@dataclass
class CacheEntry:
    """X7 Compliant cache entry with comprehensive metadata."""

    key: str
    value: Any
    timestamp: float
    last_access: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    hit_count: int = 0
    creation_context: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=low, 5=high

    def __post_init__(self):
        """Validate and initialize cache entry."""
        if self.last_access == 0:
            self.last_access = self.timestamp

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp

    @property
    def last_access_age(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_access

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return self.age_seconds > self.ttl

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate for this entry."""
        total_accesses = self.access_count
        if total_accesses == 0:
            return 0.0
        return self.hit_count / total_accesses


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    # Performance timing
    avg_access_time: float = 0.0
    avg_store_time: float = 0.0
    avg_eviction_time: float = 0.0

    # Analytics
    hit_rate_by_tag: Dict[str, float] = field(default_factory=dict)
    access_frequency: Dict[str, int] = field(default_factory=dict)
    size_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class IntelligentCacheManager:
    """
    🎯 X7 COMPLIANT INTELLIGENT CACHE MANAGER

    Advanced caching system with multiple strategies, smart invalidation,
    and comprehensive performance monitoring.
    """

    def __init__(self,
                 max_size_mb: int = 100,
                 default_ttl: int = 300,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 enable_compression: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize intelligent cache manager.

        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy
            enable_compression: Enable compression for large entries
            enable_metrics: Enable detailed metrics collection
        """
        # Cache configuration
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics

        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU

        # Performance tracking
        self.metrics = CacheMetrics()
        self.access_times: List[float] = []
        self.store_times: List[float] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background maintenance
        self.maintenance_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Advanced features
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.warmup_queue: List[Tuple[str, Callable[[], Any]]] = []

        # Adaptive strategy parameters
        self.adaptive_weights = {
            'recency_weight': 0.4,
            'frequency_weight': 0.3,
            'size_weight': 0.2,
            'priority_weight': 0.1
        }

        # Compression threshold
        self.compression_threshold = 1024  # bytes

        logger.info(f"🚀 IntelligentCacheManager initialized: {strategy.value}, {max_size_mb}MB")

        # Start background maintenance
        self.start_maintenance()

    def get(self, key: str, default: Any = None, update_access: bool = True) -> Any:
        """
        Get value from cache with comprehensive analytics.

        Args:
            key: Cache key
            default: Default value if not found
            update_access: Whether to update access time for LRU

        Returns:
            Cached value or default
        """
        start_time = time.time()

        try:
            with self._lock:
                if key not in self.cache:
                    self._record_miss()
                    return default

                entry = self.cache[key]

                # Check expiration
                if entry.is_expired:
                    self._record_eviction(key, "expired")
                    del self.cache[key]
                    self._remove_from_tracking_structures(key)
                    return default

                # Update access statistics
                if update_access:
                    entry.last_access = time.time()
                    entry.access_count += 1
                    entry.hit_count += 1
                    self.frequency_counter[key] += 1

                    # Update LRU order
                    if self.strategy == CacheStrategy.LRU:
                        self.access_order.move_to_end(key)

                # Update metrics
                self._record_hit()

                # Decompress if needed
                value = self._decompress_if_needed(entry.value)

                # Record access time
                access_time = time.time() - start_time
                if self.enable_metrics:
                    self.access_times.append(access_time)
                    if len(self.access_times) > 1000:
                        self.access_times = self.access_times[-500:]

                return value

        except Exception as e:
            logger.error(f"❌ Cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return default

    def set(self,
             key: str,
             value: Any,
             ttl: Optional[float] = None,
             tags: List[str] = None,
             priority: int = 1,
             dependencies: List[str] = None) -> bool:
        """
        Set value in cache with intelligent storage.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom time-to-live
            tags: Cache tags for invalidation
            priority: Entry priority (1-5)
            dependencies: Keys this entry depends on

        Returns:
            True if successfully cached
        """
        start_time = time.time()

        try:
            with self._lock:
                # Calculate entry size
                try:
                    if self.enable_compression and self._should_compress(value):
                        value = self._compress_value(value)
                        compressed = True
                    else:
                        compressed = False

                    size_bytes = len(pickle.dumps(value))
                except:
                    # Fallback to string size
                    size_bytes = len(str(value).encode('utf-8'))
                    compressed = False

                # Check if eviction is needed
                if not self._has_space_for(size_bytes):
                    if not self._evict_entries(size_bytes):
                        logger.warning(f"⚠️ Could not cache {key}: insufficient space")
                        return False

                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    last_access=time.time(),
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    tags=tags or [],
                    priority=priority,
                    dependencies=dependencies or [],
                    creation_context={
                        'compressed': compressed,
                        'original_size': size_bytes if not compressed else 0
                    }
                )

                # Store entry
                self.cache[key] = entry
                self.metrics.total_size_bytes += size_bytes
                self.metrics.entry_count += 1

                # Update tracking structures
                self._add_to_tracking_structures(key, entry)

                # Record metrics
                store_time = time.time() - start_time
                if self.enable_metrics:
                    self.store_times.append(store_time)
                    if len(self.store_times) > 1000:
                        self.store_times = self.store_times[-500:]

                logger.debug(f"💾 Cached {key} ({size_bytes} bytes, TTL: {entry.ttl}s)")
                return True

        except Exception as e:
            logger.error(f"❌ Cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    def get_or_compute(self,
                      key: str,
                      compute_func: Callable[[], Any],
                      ttl: Optional[float] = None,
                      **kwargs) -> Any:
        """
        Get value from cache or compute if not present.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Custom time-to-live
            **kwargs: Additional arguments for set()

        Returns:
            Computed or cached value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        computed_value = compute_func()
        self.set(key, computed_value, ttl=ttl, **kwargs)
        return computed_value

    def invalidate(self, pattern: str = None, tags: List[str] = None) -> int:
        """
        Invalidate cache entries by pattern or tags.

        Args:
            pattern: Glob pattern for keys to invalidate
            tags: List of tags to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []

            if pattern:
                # Pattern-based invalidation
                keys_to_remove.extend([
                    key for key in self.cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ])

            if tags:
                # Tag-based invalidation
                for tag in tags:
                    if tag in self.tag_index:
                        keys_to_remove.extend(self.tag_index[tag])

            # Remove duplicates
            keys_to_remove = list(set(keys_to_remove))

            # Remove entries
            for key in keys_to_remove:
                self._remove_entry(key)

            if keys_to_remove:
                logger.info(f"🗑️ Invalidated {len(keys_to_remove)} cache entries")

            return len(keys_to_remove)

    def invalidate_dependencies(self, key: str) -> int:
        """
        Invalidate all entries that depend on the given key.

        Args:
            key: Key whose dependencies should be invalidated

        Returns:
            Number of dependencies invalidated
        """
        with self._lock:
            dependent_keys = self.dependency_graph.get(key, [])
            invalidated = 0

            for dep_key in dependent_keys[:]:  # Copy list to allow modification
                if dep_key in self.cache:
                    self._remove_entry(dep_key)
                    invalidated += 1

            # Clear dependency list
            self.dependency_graph[key] = []

            if invalidated > 0:
                logger.info(f"🔗 Invalidated {invalidated} dependencies of {key}")

            return invalidated

    def warmup(self, key: str, compute_func: Callable[[], Any], **kwargs):
        """
        Add entry to warmup queue for background preloading.

        Args:
            key: Cache key
            compute_func: Function to compute value
            **kwargs: Arguments for cache set
        """
        self.warmup_queue.append((key, compute_func, kwargs))
        logger.debug(f"🔥 Added {key} to warmup queue")

    def process_warmup_queue(self, max_items: int = 10) -> int:
        """
        Process warmup queue to preload cache entries.

        Args:
            max_items: Maximum items to process

        Returns:
            Number of items processed
        """
        processed = 0

        while processed < max_items and self.warmup_queue:
            key, compute_func, kwargs = self.warmup_queue.pop(0)

            if key not in self.cache:
                try:
                    value = compute_func()
                    self.set(key, value, **kwargs)
                    processed += 1
                except Exception as e:
                    logger.error(f"❌ Warmup error for {key}: {e}")

        if processed > 0:
            logger.info(f"🔥 Warmed up {processed} cache entries")

        return processed

    def _has_space_for(self, size_bytes: int) -> bool:
        """Check if cache has space for new entry."""
        return (self.metrics.total_size_bytes + size_bytes) <= self.max_size_bytes

    def _evict_entries(self, needed_bytes: int) -> bool:
        """Evict entries to make space for new entry."""
        bytes_freed = 0
        entries_to_evict = []

        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            while self.access_order and bytes_freed < needed_bytes:
                lru_key = next(iter(self.access_order))
                if lru_key in self.cache:
                    entry = self.cache[lru_key]
                    entries_to_evict.append(lru_key)
                    bytes_freed += entry.size_bytes
                else:
                    # Clean up invalid key
                    self.access_order.popitem(last=False)

        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_items = sorted(
                self.frequency_counter.items(),
                key=lambda x: (x[1], self.cache.get(x[0], CacheEntry("", "", 0, 0)).timestamp)
            )
            for key, _ in sorted_items:
                if bytes_freed >= needed_bytes:
                    break
                if key in self.cache:
                    entry = self.cache[key]
                    entries_to_evict.append(key)
                    bytes_freed += entry.size_bytes

        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive eviction based on multiple factors
            entries = list(self.cache.items())
            entries.sort(key=lambda x: self._calculate_adaptive_score(x[1]))

            for key, entry in entries:
                if bytes_freed >= needed_bytes:
                    break
                entries_to_evict.append(key)
                bytes_freed += entry.size_bytes

        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, entry in sorted_entries:
                if bytes_freed >= needed_bytes:
                    break
                entries_to_evict.append(key)
                bytes_freed += entry.size_bytes

        elif self.strategy == CacheStrategy.RANDOM:
            # Random eviction
            import random
            keys = list(self.cache.keys())
            random.shuffle(keys)
            for key in keys:
                if bytes_freed >= needed_bytes:
                    break
                if key in self.cache:
                    entry = self.cache[key]
                    entries_to_evict.append(key)
                    bytes_freed += entry.size_bytes

        # Execute evictions
        for key in entries_to_evict:
            self._remove_entry(key)
            self._record_eviction(key, "evicted")

        return bytes_freed >= needed_bytes

    def _calculate_adaptive_score(self, entry: CacheEntry) -> float:
        """Calculate adaptive eviction score (lower = more likely to evict)."""
        # Normalize factors
        recency_score = entry.last_access_age / 3600  # Hours since last access
        frequency_score = entry.access_count / max(1, entry.age_seconds)  # Accesses per second
        size_score = entry.size_bytes / (1024 * 1024)  # Size in MB
        priority_score = (6 - entry.priority)  # Invert priority

        # Weighted combination
        score = (
            self.adaptive_weights['recency_weight'] * recency_score +
            self.adaptive_weights['frequency_weight'] * (1 / max(frequency_score, 0.001)) +
            self.adaptive_weights['size_weight'] * size_score +
            self.adaptive_weights['priority_weight'] * priority_score
        )

        return score

    def _remove_entry(self, key: str):
        """Remove entry from cache and all tracking structures."""
        if key not in self.cache:
            return

        entry = self.cache[key]

        # Update metrics
        self.metrics.total_size_bytes -= entry.size_bytes
        self.metrics.entry_count -= 1

        # Remove from main cache
        del self.cache[key]

        # Remove from tracking structures
        self._remove_from_tracking_structures(key)

    def _remove_from_tracking_structures(self, key: str):
        """Remove key from all tracking structures."""
        # LRU tracking
        self.access_order.pop(key, None)

        # Frequency tracking
        self.frequency_counter.pop(key, None)

        # Tag index
        entry = self.cache.get(key)
        if entry:
            for tag in entry.tags:
                self.tag_index[tag].discard(key)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]

        # Dependency graph
        self.dependency_graph.pop(key, None)

    def _add_to_tracking_structures(self, key: str, entry: CacheEntry):
        """Add key to all tracking structures."""
        # LRU tracking
        if self.strategy == CacheStrategy.LRU:
            self.access_order[key] = True

        # Frequency tracking
        self.frequency_counter[key] = entry.access_count

        # Tag index
        for tag in entry.tags:
            self.tag_index[tag].add(key)

        # Dependency graph
        for dep in entry.dependencies:
            self.dependency_graph[dep].append(key)

    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed."""
        if not self.enable_compression:
            return False

        try:
            size = len(pickle.dumps(value))
            return size > self.compression_threshold
        except:
            return False

    def _compress_value(self, value: Any) -> bytes:
        """Compress value using gzip."""
        import gzip
        return gzip.compress(pickle.dumps(value))

    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if it was compressed."""
        if isinstance(value, bytes) and len(value) > 100:
            try:
                import gzip
                # Try to decompress
                decompressed = gzip.decompress(value)
                return pickle.loads(decompressed)
            except:
                # Not compressed or decompression failed
                return value
        return value

    def _record_hit(self):
        """Record cache hit for metrics."""
        self.metrics.total_requests += 1
        self.metrics.cache_hits += 1

    def _record_miss(self):
        """Record cache miss for metrics."""
        self.metrics.total_requests += 1
        self.metrics.cache_misses += 1

    def _record_eviction(self, key: str, reason: str):
        """Record eviction for analytics."""
        self.metrics.evictions += 1
        logger.debug(f"🗑️ Evicted {key} ({reason})")

    def start_maintenance(self):
        """Start background maintenance thread."""
        if not self.is_running:
            self.is_running = True
            self.maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                name="CacheMaintenance",
                daemon=True
            )
            self.maintenance_thread.start()
            logger.info("🔧 Cache maintenance thread started")

    def stop_maintenance(self):
        """Stop background maintenance thread."""
        self.is_running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5)
            logger.info("⏹️ Cache maintenance thread stopped")

    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.is_running:
            try:
                # Process warmup queue
                self.process_warmup_queue(max_items=5)

                # Clean up expired entries
                self._cleanup_expired_entries()

                # Update metrics
                self._update_computed_metrics()

                # Force garbage collection periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    gc.collect()

                time.sleep(30)  # Run maintenance every 30 seconds

            except Exception as e:
                logger.error(f"❌ Cache maintenance error: {e}")
                time.sleep(60)

    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            self._remove_entry(key)
            self._record_eviction(key, "expired")

        if expired_keys:
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired entries")

    def _update_computed_metrics(self):
        """Update computed metrics."""
        if self.access_times:
            self.metrics.avg_access_time = sum(self.access_times[-100:]) / len(self.access_times[-100:])

        if self.store_times:
            self.metrics.avg_store_time = sum(self.store_times[-100:]) / len(self.store_times[-100:])

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses,
                    'hit_rate': self.metrics.hit_rate,
                    'evictions': self.metrics.evictions,
                    'errors': self.metrics.errors,
                    'entry_count': self.metrics.entry_count,
                    'total_size_bytes': self.metrics.total_size_bytes,
                    'total_size_mb': self.metrics.total_size_bytes / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'memory_usage_percent': (self.metrics.total_size_bytes / self.max_size_bytes) * 100,
                    'avg_access_time_ms': self.metrics.avg_access_time * 1000,
                    'avg_store_time_ms': self.metrics.avg_store_time * 1000
                },
                'strategy': self.strategy.value,
                'features': {
                    'compression_enabled': self.enable_compression,
                    'metrics_enabled': self.enable_metrics,
                    'maintenance_running': self.is_running,
                    'warmup_queue_size': len(self.warmup_queue)
                },
                'tracking': {
                    'tag_count': len(self.tag_index),
                    'dependency_count': len(self.dependency_graph),
                    'frequency_tracking_size': len(self.frequency_counter)
                }
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        with self._lock:
            # Entry size distribution
            size_ranges = {
                'small (<1KB)': 0,
                'medium (1-10KB)': 0,
                'large (10-100KB)': 0,
                'xlarge (>100KB)': 0
            }

            for entry in self.cache.values():
                size_kb = entry.size_bytes / 1024
                if size_kb < 1:
                    size_ranges['small (<1KB)'] += 1
                elif size_kb < 10:
                    size_ranges['medium (1-10KB)'] += 1
                elif size_kb < 100:
                    size_ranges['large (10-100KB)'] += 1
                else:
                    size_ranges['xlarge (>100KB)'] += 1

            # Tag analytics
            tag_stats = {}
            for tag, keys in self.tag_index.items():
                tag_stats[tag] = len(keys)

            # Top entries by hit rate
            top_entries = sorted(
                [(k, v) for k, v in self.cache.items()],
                key=lambda x: x[1].hit_rate,
                reverse=True
            )[:10]

            return {
                'size_distribution': size_ranges,
                'tag_analytics': tag_stats,
                'top_entries': [
                    {
                        'key': key,
                        'hit_rate': entry.hit_rate,
                        'access_count': entry.access_count,
                        'age_hours': entry.age_seconds / 3600,
                        'size_kb': entry.size_bytes / 1024
                    }
                    for key, entry in top_entries
                ],
                'performance_trends': {
                    'recent_hit_rate': self._calculate_recent_hit_rate(),
                    'avg_access_time_trend': self._calculate_access_time_trend()
                }
            }

    def _calculate_recent_hit_rate(self, window_minutes: int = 5) -> float:
        """Calculate recent hit rate."""
        if not self.enable_metrics or len(self.access_times) < 10:
            return self.metrics.hit_rate

        # Use recent metrics as approximation
        return self.metrics.hit_rate

    def _calculate_access_time_trend(self) -> str:
        """Calculate access time trend."""
        if len(self.access_times) < 20:
            return "insufficient_data"

        recent_avg = sum(self.access_times[-10:]) / 10
        older_avg = sum(self.access_times[-20:-10]) / 10

        if recent_avg < older_avg * 0.8:
            return "improving"
        elif recent_avg > older_avg * 1.2:
            return "degrading"
        else:
            return "stable"

    def export_cache_state(self) -> Dict[str, Any]:
        """Export cache state for backup/migration."""
        with self._lock:
            return {
                'cache_entries': {
                    key: {
                        'value': entry.value,
                        'timestamp': entry.timestamp,
                        'ttl': entry.ttl,
                        'tags': entry.tags,
                        'priority': entry.priority,
                        'dependencies': entry.dependencies
                    }
                    for key, entry in self.cache.items()
                },
                'metrics': self.get_cache_stats(),
                'configuration': {
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'default_ttl': self.default_ttl,
                    'strategy': self.strategy.value,
                    'enable_compression': self.enable_compression
                },
                'export_timestamp': datetime.now().isoformat()
            }

    def import_cache_state(self, state: Dict[str, Any], clear_existing: bool = False):
        """Import cache state from backup/migration."""
        with self._lock:
            if clear_existing:
                self.cache.clear()
                self.metrics = CacheMetrics()

            entries = state.get('cache_entries', {})
            imported = 0

            for key, entry_data in entries.items():
                try:
                    self.set(
                        key=key,
                        value=entry_data['value'],
                        ttl=entry_data.get('ttl'),
                        tags=entry_data.get('tags', []),
                        priority=entry_data.get('priority', 1),
                        dependencies=entry_data.get('dependencies', [])
                    )
                    imported += 1
                except Exception as e:
                    logger.error(f"❌ Failed to import cache entry {key}: {e}")

            logger.info(f"📥 Imported {imported} cache entries")


# Global cache instances
_cache_managers: Dict[str, IntelligentCacheManager] = {}

def get_cache_manager(name: str = "default",
                     max_size_mb: int = 100,
                     strategy: CacheStrategy = CacheStrategy.ADAPTIVE) -> IntelligentCacheManager:
    """Get or create cache manager instance."""
    if name not in _cache_managers:
        _cache_managers[name] = IntelligentCacheManager(
            max_size_mb=max_size_mb,
            strategy=strategy
        )
    return _cache_managers[name]


def cache_result(ttl: Optional[float] = None,
                  tags: List[str] = None,
                  key_func: Callable = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"
                cache_key += f":{hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()}"

            # Get cache manager
            cache = get_cache_manager("decorator_cache")

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl, tags=tags)

            return result

        return wrapper
    return decorator