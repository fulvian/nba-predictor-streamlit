"""Intelligent caching system for NBA Predictor.

This module implements a comprehensive caching strategy with TTL support,
prediction caching, and performance monitoring to achieve >80% cache hit rate.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import CacheError

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Intelligent cache manager for NBA Predictor with TTL support and performance monitoring.

    Features:
    - File-based caching with configurable TTL
    - Prediction result caching with model versioning
    - Performance monitoring and statistics
    - Automatic cache cleanup and optimization
    - Cache warming strategies
    """

    def __init__(self, cache_dir: str = ".nba_cache/", default_ttl: int = 3600):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "predictions").mkdir(exist_ok=True)
        (self.cache_dir / "features").mkdir(exist_ok=True)
        (self.cache_dir / "models").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)

        self.default_ttl = default_ttl
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "last_cleanup": datetime.now(),
        }

        # Cache configuration
        self.cache_config = {
            "predictions": {"ttl": 1800, "max_size": 1000},  # 30 minutes
            "features": {"ttl": 3600, "max_size": 500},  # 1 hour
            "models": {"ttl": 86400, "max_size": 100},  # 24 hours
            "analytics": {"ttl": 7200, "max_size": 200},  # 2 hours
            "metadata": {"ttl": 604800, "max_size": 50},  # 1 week
        }

        # Load cache statistics
        self._load_stats()

    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get file path for cache entry."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / cache_type / f"{key_hash}.cache"

    def _get_metadata_path(self, cache_type: str, key: str) -> Path:
        """Get metadata file path for cache entry."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / "metadata" / f"{cache_type}_{key_hash}.meta"

    def _is_cache_valid(
        self, cache_path: Path, metadata_path: Path, ttl: Optional[int] = None
    ) -> bool:
        """Check if cache entry is valid."""
        if not cache_path.exists() or not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check TTL
            cache_ttl = ttl or self.cache_config.get(metadata.get("type", {}), {}).get(
                "ttl", self.default_ttl
            )
            created_time = datetime.fromisoformat(metadata["created_at"])

            if datetime.now() - created_time > timedelta(seconds=cache_ttl):
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def _cleanup_cache_type(self, cache_type: str) -> None:
        """Clean up expired cache entries for a specific type."""
        cache_dir = self.cache_dir / cache_type
        if not cache_dir.exists():
            return

        ttl = self.cache_config.get(cache_type, {}).get("ttl", self.default_ttl)
        max_size = self.cache_config.get(cache_type, {}).get("max_size", 100)

        # Remove expired entries
        expired_count = 0
        for cache_file in cache_dir.glob("*.cache"):
            metadata_file = (
                self.cache_dir / "metadata" / f"{cache_type}_{cache_file.stem}.meta"
            )

            if not self._is_cache_valid(cache_file, metadata_file, ttl):
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                expired_count += 1

        # Remove oldest entries if over size limit
        cache_files = list(cache_dir.glob("*.cache"))
        if len(cache_files) > max_size:
            # Sort by modification time
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = cache_files[:-max_size]

            for cache_file in files_to_remove:
                metadata_file = (
                    self.cache_dir / "metadata" / f"{cache_type}_{cache_file.stem}.meta"
                )
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                self.cache_stats["evictions"] += 1

        if expired_count > 0:
            logger.info(
                f"Cleaned up {expired_count} expired {cache_type} cache entries"
            )

    def cleanup_cache(self) -> None:
        """Clean up all expired cache entries and enforce size limits."""
        for cache_type in self.cache_config.keys():
            self._cleanup_cache_type(cache_type)

        self.cache_stats["last_cleanup"] = datetime.now()
        self._save_stats()

    def get_cached_prediction(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction if available.

        Args:
            key: Cache key for the prediction

        Returns:
            Cached prediction data or None if not available/expired
        """
        cache_path = self._get_cache_path("predictions", key)
        metadata_path = self._get_metadata_path("predictions", key)

        if not self._is_cache_valid(cache_path, metadata_path):
            self.cache_stats["misses"] += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                prediction_data = pickle.load(f)

            self.cache_stats["hits"] += 1
            logger.debug(f"Cache hit for prediction key: {key}")
            return prediction_data

        except Exception as e:
            logger.error(f"Error loading cached prediction: {e}")
            self.cache_stats["misses"] += 1
            return None

    def cache_prediction(
        self, key: str, prediction: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """
        Cache prediction with TTL.

        Args:
            key: Cache key for the prediction
            prediction: Prediction data to cache
            ttl: Time to live in seconds (optional)
        """
        try:
            cache_path = self._get_cache_path("predictions", key)
            metadata_path = self._get_metadata_path("predictions", key)

            # Save prediction data
            with open(cache_path, "wb") as f:
                pickle.dump(prediction, f)

            # Save metadata
            metadata = {
                "type": "predictions",
                "created_at": datetime.now().isoformat(),
                "ttl": ttl or self.cache_config["predictions"]["ttl"],
                "key": key,
                "model_version": prediction.get("model_version", "unknown"),
                "prediction_type": prediction.get("prediction_type", "standard"),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.cache_stats["sets"] += 1
            logger.debug(f"Cached prediction with key: {key}")

        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
            raise CacheError(f"Failed to cache prediction: {e}") from e

    def get_cached_features(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get cached features if available.

        Args:
            key: Cache key for the features

        Returns:
            Cached features DataFrame or None if not available/expired
        """
        cache_path = self._get_cache_path("features", key)
        metadata_path = self._get_metadata_path("features", key)

        if not self._is_cache_valid(cache_path, metadata_path):
            self.cache_stats["misses"] += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                features_data = pickle.load(f)

            self.cache_stats["hits"] += 1
            logger.debug(f"Cache hit for features key: {key}")
            return features_data

        except Exception as e:
            logger.error(f"Error loading cached features: {e}")
            self.cache_stats["misses"] += 1
            return None

    def cache_features(
        self, key: str, features: pd.DataFrame, ttl: Optional[int] = None
    ) -> None:
        """
        Cache features with TTL.

        Args:
            key: Cache key for the features
            features: Features DataFrame to cache
            ttl: Time to live in seconds (optional)
        """
        try:
            cache_path = self._get_cache_path("features", key)
            metadata_path = self._get_metadata_path("features", key)

            # Save features data
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)

            # Save metadata
            metadata = {
                "type": "features",
                "created_at": datetime.now().isoformat(),
                "ttl": ttl or self.cache_config["features"]["ttl"],
                "key": key,
                "feature_count": len(features.columns)
                if hasattr(features, "columns")
                else 0,
                "row_count": len(features) if hasattr(features, "__len__") else 0,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.cache_stats["sets"] += 1
            logger.debug(f"Cached features with key: {key}")

        except Exception as e:
            logger.error(f"Error caching features: {e}")
            raise CacheError(f"Failed to cache features: {e}") from e

    def cache_model(
        self, key: str, model: Any, ttl: Optional[int] = None
    ) -> None:
        """
        Cache model with TTL.

        Args:
            key: Cache key for model
            model: Model object to cache
            ttl: Time to live in seconds (optional)
        """
        try:
            cache_path = self._get_cache_path("models", key)
            metadata_path = self._get_metadata_path("models", key)

            # Save model data
            with open(cache_path, "wb") as f:
                pickle.dump(model, f)

            # Save metadata
            metadata = {
                "type": "models",
                "created_at": datetime.now().isoformat(),
                "ttl": ttl or self.cache_config["models"]["ttl"],
                "key": key,
                "model_type": str(type(model).__name__),
                "model_size": len(str(model)) if model else 0,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.cache_stats["sets"] += 1
            logger.debug(f"Cached model with key: {key}")

        except Exception as e:
            logger.error(f"Error caching model: {e}")
            raise CacheError(f"Failed to cache model: {e}") from e

    def get_cached_model(self, key: str) -> Optional[Any]:
        """
        Get cached model if available.

        Args:
            key: Cache key for model

        Returns:
            Cached model object or None if not available/expired
        """
        cache_path = self._get_cache_path("models", key)
        metadata_path = self._get_metadata_path("models", key)

        if not self._is_cache_valid(cache_path, metadata_path):
            self.cache_stats["misses"] += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                model_data = pickle.load(f)

            self.cache_stats["hits"] += 1
            logger.debug(f"Cache hit for model key: {key}")
            return model_data

        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            self.cache_stats["misses"] += 1
            return None

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        # Get cache sizes
        cache_sizes = {}
        for cache_type in self.cache_config.keys():
            cache_dir = self.cache_dir / cache_type
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.cache"))
                cache_sizes[cache_type] = {
                    "file_count": len(cache_files),
                    "total_size_mb": sum(f.stat().st_size for f in cache_files)
                    / (1024 * 1024),
                }

        return {
            "performance": {
                "cache_hits": self.cache_stats["hits"],
                "cache_misses": self.cache_stats["misses"],
                "cache_sets": self.cache_stats["sets"],
                "cache_evictions": self.cache_stats["evictions"],
                "hit_rate": f"{hit_rate:.2%}",
                "total_requests": total_requests,
            },
            "storage": cache_sizes,
            "maintenance": {
                "last_cleanup": self.cache_stats["last_cleanup"].isoformat(),
                "cache_directory": str(self.cache_dir),
                "default_ttl": self.default_ttl,
            },
        }

    def warm_cache(
        self, data_store: UnifiedDataStore, warmup_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Warm up cache with commonly accessed data.

        Args:
            data_store: UnifiedDataStore instance
            warmup_data: Optional data to warm up cache with
        """
        logger.info("Starting cache warmup...")

        try:
            # Warm up with recent games
            if not warmup_data:
                warmup_data = {}

                # Get recent games for cache warming
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)

                query = f"""
                SELECT DISTINCT home_team, away_team, game_date
                FROM read_parquet('{data_store.games_dir}/*.parquet')
                WHERE game_date BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
                LIMIT 50
                """

                try:
                    result = data_store.query_analytics(query)
                    if result is not None and len(result) > 0:
                        warmup_data["recent_games"] = result.to_pandas()
                except Exception as e:
                    logger.warning(f"Failed to get warmup data: {e}")

            # Cache recent games data
            if "recent_games" in warmup_data:
                games_df = warmup_data["recent_games"]
                for _, game in games_df.iterrows():
                    key = f"game_features_{game['home_team']}_{game['away_team']}_{game['game_date']}"
                    # This would typically compute and cache features
                    # For now, we'll just create empty cache entries
                    self.cache_features(key, pd.DataFrame(), ttl=1800)

            logger.info("Cache warmup completed")

        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")
            raise CacheError(f"Failed to warm up cache: {e}") from e

    def invalidate_cache_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match cache keys

        Returns:
            Number of invalidated entries
        """
        invalidated_count = 0

        try:
            for cache_type in self.cache_config.keys():
                cache_dir = self.cache_dir / cache_type
                if not cache_dir.exists():
                    continue

                for cache_file in cache_dir.glob("*.cache"):
                    metadata_file = (
                        self.cache_dir
                        / "metadata"
                        / f"{cache_type}_{cache_file.stem}.meta"
                    )

                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)

                            if pattern in metadata.get("key", ""):
                                cache_file.unlink(missing_ok=True)
                                metadata_file.unlink(missing_ok=True)
                                invalidated_count += 1

                        except Exception as e:
                            logger.warning(
                                f"Error processing metadata file {metadata_file}: {e}"
                            )

            logger.info(
                f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}"
            )
            return invalidated_count

        except Exception as e:
            logger.error(f"Error invalidating cache pattern: {e}")
            raise CacheError(f"Failed to invalidate cache pattern: {e}") from e

    def _load_stats(self) -> None:
        """Load cache statistics from file."""
        stats_file = self.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    saved_stats = json.load(f)

                # Convert string dates back to datetime objects
                if "last_cleanup" in saved_stats:
                    saved_stats["last_cleanup"] = datetime.fromisoformat(
                        saved_stats["last_cleanup"]
                    )

                self.cache_stats.update(saved_stats)

            except Exception as e:
                logger.warning(f"Error loading cache stats: {e}")

    def _save_stats(self) -> None:
        """Save cache statistics to file."""
        stats_file = self.cache_dir / "cache_stats.json"
        try:
            # Convert datetime objects to strings for JSON serialization
            stats_to_save = self.cache_stats.copy()
            stats_to_save["last_cleanup"] = stats_to_save["last_cleanup"].isoformat()

            with open(stats_file, "w") as f:
                json.dump(stats_to_save, f, indent=2)

        except Exception as e:
            logger.warning(f"Error saving cache stats: {e}")

    def clear_all_cache(self) -> None:
        """Clear all cache data and reset statistics."""
        try:
            # Remove all cache directories
            for cache_type in self.cache_config.keys():
                cache_dir = self.cache_dir / cache_type
                if cache_dir.exists():
                    for file in cache_dir.glob("*"):
                        file.unlink()

            # Remove metadata
            metadata_dir = self.cache_dir / "metadata"
            if metadata_dir.exists():
                for file in metadata_dir.glob("*"):
                    file.unlink()

            # Reset statistics
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "evictions": 0,
                "last_cleanup": datetime.now(),
            }

            self._save_stats()
            logger.info("All cache cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise CacheError(f"Failed to clear cache: {e}") from e


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(
    cache_dir: str = ".nba_cache/", default_ttl: int = 3600
) -> CacheManager:
    """
    Get or create global cache manager instance.

    Args:
        cache_dir: Directory for cache storage
        default_ttl: Default TTL in seconds

    Returns:
        CacheManager instance
    """
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir, default_ttl)

    return _cache_manager


def setup_cache_for_nba_predictor(
    cache_dir: str = ".nba_cache/", enable_warmup: bool = True
) -> CacheManager:
    """
    Setup cache manager for NBA Predictor with optimal configuration.

    Args:
        cache_dir: Directory for cache storage
        enable_warmup: Whether to enable cache warmup

    Returns:
        Configured CacheManager instance
    """
    cache_manager = get_cache_manager(cache_dir)

    # Perform cleanup on startup
    cache_manager.cleanup_cache()

    # Optional warmup
    if enable_warmup:
        try:
            from ..core.data_store import UnifiedDataStore

            data_store = UnifiedDataStore()
            data_store.initialize()
            cache_manager.warm_cache(data_store)
        except Exception as e:
            logger.warning(f"Cache warmup failed: {e}")

    logger.info("Cache manager configured for NBA Predictor")
    return cache_manager
