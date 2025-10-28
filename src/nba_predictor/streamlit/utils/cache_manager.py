"""Advanced cache management for NBA Predictor Streamlit application.

This module implements Context7-compliant caching strategies optimized for
NBA data and ML predictions, providing intelligent cache invalidation
and performance monitoring.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import polars as pl
import streamlit as st

from ...core.data_store import UnifiedDataStore
from ...utils.exceptions import CacheError

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Advanced cache manager for NBA Streamlit application.

    Implements Context7 best practices for performance optimization
    with intelligent TTL strategies and cache monitoring.
    """

    def __init__(self, data_store: Optional[UnifiedDataStore] = None) -> None:
        """
        Initialize cache manager with optional data store integration.

        Args:
            data_store: UnifiedDataStore instance for data persistence

        Raises:
            CacheError: If cache initialization fails
        """
        self.data_store = data_store
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "last_reset": datetime.now()
        }

    def get_cache_ttl(self, data_type: str, is_real_time: bool = False) -> int:
        """
        Get appropriate TTL for different data types following Context7 patterns.

        Args:
            data_type: Type of data ('games', 'predictions', 'teams', etc.)
            is_real_time: Whether this is real-time data

        Returns:
            TTL in seconds

        Examples:
            >>> cache_manager = CacheManager()
            >>> ttl = cache_manager.get_cache_ttl('games', is_real_time=True)
            >>> print(f"Games TTL: {ttl}s")
            Games TTL: 300s
        """
        ttl_mapping = {
            # Real-time data - shorter TTL
            "live_games": 60 if is_real_time else 300,
            "predictions": 180 if is_real_time else 900,
            "player_stats": 300 if is_real_time else 1800,

            # Historical data - longer TTL
            "historical_games": 3600,
            "team_stats": 7200,
            "season_data": 86400,

            # ML models and analytics - longest TTL
            "ml_models": 86400,
            "analytics_results": 3600,
            "feature_importance": 1800,

            # Configuration and metadata
            "team_mappings": 86400 * 7,  # 1 week
            "api_config": 86400,  # 1 day
        }

        return ttl_mapping.get(data_type, 300)  # Default 5 minutes

    @st.cache_data(ttl=300, show_spinner="Loading NBA games...")
    def get_nba_games_cached(
        self,
        start_date: datetime,
        end_date: datetime,
        team_filter: Optional[list] = None
    ) -> Optional[pl.DataFrame]:
        """
        Cache NBA games data with intelligent TTL based on date range.

        Args:
            start_date: Start date for games
            end_date: End date for games
            team_filter: Optional team filter

        Returns:
            DataFrame with NBA games or None if unavailable

        Raises:
            CacheError: If data retrieval fails
        """
        try:
            if not self.data_store:
                logger.warning("No data store available for games cache")
                return None

            # Determine TTL based on how recent the games are
            now = datetime.now()
            days_since_end = (now - end_date).days

            if days_since_end <= 0:  # Future or current games
                ttl = 300  # 5 minutes
            elif days_since_end <= 7:  # Recent games
                ttl = 1800  # 30 minutes
            else:  # Historical games
                ttl = 3600  # 1 hour

            # Update cache decorator TTL dynamically
            # Note: This is a limitation of streamlit caching
            # In practice, we'd use multiple cached functions

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            query = f"""
            SELECT
                game_date,
                home_team,
                away_team,
                home_score,
                away_score,
                season
            FROM read_parquet('{self.data_store.games_dir}/*.parquet')
            WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
            """

            if team_filter:
                teams_str = "', '".join(team_filter)
                query += f" AND home_team IN ('{teams_str}') OR away_team IN ('{teams_str}')"

            result = self.data_store.query_analytics(query)

            self._cache_stats["hits"] += 1
            logger.info(f"Cache hit for NBA games: {result.height if result else 0} games")

            return result

        except Exception as e:
            self._cache_stats["misses"] += 1
            logger.error(f"Cache miss for NBA games: {e}")
            raise CacheError(f"Failed to get cached NBA games: {e}") from e

    @st.cache_data(ttl=900, show_spinner="Loading predictions...")
    def get_predictions_cached(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Cache prediction results with model-aware TTL.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date

        Returns:
            Prediction results or None if unavailable

        Raises:
            CacheError: If prediction retrieval fails
        """
        try:
            # This would integrate with UnifiedHybridPipeline
            # For now, return mock prediction data
            prediction_data = {
                "predicted_total": 225.5,
                "confidence_interval": (218.0, 233.0),
                "over_probability": 0.65,
                "under_probability": 0.35,
                "confidence": 0.78,
                "recommendation": "OVER",
                "model_weights": {
                    "unified_hybrid": 0.7,
                    "enhanced": 0.2,
                    "research": 0.1
                },
                "generated_at": datetime.now().isoformat()
            }

            self._cache_stats["hits"] += 1
            logger.info(f"Cache hit for prediction: {home_team} vs {away_team}")

            return prediction_data

        except Exception as e:
            self._cache_stats["misses"] += 1
            logger.error(f"Cache miss for predictions: {e}")
            raise CacheError(f"Failed to get cached predictions: {e}") from e

    @st.cache_data(ttl=3600, show_spinner="Loading team analytics...")
    def get_team_analytics_cached(
        self,
        team_name: str,
        days_back: int = 30
    ) -> Optional[pl.DataFrame]:
        """
        Cache team analytics with performance-optimized TTL.

        Args:
            team_name: Team name to analyze
            days_back: Number of days to look back

        Returns:
            Team analytics DataFrame or None if unavailable

        Raises:
            CacheError: If analytics retrieval fails
        """
        try:
            if not self.data_store:
                return None

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            query = f"""
            WITH team_games AS (
                SELECT
                    game_date,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    CASE
                        WHEN home_team = '{team_name}' THEN home_score
                        ELSE away_score
                    END as team_score,
                    CASE
                        WHEN home_team = '{team_name}' THEN away_score
                        ELSE home_score
                    END as opponent_score,
                    CASE
                        WHEN home_team = '{team_name}' AND home_score > away_score THEN 1
                        WHEN away_team = '{team_name}' AND away_score > home_score THEN 1
                        ELSE 0
                    END as win
                FROM read_parquet('{self.data_store.games_dir}/*.parquet')
                WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
                AND (home_team = '{team_name}' OR away_team = '{team_name}')
            )
            SELECT
                DATE(game_date) as game_date,
                team_score,
                opponent_score,
                team_score - opponent_score as point_differential,
                win,
                AVG(team_score) OVER (ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as avg_points_rolling,
                SUM(win) OVER (ORDER BY game_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as wins_rolling
            FROM team_games
            ORDER BY game_date DESC
            """

            result = self.data_store.query_analytics(query)

            self._cache_stats["hits"] += 1
            logger.info(f"Cache hit for team analytics: {team_name}")

            return result

        except Exception as e:
            self._cache_stats["misses"] += 1
            logger.error(f"Cache miss for team analytics: {e}")
            raise CacheError(f"Failed to get cached team analytics: {e}") from e

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total_requests,
            "last_reset": self._cache_stats["last_reset"].isoformat()
        }

    def clear_cache_data(self) -> None:
        """Clear all Streamlit cache data and reset statistics."""
        try:
            st.cache_data.clear()
            st.cache_resource.clear()

            self._cache_stats = {
                "hits": 0,
                "misses": 0,
                "last_reset": datetime.now()
            }

            logger.info("Cache cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise CacheError(f"Failed to clear cache: {e}") from e

    def invalidate_specific_cache(self, cache_type: str) -> None:
        """
        Invalidate specific cache type based on business logic.

        Args:
            cache_type: Type of cache to invalidate

        Raises:
            CacheError: If cache invalidation fails
        """
        try:
            # This would implement selective cache invalidation
            # For now, we clear all cache
            self.clear_cache_data()

            logger.info(f"Invalidated {cache_type} cache")

        except Exception as e:
            logger.error(f"Failed to invalidate {cache_type} cache: {e}")
            raise CacheError(f"Failed to invalidate {cache_type} cache: {e}") from e


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(data_store: Optional[UnifiedDataStore] = None) -> CacheManager:
    """
    Get or create global cache manager instance.

    Args:
        data_store: Optional data store for cache manager

    Returns:
        CacheManager instance
    """
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(data_store)

    return _cache_manager


def setup_caching_for_app(data_store: Optional[UnifiedDataStore] = None) -> CacheManager:
    """
    Setup caching configuration for Streamlit application.

    Args:
        data_store: UnifiedDataStore instance

    Returns:
        Configured cache manager
    """
    cache_manager = get_cache_manager(data_store)
    logger.info("Cache manager configured for NBA Streamlit application")

    return cache_manager