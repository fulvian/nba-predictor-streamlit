#!/usr/bin/env python3
"""
🏀 Data Persistence Bridge - NBA System

Bridge component that connects the API cache (data_provider.py) with the
persistent storage (UnifiedDataStore) to automatically save NBA data.

Key Features:
- Auto-persistenza dei dati API da cache a storage permanente
- Integrazione trasparente con data_provider.py esistente
- Schema mapping tra API format e UnifiedDataStore format
- Background sync per dati storici
- Smart caching per evitare duplicazioni
"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import polars as pl

# Import existing components (no circular imports)
# from data_provider import NBADataProvider, game_cache  # Avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_provider import NBADataProvider
from src.nba_predictor.core.data_store import UnifiedDataStore
from src.nba_predictor.core.sync_engine import AutomaticSyncEngine
from src.nba_predictor.utils.exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)


class DataPersistenceBridge:
    """
    Bridge that connects API cache with persistent storage.

    This component automatically saves NBA data from the API cache
    to the UnifiedDataStore for long-term persistence and analytics.
    """

    def __init__(
        self,
        data_provider: Union["NBADataProvider", Any],  # Use Union to avoid circular import
        storage_path: str = "data/persistent",
        auto_persist: bool = True
    ) -> None:
        """
        Initialize the data persistence bridge.

        Args:
            data_provider: NBADataProvider instance for API access
            storage_path: Path for persistent data storage
            auto_persist: Enable automatic persistence of API data
        """
        self.data_provider = data_provider
        self.auto_persist = auto_persist

        # Initialize storage components
        self.storage_path = Path(storage_path)
        self.data_store = UnifiedDataStore(
            base_path=str(self.storage_path),
            cache_enabled=True
        )

        # Initialize sync engine
        self.sync_engine = AutomaticSyncEngine(
            data_store=self.data_store,
            sync_interval=3600,  # 1 hour
            retry_attempts=3,
            batch_size=1000
        )

        # Bridge status
        self._is_initialized = False
        self._persistence_stats = {
            "total_games_saved": 0,
            "last_persist_date": None,
            "cache_hits": 0,
            "api_calls": 0
        }

        logger.info(
            "DataPersistenceBridge initialized",
            extra={
                "storage_path": str(self.storage_path),
                "auto_persist": auto_persist
            }
        )

    def initialize(self) -> None:
        """Initialize the data store and create necessary directories."""
        try:
            self.data_store.initialize()
            self._is_initialized = True

            logger.info("DataPersistenceBridge initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DataPersistenceBridge: {e}")
            raise DatabaseError(f"Bridge initialization failed: {e}") from e

    def get_scheduled_games_with_persistence(
        self,
        days_ahead: int = 7,
        specific_date: Optional[str] = None,
        force_api: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get scheduled games with automatic persistence.

        This method extends the data_provider.get_scheduled_games() method
        by adding automatic persistence to the UnifiedDataStore.

        Args:
            days_ahead: Number of days ahead to fetch
            specific_date: Specific date to fetch
            force_api: Force API call even if data exists in store

        Returns:
            List of NBA games with enhanced metadata
        """
        if not self._is_initialized:
            self.initialize()

        # Convert date format for storage lookup
        if specific_date:
            try:
                target_date = datetime.strptime(specific_date, '%Y-%m-%d').date()
            except ValueError:
                target_date = date.today()
        else:
            target_date = date.today()

        date_str = target_date.strftime('%Y-%m-%d')

        # Step 1: Try to get from persistent storage first
        if not force_api:
            stored_games = self._get_from_persistent_storage(date_str)
            if stored_games:
                self._persistence_stats["cache_hits"] += 1
                logger.info(
                    f"Retrieved {len(stored_games)} games from persistent storage",
                    extra={"date": date_str, "source": "persistent_storage"}
                )
                return stored_games

        # Step 2: Get from API (avoid recursive call to get_scheduled_games)
        self._persistence_stats["api_calls"] += 1

        # Call API directly through data provider methods to avoid recursion
        # We need to call the internal API methods, not get_scheduled_games which calls this bridge
        games = []

        # Use BallDontLie API directly (primary source)
        if hasattr(self.data_provider, '_get_ball_dont_lie_games'):
            try:
                bdl_games = self.data_provider._get_ball_dont_lie_games(
                    days_ahead=days_ahead,
                    specific_date=specific_date
                )
                games.extend(bdl_games)
            except Exception as e:
                logger.warning(f"Failed to get BallDontLie games: {e}")

        # Use fallback sources if no BallDontLie games
        if not games:
            if hasattr(self.data_provider, '_get_odds_api_games'):
                try:
                    odds_games = self.data_provider._get_odds_api_games(days_ahead=days_ahead)
                    games.extend(odds_games)
                except Exception as e:
                    logger.warning(f"Failed to get Odds API games: {e}")

            if hasattr(self.data_provider, '_get_nba_completed_games'):
                try:
                    completed_games = self.data_provider._get_nba_completed_games(days_back=3)
                    if specific_date:
                        completed_games = [g for g in completed_games if g['date'] == specific_date]
                    games.extend(completed_games)
                except Exception as e:
                    logger.warning(f"Failed to get NBA completed games: {e}")

        # Step 3: Persist API data if enabled and we have new data
        if self.auto_persist and games:
            self._persist_api_data(games, date_str)

        return games

    def _get_from_persistent_storage(self, date_str: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try to retrieve games from persistent storage.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of games or None if not found
        """
        try:
            # Query UnifiedDataStore for games on specific date
            games_df = self.data_store.get_games_data(date_range=(date_str, date_str))

            if games_df.height > 0:
                # Convert DataFrame back to list of dicts (API format)
                games_list = self._dataframe_to_api_format(games_df)
                logger.debug(
                    f"Found {len(games_list)} games in persistent storage",
                    extra={"date": date_str}
                )
                return games_list

            return None

        except Exception as e:
            logger.warning(
                f"Failed to retrieve from persistent storage: {e}",
                extra={"date": date_str}
            )
            return None

    def _persist_api_data(self, games: List[Dict[str, Any]], date_str: str) -> None:
        """
        Persist API data to UnifiedDataStore.

        Args:
            games: List of games from API
            date_str: Date string for partitioning
        """
        try:
            # Convert API format to Polars DataFrame
            games_df = self._api_format_to_dataframe(games)

            # Store in UnifiedDataStore
            file_path = self.data_store.store_games_data(games_df, date_str)

            # Update statistics
            self._persistence_stats["total_games_saved"] += len(games)
            self._persistence_stats["last_persist_date"] = datetime.now()

            logger.info(
                f"Persisted {len(games)} games to storage",
                extra={
                    "date": date_str,
                    "file_path": file_path,
                    "total_saved": self._persistence_stats["total_games_saved"]
                }
            )

        except Exception as e:
            logger.error(
                f"Failed to persist API data: {e}",
                extra={"date": date_str, "games_count": len(games)}
            )

    def _api_format_to_dataframe(self, games: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        Convert API format games list to Polars DataFrame.

        Args:
            games: List of games in API format

        Returns:
            Polars DataFrame with UnifiedDataStore schema
        """
        if not games:
            return pl.DataFrame()

        # Convert each game to UnifiedDataStore format
        unified_games = []
        for game in games:
            unified_game = {
                "game_id": game.get("game_id", ""),
                "game_date": game.get("date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "season": game.get("season", 2025),
                "home_score": game.get("home_score", 0),
                "away_score": game.get("away_score", 0),
                "status": game.get("status", "Scheduled"),
                "game_time": game.get("time_utc", ""),
                "venue": game.get("venue", ""),
                "source": game.get("source", "API"),
                "created_at": datetime.now().isoformat()
            }
            unified_games.append(unified_game)

        return pl.DataFrame(unified_games)

    def _dataframe_to_api_format(self, games_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert Polars DataFrame back to API format.

        Args:
            games_df: Polars DataFrame from storage

        Returns:
            List of games in API format
        """
        if games_df.height == 0:
            return []

        # Convert DataFrame rows back to API format
        api_games = []
        for row in games_df.to_dicts():
            api_game = {
                "game_id": row.get("game_id", ""),
                "date": row.get("game_date", ""),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "season": row.get("season", 2025),
                "home_score": row.get("home_score", 0),
                "away_score": row.get("away_score", 0),
                "status": row.get("status", "Scheduled"),
                "time_utc": row.get("game_time", ""),
                "source": f"Persistent Storage ({row.get('source', 'Unknown')})",
                "persisted_at": row.get("created_at", "")
            }
            api_games.append(api_game)

        return api_games

    def get_persistence_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive persistence statistics.

        Returns:
            Dict with persistence metrics and status
        """
        try:
            # Get data store metadata
            metadata = self.data_store.get_metadata()

            # Get sync engine statistics
            sync_stats = self.sync_engine.get_sync_statistics()

            return {
                "bridge_status": {
                    "initialized": self._is_initialized,
                    "auto_persist": self.auto_persist
                },
                "persistence_stats": self._persistence_stats,
                "data_store_stats": {
                    "total_tables": metadata.height,
                    "total_records": metadata["record_count"].sum() if metadata.height > 0 else 0,
                    "last_updated": metadata["last_updated"].max() if metadata.height > 0 else None
                },
                "sync_engine_stats": sync_stats
            }

        except Exception as e:
            logger.error(f"Failed to get persistence statistics: {e}")
            return {
                "bridge_status": {"initialized": self._is_initialized, "auto_persist": self.auto_persist},
                "persistence_stats": self._persistence_stats,
                "data_store_stats": {"error": str(e)},
                "sync_engine_stats": {"error": str(e)}
            }

    def force_full_sync(self) -> Dict[str, Any]:
        """
        Force a full synchronization of all available data.

        Returns:
            Dict with sync results
        """
        if not self._is_initialized:
            self.initialize()

        try:
            logger.info("Starting forced full synchronization")

            # Use the sync engine to sync all data
            import asyncio

            # Run sync in sync mode (not async for simplicity)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                results = loop.run_until_complete(
                    self.sync_engine.sync_all_data(force_refresh=True)
                )
            finally:
                loop.close()

            logger.info(
                "Forced full synchronization completed",
                extra={
                    "success": results.get("success", False),
                    "duration": results.get("duration_seconds", 0),
                    "games_count": results.get("games_count", 0)
                }
            )

            return results

        except Exception as e:
            logger.error(f"Forced full sync failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "games_count": 0,
                "duration_seconds": 0
            }

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old data from persistent storage.

        Args:
            days_to_keep: Number of days to keep data

        Returns:
            Dict with cleanup results
        """
        try:
            # Calculate cutoff date
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')

            # Get old data from metadata
            metadata = self.data_store.get_metadata()

            old_tables = metadata.filter(
                pl.col("table_name") < f"games_{cutoff_str}"
            )

            # TODO: Implement actual file deletion
            # This would require extending UnifiedDataStore with delete functionality

            logger.info(
                "Old data cleanup requested",
                extra={
                    "days_to_keep": days_to_keep,
                    "cutoff_date": cutoff_str,
                    "old_tables_count": old_tables.height
                }
            )

            return {
                "success": True,
                "days_to_keep": days_to_keep,
                "cutoff_date": cutoff_str,
                "old_tables_found": old_tables.height,
                "deleted_files": 0  # TODO: Implement actual deletion
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "days_to_keep": days_to_keep
            }

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            self.data_store.close()
            logger.info("DataPersistenceBridge closed successfully")
        except Exception as e:
            logger.error(f"Error closing DataPersistenceBridge: {e}")


# Global instance for easy access
_persistence_bridge: Optional[DataPersistenceBridge] = None


def get_persistence_bridge() -> Optional[DataPersistenceBridge]:
    """
    Get the global persistence bridge instance.

    Returns:
        DataPersistenceBridge instance or None if not initialized
    """
    global _persistence_bridge
    return _persistence_bridge


def initialize_persistence_bridge(
    data_provider: Union["NBADataProvider", Any],  # Use Union to avoid circular import
    storage_path: str = "data/persistent",
    auto_persist: bool = True
) -> DataPersistenceBridge:
    """
    Initialize the global persistence bridge.

    Args:
        data_provider: NBADataProvider instance
        storage_path: Path for persistent data storage
        auto_persist: Enable automatic persistence

    Returns:
        DataPersistenceBridge instance
    """
    global _persistence_bridge

    if _persistence_bridge is None:
        _persistence_bridge = DataPersistenceBridge(
            data_provider=data_provider,
            storage_path=storage_path,
            auto_persist=auto_persist
        )
        _persistence_bridge.initialize()

    return _persistence_bridge


def close_persistence_bridge() -> None:
    """Close the global persistence bridge."""
    global _persistence_bridge

    if _persistence_bridge is not None:
        _persistence_bridge.close()
        _persistence_bridge = None