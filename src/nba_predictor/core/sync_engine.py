"""Automatic data synchronization engine for NBA system.

This module provides an automatic data synchronization engine that can
fetch data from various APIs and store it in the unified data store.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import polars as pl

from .data_store import UnifiedDataStore
from ..utils.exceptions import APIError, SyncError, ValidationError

logger = logging.getLogger(__name__)


class AutomaticSyncEngine:
    """Automatic data synchronization engine for NBA system.

    This engine provides automatic synchronization of NBA data including
    games, players, teams, and betting odds from various API sources.
    """

    def __init__(
        self,
        data_store: UnifiedDataStore,
        sync_interval: int = 3600,  # 1 hour default
        retry_attempts: int = 3,
        batch_size: int = 1000
    ) -> None:
        """
        Initialize the automatic sync engine.

        Args:
            data_store: UnifiedDataStore instance for data persistence
            sync_interval: Time interval between syncs in seconds
            retry_attempts: Number of retry attempts for failed operations
            batch_size: Batch size for data processing

        Returns:
            None

        Raises:
            ValidationError: If data_store is not provided
        """
        if data_store is None:
            raise ValidationError("data_store is required")

        self.data_store = data_store
        self.sync_interval = sync_interval
        self.retry_attempts = retry_attempts
        self.batch_size = batch_size

        # Sync status tracking
        self._last_sync: Optional[datetime] = None
        self._is_syncing = False
        self._sync_stats: Dict[str, Any] = {}

        logger.info(
            "AutomaticSyncEngine initialized",
            extra={
                "sync_interval": sync_interval,
                "retry_attempts": retry_attempts,
                "batch_size": batch_size
            }
        )

    async def sync_all_data(
        self,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize all NBA data automatically.

        Args:
            force_refresh: Force refresh even if data is fresh
            progress_callback: Callback for progress updates

        Returns:
            Dict with sync results and statistics

        Raises:
            SyncError: If synchronization fails
            APIError: If API calls fail

        Example:
            >>> engine = AutomaticSyncEngine(data_store)
            >>> result = await engine.sync_all_data()
            >>> print(f"Synced {result['games_count']} games")
        """
        if self._is_syncing and not force_refresh:
            logger.info("Sync already in progress, skipping")
            return self._get_sync_status()

        self._is_syncing = True
        start_time = datetime.now()

        try:
            logger.info("Starting automatic data synchronization", extra={"force_refresh": force_refresh})

            # Initialize sync results
            sync_results: Dict[str, Any] = {
                "start_time": start_time,
                "success": False,
                "games_count": 0,
                "players_count": 0,
                "odds_count": 0,
                "teams_count": 0,
                "errors": [],
                "warnings": [],
                "duration_seconds": 0
            }

            # Calculate date range for sync (last 30 days by default)
            end_date = date.today()
            start_date = end_date - timedelta(days=30)

            if progress_callback:
                progress_callback("Initializing sync...", 0.0)

            # Step 1: Sync games data
            try:
                games_data = await self._sync_games_data(start_date, end_date, progress_callback)
                sync_results["games_count"] = len(games_data) if games_data is not None else 0

                if progress_callback:
                    progress_callback("Games data synchronized", 0.25)

            except Exception as e:
                error_msg = f"Failed to sync games data: {e}"
                logger.error(error_msg)
                sync_results["errors"].append(error_msg)

            # Step 2: Sync players data
            try:
                players_data = await self._sync_players_data(end_date.year, progress_callback)
                sync_results["players_count"] = len(players_data) if players_data is not None else 0

                if progress_callback:
                    progress_callback("Players data synchronized", 0.5)

            except Exception as e:
                error_msg = f"Failed to sync players data: {e}"
                logger.error(error_msg)
                sync_results["errors"].append(error_msg)

            # Step 3: Sync teams data
            try:
                teams_data = await self._sync_teams_data(progress_callback)
                sync_results["teams_count"] = len(teams_data) if teams_data is not None else 0

                if progress_callback:
                    progress_callback("Teams data synchronized", 0.75)

            except Exception as e:
                error_msg = f"Failed to sync teams data: {e}"
                logger.error(error_msg)
                sync_results["errors"].append(error_msg)

            # Step 4: Sync odds data
            try:
                odds_data = await self._sync_odds_data(start_date, end_date, progress_callback)
                sync_results["odds_count"] = len(odds_data) if odds_data is not None else 0

                if progress_callback:
                    progress_callback("Odds data synchronized", 1.0)

            except Exception as e:
                error_msg = f"Failed to sync odds data: {e}"
                logger.error(error_msg)
                sync_results["errors"].append(error_msg)

            # Calculate final results
            end_time = datetime.now()
            sync_results["duration_seconds"] = (end_time - start_time).total_seconds()
            sync_results["success"] = len(sync_results["errors"]) == 0

            # Update sync status
            self._last_sync = end_time
            self._sync_stats = sync_results

            if progress_callback:
                status = "completed" if sync_results["success"] else "completed_with_errors"
                progress_callback(f"Sync {status}", 1.0)

            logger.info(
                "Data synchronization completed",
                extra={
                    "success": sync_results["success"],
                    "duration": sync_results["duration_seconds"],
                    "games_count": sync_results["games_count"],
                    "errors_count": len(sync_results["errors"])
                }
            )

            return sync_results

        except Exception as e:
            logger.error("Critical sync failure", extra={"error": str(e)})
            raise SyncError(f"Sync failed: {e}") from e

        finally:
            self._is_syncing = False

    async def _sync_games_data(
        self,
        start_date: date,
        end_date: date,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Optional[pl.DataFrame]:
        """
        Synchronize NBA games data.

        Args:
            start_date: Start date for games data
            end_date: End date for games data
            progress_callback: Progress callback

        Returns:
            Polars DataFrame with games data or None if failed
        """
        try:
            # TODO: Implement actual API call to NBA data source
            # For now, return mock data
            mock_games = [
                {
                    "game_id": "0012400001",
                    "game_date": "2024-01-01",
                    "home_team": "LAL",
                    "away_team": "GSW",
                    "season": 2024,
                    "home_score": 110,
                    "away_score": 108
                },
                {
                    "game_id": "0012400002",
                    "game_date": "2024-01-02",
                    "home_team": "BOS",
                    "away_team": "MIA",
                    "season": 2024,
                    "home_score": 105,
                    "away_score": 102
                }
            ]

            games_df = pl.DataFrame(mock_games)

            # Store games data
            date_str = start_date.strftime("%Y-%m-%d")
            file_path = self.data_store.store_games_data(games_df, date_str)

            logger.info(f"Games data synchronized: {len(games_df)} games stored to {file_path}")
            return games_df

        except Exception as e:
            logger.error(f"Failed to sync games data: {e}")
            return None

    async def _sync_players_data(
        self,
        season: int,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Optional[pl.DataFrame]:
        """
        Synchronize NBA players data.

        Args:
            season: NBA season year
            progress_callback: Progress callback

        Returns:
            Polars DataFrame with players data or None if failed
        """
        try:
            # TODO: Implement actual API call to NBA data source
            # For now, return mock data
            mock_players = [
                {
                    "player_id": "2544",
                    "player_name": "LeBron James",
                    "team_id": "1610612747",
                    "season": season,
                    "position": "F",
                    "height": "6-9",
                    "weight": 250
                },
                {
                    "player_id": "1628362",
                    "player_name": "Giannis Antetokounmpo",
                    "team_id": "1610612749",
                    "season": season,
                    "position": "F",
                    "height": "6-11",
                    "weight": 242
                }
            ]

            players_df = pl.DataFrame(mock_players)

            # Store players data
            season_str = str(season)
            file_path = self.data_store.store_players_data(players_df, season_str)

            logger.info(f"Players data synchronized: {len(players_df)} players stored to {file_path}")
            return players_df

        except Exception as e:
            logger.error(f"Failed to sync players data: {e}")
            return None

    async def _sync_teams_data(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Optional[pl.DataFrame]:
        """
        Synchronize NBA teams data.

        Args:
            progress_callback: Progress callback

        Returns:
            Polars DataFrame with teams data or None if failed
        """
        try:
            # TODO: Implement actual API call to NBA data source
            # For now, return mock data
            mock_teams = [
                {
                    "team_id": "1610612747",
                    "team_name": "Los Angeles Lakers",
                    "abbreviation": "LAL",
                    "conference": "West",
                    "division": "Pacific"
                },
                {
                    "team_id": "1610612749",
                    "team_name": "Milwaukee Bucks",
                    "abbreviation": "MIL",
                    "conference": "East",
                    "division": "Central"
                }
            ]

            teams_df = pl.DataFrame(mock_teams)

            # TODO: Store teams data (need to add teams storage to data store)
            logger.info(f"Teams data synchronized: {len(teams_df)} teams")
            return teams_df

        except Exception as e:
            logger.error(f"Failed to sync teams data: {e}")
            return None

    async def _sync_odds_data(
        self,
        start_date: date,
        end_date: date,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Optional[pl.DataFrame]:
        """
        Synchronize betting odds data.

        Args:
            start_date: Start date for odds data
            end_date: End date for odds data
            progress_callback: Progress callback

        Returns:
            Polars DataFrame with odds data or None if failed
        """
        try:
            # TODO: Implement actual API call to odds data source
            # For now, return mock data
            mock_odds = [
                {
                    "game_id": "0012400001",
                    "bookmaker": "DraftKings",
                    "home_odds": -110,
                    "away_odds": -110,
                    "updated_time": datetime.now().isoformat()
                },
                {
                    "game_id": "0012400001",
                    "bookmaker": "FanDuel",
                    "home_odds": -108,
                    "away_odds": -112,
                    "updated_time": datetime.now().isoformat()
                }
            ]

            odds_df = pl.DataFrame(mock_odds)

            # Store odds data
            date_str = start_date.strftime("%Y-%m-%d")
            file_path = self.data_store.store_odds_data(odds_df, date_str)

            logger.info(f"Odds data synchronized: {len(odds_df)} odds entries stored to {file_path}")
            return odds_df

        except Exception as e:
            logger.error(f"Failed to sync odds data: {e}")
            return None

    def _get_sync_status(self) -> Dict[str, Any]:
        """
        Get current sync status.

        Returns:
            Dict with sync status information
        """
        return {
            "is_syncing": self._is_syncing,
            "last_sync": self._last_sync,
            "sync_stats": self._sync_stats,
            "sync_interval": self.sync_interval
        }

    def should_sync(self) -> bool:
        """
        Check if sync should be performed based on interval.

        Returns:
            True if sync should be performed
        """
        if self._last_sync is None:
            return True

        time_since_last_sync = datetime.now() - self._last_sync
        return time_since_last_sync.total_seconds() >= self.sync_interval

    async def start_automatic_sync(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """
        Start automatic sync in background.

        Args:
            progress_callback: Progress callback
        """
        while True:
            try:
                if self.should_sync():
                    logger.info("Starting automatic sync")
                    await self.sync_all_data(progress_callback=progress_callback)
                else:
                    logger.debug("Skipping automatic sync - data is fresh")

                # Wait for next sync interval
                await asyncio.sleep(self.sync_interval)

            except Exception as e:
                logger.error(f"Automatic sync failed: {e}")
                # Wait shorter interval on error
                await asyncio.sleep(60)  # 1 minute

    def get_sync_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sync statistics.

        Returns:
            Dict with sync statistics
        """
        status = self._get_sync_status()

        # Add data store statistics
        try:
            metadata = self.data_store.get_metadata()
            data_stats = {
                "total_tables": metadata.height,
                "total_records": metadata["record_count"].sum() if metadata.height > 0 else 0,
                "last_updated": metadata["last_updated"].max() if metadata.height > 0 else None
            }
        except Exception:
            data_stats = {
                "total_tables": 0,
                "total_records": 0,
                "last_updated": None
            }

        return {
            **status,
            "data_statistics": data_stats,
            "configuration": {
                "sync_interval": self.sync_interval,
                "retry_attempts": self.retry_attempts,
                "batch_size": self.batch_size
            }
        }