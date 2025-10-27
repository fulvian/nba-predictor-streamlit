"""Test automatic sync engine implementation."""

import asyncio
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import polars as pl
import pytest

from src.nba_predictor.core.data_store import UnifiedDataStore
from src.nba_predictor.core.sync_engine import AutomaticSyncEngine
from src.nba_predictor.utils.exceptions import SyncError, ValidationError


class TestAutomaticSyncEngine:
    """Test cases for AutomaticSyncEngine class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_store = UnifiedDataStore(self.temp_dir, cache_enabled=True)
        self.data_store.initialize()
        self.sync_engine = AutomaticSyncEngine(
            self.data_store,
            sync_interval=60,  # 1 minute for testing
            retry_attempts=2,
            batch_size=100
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.data_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_valid(self):
        """Test AutomaticSyncEngine initialization with valid parameters."""
        assert self.sync_engine.data_store == self.data_store
        assert self.sync_engine.sync_interval == 60
        assert self.sync_engine.retry_attempts == 2
        assert self.sync_engine.batch_size == 100
        assert self.sync_engine._is_syncing is False
        assert self.sync_engine._last_sync is None

    def test_init_invalid_data_store(self):
        """Test AutomaticSyncEngine initialization with invalid data store."""
        with pytest.raises(ValidationError) as exc_info:
            AutomaticSyncEngine(None)

        assert "data_store is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sync_all_data_success(self):
        """Test successful data synchronization."""
        progress_updates = []

        def mock_progress_callback(message: str, progress: float) -> None:
            progress_updates.append((message, progress))

        result = await self.sync_engine.sync_all_data(
            force_refresh=True,
            progress_callback=mock_progress_callback
        )

        # Verify sync results
        assert result["success"] is True
        assert result["games_count"] > 0
        assert result["players_count"] > 0
        assert result["teams_count"] > 0
        assert result["odds_count"] > 0
        assert result["duration_seconds"] > 0
        assert len(result["errors"]) == 0

        # Verify progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1][1] == 1.0  # Final progress should be 100%

        # Verify sync status updated
        assert self.sync_engine._last_sync is not None
        assert self.sync_engine._is_syncing is False

    @pytest.mark.asyncio
    async def test_sync_all_data_with_progress_callback(self):
        """Test data synchronization with progress callback."""
        progress_calls = []

        def progress_callback(message: str, progress: float) -> None:
            progress_calls.append((message, progress))

        await self.sync_engine.sync_all_data(
            force_refresh=True,
            progress_callback=progress_callback
        )

        # Verify progress callback was called
        assert len(progress_calls) > 0

        # Check that progress values are valid
        messages, progress_values = zip(*progress_calls)
        assert all(0.0 <= p <= 1.0 for p in progress_values)
        assert progress_values[-1] == 1.0  # Should end at 100%

        # Verify meaningful progress messages
        assert any("Initializing" in msg for msg in messages)
        assert any("synchronized" in msg for msg in messages)

    @pytest.mark.asyncio
    async def test_sync_all_data_already_syncing(self):
        """Test sync behavior when sync is already in progress."""
        # Set syncing flag manually
        self.sync_engine._is_syncing = True

        result = await self.sync_engine.sync_all_data(force_refresh=False)

        # Should return current status without starting new sync
        assert result["is_syncing"] is True

    @pytest.mark.asyncio
    async def test_sync_games_data(self):
        """Test games data synchronization."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 7)

        games_df = await self.sync_engine._sync_games_data(start_date, end_date)

        # Verify mock data structure
        assert games_df is not None
        assert isinstance(games_df, pl.DataFrame)
        assert len(games_df) > 0

        # Verify required columns exist
        required_columns = {'game_id', 'game_date', 'home_team', 'away_team', 'season'}
        assert required_columns.issubset(set(games_df.columns))

        # Verify data was stored
        metadata = self.data_store.get_metadata()
        assert metadata.height > 0
        assert any("games_" in table for table in metadata["table_name"])

    @pytest.mark.asyncio
    async def test_sync_players_data(self):
        """Test players data synchronization."""
        season = 2024

        players_df = await self.sync_engine._sync_players_data(season)

        # Verify mock data structure
        assert players_df is not None
        assert isinstance(players_df, pl.DataFrame)
        assert len(players_df) > 0

        # Verify required columns exist
        required_columns = {'player_id', 'player_name', 'team_id', 'season', 'position'}
        assert required_columns.issubset(set(players_df.columns))

        # Verify data was stored
        metadata = self.data_store.get_metadata()
        assert any("players_" in table for table in metadata["table_name"])

    @pytest.mark.asyncio
    async def test_sync_teams_data(self):
        """Test teams data synchronization."""
        teams_df = await self.sync_engine._sync_teams_data()

        # Verify mock data structure
        assert teams_df is not None
        assert isinstance(teams_df, pl.DataFrame)
        assert len(teams_df) > 0

        # Verify required columns exist
        required_columns = {'team_id', 'team_name', 'abbreviation', 'conference', 'division'}
        assert required_columns.issubset(set(teams_df.columns))

    @pytest.mark.asyncio
    async def test_sync_odds_data(self):
        """Test odds data synchronization."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 7)

        odds_df = await self.sync_engine._sync_odds_data(start_date, end_date)

        # Verify mock data structure
        assert odds_df is not None
        assert isinstance(odds_df, pl.DataFrame)
        assert len(odds_df) > 0

        # Verify required columns exist
        required_columns = {'game_id', 'bookmaker', 'home_odds', 'away_odds', 'updated_time'}
        assert required_columns.issubset(set(odds_df.columns))

        # Verify data was stored
        metadata = self.data_store.get_metadata()
        assert any("odds_" in table for table in metadata["table_name"])

    @pytest.mark.asyncio
    async def test_sync_all_data_with_errors(self):
        """Test sync behavior when some operations fail."""
        # Mock one of the sync methods to raise an exception
        with patch.object(self.sync_engine, '_sync_games_data', side_effect=Exception("API Error")):
            result = await self.sync_engine.sync_all_data(force_refresh=True)

            # Should have completed but with errors
            assert result["success"] is False
            assert len(result["errors"]) > 0
            assert "Failed to sync games data" in result["errors"][0]

            # Other data should still sync
            assert result["players_count"] > 0
            assert result["teams_count"] > 0
            assert result["odds_count"] > 0

    def test_should_sync_never_synced(self):
        """Test should_sync when never synced before."""
        assert self.sync_engine.should_sync() is True

    def test_should_sync_fresh_data(self):
        """Test should_sync with fresh data."""
        # Set last sync to recent time
        self.sync_engine._last_sync = datetime.now()
        assert self.sync_engine.should_sync() is False

    def test_should_sync_stale_data(self):
        """Test should_sync with stale data."""
        # Set last sync to old time (beyond interval)
        old_time = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
        self.sync_engine._last_sync = old_time
        assert self.sync_engine.should_sync() is True

    def test_get_sync_status(self):
        """Test getting sync status."""
        # Set some status
        self.sync_engine._last_sync = datetime.now()
        self.sync_engine._sync_stats = {"test": "value"}

        status = self.sync_engine._get_sync_status()

        assert status["is_syncing"] is False
        assert status["last_sync"] is not None
        assert status["sync_stats"]["test"] == "value"
        assert status["sync_interval"] == 60

    def test_get_sync_statistics(self):
        """Test getting comprehensive sync statistics."""
        # First run a sync to generate data
        asyncio.run(self.sync_engine.sync_all_data(force_refresh=True))

        stats = self.sync_engine.get_sync_statistics()

        # Verify basic structure
        assert "is_syncing" in stats
        assert "last_sync" in stats
        assert "sync_stats" in stats
        assert "data_statistics" in stats
        assert "configuration" in stats

        # Verify configuration
        config = stats["configuration"]
        assert config["sync_interval"] == 60
        assert config["retry_attempts"] == 2
        assert config["batch_size"] == 100

        # Verify data statistics
        data_stats = stats["data_statistics"]
        assert data_stats["total_tables"] > 0
        assert data_stats["total_records"] > 0

    @pytest.mark.asyncio
    async def test_start_automatic_sync(self):
        """Test starting automatic sync (short duration test)."""
        sync_calls = []

        # Patch sync_all_data to track calls
        async def mock_sync_all_data(*args, **kwargs):
            sync_calls.append(len(sync_calls))
            return {"success": True}

        with patch.object(self.sync_engine, 'sync_all_data', side_effect=mock_sync_all_data):
            # Start automatic sync but cancel after short time
            task = asyncio.create_task(
                self.sync_engine.start_automatic_sync()
            )

            # Wait a short time and cancel
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have attempted at least one sync
            assert len(sync_calls) >= 1

    @pytest.mark.asyncio
    async def test_critical_sync_failure(self):
        """Test handling of critical sync failures."""
        # Mock a critical failure in the main sync method
        with patch.object(
            self.sync_engine,
            '_sync_games_data',
            side_effect=Exception("Critical failure")
        ):
            with pytest.raises(SyncError) as exc_info:
                await self.sync_engine.sync_all_data(force_refresh=True)

            assert "Sync failed" in str(exc_info.value)

        # Verify sync flag is reset even on failure
        assert self.sync_engine._is_syncing is False