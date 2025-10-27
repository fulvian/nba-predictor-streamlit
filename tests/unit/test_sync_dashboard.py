"""Test sync dashboard Streamlit component."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.nba_predictor.core.data_store import UnifiedDataStore
from src.nba_predictor.core.sync_engine import AutomaticSyncEngine
from src.nba_predictor.streamlit.components.sync_dashboard import render_sync_dashboard


class TestSyncDashboard:
    """Test cases for sync dashboard component."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_store = UnifiedDataStore(self.temp_dir, cache_enabled=True)
        self.data_store.initialize()
        self.sync_engine = AutomaticSyncEngine(
            self.data_store,
            sync_interval=60,
            retry_attempts=2,
            batch_size=100
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.data_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.progress')
    @patch('streamlit.button')
    @patch('streamlit.expander')
    @patch('streamlit.spinner')
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.dataframe')
    @patch('streamlit.write')
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('streamlit.rerun')
    def test_render_sync_dashboard_success(
        self,
        mock_rerun,
        mock_empty,
        mock_progress,
        mock_write,
        mock_dataframe,
        mock_info,
        mock_error,
        mock_success,
        mock_spinner,
        mock_expander,
        mock_button,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test successful rendering of sync dashboard."""
        # Mock streamlit functions to return appropriate values
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_button.return_value = False  # Button not clicked
        mock_empty.return_value = Mock()
        mock_progress.return_value = Mock()

        # Render dashboard
        render_sync_dashboard(self.sync_engine, self.data_store)

        # Verify basic streamlit functions were called
        mock_title.assert_called_once_with("🔄 Data Synchronization Dashboard")
        mock_caption.assert_called_once()
        mock_subheader.assert_called()

        # Verify metrics were displayed
        assert mock_metric.call_count >= 4  # Should show multiple metrics

    def test_sync_dashboard_initialization(self):
        """Test dashboard initialization with valid components."""
        # This test verifies that the dashboard can be initialized without errors
        assert self.sync_engine is not None
        assert self.data_store is not None
        assert self.sync_engine.data_store == self.data_store

    def test_sync_status_overview_data(self):
        """Test sync status overview data structure."""
        sync_stats = self.sync_engine.get_sync_statistics()

        # Verify required fields exist
        required_fields = ["is_syncing", "last_sync", "sync_stats", "configuration"]
        for field in required_fields:
            assert field in sync_stats

        # Verify configuration structure
        config = sync_stats["configuration"]
        assert "sync_interval" in config
        assert "retry_attempts" in config
        assert "batch_size" in config

    def test_should_sync_logic(self):
        """Test sync timing logic."""
        # Initially should sync (never synced)
        assert self.sync_engine.should_sync() is True

        # After sync, should not sync immediately
        self.sync_engine._last_sync = datetime.now()
        assert self.sync_engine.should_sync() is False

        # After interval passes, should sync again
        old_time = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
        self.sync_engine._last_sync = old_time
        assert self.sync_engine.should_sync() is True

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_sync_status_display(
        self,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test sync status display with different states."""
        # Mock columns to return mock objects
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

        # Test with idle status
        self.sync_engine._is_syncing = False
        sync_stats = self.sync_engine.get_sync_statistics()

        # Test with syncing status
        self.sync_engine._is_syncing = True
        sync_stats_syncing = self.sync_engine.get_sync_statistics()

        # Verify different states produce different metrics
        assert mock_metric.call_count >= 8  # Should be called for both states

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.button')
    @patch('streamlit.expander')
    def test_sync_controls_rendering(
        self,
        mock_expander,
        mock_button,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test sync controls rendering."""
        # Mock streamlit functions
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        mock_button.return_value = False  # Buttons not clicked

        # Import the internal function to test it
        from src.nba_predictor.streamlit.components.sync_dashboard import _render_sync_controls

        # Render controls
        _render_sync_controls(self.sync_engine)

        # Verify buttons were created
        assert mock_button.call_count >= 3  # Should have multiple control buttons

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.write')
    @patch('streamlit.dataframe')
    def test_data_statistics_rendering(
        self,
        mock_dataframe,
        mock_write,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test data statistics rendering."""
        # Mock streamlit functions
        mock_columns.return_value = [Mock(), Mock(), Mock()]

        # Add some test data to the data store
        test_games = pl.DataFrame({
            'game_id': ['0012400001'],
            'game_date': ['2024-01-01'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'season': [2024]
        })

        self.data_store.store_games_data(test_games, "2024-01-01")

        # Import the internal function to test it
        from src.nba_predictor.streamlit.components.sync_dashboard import _render_data_statistics, set_data_store

        # Set the global data store reference
        set_data_store(self.data_store)

        # Get data statistics
        sync_stats = self.sync_engine.get_sync_statistics()
        data_stats = sync_stats.get("data_statistics", {})

        # Render statistics
        _render_data_statistics(data_stats)

        # Verify metrics were displayed
        assert mock_metric.call_count >= 3  # Should show total tables, records, etc.

    def test_fragment_decorator_configuration(self):
        """Test that the fragment decorator is properly configured."""
        # Check that the render_sync_dashboard function has the fragment decorator
        import inspect
        from src.nba_predictor.streamlit.components.sync_dashboard import render_sync_dashboard

        # Get the function's decorators
        func = render_sync_dashboard
        assert hasattr(func, '__wrapped__')  # Indicates decorator is applied

        # The function should be wrapped by streamlit.fragment
        source = inspect.getsource(func)
        assert "@st.fragment" in source
        assert "run_every=30" in source

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.write')
    def test_activity_log_rendering(
        self,
        mock_write,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test activity log rendering."""
        # Mock streamlit functions
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

        # Set some mock sync statistics
        self.sync_engine._sync_stats = {
            "start_time": datetime.now(),
            "success": True,
            "games_count": 10,
            "players_count": 50,
            "teams_count": 30,
            "odds_count": 20,
            "duration_seconds": 15.5,
            "errors": []
        }

        # Import the internal function to test it
        from src.nba_predictor.streamlit.components.sync_dashboard import _render_activity_log

        # Render activity log
        _render_activity_log(self.sync_engine._sync_stats)

        # Verify metrics and content were displayed
        assert mock_metric.call_count >= 7  # Status, duration, started, 4 data types

    def test_error_handling(self):
        """Test error handling in dashboard rendering."""
        # Test with invalid sync engine
        with pytest.raises(Exception):
            render_sync_dashboard(None, self.data_store)

        # Test with invalid data store
        with pytest.raises(Exception):
            render_sync_dashboard(self.sync_engine, None)