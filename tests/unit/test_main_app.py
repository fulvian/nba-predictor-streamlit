"""Test main Streamlit application."""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.nba_predictor.streamlit.app import create_main_app


class TestMainApp:
    """Test cases for main Streamlit application."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.subheader')
    @patch('streamlit.divider')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.sidebar')
    @patch('streamlit.tabs')
    @patch('streamlit.caption')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.error')
    @patch('streamlit.write')
    @patch('streamlit.button')
    @patch('streamlit.expander')
    @patch('streamlit.bar_chart')
    @patch('streamlit.dataframe')
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('streamlit.rerun')
    def test_create_main_app_success(
        self,
        mock_rerun,
        mock_empty,
        mock_progress,
        mock_dataframe,
        mock_bar_chart,
        mock_expander,
        mock_button,
        mock_write,
        mock_error,
        mock_info,
        mock_success,
        mock_caption,
        mock_tabs,
        mock_sidebar,
        mock_metric,
        mock_columns,
        mock_divider,
        mock_subheader,
        mock_title,
        mock_set_page_config
    ):
        """Test successful creation of main app."""
        # Mock streamlit functions
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_tabs.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        mock_button.return_value = False  # Buttons not clicked
        mock_empty.return_value = Mock()

        # Mock path.exists to return True
        with patch('pathlib.Path.exists', return_value=True):

            # Create main app
            create_main_app()

            # Verify page configuration was called
            mock_set_page_config.assert_called_once()

            # Verify basic UI elements were rendered
            mock_title.assert_called()
            mock_subheader.assert_called()
            mock_tabs.assert_called_once()

            # Verify sidebar was rendered
            mock_sidebar.assert_called_once()

    def test_page_configuration(self):
        """Test page configuration settings."""
        from src.nba_predictor.streamlit.app import _configure_page

        with patch('streamlit.set_page_config') as mock_config:
            _configure_page()

            # Verify set_page_config was called with correct parameters
            mock_config.assert_called_once()
            call_args = mock_config.call_args[1]

            assert call_args['page_title'] == "NBA Predictor Analytics"
            assert call_args['page_icon'] == "🏀"
            assert call_args['layout'] == "wide"
            assert 'menu_items' in call_args

    def test_core_components_initialization(self):
        """Test core components initialization."""
        from src.nba_predictor.streamlit.app import _initialize_core_components

        # Mock path operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir'), \
             patch('src.nba_predictor.core.data_store.UnifiedDataStore') as mock_store_class, \
             patch('src.nba_predictor.core.sync_engine.AutomaticSyncEngine') as mock_engine_class:

            # Create mock instances
            mock_store = Mock()
            mock_engine = Mock()
            mock_store_class.return_value = mock_store
            mock_engine_class.return_value = mock_engine

            # Test initialization
            data_store, sync_engine = _initialize_core_components()

            # Verify components were created
            mock_store_class.assert_called_once()
            mock_engine_class.assert_called_once()

            # Verify initialization methods were called
            mock_store.initialize.assert_called_once()

            # Verify return values
            assert data_store == mock_store
            assert sync_engine == mock_engine

    @patch('streamlit.title')
    @patch('streamlit.sidebar')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.write')
    def test_sidebar_rendering(
        self,
        mock_write,
        mock_metric,
        mock_columns,
        mock_sidebar,
        mock_title
    ):
        """Test sidebar rendering."""
        # Mock streamlit functions
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]

        # Create mock components
        mock_data_store = Mock()
        mock_sync_engine = Mock()

        # Mock sync statistics
        mock_sync_engine.get_sync_statistics.return_value = {
            'is_syncing': False,
            'last_sync': None,
            'data_statistics': {
                'total_records': 1000,
                'total_tables': 5
            }
        }

        # Import and test sidebar function
        from src.nba_predictor.streamlit.app import _render_sidebar

        _render_sidebar(mock_data_store, mock_sync_engine)

        # Verify sidebar elements were rendered
        mock_title.assert_called()
        mock_metric.assert_called()
        mock_write.assert_called()

    @patch('streamlit.header')
    @patch('streamlit.caption')
    @patch('streamlit.divider')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.subheader')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.error')
    def test_main_dashboard_rendering(
        self,
        mock_error,
        mock_info,
        mock_success,
        mock_subheader,
        mock_metric,
        mock_columns,
        mock_divider,
        mock_caption,
        mock_header
    ):
        """Test main dashboard rendering."""
        # Mock streamlit functions
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

        # Create mock components
        mock_data_store = Mock()
        mock_sync_engine = Mock()

        # Mock sync statistics
        mock_sync_engine.get_sync_statistics.return_value = {
            'sync_stats': {
                'games_count': 100,
                'players_count': 500,
                'teams_count': 30,
                'success': True
            },
            'data_statistics': {
                'total_records': 1000,
                'total_tables': 5
            }
        }

        # Import and test dashboard function
        from src.nba_predictor.streamlit.app import _render_main_dashboard

        _render_main_dashboard(mock_data_store, mock_sync_engine)

        # Verify dashboard elements were rendered
        mock_header.assert_called_once()
        mock_caption.assert_called_once()
        mock_subheader.assert_called()
        mock_metric.assert_called()

    def test_error_handling_missing_components(self):
        """Test error handling when components are missing."""
        with patch('streamlit.set_page_config'), \
             patch('src.nba_predictor.streamlit.app._initialize_core_components', side_effect=Exception("Component init failed")):

            with pytest.raises(Exception):
                create_main_app()

    def test_game_distribution_preview(self):
        """Test game distribution preview functionality."""
        from src.nba_predictor.streamlit.app import _get_game_distribution_preview

        # Create mock data store
        mock_data_store = Mock()
        mock_result = Mock()
        mock_data_store.query_analytics.return_value = mock_result

        # Test with valid date range
        date_range = (date.today() - timedelta(days=7), date.today())
        result = _get_game_distribution_preview(mock_data_store, date_range)

        # Verify query_analytics was called
        mock_data_store.query_analytics.assert_called_once()

        # Verify return value
        assert result == mock_result

    def test_top_teams_preview(self):
        """Test top teams preview functionality."""
        from src.nba_predictor.streamlit.app import _get_top_teams_preview

        # Create mock data store
        mock_data_store = Mock()
        mock_result = Mock()
        mock_data_store.query_analytics.return_value = mock_result

        # Test with valid date range
        date_range = (date.today() - timedelta(days=7), date.today())
        result = _get_top_teams_preview(mock_data_store, date_range)

        # Verify query_analytics was called
        mock_data_store.query_analytics.assert_called_once()

        # Verify return value
        assert result == mock_result

    @patch('streamlit.spinner')
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.rerun')
    def test_manual_sync_trigger(
        self,
        mock_rerun,
        mock_error,
        mock_success,
        mock_empty,
        mock_progress,
        mock_spinner
    ):
        """Test manual sync trigger functionality."""
        # Create mock sync engine
        mock_sync_engine = Mock()
        mock_sync_engine.sync_all_data.return_value = {
            'success': True,
            'duration_seconds': 15.5,
            'errors': []
        }

        # Mock asyncio
        with patch('src.nba_predictor.streamlit.app.asyncio') as mock_asyncio:
            mock_loop = Mock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_asyncio.set_event_loop.return_value = None

            # Import and test sync trigger
            from src.nba_predictor.streamlit.app import _trigger_manual_sync

            _trigger_manual_sync(mock_sync_engine)

            # Verify sync was triggered
            mock_sync_engine.sync_all_data.assert_called_once_with(
                force_refresh=True,
                progress_callback=mock_asyncio.AsyncMock()
            )

    @patch('streamlit.subheader')
    @patch('streamlit.write')
    @patch('streamlit.expander')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.selectbox')
    @patch('streamlit.info')
    def test_settings_page_rendering(
        self,
        mock_info,
        mock_selectbox,
        mock_button,
        mock_columns,
        mock_metric,
        mock_expander,
        mock_write,
        mock_subheader
    ):
        """Test settings page rendering."""
        # Mock streamlit functions
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_button.return_value = False  # Buttons not clicked

        # Create mock components
        mock_data_store = Mock()
        mock_sync_engine = Mock()

        # Mock data
        mock_data_store.get_metadata.return_value = Mock()
        mock_data_store.get_metadata.return_value.height = 5
        mock_data_store.get_metadata.return_value.__getitem__ = Mock(return_value=Mock())
        mock_data_store.get_metadata.return_value.__getitem__.return_value.sum.return_value = 1000

        mock_sync_engine.get_sync_statistics.return_value = {
            'configuration': {
                'sync_interval': 3600,
                'retry_attempts': 3,
                'batch_size': 1000
            }
        }

        # Import and test settings function
        from src.nba_predictor.streamlit.app import _render_settings_page

        _render_settings_page(mock_data_store, mock_sync_engine)

        # Verify settings elements were rendered
        mock_subheader.assert_called()
        mock_metric.assert_called()
        mock_expander.assert_called()

    def test_query_construction(self):
        """Test that analytics queries are properly constructed."""
        from src.nba_predictor.streamlit.app import _get_game_distribution_preview

        # Create mock data store
        mock_data_store = Mock()

        # Test with specific date range
        date_range = (date(2024, 1, 1), date(2024, 1, 7))
        _get_game_distribution_preview(mock_data_store, date_range)

        # Verify query_analytics was called
        mock_data_store.query_analytics.assert_called_once()

        # Check query structure
        call_args = mock_data_store.query_analytics.call_args[0][0]
        assert 'SELECT' in call_args
        assert 'FROM read_parquet' in call_args
        assert 'game_date BETWEEN' in call_args
        assert 'GROUP BY DATE(game_date)' in call_args

    def test_component_integration(self):
        """Test integration between different components."""
        # Verify that all required imports are available
        from src.nba_predictor.streamlit.app import (
            create_main_app,
            _configure_page,
            _initialize_core_components,
            _render_main_navigation
        )

        # Verify imports work
        assert callable(create_main_app)
        assert callable(_configure_page)
        assert callable(_initialize_core_components)
        assert callable(_render_main_navigation)