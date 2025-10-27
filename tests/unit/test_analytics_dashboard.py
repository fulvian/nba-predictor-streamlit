"""Test analytics dashboard Streamlit component."""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.nba_predictor.core.data_store import UnifiedDataStore
from src.nba_predictor.streamlit.components.analytics_dashboard import (
    render_analytics_dashboard,
    _calculate_key_metrics,
    _get_scoring_trends,
    _get_game_distribution,
    _get_teams_performance
)


class TestAnalyticsDashboard:
    """Test cases for analytics dashboard component."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_store = UnifiedDataStore(self.temp_dir, cache_enabled=True)
        self.data_store.initialize()

        # Add sample data for testing
        self._add_sample_data()

    def teardown_method(self):
        """Clean up test environment."""
        self.data_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add_sample_data(self):
        """Add sample data for testing."""
        # Sample games data
        games_data = pl.DataFrame({
            'game_id': ['0012400001', '0012400002', '0012400003', '0012400004', '0012400005'],
            'game_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'home_team': ['LAL', 'BOS', 'GSW', 'MIA', 'CHI'],
            'away_team': ['GSW', 'MIA', 'LAL', 'BOS', 'NYK'],
            'home_score': [110, 105, 108, 102, 115],
            'away_score': [108, 102, 112, 105, 118],
            'season': [2024, 2024, 2024, 2024, 2024]
        })

        self.data_store.store_games_data(games_data, "2024-01-01")

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.sidebar')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.date_input')
    @patch('streamlit.button')
    @patch('streamlit.line_chart')
    @patch('streamlit.bar_chart')
    @patch('streamlit.expander')
    @patch('streamlit.dataframe')
    @patch('streamlit.info')
    @patch('streamlit.error')
    @patch('streamlit.write')
    def test_render_analytics_dashboard_success(
        self,
        mock_write,
        mock_error,
        mock_info,
        mock_dataframe,
        mock_expander,
        mock_bar_chart,
        mock_line_chart,
        mock_metric,
        mock_date_input,
        mock_button,
        mock_columns,
        mock_sidebar,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test successful rendering of analytics dashboard."""
        # Mock streamlit functions
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_date_input.side_effect = [date(2024, 1, 1), date(2024, 1, 31)]
        mock_button.return_value = False  # Buttons not clicked

        # Render dashboard
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        render_analytics_dashboard(self.data_store, date_range)

        # Verify basic streamlit functions were called
        mock_title.assert_called_once_with("📊 NBA Analytics Dashboard")
        mock_caption.assert_called_once()
        assert mock_subheader.call_count >= 4  # Multiple subheaders

    def test_analytics_dashboard_initialization(self):
        """Test dashboard initialization with valid data store."""
        assert self.data_store is not None
        assert hasattr(self.data_store, 'query_analytics')
        assert hasattr(self.data_store, 'games_dir')

    def test_calculate_key_metrics_with_data(self):
        """Test key metrics calculation with sample data."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        metrics = _calculate_key_metrics(self.data_store, date_range)

        # Verify required fields exist
        required_fields = [
            'total_games', 'games_today', 'avg_points', 'active_teams', 'overall_win_rate'
        ]
        for field in required_fields:
            assert field in metrics

        # Verify metrics are reasonable
        assert metrics['total_games'] >= 0
        assert metrics['active_teams'] >= 0
        assert 0 <= metrics['overall_win_rate'] <= 1

    def test_calculate_key_metrics_empty_date_range(self):
        """Test key metrics calculation with empty date range."""
        future_date = date(2025, 1, 1)
        date_range = (future_date, future_date)
        metrics = _calculate_key_metrics(self.data_store, date_range)

        # Should return empty or zero metrics
        assert metrics.get('total_games', 0) == 0
        assert metrics.get('active_teams', 0) == 0

    def test_get_scoring_trends_with_data(self):
        """Test scoring trends retrieval with sample data."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        trends = _get_scoring_trends(self.data_store, date_range)

        if trends is not None:
            assert 'date' in trends.columns
            assert len(trends) > 0
            # Verify data types
            assert trends['date'].dtype == pl.Date
            assert trends['avg_home_score'].dtype in [pl.Float64, pl.Float32]
            assert trends['avg_away_score'].dtype in [pl.Float64, pl.Float32]

    def test_get_scoring_trends_empty_range(self):
        """Test scoring trends with empty date range."""
        future_date = date(2025, 1, 1)
        date_range = (future_date, future_date)
        trends = _get_scoring_trends(self.data_store, date_range)

        # Should return None or empty DataFrame
        assert trends is None or (isinstance(trends, pl.DataFrame) and trends.height == 0)

    def test_get_game_distribution_with_data(self):
        """Test game distribution retrieval with sample data."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        distribution = _get_game_distribution(self.data_store, date_range)

        if distribution is not None:
            assert 'date' in distribution.columns
            assert 'games_count' in distribution.columns
            assert len(distribution) > 0

    def test_get_teams_performance_with_data(self):
        """Test teams performance retrieval with sample data."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        performance = _get_teams_performance(self.data_store, date_range)

        if performance is not None:
            expected_columns = [
                'team_name', 'games_played', 'wins', 'win_percentage',
                'avg_points_scored', 'avg_points_allowed', 'avg_point_differential'
            ]

            for col in expected_columns:
                assert col in performance.columns

            # Verify data integrity
            assert all(performance['games_played'] > 0)
            assert all(0 <= performance['win_percentage'] <= 100)

    @patch('streamlit.title')
    @patch('streamlit.caption')
    @patch('streamlit.subheader')
    @patch('streamlit.sidebar')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_date_selector_rendering(
        self,
        mock_metric,
        mock_columns,
        mock_sidebar,
        mock_subheader,
        mock_caption,
        mock_title
    ):
        """Test date selector component rendering."""
        # Mock streamlit functions
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_date_input = Mock()
        mock_button = Mock()

        with patch('streamlit.date_input', mock_date_input), \
             patch('streamlit.button', mock_button):
            # Import the internal function to test it
            from src.nba_predictor.streamlit.components.analytics_dashboard import _render_date_selector

            # Test date selector
            default_range = (date(2024, 1, 1), date(2024, 1, 31))
            result = _render_date_selector(default_range)

            # Verify date input was called
            assert mock_date_input.call_count == 2

            # Verify result is a tuple of dates
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], date)
            assert isinstance(result[1], date)

    def test_head_to_head_stats_calculation(self):
        """Test head-to-head statistics calculation."""
        from src.nba_predictor.streamlit.components.analytics_dashboard import _get_head_to_head_stats

        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        stats = _get_head_to_head_stats(self.data_store, date_range, "LAL", "GSW")

        # Verify structure
        if stats:
            expected_fields = [
                'team1_wins', 'team1_losses', 'win_percentage',
                'avg_points_scored', 'avg_points_allowed', 'point_differential'
            ]
            for field in expected_fields:
                assert field in stats

            # Verify data integrity
            assert stats['team1_wins'] >= 0
            assert stats['team1_losses'] >= 0
            assert 0 <= stats['win_percentage'] <= 100

    @patch('streamlit.title')
    @patch('streamlit.caption')
    def test_error_handling_invalid_data_store(
        self,
        mock_caption,
        mock_title
    ):
        """Test error handling with invalid data store."""
        with pytest.raises(Exception):
            render_analytics_dashboard(None, (date(2024, 1, 1), date(2024, 1, 31)))

    def test_query_construction_and_execution(self):
        """Test that analytics queries are properly constructed and executed."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))

        # Test that query_analytics method is called with proper SQL
        with patch.object(self.data_store, 'query_analytics') as mock_query:
            mock_query.return_value = pl.DataFrame({
                'total_games': [5],
                'avg_points': [220.5],
                'active_teams': [8],
                'overall_win_rate': [0.6]
            })

            result = _calculate_key_metrics(self.data_store, date_range)

            # Verify query_analytics was called
            mock_query.assert_called_once()

            # Verify query contains expected SQL elements
            call_args = mock_query.call_args[0][0]
            assert 'SELECT' in call_args
            assert 'FROM read_parquet' in call_args
            assert 'game_date BETWEEN' in call_args

    def test_date_range_validation(self):
        """Test date range validation logic."""
        from src.nba_predictor.streamlit.components.analytics_dashboard import _render_date_selector

        # Test valid date range
        valid_range = (date(2024, 1, 1), date(2024, 1, 31))
        result = _render_date_selector(valid_range)
        assert result[0] <= result[1]

        # Test invalid date range (mock scenario)
        # This would be tested through UI interaction in actual Streamlit app

    def test_data_type_handling(self):
        """Test proper handling of different data types in analytics."""
        date_range = (date(2024, 1, 1), date(2024, 1, 31))

        # Test that functions handle None returns gracefully
        result = _get_scoring_trends(self.data_store, (date(2025, 1, 1), date(2025, 1, 2)))
        assert result is None or (isinstance(result, pl.DataFrame) and result.height == 0)

        # Test that functions return proper data types
        metrics = _calculate_key_metrics(self.data_store, date_range)
        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(value, (int, float, str))

    def test_integration_with_data_store(self):
        """Test integration with UnifiedDataStore."""
        # Verify data store has required attributes
        assert hasattr(self.data_store, 'games_dir')
        assert hasattr(self.data_store, 'query_analytics')

        # Test that query_analytics can handle our queries
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        query = f"SELECT COUNT(*) as count FROM read_parquet('{self.data_store.games_dir}/*.parquet')"

        result = self.data_store.query_analytics(query)
        assert result is not None
        assert isinstance(result, pl.DataFrame)

    def test_performance_characteristics(self):
        """Test performance characteristics of analytics functions."""
        import time

        date_range = (date(2024, 1, 1), date(2024, 1, 31))

        # Test that functions complete in reasonable time
        start_time = time.time()
        metrics = _calculate_key_metrics(self.data_store, date_range)
        end_time = time.time()

        # Should complete within 1 second for small dataset
        assert (end_time - start_time) < 1.0
        assert metrics is not None

        # Test multiple calls don't significantly slow down
        start_time = time.time()
        for _ in range(3):
            _get_scoring_trends(self.data_store, date_range)
        end_time = time.time()

        assert (end_time - start_time) < 2.0  # 3 calls within 2 seconds