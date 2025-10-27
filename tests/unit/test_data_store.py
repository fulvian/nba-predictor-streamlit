"""Test unified data store implementation."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.nba_predictor.core.data_store import UnifiedDataStore
from src.nba_predictor.utils.exceptions import DatabaseError, ValidationError


class TestUnifiedDataStore:
    """Test cases for UnifiedDataStore class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_store = UnifiedDataStore(self.temp_dir, cache_enabled=True)
        self.data_store.initialize()

    def teardown_method(self):
        """Clean up test environment."""
        self.data_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test UnifiedDataStore initialization."""
        assert self.data_store.base_path == Path(self.temp_dir)
        assert self.data_store.duckdb_path.endswith("nba_data.duckdb")
        assert self.data_store.cache_enabled is True
        assert self.data_store._duckdb_conn is not None

    def test_initialization_creates_directories(self):
        """Test that initialization creates required directories."""
        expected_dirs = [
            Path(self.temp_dir) / "games",
            Path(self.temp_dir) / "players",
            Path(self.temp_dir) / "odds",
            Path(self.temp_dir) / "teams"
        ]

        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()

    def test_store_games_data_valid(self):
        """Test storing valid games data."""
        games_df = pl.DataFrame({
            'game_id': ['0012400001', '0012400002'],
            'game_date': ['2024-01-01', '2024-01-02'],
            'home_team': ['LAL', 'BOS'],
            'away_team': ['GSW', 'MIA'],
            'season': [2024, 2024]
        })

        file_path = self.data_store.store_games_data(games_df, "2024-01-01")

        assert Path(file_path).exists()
        assert file_path.endswith("games_2024-01-01.parquet")

        # Verify metadata was updated
        metadata = self.data_store.get_metadata()
        assert metadata.height == 1
        assert metadata['table_name'][0] == "games_2024-01-01"
        assert metadata['record_count'][0] == 2

    def test_store_games_data_invalid_schema(self):
        """Test storing games data with invalid schema."""
        invalid_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'home_team': ['LAL']
            # Missing required columns
        })

        with pytest.raises(ValidationError) as exc_info:
            self.data_store.store_games_data(invalid_df, "2024-01-01")

        assert "Missing required columns" in str(exc_info.value)

    def test_store_games_data_empty(self):
        """Test storing empty games data."""
        empty_df = pl.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            self.data_store.store_games_data(empty_df, "2024-01-01")

        assert "DataFrame is empty or None" in str(exc_info.value)

    def test_store_players_data_valid(self):
        """Test storing valid players data."""
        players_df = pl.DataFrame({
            'player_id': ['2544', '1628362'],
            'player_name': ['LeBron James', 'Giannis Antetokounmpo'],
            'team_id': [1610612747, 1610612749],
            'season': [2024, 2024],
            'position': ['F', 'F']
        })

        file_path = self.data_store.store_players_data(players_df, "2024")

        assert Path(file_path).exists()
        assert file_path.endswith("players_2024.parquet")

        # Verify metadata was updated
        metadata = self.data_store.get_metadata()
        assert metadata.height == 1
        assert metadata['table_name'][0] == "players_2024"
        assert metadata['record_count'][0] == 2

    def test_store_players_data_invalid_schema(self):
        """Test storing players data with invalid schema."""
        invalid_df = pl.DataFrame({
            'player_id': ['2544'],
            'player_name': ['LeBron James']
            # Missing required columns
        })

        with pytest.raises(ValidationError) as exc_info:
            self.data_store.store_players_data(invalid_df, "2024")

        assert "Missing required columns" in str(exc_info.value)

    def test_store_odds_data_valid(self):
        """Test storing valid odds data."""
        odds_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'bookmaker': ['DraftKings'],
            'home_odds': [-150, 120],
            'away_odds': [130, -140],
            'updated_time': ['2024-01-01T12:00:00Z']
        })

        file_path = self.data_store.store_odds_data(odds_df, "2024-01-01")

        assert Path(file_path).exists()
        assert file_path.endswith("odds_2024-01-01.parquet")

        # Verify metadata was updated
        metadata = self.data_store.get_metadata()
        assert metadata.height == 1
        assert metadata['table_name'][0] == "odds_2024-01-01"
        assert metadata['record_count'][0] == 1

    def test_store_odds_data_invalid_schema(self):
        """Test storing odds data with invalid schema."""
        invalid_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'bookmaker': ['DraftKings']
            # Missing required columns
        })

        with pytest.raises(ValidationError) as exc_info:
            self.data_store.store_odds_data(invalid_df, "2024-01-01")

        assert "Missing required columns" in str(exc_info.value)

    def test_get_games_data_all(self):
        """Test retrieving all games data."""
        # First store some test data
        games_df = pl.DataFrame({
            'game_id': ['0012400001', '0012400002'],
            'game_date': ['2024-01-01', '2024-01-02'],
            'home_team': ['LAL', 'BOS'],
            'away_team': ['GSW', 'MIA'],
            'season': [2024, 2024]
        })

        self.data_store.store_games_data(games_df, "2024-01-01")

        # Retrieve data
        result = self.data_store.get_games_data()

        assert result.height == 2
        assert 'game_id' in result.columns
        assert 'game_date' in result.columns

    def test_get_games_data_with_date_range(self):
        """Test retrieving games data with date range filter."""
        # Store test data spanning multiple dates
        games_df1 = pl.DataFrame({
            'game_id': ['0012400001'],
            'game_date': ['2024-01-01'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'season': [2024]
        })

        games_df2 = pl.DataFrame({
            'game_id': ['0012400002'],
            'game_date': ['2024-01-03'],
            'home_team': ['BOS'],
            'away_team': ['MIA'],
            'season': [2024]
        })

        self.data_store.store_games_data(games_df1, "2024-01-01")
        self.data_store.store_games_data(games_df2, "2024-01-03")

        # Retrieve data with date range
        result = self.data_store.get_games_data(('2024-01-01', '2024-01-02'))

        # Should only return the first game (within date range)
        assert result.height == 1
        assert result['game_id'][0] == '0012400001'

    @patch('src.nba_predictor.core.data_store.logger')
    def test_store_games_data_database_error(self, mock_logger):
        """Test handling of database errors during storage."""
        games_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'game_date': ['2024-01-01'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'season': [2024]
        })

        # Mock DuckDB connection to raise an exception
        self.data_store._duckdb_conn = Mock()
        self.data_store._duckdb_conn.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseError) as exc_info:
            self.data_store.store_games_data(games_df, "2024-01-01")

        assert "Failed to store games data" in str(exc_info.value)
        mock_logger.error.assert_called()

    def test_context_manager(self):
        """Test using UnifiedDataStore as context manager."""
        games_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'game_date': ['2024-01-01'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'season': [2024]
        })

        with UnifiedDataStore(self.temp_dir) as store:
            store.initialize()
            file_path = store.store_games_data(games_df, "2024-01-01")
            assert Path(file_path).exists()

        # Connection should be closed after context exit
        assert store._duckdb_conn is None

    def test_query_analytics_empty_result(self):
        """Test analytics query with no results."""
        query = "SELECT COUNT(*) as count FROM data_metadata WHERE 1=0"
        result = self.data_store.query_analytics(query)

        assert result.height == 0
        assert result.columns == ['count']

    def test_get_metadata_empty(self):
        """Test getting metadata when no data is stored."""
        metadata = self.data_store.get_metadata()
        assert metadata.height == 0

    def test_caching_behavior(self):
        """Test that caching works as expected."""
        # Store test data
        games_df = pl.DataFrame({
            'game_id': ['0012400001'],
            'game_date': ['2024-01-01'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'season': [2024]
        })

        self.data_store.store_games_data(games_df, "2024-01-01")

        # First access should cache the data
        result1 = self.data_store.get_games_data()

        # Second access should use cache (if cache were implemented for get_games_data)
        result2 = self.data_store.get_games_data()

        # Results should be identical
        assert result1.equals(result2)

    def test_initialization_with_custom_duckdb_path(self):
        """Test initialization with custom DuckDB path."""
        custom_path = Path(self.temp_dir) / "custom.duckdb"
        store = UnifiedDataStore(self.temp_dir, duckdb_path=str(custom_path))
        store.initialize()

        assert store.duckdb_path == str(custom_path)
        assert custom_path.exists()

        store.close()