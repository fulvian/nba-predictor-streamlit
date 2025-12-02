"""Unified data store for NBA system with Polars + DuckDB + Parquet.

This module provides a high-performance data store that combines the strengths
of Polars for data manipulation, DuckDB for analytical queries, and Parquet
for efficient storage and retrieval.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from ..utils.exceptions import DatabaseError, FileNotFoundError, ValidationError

logger = logging.getLogger(__name__)


class UnifiedDataStore:
    """Unified data store for NBA system with Polars + DuckDB + Parquet.

    This class provides a unified interface for data storage and retrieval,
    leveraging Polars for high-performance DataFrame operations, DuckDB for
    analytical SQL queries, and Parquet for efficient columnar storage.
    """

    def __init__(
        self,
        base_path: str,
        duckdb_path: Optional[str] = None,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initialize the unified data store.

        Args:
            base_path: Base directory for data storage
            duckdb_path: Path to DuckDB database file
            cache_enabled: Enable caching for performance

        Returns:
            None

        Raises:
            FileNotFoundError: If base_path doesn't exist
            DatabaseError: If DuckDB initialization fails

        Example:
            >>> store = UnifiedDataStore("/data", cache_enabled=True)
            >>> store.initialize()
        """
        self.base_path = Path(base_path)
        self.duckdb_path = duckdb_path or str(self.base_path / "nba_data.duckdb")
        self.cache_enabled = cache_enabled

        # Initialize data directories
        self.games_dir = self.base_path / "games"
        self.players_dir = self.base_path / "players"
        self.odds_dir = self.base_path / "odds"
        self.teams_dir = self.base_path / "teams"

        # Connection objects
        self._duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None
        self._polars_cache: Dict[str, pl.DataFrame] = {}

        logger.info(
            "Initializing UnifiedDataStore",
            extra={
                "base_path": str(self.base_path),
                "duckdb_path": self.duckdb_path,
                "cache_enabled": self.cache_enabled,
            },
        )

    def initialize(self) -> None:
        """
        Initialize the data store and create necessary directories.

        Raises:
            FileNotFoundError: If base_path cannot be created
            DatabaseError: If DuckDB initialization fails
        """
        try:
            # Create base path if it doesn't exist
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            for directory in [
                self.games_dir,
                self.players_dir,
                self.odds_dir,
                self.teams_dir,
            ]:
                directory.mkdir(exist_ok=True)
                logger.debug(f"Created directory: {directory}")

            # Initialize DuckDB connection
            self._init_duckdb()

            # Create metadata table if not exists
            self._create_metadata_table()

            logger.info("UnifiedDataStore initialized successfully")

        except OSError as e:
            logger.error(
                "Failed to create directories",
                extra={"error": str(e), "base_path": str(self.base_path)},
            )
            raise FileNotFoundError(
                f"Failed to create base path: {self.base_path}"
            ) from e
        except Exception as e:
            logger.error(
                "Database initialization failed",
                extra={"error": str(e), "duckdb_path": self.duckdb_path},
            )
            raise DatabaseError(f"DuckDB initialization failed: {e}") from e

    def _init_duckdb(self) -> None:
        """Initialize DuckDB connection with optimal settings."""
        try:
            self._duckdb_conn = duckdb.connect(self.duckdb_path)

            # Configure DuckDB for optimal performance
            settings = {
                "memory_limit": "1GB",
                "threads": "4",
                "enable_progress_bar": "false",
                "preserve_insertion_order": "false",
            }

            for key, value in settings.items():
                self._duckdb_conn.execute(f"SET {key} = '{value}'")

            logger.debug("DuckDB connection initialized with optimized settings")

        except Exception as e:
            logger.error(
                "Failed to initialize DuckDB connection", extra={"error": str(e)}
            )
            raise DatabaseError(f"DuckDB connection failed: {e}") from e

    def _create_metadata_table(self) -> None:
        """Create metadata table for tracking data updates."""
        if self._duckdb_conn is None:
            raise DatabaseError("DuckDB connection not initialized")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS data_metadata (
            table_name VARCHAR PRIMARY KEY,
            last_updated TIMESTAMP,
            record_count INTEGER,
            file_path VARCHAR,
            checksum VARCHAR
        )
        """

        self._duckdb_conn.execute(create_table_sql)
        logger.debug("Data metadata table created or verified")

    def store_games_data(self, games_df: pl.DataFrame, date_str: str) -> str:
        """
        Store NBA games data in Parquet format.

        Args:
            games_df: Polars DataFrame containing games data
            date_str: Date string for partitioning

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If DataFrame schema is invalid
            DatabaseError: If storage operation fails
        """
        if games_df is None or games_df.height == 0:
            raise ValidationError("Games DataFrame is empty or None")

        try:
            # Validate required columns
            required_columns = {
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "season",
            }
            missing_columns = required_columns - set(games_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            # Create file path
            file_path = self.games_dir / f"games_{date_str}.parquet"

            # Store as Parquet with compression
            games_df.write_parquet(file_path, compression="snappy", statistics=True)

            # Update metadata
            self._update_metadata(
                table_name=f"games_{date_str}",
                record_count=games_df.height,
                file_path=str(file_path),
            )

            # Clear cache for this table
            cache_key = f"games_{date_str}"
            if cache_key in self._polars_cache:
                del self._polars_cache[cache_key]

            logger.info(
                "Games data stored successfully",
                extra={
                    "file_path": str(file_path),
                    "record_count": games_df.height,
                    "date": date_str,
                },
            )

            return str(file_path)

        except Exception as e:
            logger.error(
                "Failed to store games data", extra={"error": str(e), "date": date_str}
            )
            raise DatabaseError(f"Failed to store games data: {e}") from e

    def store_players_data(self, players_df: pl.DataFrame, season: str) -> str:
        """
        Store NBA players data in Parquet format.

        Args:
            players_df: Polars DataFrame containing players data
            season: NBA season identifier

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If DataFrame schema is invalid
            DatabaseError: If storage operation fails
        """
        if players_df is None or players_df.height == 0:
            raise ValidationError("Players DataFrame is empty or None")

        try:
            # Validate required columns
            required_columns = {
                "player_id",
                "player_name",
                "team_id",
                "season",
                "position",
            }
            missing_columns = required_columns - set(players_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            # Create file path
            file_path = self.players_dir / f"players_{season}.parquet"

            # Store as Parquet with compression
            players_df.write_parquet(file_path, compression="snappy", statistics=True)

            # Update metadata
            self._update_metadata(
                table_name=f"players_{season}",
                record_count=players_df.height,
                file_path=str(file_path),
            )

            # Clear cache
            cache_key = f"players_{season}"
            if cache_key in self._polars_cache:
                del self._polars_cache[cache_key]

            logger.info(
                "Players data stored successfully",
                extra={
                    "file_path": str(file_path),
                    "record_count": players_df.height,
                    "season": season,
                },
            )

            return str(file_path)

        except Exception as e:
            logger.error(
                "Failed to store players data",
                extra={"error": str(e), "season": season},
            )
            raise DatabaseError(f"Failed to store players data: {e}") from e

    def store_odds_data(self, odds_df: pl.DataFrame, date_str: str) -> str:
        """
        Store betting odds data in Parquet format.

        Args:
            odds_df: Polars DataFrame containing odds data
            date_str: Date string for partitioning

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If DataFrame schema is invalid
            DatabaseError: If storage operation fails
        """
        if odds_df is None or odds_df.height == 0:
            raise ValidationError("Odds DataFrame is empty or None")

        try:
            # Validate required columns
            required_columns = {
                "game_id",
                "bookmaker",
                "home_odds",
                "away_odds",
                "updated_time",
            }
            missing_columns = required_columns - set(odds_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            # Create file path
            file_path = self.odds_dir / f"odds_{date_str}.parquet"

            # Store as Parquet with compression
            odds_df.write_parquet(file_path, compression="snappy", statistics=True)

            # Update metadata
            self._update_metadata(
                table_name=f"odds_{date_str}",
                record_count=odds_df.height,
                file_path=str(file_path),
            )

            # Clear cache
            cache_key = f"odds_{date_str}"
            if cache_key in self._polars_cache:
                del self._polars_cache[cache_key]

            logger.info(
                "Odds data stored successfully",
                extra={
                    "file_path": str(file_path),
                    "record_count": odds_df.height,
                    "date": date_str,
                },
            )

            return str(file_path)

        except Exception as e:
            logger.error(
                "Failed to store odds data", extra={"error": str(e), "date": date_str}
            )
            raise DatabaseError(f"Failed to store odds data: {e}") from e

    def get_games_data(
        self, date_range: Optional[tuple[str, str]] = None
    ) -> pl.DataFrame:
        """
        Retrieve NBA games data, optionally filtered by date range.

        Args:
            date_range: Optional tuple of (start_date, end_date)

        Returns:
            Polars DataFrame containing games data

        Raises:
            DatabaseError: If query operation fails
        """
        try:
            if date_range:
                start_date, end_date = date_range
                # Use DuckDB for efficient filtering with schema evolution handling
                # Cast season to VARCHAR to handle mixed types (Int64 vs String)
                query = f"""
                SELECT * REPLACE (CAST(season AS VARCHAR) AS season) 
                FROM read_parquet('{self.games_dir}/*.parquet', union_by_name=true)
                WHERE game_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY game_date
                """

                if self._duckdb_conn is None:
                    raise DatabaseError("DuckDB connection not initialized")

                result = self._duckdb_conn.execute(query).fetchall()

                # Convert to Polars DataFrame
                if result:
                    columns = [desc[0] for desc in self._duckdb_conn.description]
                    df = pl.DataFrame(result, schema=columns, orient="row")
                else:
                    df = pl.DataFrame()
            else:
                # Load all games data using DuckDB to handle schema evolution and type casting
                query = f"""
                SELECT * REPLACE (CAST(season AS VARCHAR) AS season) 
                FROM read_parquet('{self.games_dir}/*.parquet', union_by_name=true)
                ORDER BY game_date
                """

                if self._duckdb_conn is None:
                    raise DatabaseError("DuckDB connection not initialized")

                result = self._duckdb_conn.execute(query).fetchall()

                if result:
                    columns = [desc[0] for desc in self._duckdb_conn.description]
                    df = pl.DataFrame(result, schema=columns, orient="row")
                else:
                    df = pl.DataFrame()

            logger.info(
                "Games data retrieved successfully",
                extra={"record_count": df.height, "date_range": date_range},
            )

            return df

        except Exception as e:
            logger.error(
                "Failed to retrieve games data",
                extra={"error": str(e), "date_range": date_range},
            )
            raise DatabaseError(f"Failed to retrieve games data: {e}") from e

    def query_analytics(self, sql_query: str) -> pl.DataFrame:
        """
        Execute analytical SQL query using DuckDB.

        Args:
            sql_query: SQL query string

        Returns:
            Polars DataFrame containing query results

        Raises:
            DatabaseError: If query execution fails
        """
        if self._duckdb_conn is None:
            raise DatabaseError("DuckDB connection not initialized")

        try:
            result = self._duckdb_conn.execute(sql_query).fetchall()

            if result:
                columns = [desc[0] for desc in self._duckdb_conn.description]
                df = pl.DataFrame(result, schema=columns, orient="row")
            else:
                df = pl.DataFrame()

            logger.debug(
                "Analytics query executed successfully",
                extra={"record_count": df.height, "query": sql_query[:100]},
            )

            return df

        except Exception as e:
            logger.error(
                "Analytics query failed",
                extra={"error": str(e), "query": sql_query[:100]},
            )
            raise DatabaseError(f"Query execution failed: {e}") from e

    def _update_metadata(
        self, table_name: str, record_count: int, file_path: str
    ) -> None:
        """Update metadata table with latest information."""
        if self._duckdb_conn is None:
            raise DatabaseError("DuckDB connection not initialized")

        try:
            # Calculate simple checksum (in production, use proper hash)
            checksum = str(record_count) + str(datetime.now().timestamp())

            upsert_sql = """
            INSERT OR REPLACE INTO data_metadata
            (table_name, last_updated, record_count, file_path, checksum)
            VALUES (?, ?, ?, ?, ?)
            """

            self._duckdb_conn.execute(
                upsert_sql,
                [table_name, datetime.now(), record_count, file_path, checksum],
            )

        except Exception as e:
            logger.warning(
                "Failed to update metadata",
                extra={"error": str(e), "table_name": table_name},
            )

    def store_player_stats(self, player_stats_df: pl.DataFrame, date_str: str) -> str:
        """
        Store NBA player statistics in Parquet format.

        Args:
            player_stats_df: Polars DataFrame containing player statistics
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If DataFrame schema is invalid
            DatabaseError: If storage operation fails
        """
        if player_stats_df is None or player_stats_df.height == 0:
            raise ValidationError("Player stats DataFrame is empty or None")

        try:
            # Validate required columns
            required_columns = {"player_id", "player_name", "team_id"}
            missing_columns = required_columns - set(player_stats_df.columns)

            if missing_columns:
                logger.warning(
                    f"Missing optional columns in player stats: {missing_columns}"
                )

            # Create file path
            file_path = self.players_dir / f"player_stats_{date_str}.parquet"

            # Ensure directory exists
            self.players_dir.mkdir(parents=True, exist_ok=True)

            # Store as Parquet
            player_stats_df.write_parquet(file_path, compression="snappy")

            # Update metadata
            self._update_metadata("player_stats", date_str, len(player_stats_df))

            logger.info(
                f"Stored player stats for {len(player_stats_df)} players to {file_path}"
            )
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store player stats: {e}")
            raise DatabaseError(f"Failed to store player stats: {e}") from e

    def store_team_stats(self, team_stats_df: pl.DataFrame, date_str: str) -> str:
        """
        Store NBA team statistics in Parquet format.

        Args:
            team_stats_df: Polars DataFrame containing team statistics
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If DataFrame schema is invalid
            DatabaseError: If storage operation fails
        """
        if team_stats_df is None or team_stats_df.height == 0:
            raise ValidationError("Team stats DataFrame is empty or None")

        try:
            # Validate required columns
            required_columns = {"team_id", "team_name"}
            missing_columns = required_columns - set(team_stats_df.columns)

            if missing_columns:
                logger.warning(
                    f"Missing optional columns in team stats: {missing_columns}"
                )

            # Create file path
            file_path = self.teams_dir / f"team_stats_{date_str}.parquet"

            # Ensure directory exists
            self.teams_dir.mkdir(parents=True, exist_ok=True)

            # Store as Parquet
            team_stats_df.write_parquet(file_path, compression="snappy")

            # Update metadata
            self._update_metadata("team_stats", date_str, len(team_stats_df))

            logger.info(
                f"Stored team stats for {len(team_stats_df)} teams to {file_path}"
            )
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store team stats: {e}")
            raise DatabaseError(f"Failed to store team stats: {e}") from e

    def get_player_stats(
        self, date_range: Optional[tuple[str, str]] = None
    ) -> pl.DataFrame:
        """
        Retrieve player statistics from persistent storage.

        Args:
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format

        Returns:
            Polars DataFrame containing player statistics
        """
        try:
            if date_range:
                start_date, end_date = date_range
                pattern = f"player_stats_*.parquet"

                # Get all player stats files
                files = list(self.players_dir.glob(pattern))

                # Filter by date range
                valid_files = []
                for file_path in files:
                    file_date = file_path.stem.replace("player_stats_", "")
                    try:
                        file_date_dt = datetime.strptime(file_date, "%Y-%m-%d").date()
                        if start_date <= file_date_dt <= end_date:
                            valid_files.append(file_path)
                    except ValueError:
                        continue

                if not valid_files:
                    return pl.DataFrame()

                # Read and combine all valid files
                dfs = []
                for file_path in valid_files:
                    try:
                        df = pl.read_parquet(file_path)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")

                if dfs:
                    return pl.concat(dfs)

            # If no date range, get most recent
            most_recent_file = max(
                self.players_dir.glob("player_stats_*.parquet"),
                key=lambda x: x.stat().st_mtime,
                default=None,
            )
            if most_recent_file:
                return pl.read_parquet(most_recent_file)

            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Failed to retrieve player stats: {e}")
            return pl.DataFrame()

    def get_team_stats(
        self, date_range: Optional[tuple[str, str]] = None
    ) -> pl.DataFrame:
        """
        Retrieve team statistics from persistent storage.

        Args:
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format

        Returns:
            Polars DataFrame containing team statistics
        """
        try:
            if date_range:
                start_date, end_date = date_range
                pattern = f"team_stats_*.parquet"

                # Get all team stats files
                files = list(self.teams_dir.glob(pattern))

                # Filter by date range
                valid_files = []
                for file_path in files:
                    file_date = file_path.stem.replace("team_stats_", "")
                    try:
                        file_date_dt = datetime.strptime(file_date, "%Y-%m-%d").date()
                        if start_date <= file_date_dt <= end_date:
                            valid_files.append(file_path)
                    except ValueError:
                        continue

                if not valid_files:
                    return pl.DataFrame()

                # Read and combine all valid files
                dfs = []
                for file_path in valid_files:
                    try:
                        df = pl.read_parquet(file_path)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")

                if dfs:
                    return pl.concat(dfs)

            # If no date range, get most recent
            most_recent_file = max(
                self.teams_dir.glob("team_stats_*.parquet"),
                key=lambda x: x.stat().st_mtime,
                default=None,
            )
            if most_recent_file:
                return pl.read_parquet(most_recent_file)

            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Failed to retrieve team stats: {e}")
            return pl.DataFrame()

    def get_metadata(self) -> pl.DataFrame:
        """
        Retrieve metadata about stored data.

        Returns:
            Polars DataFrame containing metadata
        """
        if self._duckdb_conn is None:
            raise DatabaseError("DuckDB connection not initialized")

        try:
            result = self._duckdb_conn.execute(
                "SELECT * FROM data_metadata ORDER BY last_updated DESC"
            ).fetchall()

            if result:
                columns = [desc[0] for desc in self._duckdb_conn.description]
                df = pl.DataFrame(result, schema=columns, orient="row")
            else:
                df = pl.DataFrame()

            return df

        except Exception as e:
            logger.error("Failed to retrieve metadata", extra={"error": str(e)})
            return pl.DataFrame()

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._duckdb_conn:
            self._duckdb_conn.close()
            self._duckdb_conn = None

        # Clear cache
        self._polars_cache.clear()

        logger.info("UnifiedDataStore connections closed")

    def __enter__(self) -> "UnifiedDataStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
