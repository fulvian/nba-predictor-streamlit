"""
NBA Totals Odds Schema - Data model for Over/Under betting odds.

This module defines the database schema and data structures for storing
NBA Over/Under (totals) betting odds from various sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class SeasonStage(str, Enum):
    """NBA season stage classification."""

    REGULAR_SEASON = "RS"
    PLAYOFFS = "PO"
    FINALS = "FINALS"
    PRESEASON = "PRE"


class OddsSource(str, Enum):
    """Source of odds data."""

    KAGGLE = "kaggle"
    ODDS_HARVESTER = "oddsharvester"
    ODDS_API = "oddsapi"
    MANUAL = "manual"


@dataclass
class TotalsOddsRecord:
    """Single record of Over/Under odds for a game."""

    game_id: str
    game_date: str  # YYYY-MM-DD
    season: str  # e.g., "2023-2024"
    stage: SeasonStage
    home_team_id: int
    away_team_id: int
    bookmaker: str
    total_points_line: float  # e.g., 224.5
    odds_over_decimal: float  # e.g., 1.91
    odds_under_decimal: float
    scrape_datetime: datetime
    source: OddsSource
    is_closing: bool = False
    id: Optional[int] = None


# SQL Schema for DuckDB
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nba_totals_odds (
    id INTEGER PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    game_date DATE NOT NULL,
    season VARCHAR(9) NOT NULL,
    stage VARCHAR(10) NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    total_points_line DECIMAL(5,1) NOT NULL,
    odds_over_decimal DECIMAL(6,4) NOT NULL,
    odds_under_decimal DECIMAL(6,4) NOT NULL,
    scrape_datetime TIMESTAMP NOT NULL,
    source VARCHAR(20) NOT NULL,
    is_closing BOOLEAN DEFAULT FALSE,
    
    UNIQUE(game_id, bookmaker, total_points_line, scrape_datetime)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_totals_game ON nba_totals_odds(game_id);
CREATE INDEX IF NOT EXISTS idx_totals_season ON nba_totals_odds(season, stage);
CREATE INDEX IF NOT EXISTS idx_totals_closing ON nba_totals_odds(game_id, bookmaker, is_closing);
CREATE INDEX IF NOT EXISTS idx_totals_date ON nba_totals_odds(game_date);
"""


class NbaTotalsOddsRepository:
    """Repository for managing NBA totals odds data in DuckDB."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize repository with database path.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def initialize_schema(self) -> None:
        """Create table and indexes if they don't exist."""
        conn = self._get_connection()
        try:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEXES_SQL)
            logger.info(f"Initialized nba_totals_odds schema in {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def insert_odds(self, df: pl.DataFrame) -> int:
        """
        Insert odds records from Polars DataFrame.

        Args:
            df: DataFrame with odds data matching schema

        Returns:
            Number of records inserted
        """
        conn = self._get_connection()

        # Convert to pandas for DuckDB insertion
        pdf = df.to_pandas()

        # Generate IDs manually to satisfy NOT NULL constraint
        try:
            max_id_res = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM nba_totals_odds"
            ).fetchone()
            current_max_id = max_id_res[0] if max_id_res else 0

            # Assign new IDs
            # Note: This assigns IDs even to rows that might be ignored.
            # This is acceptable for simple sequential ID generation in single-threaded context.
            pdf["id"] = range(current_max_id + 1, current_max_id + 1 + len(pdf))
        except Exception as e:
            logger.warning(f"Could not generate IDs: {e}")

        # Use INSERT OR IGNORE for upsert behavior
        try:
            # Register the dataframe as a temporary table
            conn.register("temp_odds", pdf)

            insert_sql = """
            INSERT OR IGNORE INTO nba_totals_odds 
            (id, game_id, game_date, season, stage, home_team_id, away_team_id,
             bookmaker, total_points_line, odds_over_decimal, odds_under_decimal,
             scrape_datetime, source, is_closing)
            SELECT id, game_id, game_date, season, stage, home_team_id, away_team_id,
                   bookmaker, total_points_line, odds_over_decimal, odds_under_decimal,
                   scrape_datetime, source, is_closing
            FROM temp_odds
            """

            result = conn.execute(insert_sql)
            inserted = result.fetchone()[0] if result else len(pdf)

            conn.unregister("temp_odds")

            logger.info(f"Inserted {inserted} odds records")
            return inserted

        except Exception as e:
            logger.error(f"Failed to insert odds: {e}")
            raise

    def get_odds_by_game(
        self, game_id: str, bookmaker: Optional[str] = None, closing_only: bool = False
    ) -> pl.DataFrame:
        """
        Get odds for a specific game.

        Args:
            game_id: NBA game ID
            bookmaker: Optional bookmaker filter
            closing_only: If True, return only closing odds

        Returns:
            DataFrame with odds records
        """
        conn = self._get_connection()

        query = "SELECT * FROM nba_totals_odds WHERE game_id = ?"
        params = [game_id]

        if bookmaker:
            query += " AND bookmaker = ?"
            params.append(bookmaker)

        if closing_only:
            query += " AND is_closing = TRUE"

        query += " ORDER BY scrape_datetime DESC"

        result = conn.execute(query, params).pl()
        return result

    def get_odds_by_season(
        self, season: str, bookmaker: Optional[str] = None, closing_only: bool = True
    ) -> pl.DataFrame:
        """
        Get all odds for a season.

        Args:
            season: Season string (e.g., "2023-2024")
            bookmaker: Optional bookmaker filter
            closing_only: If True, return only closing odds

        Returns:
            DataFrame with odds records
        """
        conn = self._get_connection()

        query = "SELECT * FROM nba_totals_odds WHERE season = ?"
        params = [season]

        if bookmaker:
            query += " AND bookmaker = ?"
            params.append(bookmaker)

        if closing_only:
            query += " AND is_closing = TRUE"

        query += " ORDER BY game_date, game_id"

        result = conn.execute(query, params).pl()
        return result

    def count_records(self, by_source: bool = False) -> dict:
        """
        Count total records, optionally by source.

        Args:
            by_source: If True, return counts grouped by source

        Returns:
            Dictionary with counts
        """
        conn = self._get_connection()

        if by_source:
            result = conn.execute("""
                SELECT source, COUNT(*) as count 
                FROM nba_totals_odds 
                GROUP BY source
            """).fetchall()
            return {row[0]: row[1] for row in result}
        else:
            result = conn.execute("SELECT COUNT(*) FROM nba_totals_odds").fetchone()
            return {"total": result[0] if result else 0}

    def get_available_bookmakers(self) -> list[str]:
        """Get list of unique bookmakers in database."""
        conn = self._get_connection()
        result = conn.execute(
            "SELECT DISTINCT bookmaker FROM nba_totals_odds ORDER BY bookmaker"
        ).fetchall()
        return [row[0] for row in result]

    def get_available_seasons(self) -> list[str]:
        """Get list of unique seasons in database."""
        conn = self._get_connection()
        result = conn.execute(
            "SELECT DISTINCT season FROM nba_totals_odds ORDER BY season"
        ).fetchall()
        return [row[0] for row in result]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
