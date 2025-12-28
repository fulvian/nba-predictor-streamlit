"""
NBA Spreads Odds Schema - Data model for Handicap betting odds.

This module defines the database schema and data structures for storing
NBA Spread (Handicap) betting odds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class SeasonStage(str, Enum):
    REGULAR_SEASON = "RS"
    PLAYOFFS = "PO"
    FINALS = "FINALS"
    PRESEASON = "PRE"


class OddsSource(str, Enum):
    KAGGLE = "kaggle"
    ODDS_HARVESTER = "oddsharvester"
    MANUAL = "manual"


@dataclass
class SpreadsOddsRecord:
    game_id: str
    game_date: str
    season: str
    stage: SeasonStage
    home_team_id: int
    away_team_id: int
    bookmaker: str
    handicap_home: float  # Negative = Home Favored (e.g. -5.5)
    odds_home_decimal: float
    odds_away_decimal: float
    scrape_datetime: datetime
    source: OddsSource
    is_closing: bool = False
    id: Optional[int] = None


# SQL Schema
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nba_spreads_odds (
    id INTEGER PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    game_date DATE NOT NULL,
    season VARCHAR(9) NOT NULL,
    stage VARCHAR(10) NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    handicap_home DECIMAL(5,1) NOT NULL, -- The spread relative to home team
    odds_home_decimal DECIMAL(6,4) NOT NULL,
    odds_away_decimal DECIMAL(6,4) NOT NULL,
    scrape_datetime TIMESTAMP NOT NULL,
    source VARCHAR(20) NOT NULL,
    is_closing BOOLEAN DEFAULT FALSE,
    
    UNIQUE(game_id, bookmaker, handicap_home, scrape_datetime)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_spreads_game ON nba_spreads_odds(game_id);
CREATE INDEX IF NOT EXISTS idx_spreads_season ON nba_spreads_odds(season, stage);
CREATE INDEX IF NOT EXISTS idx_spreads_closing ON nba_spreads_odds(game_id, bookmaker, is_closing);
"""


class NbaSpreadsOddsRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def initialize_schema(self) -> None:
        conn = self._get_connection()
        try:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEXES_SQL)
            logger.info(f"Initialized nba_spreads_odds schema in {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def insert_odds(self, df: pl.DataFrame) -> int:
        conn = self._get_connection()
        pdf = df.to_pandas()

        try:
            max_id_res = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM nba_spreads_odds"
            ).fetchone()
            current_max_id = max_id_res[0] if max_id_res else 0
            pdf["id"] = range(current_max_id + 1, current_max_id + 1 + len(pdf))
        except Exception as e:
            logger.warning(f"Could not generate IDs: {e}")

        try:
            conn.register("temp_spreads", pdf)
            insert_sql = """
            INSERT OR IGNORE INTO nba_spreads_odds 
            (id, game_id, game_date, season, stage, home_team_id, away_team_id,
             bookmaker, handicap_home, odds_home_decimal, odds_away_decimal,
             scrape_datetime, source, is_closing)
            SELECT id, game_id, game_date, season, stage, home_team_id, away_team_id,
                   bookmaker, handicap_home, odds_home_decimal, odds_away_decimal,
                   scrape_datetime, source, is_closing
            FROM temp_spreads
            """
            result = conn.execute(insert_sql)
            inserted = result.fetchone()[0] if result else len(pdf)
            conn.unregister("temp_spreads")
            logger.info(f"Inserted {inserted} spread records")
            return inserted
        except Exception as e:
            logger.error(f"Failed to insert spreads: {e}")
            raise

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
