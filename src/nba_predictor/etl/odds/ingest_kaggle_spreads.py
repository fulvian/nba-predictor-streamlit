"""
Ingest Kaggle Spread Data.

This script parses 'nba_2008-2025.csv' to populate the 'nba_spreads_odds' table.
It calculates 'handicap_home' based on 'spread' and 'whos_favored'.
"""

import logging
from datetime import datetime
from pathlib import Path
import sys
import os
import polars as pl

# Add project root
sys.path.append(os.getcwd())

from src.nba_predictor.models.nba_spreads_odds import NbaSpreadsOddsRepository

logger = logging.getLogger(__name__)

# Team name mapping (Reused)
KAGGLE_TEAM_MAPPING = {
    "atl": 1610612737,
    "bos": 1610612738,
    "bk": 1610612751,
    "bkn": 1610612751,
    "cha": 1610612766,
    "chi": 1610612741,
    "cle": 1610612739,
    "dal": 1610612742,
    "den": 1610612743,
    "det": 1610612765,
    "gs": 1610612744,
    "hou": 1610612745,
    "ind": 1610612754,
    "lac": 1610612746,
    "lal": 1610612747,
    "mem": 1610612763,
    "mia": 1610612748,
    "mil": 1610612749,
    "min": 1610612750,
    "no": 1610612740,
    "nop": 1610612740,
    "ny": 1610612752,
    "nyk": 1610612752,
    "okc": 1610612760,
    "orl": 1610612753,
    "phi": 1610612755,
    "phx": 1610612756,
    "pho": 1610612756,
    "por": 1610612757,
    "sac": 1610612758,
    "sa": 1610612759,
    "sas": 1610612759,
    "tor": 1610612761,
    "uta": 1610612762,
    "utah": 1610612762,
    "was": 1610612764,
}

DB_PATH = "data/nba_betting.duckdb"


def normalize_kaggle_spreads(csv_path: Path) -> pl.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    logger.info(f"Reading {csv_path}...")
    df = pl.read_csv(csv_path, infer_schema_length=10000)

    # Filter for relevant years (2020-2025 like totals)
    # Kaggle season '2021' = 2020-2021.
    df = df.filter(pl.col("season") >= 2021)

    logger.info(f"Filtered {len(df)} records for seasons 2020-2025")

    def map_team(abbr: str) -> int:
        if not abbr:
            return -1
        return KAGGLE_TEAM_MAPPING.get(abbr.lower(), -1)

    normalized = df.with_columns(
        [
            # Game ID
            pl.format(
                "KG_{}_{}_{}", pl.col("date"), pl.col("home"), pl.col("away")
            ).alias("game_id"),
            # Season string
            pl.format("{}-{}", pl.col("season") - 1, pl.col("season")).alias(
                "season_formatted"
            ),
            # Stage
            pl.when(pl.col("playoffs"))
            .then(pl.lit("PO"))
            .otherwise(pl.lit("RS"))
            .alias("stage"),
            # Teams
            pl.col("home")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("home_team_id"),
            pl.col("away")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("away_team_id"),
            # HANDICAP Calculation
            # whos_favored = 'home' -> handicap is negative (-spread)
            # whos_favored = 'away' -> handicap is positive (+spread)
            pl.when(pl.col("whos_favored") == "home")
            .then(pl.col("spread") * -1)
            .otherwise(pl.col("spread"))
            .cast(pl.Float64)
            .alias("handicap_home"),
            # Odds (Default 1.909)
            pl.lit(1.909).alias("odds_home_decimal"),
            pl.lit(1.909).alias("odds_away_decimal"),
            # Metadata
            pl.lit("Kaggle_v2").alias("bookmaker"),
            pl.lit("kaggle_recovery").alias("source"),
            pl.lit(datetime.now()).alias("scrape_datetime"),
            pl.lit(True).alias("is_closing"),
            pl.col("date").alias("game_date"),
        ]
    )

    normalized = normalized.filter(
        (pl.col("home_team_id") != -1)
        & (pl.col("away_team_id") != -1)
        & (pl.col("handicap_home").is_not_null())
    )

    final_cols = [
        "game_id",
        "game_date",
        "season_formatted",
        "stage",
        "home_team_id",
        "away_team_id",
        "bookmaker",
        "handicap_home",
        "odds_home_decimal",
        "odds_away_decimal",
        "scrape_datetime",
        "source",
        "is_closing",
    ]

    return normalized.select(
        [
            pl.col(c) if c != "season_formatted" else pl.col(c).alias("season")
            for c in final_cols
        ]
    )


def main():
    logging.basicConfig(level=logging.INFO)
    file_path = Path("data/kaggle_temp/nba_2008-2025.csv")

    try:
        repo = NbaSpreadsOddsRepository(DB_PATH)
        repo.initialize_schema()

        df_new = normalize_kaggle_spreads(file_path)
        logger.info(f"Ready to ingest {len(df_new)} spread records.")

        inserted = repo.insert_odds(df_new)
        repo.close()

        logger.info(f"✅ Successfully ingested {inserted} spread records!")

    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        raise


if __name__ == "__main__":
    main()
