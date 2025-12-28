"""
Ingest Kaggle Game Results (Scores).

This script parses 'nba_2008-2025.csv' to populate the 'nba_games' table in DuckDB.
It provides the 'Results' side of the Unified Training Dataset.
"""

import logging
from datetime import datetime
from pathlib import Path
import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# Team name mapping from Kaggle Abbrev to NBA team IDs (Same as ingest_kaggle_2025.py)
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


def setup_games_table():
    """Create nba_games table if not exists."""
    with duckdb.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_games (
                game_id VARCHAR PRIMARY KEY,
                game_date DATE,
                season VARCHAR,
                stage VARCHAR,
                home_team_id BIGINT,
                away_team_id BIGINT,
                home_score INTEGER,
                away_score INTEGER,
                total_score INTEGER,
                ingested_at TIMESTAMP
            )
        """)
        logger.info("Ensured nba_games table exists.")


def ingest_results(csv_path: Path):
    """Read CSV, normalize, and ingest into DuckDB."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read CSV
    df = pl.read_csv(csv_path)
    logger.info(f"Read {len(df)} rows from CSV.")

    # Mapping function
    def map_team(abbr: str) -> int:
        if not abbr:
            return -1
        return KAGGLE_TEAM_MAPPING.get(abbr.lower(), -1)

    # Normalize
    # 1. Map Teams
    # 2. Create Game ID (KG_{date}_{home}_{away})
    # 3. Format Date/Season

    normalized = df.with_columns(
        [
            # Game ID
            pl.format(
                "KG_{}_{}_{}", pl.col("date"), pl.col("home"), pl.col("away")
            ).alias("game_id"),
            # Teams
            pl.col("home")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("home_team_id"),
            pl.col("away")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("away_team_id"),
            # Scores
            pl.col("score_home").alias("home_score"),
            pl.col("score_away").alias("away_score"),
            (pl.col("score_home") + pl.col("score_away")).alias("total_score"),
            # Dates/Season
            pl.col("date").cast(pl.Date).alias("game_date"),
            pl.when(pl.col("playoffs"))
            .then(pl.lit("PO"))
            .otherwise(pl.lit("RS"))
            .alias("stage"),
            # Season (Kaggle uses '2025' for 2024-25, etc. Convert to YYYY-YYYY format ideally,
            # or keep consistent with odds ingestion. ingest_kaggle_2025 used '2024-2025' LIT,
            # ingest_kaggle_history used '2020-2021'.
            # We need to calculate it: If season=2025 -> '2024-2025')
            pl.format("{}-{}", pl.col("season") - 1, pl.col("season")).alias(
                "season_formatted"
            ),
            pl.lit(datetime.now()).alias("ingested_at"),
        ]
    )

    # Filter invalid teams
    normalized = normalized.filter(
        (pl.col("home_team_id") != -1) & (pl.col("away_team_id") != -1)
    )

    # Select final columns
    final_cols = [
        "game_id",
        "game_date",
        "season_formatted",
        "stage",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "total_score",
        "ingested_at",
    ]

    # Rename 'season_formatted' to 'season'
    final_df = normalized.select(
        [
            pl.col(c) if c != "season_formatted" else pl.col(c).alias("season")
            for c in final_cols
        ]
    )

    logger.info(f"Normalized {len(final_df)} valid games.")

    # Write to DuckDB
    with duckdb.connect(DB_PATH) as conn:
        # Use simple INSERT OR REPLACE/IGNORE logic
        # DuckDB generic insert
        # We can implement an UPSERT or just DELETE/INSERT for simplicity since this is reference data

        # Delete existing IDs to update them (Pseudo-Upsert)
        # Assuming we want to reload or update based on game_id
        # For bulk loading, DELETE WHERE TRUE is unnecessary if we trust the IDs,
        # but safely we can use INSERT OR IGNORE or a temp table merge.
        # Let's use generic SQL with params via executemany or register the DF

        conn.register("temp_games", final_df.to_arrow())

        # Insert or Replace logic
        conn.execute("""
            INSERT OR REPLACE INTO nba_games 
            SELECT * FROM temp_games
        """)

        count = conn.execute("SELECT COUNT(*) FROM nba_games").fetchone()[0]
        logger.info(f"Total games in DB: {count}")


def main():
    logging.basicConfig(level=logging.INFO)
    file_path = Path("data/kaggle_temp/nba_2008-2025.csv")

    try:
        setup_games_table()
        ingest_results(file_path)
        logger.info("✅ Game results ingestion complete.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
