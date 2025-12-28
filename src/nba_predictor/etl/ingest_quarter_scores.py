import logging
from pathlib import Path
import sys
import os
import polars as pl
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

from src.nba_predictor.models.nba_quarter_scores import NbaQuarterScoresRepository
from src.nba_predictor.etl.odds.ingest_kaggle_spreads import KAGGLE_TEAM_MAPPING

logger = logging.getLogger("IngestQuarters")
logging.basicConfig(level=logging.INFO)

DB_PATH = "data/nba_betting.duckdb"
CSV_PATH = Path("data/kaggle_temp/nba_2008-2025.csv")


def main():
    if not CSV_PATH.exists():
        logger.error(f"File not found: {CSV_PATH}")
        return

    logger.info("Reading Kaggle CSV...")
    df = pl.read_csv(CSV_PATH, infer_schema_length=10000)

    # Filter 2020-2025 (Season >= 2021)
    df = df.filter(pl.col("season") >= 2021)
    logger.info(f"Filtered {len(df)} records (2020-2025).")

    # Generate Game ID (KG_date_home_away) and Select/Rename
    processed = df.with_columns(
        [
            pl.format(
                "KG_{}_{}_{}", pl.col("date"), pl.col("home"), pl.col("away")
            ).alias("game_id"),
            pl.col("q1_home").cast(pl.Int32),
            pl.col("q2_home").cast(pl.Int32),
            pl.col("q3_home").cast(pl.Int32),
            pl.col("q4_home").cast(pl.Int32),
            pl.col("ot_home").cast(pl.Int32),
            pl.col("q1_away").cast(pl.Int32),
            pl.col("q2_away").cast(pl.Int32),
            pl.col("q3_away").cast(pl.Int32),
            pl.col("q4_away").cast(pl.Int32),
            pl.col("ot_away").cast(pl.Int32),
            # Calculated Halftime Scores
            (pl.col("q1_home") + pl.col("q2_home")).alias("half_home").cast(pl.Int32),
            (pl.col("q1_away") + pl.col("q2_away")).alias("half_away").cast(pl.Int32),
        ]
    ).select(
        [
            "game_id",
            "q1_home",
            "q2_home",
            "q3_home",
            "q4_home",
            "ot_home",
            "q1_away",
            "q2_away",
            "q3_away",
            "q4_away",
            "ot_away",
            "half_home",
            "half_away",
        ]
    )

    # Init Repo and Insert
    repo = NbaQuarterScoresRepository(DB_PATH)
    repo.initialize_schema()

    logger.info("Inserting scores...")
    count = repo.insert_scores(processed)
    logger.info(f"✅ Ingested {count} quarter score records.")

    repo.close()


if __name__ == "__main__":
    main()
