"""
Ingest Kaggle NBA Betting Data (Historical 2020-2024).

This script parses the 'nba_2008-2025.csv' dataset to ingest historical
odds data for seasons 2020-2021 through 2023-2024.
"""

import logging
from datetime import datetime
from pathlib import Path
import duckdb
import polars as pl
from src.nba_predictor.services.nba_totals_service import get_totals_service

logger = logging.getLogger(__name__)

# Team name mapping from Kaggle Abbrev to NBA team IDs
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

TARGET_SEASONS = [2021, 2022, 2023, 2024]  # 2020-21 to 2023-24


def normalize_kaggle_history(csv_path: Path) -> pl.DataFrame:
    """Normalize the Kaggle CSV for historical seasons."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pl.read_csv(csv_path)

    # Filter for target seasons
    history_df = df.filter(pl.col("season").is_in(TARGET_SEASONS))
    logger.info(f"Filtered {len(history_df)} records for seasons {TARGET_SEASONS}")

    def map_team(abbr: str) -> int:
        if not abbr:
            return -1
        return KAGGLE_TEAM_MAPPING.get(abbr.lower(), -1)

    def format_season(year: int) -> str:
        return f"{year - 1}-{year}"

    normalized = history_df.with_columns(
        [
            pl.format(
                "KG_{}_{}_{}", pl.col("date"), pl.col("home"), pl.col("away")
            ).alias("game_id"),
            pl.col("season")
            .map_elements(format_season, return_dtype=pl.Utf8)
            .alias("season_formatted"),
            pl.when(pl.col("playoffs"))
            .then(pl.lit("PO"))
            .otherwise(pl.lit("RS"))
            .alias("stage"),
            pl.col("home")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("home_team_id"),
            pl.col("away")
            .map_elements(map_team, return_dtype=pl.Int64)
            .alias("away_team_id"),
            pl.col("total").cast(pl.Float64).alias("total_points_line"),
            pl.lit(1.909).alias("odds_over_decimal"),
            pl.lit(1.909).alias("odds_under_decimal"),
            pl.lit("Kaggle_v2").alias("bookmaker"),
            pl.lit("kaggle_history").alias("source"),
            pl.lit(datetime.now()).alias("scrape_datetime"),
            pl.lit(True).alias("is_closing"),
            pl.col("date").alias("game_date"),
        ]
    )

    normalized = normalized.filter(
        (pl.col("home_team_id") != -1) & (pl.col("away_team_id") != -1)
    )

    logger.info(f"Mapped {len(normalized)} records successfully")

    final_cols = [
        "game_id",
        "game_date",
        "season_formatted",
        "stage",
        "home_team_id",
        "away_team_id",
        "bookmaker",
        "total_points_line",
        "odds_over_decimal",
        "odds_under_decimal",
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
    file_path = Path(
        "/Users/fulvioventura/nba-predictor-streamlit/data/kaggle_temp/nba_2008-2025.csv"
    )

    try:
        logger.info("Starting ingestion of Kaggle Historical data (2020-2024)...")

        # 1. Normalize
        df_new = normalize_kaggle_history(file_path)

        # 2. Ingest
        service = get_totals_service()

        import duckdb

        with duckdb.connect("data/nba_betting.duckdb") as conn:
            try:
                pre_stats = conn.execute(
                    "SELECT COUNT(*) FROM nba_totals_odds WHERE source = 'kaggle_history'"
                ).fetchone()[0]
                logger.info(f"Current Kaggle history records in DB: {pre_stats}")
            except Exception:
                pass

        inserted = service.repository.insert_odds(df_new)
        logger.info(f"✅ Successfully inserted {inserted} records!")

        with duckdb.connect("data/nba_betting.duckdb") as conn:
            post_stats = conn.execute(
                "SELECT season, COUNT(*) FROM nba_totals_odds WHERE source = 'kaggle_history' GROUP BY season ORDER BY season"
            ).fetchall()
            logger.info(f"New Kaggle history distribution: {post_stats}")

    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        raise


if __name__ == "__main__":
    main()
