"""
Ingest Kaggle NBA Betting Data (2024-2025 Recovery).

This script parses the specific 'nba_2008-2025.csv' dataset to recover
missing odds data for the 2024-2025 season. It handles team mapping
from 2-3 letter abbreviations and normalizes the data to the system schema.
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from src.nba_predictor.services.nba_totals_service import get_totals_service

logger = logging.getLogger(__name__)

# Team name mapping from Kaggle Abbrev to NBA team IDs
KAGGLE_TEAM_MAPPING = {
    "atl": 1610612737,  # Atlanta Hawks
    "bos": 1610612738,  # Boston Celtics
    "bk": 1610612751,  # Brooklyn Nets (ck check 'bk' vs 'bkn')
    "bkn": 1610612751,
    "cha": 1610612766,  # Charlotte Hornets
    "chi": 1610612741,  # Chicago Bulls
    "cle": 1610612739,  # Cleveland Cavaliers
    "dal": 1610612742,  # Dallas Mavericks
    "den": 1610612743,  # Denver Nuggets
    "det": 1610612765,  # Detroit Pistons
    "gs": 1610612744,  # Golden State Warriors
    "hou": 1610612745,  # Houston Rockets
    "ind": 1610612754,  # Indiana Pacers
    "lac": 1610612746,  # LA Clippers
    "lal": 1610612747,  # LA Lakers
    "mem": 1610612763,  # Memphis Grizzlies
    "mia": 1610612748,  # Miami Heat
    "mil": 1610612749,  # Milwaukee Bucks
    "min": 1610612750,  # Minnesota Timberwolves
    "no": 1610612740,  # New Orleans Pelicans
    "nop": 1610612740,
    "ny": 1610612752,  # New York Knicks
    "nyk": 1610612752,
    "okc": 1610612760,  # Oklahoma City Thunder
    "orl": 1610612753,  # Orlando Magic
    "phi": 1610612755,  # Philadelphia 76ers
    "phx": 1610612756,  # Phoenix Suns
    "pho": 1610612756,
    "por": 1610612757,  # Portland Trail Blazers
    "sac": 1610612758,  # Sacramento Kings
    "sa": 1610612759,  # San Antonio Spurs
    "sas": 1610612759,
    "tor": 1610612761,  # Toronto Raptors
    "uta": 1610612762,  # Utah Jazz
    "utah": 1610612762,
    "was": 1610612764,  # Washington Wizards
}


def normalize_kaggle_2025(csv_path: Path) -> pl.DataFrame:
    """Normalize the Kaggle CSV for 2024-2025 season."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read CSV
    df = pl.read_csv(csv_path)

    # Filter for 2025 season (which represents 2024-2025)
    # Also handle team mapping early to filter invalid rows if any
    season_df = df.filter(pl.col("season") == 2025)

    logger.info(f"Filtered {len(season_df)} records for season 2025")

    # Mapping function
    def map_team(abbr: str) -> int:
        if not abbr:
            return -1
        return KAGGLE_TEAM_MAPPING.get(abbr.lower(), -1)

    normalized = season_df.with_columns(
        [
            # Generate Game ID
            pl.format(
                "KG_{}_{}_{}", pl.col("date"), pl.col("home"), pl.col("away")
            ).alias("game_id"),
            # Consistent Season
            pl.lit("2024-2025").alias("season_formatted"),
            # Stage (All RS for now, dataset doesn't seem to have PO for 2025 yet or filter needed)
            # Using column 'playoffs' (bool)
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
            # Odds (Assuming standard -110 / 1.909 as prices are missing)
            pl.col("total").cast(pl.Float64).alias("total_points_line"),
            pl.lit(1.909).alias("odds_over_decimal"),
            pl.lit(1.909).alias("odds_under_decimal"),
            # Metadata
            pl.lit("Kaggle_v2").alias("bookmaker"),  # Distinguish from v1
            pl.lit("kaggle_recovery").alias("source"),
            pl.lit(datetime.now()).alias("scrape_datetime"),
            pl.lit(True).alias("is_closing"),  # Dataset implies final lines
            # Date
            pl.col("date").alias("game_date"),
        ]
    )

    # Filter out unmapped teams
    normalized = normalized.filter(
        (pl.col("home_team_id") != -1) & (pl.col("away_team_id") != -1)
    )

    logger.info(f"Mapped {len(normalized)} records successfully")

    # Select final schema columns
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

    # Rename season_formatted back to season for schema
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
        logger.info("Starting ingestion of Kaggle 2024-2025 Recovery data...")

        # 1. Normalize
        df_new = normalize_kaggle_2025(file_path)

        # 2. Ingest
        service = get_totals_service()

        # Check current stats via direct connection to avoid private member access
        import duckdb

        with duckdb.connect("data/nba_betting.duckdb") as conn:
            try:
                pre_stats = conn.execute(
                    "SELECT COUNT(*) FROM nba_totals_odds WHERE season = '2024-2025'"
                ).fetchone()[0]
                logger.info(f"Current 2024-2025 records in DB: {pre_stats}")
            except Exception:
                logger.warning("Could not query pre-stats (table might not exist)")

        # Insert
        inserted = service.repository.insert_odds(df_new)
        logger.info(f"✅ Successfully inserted {inserted} records!")

        # Verify
        with duckdb.connect("data/nba_betting.duckdb") as conn:
            post_stats = conn.execute(
                "SELECT COUNT(*) FROM nba_totals_odds WHERE season = '2024-2025'"
            ).fetchone()[0]
            logger.info(f"New 2024-2025 records in DB: {post_stats}")

    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        raise


if __name__ == "__main__":
    main()
