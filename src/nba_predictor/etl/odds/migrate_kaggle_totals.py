"""
Migrate Kaggle NBA Betting Totals Dataset.

This module migrates the Kaggle nba_betting_totals.csv dataset into the
normalized nba_totals_odds schema. Handles American to Decimal odds conversion
and season/stage extraction from game_id.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to Decimal odds.

    Args:
        american_odds: American format odds (e.g., -110, +150)

    Returns:
        Decimal format odds (e.g., 1.909, 2.5)

    Examples:
        >>> american_to_decimal(-110)
        1.909090909090909
        >>> american_to_decimal(150)
        2.5
        >>> american_to_decimal(-200)
        1.5
    """
    if american_odds is None or american_odds == 0:
        return 2.0  # Default even odds

    if american_odds >= 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def extract_season_from_game_id(game_id: str) -> str:
    """
    Extract season string from NBA game_id.

    NBA game_id formats vary but typically:
    - Modern (10 digits): 00XXYYGGGGG where XX=type, YY=year
    - Older (8-9 digits): XYYGGGGGG where X=type, YY=year

    Examples seen in data:
    - 21100131 -> season 2011-12 (2=regular, 11=year)
    - 20800741 -> season 2008-09 (2=regular, 08=year)
    - 0021100001 -> season 2011-12

    Args:
        game_id: NBA game ID string

    Returns:
        Season string in format "YYYY-YYYY" (e.g., "2011-2012")
    """
    if not game_id:
        return "unknown"

    game_id_str = str(game_id).strip()

    # Handle different lengths
    try:
        if len(game_id_str) == 10:
            # Format: 00XXYYGGGGG -> extract YY from positions 3-5
            year_short = int(game_id_str[3:5])
        elif len(game_id_str) == 9:
            # Format: 0XXYYGGGGG -> extract YY from positions 2-4
            year_short = int(game_id_str[2:4])
        elif len(game_id_str) == 8:
            # Format: XYYGGGGGG -> extract YY from positions 1-3
            year_short = int(game_id_str[1:3])
        elif len(game_id_str) >= 5:
            # Fallback: try positions 1-3
            year_short = int(game_id_str[1:3])
        else:
            return "unknown"

        # Handle century (98, 99 -> 1998, 1999; 00-50 -> 2000-2050)
        if year_short >= 50:
            start_year = 1900 + year_short
        else:
            start_year = 2000 + year_short

        return f"{start_year}-{start_year + 1}"
    except (ValueError, IndexError):
        return "unknown"


def extract_stage_from_game_id(game_id: str) -> str:
    """
    Extract season stage from NBA game_id.

    The first digit(s) indicate game type:
    - 0: Preseason
    - 1: Preseason
    - 2: Regular Season
    - 3: All-Star
    - 4: Playoffs
    - 5: Play-in

    Args:
        game_id: NBA game ID string

    Returns:
        Stage code: "PRE", "RS", "PO", or "FINALS"
    """
    if not game_id:
        return "RS"

    game_id_str = str(game_id).strip()

    # For 10-digit IDs (00XXYY...), look at position 2
    # For 8-9 digit IDs (XYY...), look at position 0
    try:
        if len(game_id_str) == 10:
            game_type = game_id_str[2]
        else:
            game_type = game_id_str[0]
    except IndexError:
        return "RS"

    stage_map = {
        "0": "PRE",  # Preseason
        "1": "PRE",  # Preseason
        "2": "RS",  # Regular Season
        "3": "RS",  # All-Star related
        "4": "PO",  # Playoffs
        "5": "PO",  # Play-in
    }

    return stage_map.get(game_type, "RS")


def normalize_bookmaker_name(book_name: str) -> str:
    """
    Normalize bookmaker names for consistency.

    Args:
        book_name: Raw bookmaker name from Kaggle dataset

    Returns:
        Normalized bookmaker name
    """
    name_map = {
        "Pinnacle Sports": "Pinnacle",
        "5Dimes": "5Dimes",
        "Bookmaker": "Bookmaker",
        "BetOnline": "BetOnline",
        "Bovada": "Bovada",
        "Heritage": "Heritage",
        "Intertops": "Intertops",
        "YouWager": "YouWager",
        "JustBet": "JustBet",
        "Sportsbetting": "Sportsbetting",
    }
    return name_map.get(book_name, book_name)


def migrate_kaggle_totals(
    kaggle_path: Path | str = Path("data/nba_odds_csv/nba_betting_totals.csv"),
    games_path: Optional[Path | str] = Path("data/nba_odds_csv/nba_games_all.csv"),
    output_path: Optional[Path | str] = None,
    min_season: str = "2011-2012",
    max_season: Optional[str] = None,
) -> pl.DataFrame:
    """
    Migrate Kaggle nba_betting_totals.csv to normalized schema.

    The Kaggle dataset has columns:
    - game_id: NBA official game ID
    - book_name: Bookmaker name
    - book_id: Bookmaker ID
    - team_id: Home team ID
    - a_team_id: Away team ID
    - total1, total2: Over/Under line (usually same value)
    - price1, price2: American odds for Over/Under

    Args:
        kaggle_path: Path to Kaggle totals CSV
        games_path: Optional path to games CSV for date lookup
        output_path: Optional path to save normalized Parquet
        min_season: Minimum season to include
        max_season: Maximum season to include (None = all)

    Returns:
        Polars DataFrame in normalized schema
    """
    kaggle_path = Path(kaggle_path)

    if not kaggle_path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found: {kaggle_path}")

    logger.info(f"Loading Kaggle totals from {kaggle_path}")

    # Load Kaggle totals
    df = pl.read_csv(kaggle_path)

    logger.info(f"Loaded {len(df)} raw records")

    # Load games for date lookup if available
    game_dates = {}
    if games_path and Path(games_path).exists():
        games_df = pl.read_csv(games_path)
        if "game_id" in games_df.columns and "game_date" in games_df.columns:
            game_dates = dict(
                zip(
                    games_df["game_id"].cast(pl.Utf8).to_list(),
                    games_df["game_date"].to_list(),
                )
            )
            logger.info(f"Loaded {len(game_dates)} game dates for lookup")

    # Transform to normalized schema
    normalized = df.with_columns(
        [
            # Convert game_id to string
            pl.col("game_id").cast(pl.Utf8).alias("game_id"),
            # Extract season and stage
            pl.col("game_id")
            .cast(pl.Utf8)
            .map_elements(extract_season_from_game_id, return_dtype=pl.Utf8)
            .alias("season"),
            pl.col("game_id")
            .cast(pl.Utf8)
            .map_elements(extract_stage_from_game_id, return_dtype=pl.Utf8)
            .alias("stage"),
            # Normalize bookmaker name
            pl.col("book_name")
            .map_elements(normalize_bookmaker_name, return_dtype=pl.Utf8)
            .alias("bookmaker"),
            # Use total1 as the line (total1 and total2 are usually equal)
            pl.col("total1").alias("total_points_line"),
            # Convert American odds to Decimal
            pl.col("price1")
            .map_elements(american_to_decimal, return_dtype=pl.Float64)
            .alias("odds_over_decimal"),
            pl.col("price2")
            .map_elements(american_to_decimal, return_dtype=pl.Float64)
            .alias("odds_under_decimal"),
            # Team IDs
            pl.col("team_id").cast(pl.Int64).alias("home_team_id"),
            pl.col("a_team_id").cast(pl.Int64).alias("away_team_id"),
            # Source
            pl.lit("kaggle").alias("source"),
            # Default scrape datetime (use epoch for historical)
            pl.lit(datetime(2018, 11, 23, 12, 0, 0)).alias("scrape_datetime"),
            # Is closing (Kaggle data is typically closing odds)
            pl.lit(True).alias("is_closing"),
        ]
    )

    # Add game_date from lookup or extract from season
    if game_dates:
        normalized = normalized.with_columns(
            pl.col("game_id")
            .map_elements(
                lambda gid: game_dates.get(str(gid), None), return_dtype=pl.Utf8
            )
            .alias("game_date")
        )
    else:
        # Fallback: use season start date as placeholder
        normalized = normalized.with_columns(
            pl.col("season")
            .map_elements(
                lambda s: f"{s.split('-')[0]}-10-01" if s != "unknown" else None,
                return_dtype=pl.Utf8,
            )
            .alias("game_date")
        )

    # Select final columns in order
    final_columns = [
        "game_id",
        "game_date",
        "season",
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

    normalized = normalized.select(final_columns)

    # Filter by season if specified
    if min_season:
        normalized = normalized.filter(pl.col("season") >= min_season)
    if max_season:
        normalized = normalized.filter(pl.col("season") <= max_season)

    # Remove rows with null essential fields
    normalized = normalized.drop_nulls(subset=["game_id", "total_points_line"])

    logger.info(f"Normalized to {len(normalized)} records")

    # Save to Parquet if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_parquet(output_path, compression="snappy")
        logger.info(f"Saved normalized data to {output_path}")

    return normalized


if __name__ == "__main__":
    # Test migration
    logging.basicConfig(level=logging.INFO)

    result = migrate_kaggle_totals(
        kaggle_path=Path("data/nba_odds_csv/nba_betting_totals.csv"),
        games_path=Path("data/nba_odds_csv/nba_games_all.csv"),
        output_path=Path("data/odds/kaggle_totals_normalized.parquet"),
    )

    print(f"\nMigrated {len(result)} records")
    print(f"\nSeasons: {result['season'].unique().to_list()}")
    print(f"Bookmakers: {result['bookmaker'].unique().to_list()}")
    print(f"\nSample records:")
    print(result.head(5))
