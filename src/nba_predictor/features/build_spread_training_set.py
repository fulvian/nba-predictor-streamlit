"""
Build Spread Training Set (Alpha Strategy).

This script:
1. Loads 'unified_spread_dataset' (Spreads + Scores).
2. Calculates Rolling Stats (Efficiency, Pace, Density).
3. Adds Alpha Features:
   - Altitude (Den/Uta)
   - Schedule Density (3in4, 5in7)
   - Style Clash (Home Pace - Away Pace)
4. Saves to 'data/nba_spread_features_v1.parquet'.
"""

import logging
import sys
import os
import duckdb
import polars as pl
import numpy as np
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

from src.nba_predictor.features.rolling_stats_engine import RollingStatsEngine

logger = logging.getLogger(__name__)

DB_PATH = "data/nba_betting.duckdb"
OUTPUT_PATH = "data/nba_spread_features_v1.parquet"

# Altitude Mapping (High Altitude = Advantage)
# Denver (1610612743), Utah (1610612762)
HIGH_ALTITUDE_TEAMS = [1610612743, 1610612762]


def load_data():
    logger.info("Loading unified_spread_dataset...")
    with duckdb.connect(DB_PATH) as conn:
        df = conn.execute("SELECT * FROM unified_spread_dataset").pl()

    # Sort
    df = df.sort(["game_date", "game_id"])
    return df


def add_altitude_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add boolean flag for High Altitude Home Games."""
    return df.with_columns(
        [
            pl.col("home_team_id")
            .is_in(HIGH_ALTITUDE_TEAMS)
            .cast(pl.Int8)
            .alias("is_high_altitude_home")
        ]
    )


def build_dataset():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # 1. Load Match Level Data
    match_df = load_data()
    logger.info(f"Loaded {len(match_df)} matches.")

    # 2. Construct Team Log (Home perspective) for Rolling Stats
    home_log = match_df.select(
        [
            pl.col("game_id"),
            pl.col("game_date").alias("date"),
            pl.col("season"),
            pl.col("home_team_id").alias("team_id"),
            pl.col("away_team_id").alias("opponent_id"),
            pl.lit(1).alias("is_home"),
            pl.col("home_score").alias("score"),
            pl.col("away_score").alias("opponent_score"),
            pl.col("away_score").alias("points_allowed"),
        ]
    )

    away_log = match_df.select(
        [
            pl.col("game_id"),
            pl.col("game_date").alias("date"),
            pl.col("season"),
            pl.col("away_team_id").alias("team_id"),
            pl.col("home_team_id").alias("opponent_id"),
            pl.lit(0).alias("is_home"),
            pl.col("away_score").alias("score"),
            pl.col("home_score").alias("opponent_score"),
            pl.col("home_score").alias("points_allowed"),
        ]
    )

    team_log = pl.concat([home_log, away_log]).sort(["date"])

    # 3. Add Dummy Stats for Engine
    # Since we lack FGA/FTA, we add 0s. Engine will produce invalid Pace/Rtg but valid Scores.
    team_log = team_log.with_columns(
        [
            pl.lit(1).alias("is_game"),
            pl.lit(0).alias("fga"),
            pl.lit(0).alias("fta"),
            pl.lit(0).alias("orb"),
            pl.lit(0).alias("tov"),
            pl.lit(0).alias("fgm"),
            pl.lit(0).alias("fg3m"),
            pl.lit(0).alias("ftm"),
            pl.lit(240).alias("minutes"),
        ]
    )

    engine = RollingStatsEngine()

    # CALCULATE FEATURES
    # Using 'compute_rolling_stats' explicitly
    team_features = engine.compute_rolling_stats(team_log)

    # 4. Join Back to Match
    # Use L10_score (Offense) and L10_points_allowed (Defense) since Rtg is invalid
    # Also include Rest and Density

    # Filter Home/Away to avoid cartesian product on Join (since game_id is in both rows)
    # We must ensure 'is_home' is available in team_features.
    # RollingStatsEngine usually preserves input columns? yes.

    home_feats = team_features.filter(pl.col("is_home") == 1).select(
        [
            pl.col("game_id"),
            # pl.col("team_id") is redundant if we trust game_id + is_home mapping,
            # but let's keep aliases for sanity check
            pl.col("team_id").alias("home_team_id"),
            pl.col("L10_score").alias("home_L10_score"),
            pl.col("L10_points_allowed").alias("home_L10_points_allowed"),
            # Proxy Pace
            (pl.col("L10_score") + pl.col("L10_points_allowed")).alias(
                "home_L10_pace_proxy"
            ),
            pl.col("rest_days").alias("home_rest"),
            pl.col("games_in_last_4d").alias("home_density_4d"),
            pl.col("games_in_last_7d").alias("home_density_7d"),
        ]
    )

    away_feats = team_features.filter(pl.col("is_home") == 0).select(
        [
            pl.col("game_id"),
            pl.col("team_id").alias("away_team_id"),
            pl.col("L10_score").alias("away_L10_score"),
            pl.col("L10_points_allowed").alias("away_L10_points_allowed"),
            (pl.col("L10_score") + pl.col("L10_points_allowed")).alias(
                "away_L10_pace_proxy"
            ),
            pl.col("rest_days").alias("away_rest"),
            pl.col("games_in_last_4d").alias("away_density_4d"),
            pl.col("games_in_last_7d").alias("away_density_7d"),
        ]
    )

    final_df = match_df.join(home_feats, on="game_id").join(away_feats, on="game_id")

    # 5. Add Alpha Features
    # Altitude
    final_df = add_altitude_features(final_df)

    # Style Clash (Pace Proxy Diff)
    # Positive means Home plays faster games than Away
    final_df = final_df.with_columns(
        [
            (pl.col("home_L10_pace_proxy") - pl.col("away_L10_pace_proxy")).alias(
                "pace_mismatch"
            ),
            (pl.col("home_density_4d") - pl.col("away_density_4d")).alias(
                "density_mismatch_4d"
            ),
        ]
    )

    # Filter Data (Training Validation)
    # Drop rows where L10 stats are null (start of season)
    final_df = final_df.drop_nulls()

    # Save
    logger.info(f"Saving {len(final_df)} training records to {OUTPUT_PATH}")
    final_df.write_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    build_dataset()
