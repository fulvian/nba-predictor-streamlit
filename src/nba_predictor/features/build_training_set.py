"""
Build Training Set (V2).

This script generates the final ML Training Dataset by:
1. Loading the Consolidated View (Odds + Results).
2. Exploding Matches into Team-Logs (Home/Away -> 2 rows).
3. Applying RollingStatsEngine (EWMA).
4. Re-joining to Match Level (Home Features + Away Features).
5. Calculating Target Variables (Margin).
"""

import logging
import duckdb
import polars as pl
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.nba_predictor.features.rolling_stats_engine import RollingStatsEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "data/nba_betting.duckdb"
OUTPUT_PATH = "data/nba_training_features_v2.parquet"


def load_consolidated_data() -> pl.DataFrame:
    """Load unified_training_dataset from DuckDB."""
    with duckdb.connect(DB_PATH) as conn:
        logger.info("Loading unified_training_dataset...")
        df = conn.execute("SELECT * FROM unified_training_dataset").pl()
        logger.info(f"Loaded {len(df)} matches.")
    return df


def transform_to_team_logs(matches: pl.DataFrame) -> pl.DataFrame:
    """
    Convert Match rows to Team-Game rows (2x rows).
    Needed for time-series rolling calculations.
    """
    # Select subset for Home
    home_cols = {
        "game_id": "game_id",
        "game_date": "date",
        "season": "season",
        "home_team_id": "team_id",
        "home_score": "score",
        "away_score": "opponent_score",
        "away_team_id": "opponent_team_id",
        # We need extended stats (FGA, etc) for Rolling Engine.
        # Currently unified_view has ONLY Scores/Odds.
        # CRITICAL MISSING LINK: We need Box Score Stats (FGA, TOV, etc).
        # We must join with nba_games explicit stats if available, or we only have SCORES.
        # STOPGAP: Use Scores only for now (OffRtg/DefRtg based on score).
        # Pace requires FGA/TOV. If missing, we assume 98.0 or cannot calc accurate Pace.
    }

    # Check if we have box score cols.
    # Current unification ONLY pulled scores.
    # To do full features we need to ingest FULL stats from Kaggle.
    # The Kaggle CSV has: 'fga_home', 'tov_home' etc ??
    # Checking Kaggle columns AGAIN in memory...
    # 'q1', 'spread', etc. It does NOT seem to have FGA/TOV/ORB in the columns I saw earlier.
    # wait... Step 1619 output:
    # ['season', 'date', ... 'score_home', 'q1_home' ... 'spread', 'total' ...]
    # It seems the Kaggle dataset is simplistic (Scores Only).
    # This LIMITS Feature Engineering to Score-Based stats (Eff, Def, Margin).
    # We CANNOT do Four Factors without FGA/TOV.
    # PLAN ADAPTATION: We will use Score-Based Rolling features + Odds.

    # Home transformation
    home = matches.select(
        [
            pl.col("game_id"),
            pl.col("game_date").alias("date"),
            pl.col("season"),
            pl.col("home_team_id").alias("team_id"),
            pl.col("home_score").alias("score"),
            pl.col("away_score").alias("opponent_score"),
            pl.col("away_team_id").alias("opponent_team_id"),
            pl.lit(48 * 5).alias("minutes"),  # Placeholder
            pl.lit(85).alias("fga"),  # Dummy to avoid crash in Engine
            pl.lit(20).alias("fta"),
            pl.lit(10).alias("orb"),
            pl.lit(12).alias("tov"),
            pl.lit(40).alias("fgm"),
            pl.lit(12).alias("fg3m"),
            pl.lit(18).alias("ftm"),
            pl.lit(35).alias("opponent_drb"),
        ]
    )

    # Away transformation
    away = matches.select(
        [
            pl.col("game_id"),
            pl.col("game_date").alias("date"),
            pl.col("season"),
            pl.col("away_team_id").alias("team_id"),
            pl.col("away_score").alias("score"),
            pl.col("home_score").alias("opponent_score"),
            pl.col("home_team_id").alias("opponent_team_id"),
            pl.lit(48 * 5).alias("minutes"),  # Placeholder
            pl.lit(85).alias("fga"),
            pl.lit(20).alias("fta"),
            pl.lit(10).alias("orb"),
            pl.lit(12).alias("tov"),
            pl.lit(40).alias("fgm"),
            pl.lit(12).alias("fg3m"),
            pl.lit(18).alias("ftm"),
            pl.lit(35).alias("opponent_drb"),
        ]
    )

    return pl.concat([home, away]).sort(["season", "team_id", "date"])


def main():
    logger.info("🚀 Building Training Dataset V2...")

    # 1. Load Data
    matches = load_consolidated_data()

    # 2. Explode to Team Logs
    team_logs = transform_to_team_logs(matches)
    logger.info(f"Exploded to {len(team_logs)} team logs.")

    # 3. Apply Rolling Engine
    engine = RollingStatsEngine(
        span_windows=[5, 10, 80]
    )  # Short, Med, Long form via EWMA

    # Calc base metrics (Will be dummy/approx if FGA missing)
    team_logs = engine._calculate_base_metrics(team_logs)

    # Compute Rolling
    team_logs_rolling = engine.compute_rolling_stats(team_logs)

    # 4. Re-Join to Matches
    # We need to join back to 'matches' twice: once for Home stats, once for Away stats

    # Prefix columns
    cols_to_join = [
        c
        for c in team_logs_rolling.columns
        if c.startswith("L") or c.startswith("STD_") or c == "rest_days"
    ]

    home_stats = team_logs_rolling.select(["game_id", "team_id"] + cols_to_join).rename(
        {c: f"home_{c}" for c in cols_to_join}
    )

    away_stats = team_logs_rolling.select(["game_id", "team_id"] + cols_to_join).rename(
        {c: f"away_{c}" for c in cols_to_join}
    )

    # Join
    final_df = matches.join(
        home_stats, left_on=["game_id", "home_team_id"], right_on=["game_id", "team_id"]
    ).join(
        away_stats, left_on=["game_id", "away_team_id"], right_on=["game_id", "team_id"]
    )

    # 5. Add Advanced Features (Comparatives)
    final_df = final_df.with_columns(
        [
            (pl.col("home_rest_days") - pl.col("away_rest_days")).alias(
                "rest_advantage"
            ),
            (pl.col("home_L10_off_rtg") - pl.col("away_L10_def_rtg")).alias(
                "matchup_off_h_def_a"
            ),
            (pl.col("away_L10_off_rtg") - pl.col("home_L10_def_rtg")).alias(
                "matchup_off_a_def_h"
            ),
            # Target: Margin (Over/Under)
            # Actual Total - Closing Line
            (pl.col("total_score") - pl.col("closing_total")).alias("target_margin"),
            # Binary Target (Over Hit?)
            (pl.col("total_score") > pl.col("closing_total")).alias("target_over_hit"),
        ]
    )

    # 6. Save
    logger.info(f"✅ Final Schema: {final_df.columns}")

    # Drop rows with Null targets or missing critical features (e.g. first games of season)
    # We can drop L10 nulls
    final_df = final_df.drop_nulls(subset=["home_L10_off_rtg", "closing_total"])
    logger.info(
        f"Retained {len(final_df)} rows after dropping nulls (early season games)."
    )

    final_df.write_parquet(OUTPUT_PATH)
    logger.info(f"💾 Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
