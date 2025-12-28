"""
Rolling Stats Engine (EWMA).

This module calculates advanced time-series features for NBA teams using
Exponential Weighted Moving Averages (EWMA).

It calculates:
1. Pace (Possessions/48m)
2. Four Factors (eFG%, TOV%, ORB%, FTR)
3. Offensive/Defensive Ratings
4. Margin & Score Trends

It generates 'pre-game' snapshots to avoid data leakage.
"""

import logging
import polars as pl
from typing import List, Optional

logger = logging.getLogger(__name__)


class RollingStatsEngine:
    def __init__(self, span_windows: List[int] = [5, 10]):
        """
        Initialize Engine.
        Args:
            span_windows: List of spans for EWMA (e.g., [5, 10] for L5, L10).
        """
        self.windows = span_windows
        self.metrics = [
            "pace",
            "off_rtg",
            "def_rtg",
            "net_rtg",
            "efg_pct",
            "tov_pct",
            "orb_pct",
            "ft_rate",
            "score",
            "points_allowed",
        ]

    def _calculate_base_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate per-game advanced metrics from raw box scores."""
        # Assume df has: season, date, team_id, score, opponent_score,
        # fga, fta, orb, tov, fgm, fg3m, ftm, minutes

        # Calculate Possessions (Basic Formula)
        # Poss = FGA + 0.44*FTA - ORB + TOV
        df = df.with_columns(
            [
                (
                    pl.col("fga") + 0.44 * pl.col("fta") - pl.col("orb") + pl.col("tov")
                ).alias("possessions"),
                (pl.col("minutes") / 5).alias(
                    "game_duration_ratio"
                ),  # standard is 48m = 1.0 (approx)
            ]
        )

        # Pace: Poss / Minutes * 48
        df = df.with_columns(
            [(pl.col("possessions") / pl.col("minutes") * 48).alias("pace")]
        )

        # Ratings
        df = df.with_columns(
            [
                (pl.col("score") / pl.col("possessions") * 100).alias("off_rtg"),
                (pl.col("opponent_score") / pl.col("possessions") * 100).alias(
                    "def_rtg"
                ),
            ]
        )

        df = df.with_columns([(pl.col("off_rtg") - pl.col("def_rtg")).alias("net_rtg")])

        # Four Factors (Simplified for this stage if raw components exist, else assume pre-calc)
        # We assume base stats exist.
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        df = df.with_columns(
            [
                ((pl.col("fgm") + 0.5 * pl.col("fg3m")) / pl.col("fga")).alias(
                    "efg_pct"
                ),
                (pl.col("tov") / pl.col("possessions")).alias("tov_pct"),
                (pl.col("orb") / (pl.col("orb") + pl.col("opponent_drb"))).alias(
                    "orb_pct"
                ),  # Requires opp_drb
                (pl.col("ftm") / pl.col("fga")).alias("ft_rate"),
            ]
        )

        # Fill NaNs/Infs
        df = df.fill_nan(0).fill_null(0)

        return df

    def compute_rolling_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute rolling stats for the entire dataset.
        Returns DataFrame with 'L{span}_{metric}' columns.
        SAFE: Shifts data to ensure row[i] contains stats from games[0...i-1].
        """
        # 1. Sort by Team, Date
        df = df.sort(["season", "team_id", "date"])

        # 2. Iterate windows
        ops = []
        for span in self.windows:
            for metric in self.metrics:
                if metric in df.columns:
                    # EWMA Calculation:
                    # Polars ewm_mean is dynamic.
                    # CRITICAL: We calculate EWM on current row, THEN SHIFT by 1.
                    # Because for game G, we only know stats from G-1 backwards.

                    # We group by Season + Team to avoid bleeding across seasons?
                    # Ideally yes, or reset index.

                    ops.append(
                        pl.col(metric)
                        .ewm_mean(span=span, adjust=False, min_periods=1)
                        .shift(1)  # SHIFT IS KEY FOR NO PREDICTION LEAKAGE
                        .over(["season", "team_id"])
                        .alias(f"L{span}_{metric}")
                    )

        # Season-to-Date (STD) Cumulative Average
        for metric in self.metrics:
            if metric in df.columns:
                ops.append(
                    pl.col(metric)
                    .rolling_mean(
                        window_size=1000, min_periods=1
                    )  # effectively expanding
                    .shift(1)
                    .over(["season", "team_id"])
                    .alias(f"STD_{metric}")
                )

        result = df.with_columns(ops)

        # Add Rest Days
        # Date diff from prev row
        result = result.with_columns(
            [
                (pl.col("date") - pl.col("date").shift(1).over(["season", "team_id"]))
                .dt.total_days()
                .fill_null(3)  # First game of season = rest
                .alias("rest_days")
            ]
        )

        # Cap Rest Days at 7 to avoid outliers (all-star break, etc) skewing models
        result = result.with_columns(pl.col("rest_days").clip(0, 7))

        # Schedule Density (3-in-4, 5-in-7)
        # We need a rolling count of games in the last X days.
        # This is hard in pure Polars without `rolling` over date, but we can approx with `rolling_sum` over windows if we had daily rows.
        # Alternative: Rolling count of index?
        # Easier: Rolling Sum of 'IsGame' dummy variable over a Date Window.
        # However, Polars `rolling` supports `period` argument!

        result = result.sort(["season", "team_id", "date"])
        # Schedule Density using rolling(by=...) aggregation (Polars robust method)
        # We calculate separately and join back to avoid 'rolling in aggregation' error

        # 4-Day Density
        density_4d = (
            result.sort("date")
            .rolling(
                index_column="date",
                period="4d",
                by=["season", "team_id"],
                closed="left",
            )
            .agg(pl.col("is_game").sum().alias("games_in_last_4d"))
        )

        # 7-Day Density
        density_7d = (
            result.sort("date")
            .rolling(
                index_column="date",
                period="7d",
                by=["season", "team_id"],
                closed="left",
            )
            .agg(pl.col("is_game").sum().alias("games_in_last_7d"))
        )

        # Join back - groupby_rolling preserves rows?
        # Yes, rolling(by) returns one row per input row if done right?
        # Actually rolling() returns a DataFrame matching input rows if by is used?
        # Polars rolling() returns the aggregated result.
        # It includes the grouping keys and index column.

        # To be safe, we join on [season, team_id, date].
        # But 'result' might have multiple games per date/team? (Rare in NBA)
        # Yes, double headers or data dupes. Assuming unique team/date.

        result = result.join(density_4d, on=["season", "team_id", "date"], how="left")
        result = result.join(density_7d, on=["season", "team_id", "date"], how="left")

        # Fill Nulls (start of rolling window might be null?)
        result = result.fill_null(0)

        # Fill Nulls
        result = result.fill_null(0)

        return result
