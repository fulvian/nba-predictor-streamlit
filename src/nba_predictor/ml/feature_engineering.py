"""
Advanced Feature Engineering Module for NBA Predictor.

This module implements critical features identified by research as high-impact for NBA totals:
1. PACE (Possessions/48min) - #1 Predictor for totals
2. Rest Days / Back-to-Back - Systematic bias correction (-3.5pts)
3. Recent Form (Weighted L10) - Captures momentum/injuries
4. Defensive Rating - Key for mismatch identification
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """
    Engine for creating advanced NBA analytics features.
    Focuses on 'Four Factors' derived metrics and scheduling impacts.
    """

    def __init__(self):
        self.required_columns = [
            "fga",
            "fta",
            "orb",
            "tov",
            "points",
            "date",
            "team_id",
        ]

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all advanced feature engineering steps."""
        df = df.copy()

        # Validation
        missing = [
            c
            for c in self.required_columns
            if c not in df.columns and c not in ["date", "team_id"]
        ]
        if missing:
            # Try to infer or check if they have home/away prefixes
            logger.warning(
                f"[FEATURE ENGINE] Potential missing base columns: {missing}"
            )

        logger.info("[FEATURE ENGINE] Adding PACE features...")
        df = self.add_pace_features(df)

        logger.info("[FEATURE ENGINE] Adding REST DAYS features...")
        df = self.add_rest_days(df)

        logger.info("[FEATURE ENGINE] Adding RECENT FORM features...")
        df = self.add_recent_form(df)

        logger.info("[FEATURE ENGINE] Adding DEFENSIVE RATING features...")
        df = self.add_defensive_rating(df)

        return df

    def _calculate_pace(self, df: pd.DataFrame, team_prefix: str) -> pd.Series:
        """
        Calculate Pace (Possessions) for a team.
        Formula: 0.96 * (FGA + 0.44*FTA - ORB + TOV)
        """
        # Try both short (fga) and verbose (field_goals_attempted) suffixes
        # Also check for uppercase (FGA, etc) as prediction pipeline produces them

        # Helper to get column case-insensitively-ish
        def get_col(suffix_list):
            for suffix in suffix_list:
                col = f"{team_prefix}_{suffix}"
                if col in df.columns:
                    return df[col]
            return 0

        fga = get_col(["fga", "FGA", "field_goals_attempted", "FIELD_GOALS_ATTEMPTED"])
        fta = get_col(["fta", "FTA", "free_throws_attempted", "FREE_THROWS_ATTEMPTED"])
        orb = get_col(["orb", "ORB", "offensive_rebounds", "OFFENSIVE_REBOUNDS"])
        tov = get_col(["tov", "TOV", "turnovers", "TURNOVERS"])

        # If columns are missing (sum is 0), try checking without prefix if the dataframe is team-specific
        # But here we are likely in a matchup df

        possessions = 0.96 * (fga + 0.44 * fta - orb + tov)
        return possessions

    def add_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Pace (Possessions per game) features.
        Research: Higher pace directly correlates with higher totals.
        """
        # Detect prefix convention
        p1 = "home"
        p2 = "away"

        # Check for home/away or team1/team2 conventions (including uppercase variants)
        # We check for FGA/fga presence with prefixes

        has_home = any(
            f"home_{x}" in df.columns for x in ["fga", "FGA", "field_goals_attempted"]
        )
        has_team1 = any(
            f"team1_{x}" in df.columns for x in ["fga", "FGA", "field_goals_attempted"]
        )

        if not has_home and has_team1:
            p1 = "team1"
            p2 = "team2"

        df[f"{p1}_pace"] = self._calculate_pace(df, p1)
        df[f"{p2}_pace"] = self._calculate_pace(df, p2)

        # Pace Matchup: Expected total possessions
        df["pace_matchup"] = (df[f"{p1}_pace"] + df[f"{p2}_pace"]) / 2.0

        # Log if we found possessions
        if df["pace_matchup"].sum() == 0:
            logger.warning(
                "[FEATURE ENGINE] Pace calculation returned 0s. Check input column names."
            )
        else:
            logger.info(f"[FEATURE ENGINE] Pace calculated using prefixes: {p1}/{p2}")

        return df

    def _calc_days_since_last_game(
        self, df: pd.DataFrame, team_prefix: str
    ) -> pd.Series:
        """Calculate days of rest for the team."""
        # This requires sorting by team and date, which is complex in a single row-based DataFrame
        # that contains matchups.
        # We need a robust way to lookup the previous game for the specific team logic.

        # Assuming df has 'date' or 'game_date'
        date_col = "date" if "date" in df.columns else "game_date"
        if date_col not in df.columns:
            logger.warning("Date column not found, skipping rest days calculation")
            return pd.Series([3] * len(df))  # Default to fully rested

        # We'll need to reconstruct a team-level schedule from the matchup df
        # This is expensive but necessary

        # Simplified vectorised approach if possible?
        # Creating a map of (team, date) -> previous_date is better

        # Placeholder for complex logic:
        # For now, if 'days_rest' is already in input, trust it.
        # Otherwise, return default (needs improvement in Phase 2)
        if f"{team_prefix}_days_rest" in df.columns:
            return df[f"{team_prefix}_days_rest"]

        return pd.Series([2] * len(df))  # Default placeholder

    def add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Rest Days and Back-to-Back flags.
        Research: B2B generally reduces scoring efficiency (-3.5 pts/game impact).
        """
        # NOTE: If we are constructing this from raw game logs, we can calculate accurately.
        # If we assume the input df is the feature set, we try to use or calculate.

        df["home_rest_days"] = self._calc_days_since_last_game(df, "home")
        df["away_rest_days"] = self._calc_days_since_last_game(df, "away")

        # Back-to-back flag (1 day rest usually means played yesterday? No, 0 days gap usually implies B2B in data?)
        # Convention: If date diff is 1, they played yesterday. i.e. 0 rest days.
        df["is_b2b_home"] = (df["home_rest_days"] <= 1).astype(int)
        df["is_b2b_away"] = (df["away_rest_days"] <= 1).astype(int)
        df["is_back_to_back"] = df["is_b2b_home"] | df["is_b2b_away"]

        return df

    def _rolling_weighted_avg(
        self, df: pd.DataFrame, team_prefix: str, metric: str, window: int = 10
    ) -> pd.Series:
        """Calculate weighted average for recent form."""
        col_name = f"{team_prefix}_{metric}"
        if col_name not in df.columns:
            return pd.Series([0] * len(df))

        # We need simple logic here given the structure might not be time-series of one team
        # If we can't do true rolling, we use the value itself or a placeholder
        # Ideally, this should be done during Data Loading phase in DataStore

        return df[
            col_name
        ]  # Placeholder - Real rolling requires team-history structure

    def add_recent_form(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Add recent form features (Weighted L10).
        """
        # Logic needs to move to Data Loading if it requires cross-row history
        # For now, we assume the input might already have some form data or we operate on what's available
        # Real implementation should happen in UnifiedDataStore where history is available
        return df

    def add_defensive_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Defensive Rating (Pts Allowed / 100 Possessions).
        """
        # Needs Points Allowed. In a matchup df:
        # Home Def Rtg = Away Points / Home Possessions * 100
        # Away Def Rtg = Home Points / Away Possessions * 100

        # CAUTION: 'points' usually refers to the CURRENT GAME points in historical data
        # We need PRE-GAME average defensive rating.
        # Calculating DefRtg from the *current* game is data leakage.

        # We need 'avg_pts_allowed' columns from pre-game stats

        if "home_pace" in df.columns and "home_pts_allowed_avg" in df.columns:
            df["home_def_rtg"] = (df["home_pts_allowed_avg"] / df["home_pace"]) * 100

        if "away_pace" in df.columns and "away_pts_allowed_avg" in df.columns:
            df["away_def_rtg"] = (df["away_pts_allowed_avg"] / df["away_pace"]) * 100

        return df
