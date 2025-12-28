import polars as pl
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List

from src.nba_predictor.features.rolling_stats_engine import RollingStatsEngine
from src.nba_predictor.models.nba_spreads_odds import NbaSpreadsOddsRepository
from src.nba_predictor.etl.odds.ingest_kaggle_spreads import KAGGLE_TEAM_MAPPING

# Map Team Full Names (from NBA API) to ID
# Extending KAGGLE_TEAM_MAPPING inverse or just hardcoding common ones if needed
# Actually, nba_api returns team IDs directly. We should use those if they match.
# Our IDs are standard NBA IDs (10 digits).


class LiveContextLoader:
    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = db_path
        self.engine = RollingStatsEngine()
        self.repo = NbaSpreadsOddsRepository(db_path)

    def load_todays_context(
        self, scheduled_games: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """
        Generates feature context (Rest, Density, Altitude) for the given scheduled games.

        Args:
            scheduled_games: List of dicts from nba_api scoreboard (homeTeam, awayTeam, etc.)

        Returns:
            pl.DataFrame: DataFrame keyed by game_id/team_id with alpha features.
        """
        # 1. Load History (Last 30 days is enough for rolling density?)
        # Actually RollingStatsEngine usually loads everything.
        # For performance, we might want to optimize, but loading parquet feature file is fast.

        # We need the RAW logs to calculate current density, OR we can load the feature file
        # and checking the LAST date for each team.
        # But wait, Density is calculated FROM the logs.

        # Approach:
        # 1. Load existing 'nba_spread_features_v1.parquet' to get recent state.
        # 2. Extract last game date per team.
        # 3. Calculate 'Rest Days' = Today - Last Game Date.
        # 4. Calculate 'Density' = Count games in [Today-4, Today-1].

        df_history = pl.read_parquet("data/nba_spread_features_v1.parquet")

        # We also need a mapping of Team ID -> Altitude info
        altitude_map = self._get_altitude_map(df_history)

        today = datetime.now().date()
        today_context = []

        for game in scheduled_games:
            game_id = game["gameId"]
            home_id = game["homeTeam"]["teamId"]
            away_id = game["awayTeam"]["teamId"]

            # HOME CONTEXT
            home_stats = self._calc_team_context(df_history, home_id, today)
            home_stats.update(
                {
                    "game_id": game_id,
                    "team_id": home_id,
                    "is_home": 1,
                    "is_high_altitude": altitude_map.get(home_id, 0),
                }
            )

            # AWAY CONTEXT
            away_stats = self._calc_team_context(df_history, away_id, today)
            away_stats.update(
                {
                    "game_id": game_id,
                    "team_id": away_id,
                    "is_home": 0,
                    "is_high_altitude": altitude_map.get(
                        away_id, 0
                    ),  # Usually irrelevant for Away
                }
            )

            today_context.append(home_stats)
            today_context.append(away_stats)

        return pl.DataFrame(today_context)

    def _calc_team_context(
        self, history: pl.DataFrame, team_id: int, today: datetime.date
    ) -> Dict[str, Any]:
        """Calculates Rest and Density for a specific team relative to Today."""

        # Filter games for this team (Home or Away)
        team_games = history.filter(
            (pl.col("home_team_id") == team_id) | (pl.col("away_team_id") == team_id)
        ).sort("game_date")

        if len(team_games) == 0:
            return {
                "rest_days": 3,
                "density_4d": 0,
            }  # Default if no history (start of season)

        last_game_date = team_games[-1]["game_date"][0]

        # Rest Days
        delta = (today - last_game_date).days
        rest_days = max(0, delta - 1)  # If played yesterday (delta=1), rest=0.

        # Density (Games in last 4 days EXCLUDING today)
        # Window: [Today - 4 days, Today - 1 day]
        # Example: Today is Friday. Window: Mon, Tue, Wed, Thu.

        window_start = today - pd.Timedelta(days=4)
        window_end = today - pd.Timedelta(days=1)

        games_in_window = team_games.filter(
            pl.col("game_date").is_between(window_start, window_end)
        )

        density_4d = len(games_in_window)

        return {"rest_days": rest_days, "density_4d": density_4d}

    def _get_altitude_map(self, df: pl.DataFrame) -> Dict[int, int]:
        """Extracts Altitude mapping (TeamID -> 1/0) from history."""
        # Find rows where is_high_altitude_home == 1
        high_alt_teams = (
            df.filter(pl.col("is_high_altitude_home") == 1)
            .select("home_team_id")
            .unique()
        )

        mapping = {}
        for row in high_alt_teams.iter_rows():
            mapping[row[0]] = 1
        return mapping
