import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import polars as pl
from nba_api.live.nba.endpoints import scoreboard

from src.nba_predictor.live.context_loader import LiveContextLoader

logger = logging.getLogger(__name__)


class LiveMonitor:
    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.context_loader = LiveContextLoader(db_path)
        self.cached_context: Optional[pl.DataFrame] = None
        self.last_context_update = None

    def fetch_current_state(self) -> List[Dict[str, Any]]:
        """
        Fetches live scores and merges with pre-computed physical context.
        Returns a list of game objects enriched with 'context' (Rest, Density).
        """
        # 1. Fetch Scoreboard
        try:
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
        except Exception as e:
            logger.error(f"NBA API Error: {e}")
            return []

        # 2. Update Context if needed (Once per day usually)
        if self.cached_context is None or self._is_new_day():
            logger.info("Refreshing Daily Context...")
            try:
                self.cached_context = self.context_loader.load_todays_context(games)
                self.last_context_update = datetime.now()
            except Exception as e:
                logger.error(f"Context Load Error: {e}")
                # Don't fail completely, just run without context (or empty)
                self.cached_context = pl.DataFrame()

        # 3. Merge Live Data with Context
        enriched_games = []
        for game in games:
            game_id = game["gameId"]

            # Extract basic live state
            live_state = {
                "game_id": game_id,
                "status": game["gameStatus"],  # 1=Scheduled, 2=Live, 3=Final
                "period": game["period"],
                "clock": self._parse_nba_clock(game["gameClock"]),
                "home_team": game["homeTeam"]["teamName"],
                "away_team": game["awayTeam"]["teamName"],
                "home_score": game["homeTeam"]["score"],
                "away_score": game["awayTeam"]["score"],
                "home_id": game["homeTeam"]["teamId"],
                "away_id": game["awayTeam"]["teamId"],
                # Quarter Scores (if available in this endpoint breakdown)
                # Scoreboard endpoint usually gives total score + period info.
                # For specific Quarter/Half scores we might need deep inspection or boxscore.
                # Assuming 'periods' dict exists in game payload?
                # Usually: game['homeTeam']['periods'] = [{'period':1, 'score': 25}, ...]
            }

            # Enrich with Quarter Scores if available
            # Note: We need this for "Tired Lead" strategy (Margin at Q4 Start)
            # nba_api scoreboard periods structure:
            # homeTeam: { periods: [{period:1, score:25}, ...]}

            periods_home = game["homeTeam"].get("periods", [])
            periods_away = game["awayTeam"].get("periods", [])

            live_state["periods_home"] = periods_home
            live_state["periods_away"] = periods_away

            # Attach Context (Lookups in Polars DF)
            # Find context rows for this game
            if not self.cached_context.is_empty():
                home_ctx = self.cached_context.filter(
                    (pl.col("game_id") == game_id) & (pl.col("is_home") == 1)
                )
                away_ctx = self.cached_context.filter(
                    (pl.col("game_id") == game_id) & (pl.col("is_home") == 0)
                )

                if not home_ctx.is_empty():
                    live_state["home_context"] = home_ctx.to_dicts()[0]
                if not away_ctx.is_empty():
                    live_state["away_context"] = away_ctx.to_dicts()[0]

            enriched_games.append(live_state)

        return enriched_games

    def _is_new_day(self) -> bool:
        if not self.last_context_update:
            return True
        return self.last_context_update.date() < datetime.now().date()

    def _parse_nba_clock(self, clock_str: str) -> str:
        """Parses ISO 8601 duration (PT12M00.00S) or raw string to MM:SS."""
        if not clock_str:
            return ""

        # If already simple format like "12:00", return as is
        if ":" in clock_str and "PT" not in clock_str:
            return clock_str

        import re

        # Regex for PT#M#S pattern
        match = re.search(r"PT(\d+)M(\d+(\.\d+)?)S", clock_str)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return f"{minutes:02d}:{int(seconds):02d}"

        return clock_str
