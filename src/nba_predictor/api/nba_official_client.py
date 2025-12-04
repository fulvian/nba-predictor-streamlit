"""
NBA Official API Client
Robust client for NBA.com official API with proper rate limiting and retry logic.
This replaces BallDontLie API to avoid rate limiting issues.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class NBAOfficialClient:
    """
    Official NBA API client with robust error handling and rate limiting.
    """

    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Host": "stats.nba.com",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Referer": "https://stats.nba.com/",
                "Origin": "https://stats.nba.com",
                "Cache-Control": "max-age=0",
                "Upgrade-Insecure-Requests": "1",
                "x-nba-stats-origin": "stats",
                "x-nba-stats-token": "true",
            }
        )

        # Rate limiting: conservative approach
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with rate limiting and retry logic.
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=30)

                # Update rate limiting
                self.last_request_time = time.time()

                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        return None
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (attempt {attempt + 1}), waiting...")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                continue

        logger.error(f"All {self.max_retries} attempts failed for {endpoint}")
        return None

    def get_games_by_date(self, game_date: str) -> List[Dict]:
        """
        Get games scheduled for a specific date.
        """
        logger.info(f"Fetching NBA games for date: {game_date}")

        # Use scoreboardv2 endpoint (more reliable than scoreboard)
        endpoint = "scoreboardv2"
        params = {
            "GameDate": game_date,
            "LeagueID": "00",  # NBA
            "DayOffset": "0",
        }

        response_data = self._make_request(endpoint, params)
        if not response_data:
            return []

        games = []
        try:
            if "resultSets" in response_data and len(response_data["resultSets"]) > 0:
                game_set = response_data["resultSets"][0]
                if "rowSet" in game_set:
                    for game_row in game_set["rowSet"]:
                        game_info = {
                            "game_id": game_row.get("GAME_ID"),
                            "game_date": game_date,
                            "home_team": game_row.get("HOME_TEAM_NAME"),
                            "away_team": game_row.get("VISITOR_TEAM_NAME"),
                            "home_score": int(game_row.get("HOME_TEAM_SCORE", 0)),
                            "away_score": int(game_row.get("VISITOR_TEAM_SCORE", 0)),
                            "game_time": game_row.get("GAME_TIME_ET", "TBD"),
                            "status": self._determine_game_status(game_row),
                            "arena": game_row.get("ARENA_NAME"),
                            "city": game_row.get("CITY"),
                            "state": game_row.get("STATE"),
                            "match_id": self._generate_match_id(game_row, game_date),
                        }
                        games.append(game_info)
        except Exception as e:
            logger.error(f"Error parsing games response: {e}")

        logger.info(f"Found {len(games)} games for {game_date}")
        return games

    def get_game_results(self, game_date: str) -> List[Dict]:
        """
        Get completed game results for a specific date.
        """
        logger.info(f"Fetching NBA game results for date: {game_date}")

        # Use same endpoint but filter for completed games
        games = self.get_games_by_date(game_date)

        # Filter for completed games
        completed_games = []
        for game in games:
            if game["status"] in ["Final", "Completed", "Finished"]:
                completed_games.append(game)

        logger.info(f"Found {len(completed_games)} completed games for {game_date}")
        return completed_games

    def _determine_game_status(self, game_row: Dict) -> str:
        """
        Determine game status from NBA API response.
        """
        # NBA API uses different status codes
        game_status_code = game_row.get("GAME_STATUS_TEXT", "")

        # Map NBA status codes to our standard status
        status_mapping = {
            "Final": "Final",
            "Completed": "Final",
            "Finished": "Final",
            "In Progress": "In Progress",
            "Scheduled": "Scheduled",
            "Pre-Game": "Scheduled",
            "Halftime": "In Progress",
            "End of Q1": "In Progress",
            "End of Q2": "In Progress",
            "End of Q3": "In Progress",
            "End of Q4": "In Progress",
        }

        return status_mapping.get(game_status_code, "Scheduled")

    def _generate_match_id(self, game_row: Dict, game_date: str) -> str:
        """
        Generate canonical match ID for consistent identification.
        """
        home_team = game_row.get("HOME_TEAM_NAME", "").replace(" ", "_").upper()
        away_team = game_row.get("VISITOR_TEAM_NAME", "").replace(" ", "_").upper()

        return f"{game_date}_{away_team}_{home_team}"

    def get_team_stats(self, team_id: int, season: str = "2024-25") -> Optional[Dict]:
        """
        Get team statistics for a specific team and season.
        """
        logger.info(f"Fetching team stats for team {team_id}, season {season}")

        endpoint = "teamgamelogs"
        params = {
            "TeamID": team_id,
            "Season": season,
            "SeasonType": "Regular Season",
            "MeasureType": "Base",  # Basic stats
        }

        response_data = self._make_request(endpoint, params)
        if not response_data:
            return None

        try:
            if "resultSets" in response_data and len(response_data["resultSets"]) > 0:
                return response_data["resultSets"][0]
        except Exception as e:
            logger.error(f"Error parsing team stats response: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test API connection with a simple request.
        """
        logger.info("Testing NBA Official API connection...")

        # Try to get today's games as a test
        today = datetime.now().strftime("%Y-%m-%d")
        games = self.get_games_by_date(today)

        if games is not None:
            logger.info("✅ NBA Official API connection successful")
            return True
        else:
            logger.error("❌ NBA Official API connection failed")
            return False


# Convenience function for backward compatibility
def get_nba_official_client() -> NBAOfficialClient:
    """Get NBA Official API client instance."""
    return NBAOfficialClient()
