"""
NBA Timezone Utilities
Utility functions for NBA Official API integration.
"""

import os
import time
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any

# NBA API Imports
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.library.parameters import LeagueID
from nba_api.stats.static import teams as nba_teams

# Internal Imports
from nba_predictor.api.ball_dont_lie_client import NBABallDontLieClient

logger = logging.getLogger(__name__)

# Configure NBA API Headers globally to avoid timeouts
# Try different import paths for NBAHTTP as it varies by version
try:
    try:
        from nba_api.stats.library.http import NBAHTTP
    except ImportError:
        from nba_api.library.http import NBAHTTP

    # Robust Headers matching browser behavior
    custom_headers = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://stats.nba.com/",
        "Origin": "https://stats.nba.com",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
        "Upgrade-Insecure-Requests": "1",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

    if hasattr(NBAHTTP, "headers"):
        if NBAHTTP.headers is None:
            NBAHTTP.headers = custom_headers
        else:
            NBAHTTP.headers.update(custom_headers)
        logger.info("✅ NBA API Headers configured for robustness")
    else:
        logger.warning("⚠️ NBAHTTP class does not have 'headers' attribute")

except ImportError:
    logger.warning("⚠️ Could not configure NBA API headers directly (NBAHTTP not found)")
except Exception as e:
    logger.warning(f"⚠️ Error configuring NBA API headers: {e}")


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Retry decorator with exponential backoff.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"❌ All {retries} retries failed: {e}")
                        raise e
                    sleep = backoff_in_seconds * 2**x + (time.time() % 1)  # Add jitter
                    logger.warning(
                        f"⚠️ Request failed: {e}. Retrying in {sleep:.2f}s..."
                    )
                    time.sleep(sleep)
                    x += 1

        return wrapper

    return decorator


@retry_with_backoff(retries=3, backoff_in_seconds=2)
def get_nba_games_official_api(target_date: date) -> List[Dict[str, Any]]:
    """
    Get NBA games using official NBA.com API (ScoreboardV2) with fallback to BallDontLie.

    Args:
        target_date: Date to fetch games for

    Returns:
        List of NBA games with official schedule information
    """
    # 1. Try Official NBA API (ScoreboardV2)
    try:
        logger.info(
            f"🏀 NBA Official API: Fetching games for {target_date} using ScoreboardV2"
        )

        # Use ScoreboardV2 which is more reliable for historical/completed games
        board = scoreboardv2.ScoreboardV2(
            game_date=target_date.strftime("%Y-%m-%d"),
            league_id=LeagueID.nba,
            timeout=30,  # Increased timeout for robustness
        )

        # Get GameHeader and LineScore
        games_list = board.game_header.get_dict()["data"]
        line_score = board.line_score.get_dict()["data"]

        # Pre-fetch team info for name resolution
        nba_teams_list = nba_teams.get_teams()
        team_id_map = {team["id"]: team["full_name"] for team in nba_teams_list}

        if games_list:
            games = []

            # Create a map for line scores (scores are here)
            scores_map = {}
            for ls in line_score:
                # LineScore indices: GAME_ID=2, TEAM_ID=3, PTS=22
                game_id = ls[2]
                team_id = ls[3]
                pts = ls[22]
                if game_id not in scores_map:
                    scores_map[game_id] = {}
                scores_map[game_id][team_id] = pts

            for game_row in games_list:
                # GameHeader indices:
                # GAME_DATE_EST=0, GAME_SEQUENCE=1, GAME_ID=2, GAME_STATUS_ID=3,
                # GAME_STATUS_TEXT=4, GAMECODE=5, HOME_TEAM_ID=6, VISITOR_TEAM_ID=7

                game_id = game_row[2]
                home_team_id = game_row[6]
                away_team_id = game_row[7]

                # Get scores from map
                home_score = scores_map.get(game_id, {}).get(home_team_id, 0)
                away_score = scores_map.get(game_id, {}).get(away_team_id, 0)

                # Determine status
                status_text = game_row[4]
                status = _map_status(status_text)

                # Resolve Team Names using static map
                home_team_name = team_id_map.get(
                    home_team_id, f"Unknown ({home_team_id})"
                )
                away_team_name = team_id_map.get(
                    away_team_id, f"Unknown ({away_team_id})"
                )

                game_info = {
                    "game_id": game_id,
                    "game_date": target_date.strftime("%Y-%m-%d"),
                    "home_team": home_team_name,
                    "away_team": away_team_name,
                    "home_score": int(home_score or 0),
                    "away_score": int(away_score or 0),
                    "game_time": status_text,  # Use status text as time for now (e.g. "Final", "7:00 pm ET")
                    "status": status,
                    "arena": "Unknown",
                    "city": "Unknown",
                    "state": "Unknown",
                }
                games.append(game_info)

            logger.info(
                f"✅ NBA Official API: Found {len(games)} games for {target_date}"
            )
            return games
        else:
            logger.warning(f"❌ NBA Official API: No games found in response")

    except Exception as e:
        logger.error(f"❌ NBA Official API (ScoreboardV2) failed: {e}")
        # Continue to fallback

    # 2. Fallback to BallDontLie API
    logger.warning("⚠️ NBA Official API failed. Falling back to BallDontLie API...")
    return _get_games_from_balldontlie(target_date)


def _get_games_from_balldontlie(target_date: date) -> List[Dict[str, Any]]:
    """
    Fallback method to get games from BallDontLie API.
    """
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        logger.error("❌ BALLDONTLIE_API_KEY not found. Cannot use fallback.")
        return []

    try:
        client = NBABallDontLieClient(api_key)
        bdl_games = client.get_games_for_date_range(target_date)

        games = []
        for g in bdl_games:
            game_info = {
                "game_id": g.get("game_id"),
                "game_date": target_date.strftime("%Y-%m-%d"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_score": int(g.get("home_score", 0)),
                "away_score": int(g.get("away_score", 0)),
                "game_time": g.get("time", "TBD"),
                "status": g.get("status"),
                "arena": "Unknown",
                "city": "Unknown",
                "state": "Unknown",
            }
            games.append(game_info)

        logger.info(
            f"✅ BallDontLie Fallback: Found {len(games)} games for {target_date}"
        )
        return games

    except Exception as e:
        logger.error(f"❌ BallDontLie Fallback failed: {e}")
        return []


def _map_status(status_text: str) -> str:
    """Map NBA status text to standard status."""
    status_mapping = {
        "Final": "Final",
        "Completed": "Final",
        "Finished": "Final",
        "In Progress": "In Progress",
        "Scheduled": "Scheduled",
        "Pre-Game": "Scheduled",
        "Halftime": "In Progress",
    }
    # Handle "Final" or "Final/OT" etc
    if "Final" in status_text:
        return "Final"
    return status_mapping.get(status_text, status_text)


def test_nba_official_api() -> bool:
    """
    Test NBA Official API connection.
    """
    try:
        logger.info("🧪 Testing NBA Official API connection...")
        today = date.today()
        games = get_nba_games_official_api(today)

        if games is not None:
            logger.info("✅ NBA Official API connection successful")
            return True
        else:
            logger.error("❌ NBA Official API connection failed")
            return False

    except Exception as e:
        logger.error(f"❌ NBA Official API test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_nba_official_api()
    if success:
        print("🎉 NBA Official API is working correctly!")
    else:
        print("❌ NBA Official API test failed")
