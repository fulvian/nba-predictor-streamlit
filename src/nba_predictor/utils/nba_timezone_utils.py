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
    Prioritizes LeagueGameLog for past dates to ensure 'Final' status.

    Args:
        target_date: Date to fetch games for

    Returns:
        List of NBA games with official schedule information
    """
    # 0. Check if date is in the past (yesterday or older)
    # If so, try LeagueGameLog first as it's more reliable for completed games status
    if target_date < date.today():
        try:
            logger.info(
                f"📅 Date {target_date} is in the past. Attempting LeagueGameLog fetch first..."
            )
            games = _get_games_from_leaguegamelog(target_date)
            if games:
                logger.info(
                    f"✅ LeagueGameLog: Found {len(games)} completed games for {target_date}"
                )
                return games
        except Exception as e:
            logger.warning(
                f"⚠️ LeagueGameLog fetch failed: {e}. Falling back to ScoreboardV2."
            )

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


def _get_games_from_leaguegamelog(target_date: date) -> List[Dict[str, Any]]:
    """
    Fetch games from LeagueGameLog for robust 'Final' status on past dates.
    """
    from nba_api.stats.endpoints import leaguegamelog

    # Determine season (e.g. 2025-12-04 -> 2025-26)
    # Simple logic: if month > 9, use year-(year+1), else (year-1)-year
    year = target_date.year
    if target_date.month >= 10:
        season = f"{year}-{str(year + 1)[-2:]}"
    else:
        season = f"{year - 1}-{str(year)[-2:]}"

    logger.info(f"🔍 Fetching LeagueGameLog for season {season}...")
    log = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="T", timeout=30
    )
    data = log.get_normalized_dict().get("LeagueGameLog", [])

    # Filter for target date
    target_str = target_date.strftime("%Y-%m-%d")
    daily_games = [g for g in data if g.get("GAME_DATE") == target_str]

    if not daily_games:
        return []

    # Map game_id to team entries
    games_map = {}
    for entry in daily_games:
        gid = entry["GAME_ID"]
        if gid not in games_map:
            games_map[gid] = []
        games_map[gid].append(entry)

    formatted_games = []

    for gid, entries in games_map.items():
        if len(entries) < 2:
            continue  # Incomplete data

        # Identify Home/Away
        # MATCHUP: "GSW @ PHI" -> GSW is Away
        # MATCHUP: "GSW vs. PHI" -> GSW is Home (usually)

        team_a = entries[0]
        team_b = entries[1]

        if "@" in team_a["MATCHUP"]:
            away = team_a
            home = team_b
        else:
            home = team_a
            away = team_b

        game_info = {
            "game_id": gid,
            "game_date": target_str,
            "home_team": home.get("TEAM_NAME"),
            "away_team": away.get("TEAM_NAME"),
            "home_score": int(home.get("PTS", 0)),
            "away_score": int(away.get("PTS", 0)),
            "game_time": "Final",
            "status": "Final",
            "arena": "Unknown",
            "city": "Unknown",
            "state": "Unknown",
            "source": "LeagueGameLog",
        }
        formatted_games.append(game_info)

    return formatted_games


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
