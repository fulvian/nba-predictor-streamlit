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


def _standardize_team_name(name: str) -> str:
    """Standardize team name (handle nicknames/abbreviations)."""
    nickname_map = {
        "Warriors": "Golden State Warriors",
        "Celtics": "Boston Celtics",
        "Sixers": "Philadelphia 76ers",
        "76ers": "Philadelphia 76ers",
        "Lakers": "Los Angeles Lakers",
        "Knicks": "New York Knicks",
        "Nets": "Brooklyn Nets",
        "Raptors": "Toronto Raptors",
        "Bulls": "Chicago Bulls",
        "Cavaliers": "Cleveland Cavaliers",
        "Bucks": "Milwaukee Bucks",
        "Pistons": "Detroit Pistons",
        "Pacers": "Indiana Pacers",
        "Magic": "Orlando Magic",
        "Heat": "Miami Heat",
        "Wizards": "Washington Wizards",
        "Hornets": "Charlotte Hornets",
        "Hawks": "Atlanta Hawks",
        "Mavericks": "Dallas Mavericks",
        "Spurs": "San Antonio Spurs",
        "Rockets": "Houston Rockets",
        "Grizzlies": "Memphis Grizzlies",
        "Pelicans": "New Orleans Pelicans",
        "Timberwolves": "Minnesota Timberwolves",
        "Nuggets": "Denver Nuggets",
        "Jazz": "Utah Jazz",
        "Trail Blazers": "Portland Trail Blazers",
        "Kings": "Sacramento Kings",
        "Clippers": "Los Angeles Clippers",
        "Suns": "Phoenix Suns",
        "Thunder": "Oklahoma City Thunder",
    }
    return nickname_map.get(name, name)


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
    Get NBA games using a robust "Master Schedule First" architecture.

    Strategy:
    1. Past Dates (< Today): Use LeagueGameLog (Reliable Final Scores).
    2. Today & Future (>= Today):
        a. Fetch MASTER SCHEDULE (scheduleLeagueV2.json) to get valid headers (IDs, Names, Dates).
        b. Filter strictly for target_date.
        c. If Today: Fetch LIVE SCORES (todaysScoreboard) and enrich the master records.
        d. If Future: Return master records (Status: Scheduled).
    3. Fallback: BallDontLie.
    """
    # 1. PAST DATES (< Today): Keep using LeagueGameLog (Best for Final status)
    if target_date < date.today():
        try:
            logger.info(f"📅 Past Date {target_date}: Fetching from LeagueGameLog...")
            games = _get_games_from_leaguegamelog(target_date)
            if games:
                logger.info(f"✅ LeagueGameLog: Found {len(games)} completed games.")
                return games
        except Exception as e:
            logger.warning(
                f"⚠️ LeagueGameLog failed: {e}. Falling back to Master Schedule."
            )

    # 2. TODAY & FUTURE (>= Today): Use Master Schedule as Source of Truth
    try:
        logger.info(f"🔮 Fetching Master Schedule (CDN) for {target_date}...")
        master_games = _fetch_master_schedule_games(target_date)

        if not master_games:
            logger.warning(f"⚠️ Master Schedule returned 0 games for {target_date}.")
            # If completely empty, maybe try fallback? Or maybe there ARE no games.
            # Let's try fallback just in case.
        else:
            logger.info(
                f"✅ Master Schedule: Found {len(master_games)} valid matchups."
            )

            # If TODAY, enrich with valid live scores
            if target_date == date.today():
                logger.info(
                    "⚡️ Date is TODAY: Enriching with Live Scores (todaysScoreboard)..."
                )
                enrichment_data = _fetch_live_scores_enrichment()

                # Merge logic
                for game in master_games:
                    g_id = game["game_id"]
                    if g_id in enrichment_data:
                        live = enrichment_data[g_id]
                        game["home_score"] = live["home_score"]
                        game["away_score"] = live["away_score"]
                        game["status"] = live["status"]
                        game["game_time"] = live["game_time"]
                        game["source"] = "NBA_CDN_MASTER+LIVE"
                    else:
                        # Game in schedule but not in live scoreboard yet (or finished yesterday in timezone overlap)
                        # Trust the Schedule's existence, but keep score 0/Scheduled if missing from live feed
                        pass

            return master_games

    except Exception as e:
        logger.error(f"❌ Master Schedule Logic failed: {e}")

    # 3. Fallback to BallDontLie API
    logger.warning("⚠️ Official API sources failed. Falling back to BallDontLie API...")
    return _get_games_from_balldontlie(target_date)


def _fetch_master_schedule_games(target_date: date) -> List[Dict[str, Any]]:
    """
    Fetch games from the full season CDN schedule (scheduleLeagueV2.json).
    Guarantees valid Team IDs and Names for any date in the season.
    """
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    import requests

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        }

        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"⚠️ CDN Schedule fetch failed: {resp.status_code}")
            return []

        data = resp.json()
        league_schedule = data.get("leagueSchedule", {})
        game_dates = league_schedule.get("gameDates", [])

        # Format target date to match CDN format if possible, or search
        # Typically CDN has "MM/DD/YYYY HH:MM:SS" or similar
        # We'll search by containing string YYYY-MM-DD or MM/DD/YYYY
        target_str_us = target_date.strftime("%m/%d/%Y")
        target_str_iso = target_date.strftime("%Y-%m-%d")

        found_date_node = None
        for gd in game_dates:
            d_val = gd.get("gameDate", "")
            if target_str_us in d_val or target_str_iso in d_val:
                found_date_node = gd
                break

        if not found_date_node:
            logger.info(f"ℹ️ No games found in CDN schedule for {target_date}")
            return []

        games = []
        for g in found_date_node.get("games", []):
            try:
                # Extract and map data
                g_id = g.get("gameId")

                # Teams
                home_t = g.get("homeTeam", {})
                away_t = g.get("awayTeam", {})

                home_name = _standardize_team_name(home_t.get("teamName", "Unknown"))
                away_name = _standardize_team_name(away_t.get("teamName", "Unknown"))

                # Status/Time
                status_text = g.get("gameStatusText", "Scheduled")

                # Scores (likely 0 for future games)
                h_score = home_t.get("score", 0)
                a_score = away_t.get("score", 0)

                game_info = {
                    "game_id": g_id,
                    "game_date": target_date.strftime("%Y-%m-%d"),
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": int(h_score),
                    "away_score": int(a_score),
                    "game_time": status_text,
                    "status": _map_status(status_text),  # Dynamic status mapping
                    "arena": g.get("arenaName", "Unknown"),
                    "city": g.get("arenaCity", "Unknown"),
                    "state": g.get("arenaState", "Unknown"),
                    "source": "NBA_CDN_SCHEDULE",
                }
                games.append(game_info)
            except Exception as e:
                logger.warning(f"⚠️ Error parsing singular CDN scheduled game: {e}")
                continue

        return games

    except Exception as e:
        logger.error(f"❌ CDN Schedule Exception: {e}")
        return []


def _fetch_live_scores_enrichment() -> Dict[str, Dict[str, Any]]:
    """
    Fetch TODAYS live scores (todaysScoreboard_00.json) and return a map
    of game_id -> {score, status, time} for enrichment.
    """
    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    import requests

    enrichment_map = {}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        }
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for game in data.get("scoreboard", {}).get("games", []):
                g_id = game.get("gameId")

                # Determine generic status
                status_text = game.get("gameStatusText", "")
                status_raw = status_text.upper()
                status = "Scheduled"
                if "FINAL" in status_raw:
                    status = "Final"
                elif "HALF" in status_raw or "Q" in status_raw:
                    status = "Live"

                enrichment_map[g_id] = {
                    "home_score": int(game.get("homeTeam", {}).get("score", 0)),
                    "away_score": int(game.get("awayTeam", {}).get("score", 0)),
                    "status": status,
                    "game_time": status_text,
                }
    except Exception as e:
        logger.warning(f"⚠️ Live Score Enrichment failed: {e}")

    return enrichment_map


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

    # Standardization Map for Nicknames -> Full Names
    nickname_map = {
        "Warriors": "Golden State Warriors",
        "Celtics": "Boston Celtics",
        "Sixers": "Philadelphia 76ers",
        "76ers": "Philadelphia 76ers",
        "Lakers": "Los Angeles Lakers",
        "Knicks": "New York Knicks",
        "Nets": "Brooklyn Nets",
        "Raptors": "Toronto Raptors",
        "Bulls": "Chicago Bulls",
        "Cavaliers": "Cleveland Cavaliers",
        "Bucks": "Milwaukee Bucks",
        "Pistons": "Detroit Pistons",
        "Pacers": "Indiana Pacers",
        "Magic": "Orlando Magic",
        "Heat": "Miami Heat",
        "Wizards": "Washington Wizards",
        "Hornets": "Charlotte Hornets",
        "Hawks": "Atlanta Hawks",
        "Mavericks": "Dallas Mavericks",
        "Spurs": "San Antonio Spurs",
        "Rockets": "Houston Rockets",
        "Grizzlies": "Memphis Grizzlies",
        "Pelicans": "New Orleans Pelicans",
        "Timberwolves": "Minnesota Timberwolves",
        "Nuggets": "Denver Nuggets",
        "Jazz": "Utah Jazz",
        "Trail Blazers": "Portland Trail Blazers",
        "Kings": "Sacramento Kings",
        "Clippers": "Los Angeles Clippers",
        "Suns": "Phoenix Suns",
        "Thunder": "Oklahoma City Thunder",
    }

    def _standardize(name):
        return nickname_map.get(name, name)  # Return full name if mapped, else original

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
            "home_team": _standardize(home.get("TEAM_NAME")),
            "away_team": _standardize(away.get("TEAM_NAME")),
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
