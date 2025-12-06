import logging
import sys
import os
from datetime import date, datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScheduleDebug")

# Add src to path
sys.path.append(os.path.abspath("src"))

from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api
from nba_predictor.api.data_provider import NBADataProvider


def debug_schedule():
    target_date_str = "2025-12-05"
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    print(f"--- DEBUGGING SCHEDULE FOR {target_date} ---")

    # 1. Test Official API
    print("\n[1] Testing Official API (ScoreboardV2)...")
    try:
        games = get_nba_games_official_api(target_date)
        if games:
            print(f"✅ Official API returned {len(games)} games:")
            for g in games:
                print(
                    f"   - {g['away_team']} @ {g['home_team']} ({g['game_time']}) [{g['status']}]"
                )
        else:
            print("❌ Official API returned NO games.")
    except Exception as e:
        print(f"❌ Official API Error: {e}")

    # 2. Test Data Provider Fallback flow
    print("\n[2] Testing DataProvider (Full Flow)...")
    provider = NBADataProvider()
    # Force specific date fetch
    # We need to access the private methods or use get_scheduled_games with date
    # But get_scheduled_games uses "days_ahead".
    # Let's inspect _get_odds_api_games behavior specifically.

    print("\n[3] Testing The Odds API (Directly)...")
    odds_games = provider._get_odds_api_games(days_ahead=3)
    # Filter for our date if logic does that
    relevant_odds = [g for g in odds_games if g["date"] == target_date_str]

    if relevant_odds:
        print(f"✅ Odds API for {target_date_str}: {len(relevant_odds)} games found")
        for g in relevant_odds:
            print(
                f"   - {g['away_team']} @ {g['home_team']} (Source: {g.get('source')})"
            )
    else:
        print(
            f"⚠️ Odds API returned games, but none for {target_date_str}. Total futures: {len(odds_games)}"
        )
        if odds_games:
            print(
                "Sample future game:",
                odds_games[0]["date"],
                odds_games[0]["away_team"],
                "@",
                odds_games[0]["home_team"],
            )


if __name__ == "__main__":
    debug_schedule()
