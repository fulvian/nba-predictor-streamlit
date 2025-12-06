import sys
import os
import json
from datetime import datetime, timedelta, date
from dateutil import parser, tz

# Add src to path
sys.path.append(os.path.abspath("src"))

# Import Providers
from nba_predictor.api.data_provider import NBADataProvider
from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api


def debug_schedule():
    print(f"--- DEBUG SCHEDULE: {datetime.now()} ---")
    provider = NBADataProvider()

    # Range: Today + 3 days
    start_date = date.today()
    days_to_check = 3

    print(
        f"\nChecking range: {start_date} to {start_date + timedelta(days=days_to_check)}"
    )

    # 1. Test The Odds API Raw Logic (Mock if needed, but provider has key)
    print("\n--- 1. THE ODDS API (Raw Check) ---")
    try:
        games = provider._get_odds_api_games(days_ahead=days_to_check)
        for g in games[:5]:  # Show first 5
            print(f"ID: {g['game_id']}")
            print(f"  Matchup: {g['away_team']} @ {g['home_team']}")
            print(f"  UTC: {g['time_utc']}")
            print(f"  Calc Date: {g['date']} (Should be ET)")
            print(f"  Calc Time: {g['time']}")
    except Exception as e:
        print(f"Error checking Odds API: {e}")

    # 2. Test Provider Logic (The one we fixed)
    print("\n--- 2. NBA DATA PROVIDER (get_scheduled_games) ---")
    try:
        games = provider.get_scheduled_games(days_ahead=days_to_check)
        if not games:
            print("  No games found via Provider.")
        for g in games:
            # Only show ones in our target date range to avoid noise
            if str(g["date"]) >= str(start_date) and str(g["date"]) <= str(
                start_date + timedelta(days=days_to_check)
            ):
                print(
                    f"  [{g.get('source', '?')}] {g['away_team']} @ {g['home_team']} | Date: {g['date']} | Time: {g.get('time', '')}"
                )
    except Exception as e:
        print(f"Error checking Provider: {e}")


if __name__ == "__main__":
    debug_schedule()
