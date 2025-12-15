import sys
import os
from datetime import date
import logging

# Setup path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)

from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api

def verify():
    print("--- Verifying TODAY (2025-12-09) ---")
    today = date(2025, 12, 9)
    games_today = get_nba_games_official_api(today)
    
    found_heat_magic = False
    for g in games_today:
        print(f"Game: {g['away_team']} @ {g['home_team']} | Status: {g['status']} | Time: {g['game_time']}")
        if "Heat" in g['away_team'] or "Heat" in g['home_team']:
            if "Magic" in g['away_team'] or "Magic" in g['home_team']:
                found_heat_magic = True
                
    if found_heat_magic:
        print("✅ SUCCESS: Found Heat vs Magic for today!")
    else:
        print("❌ FAILURE: Did NOT find Heat vs Magic for today.")

    print("\n--- Verifying TOMORROW (2025-12-10) ---")
    tomorrow = date(2025, 12, 10)
    games_tomorrow = get_nba_games_official_api(tomorrow)
    
    found_unknown = False
    for g in games_tomorrow:
        print(f"Game: {g['away_team']} @ {g['home_team']} | Status: {g['status']}")
        if "Unknown" in g['away_team'] or "Unknown" in g['home_team']:
            found_unknown = True
            
    if not found_unknown and len(games_tomorrow) > 0:
        print("✅ SUCCESS: No Unknown teams found for tomorrow!")
    else:
        print(f"❌ FAILURE: Found Unknown teams or no games. Count: {len(games_tomorrow)}")

if __name__ == "__main__":
    verify()
