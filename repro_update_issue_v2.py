import sys
import logging
from datetime import date
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro_v2")

from nba_predictor.utils import nba_timezone_utils
from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api

# Access protected function for testing
_fetch_master_schedule_games = nba_timezone_utils._fetch_master_schedule_games


def test_fallback_behavior():
    target_date = date(2025, 12, 9)
    print(f"Testing Fallback (Master Schedule) for {target_date}...")

    # Check what Master Schedule returns for yesterday
    games = _fetch_master_schedule_games(target_date)
    print(f"Found {len(games)} games from Master Schedule (CDN)")
    for g in games:
        print(f"Game: {g.get('home_team')} vs {g.get('away_team')}")
        print(f"  Status: {g.get('status')} (Raw Time: {g.get('game_time')})")
        print(f"  Score: {g.get('home_score')} - {g.get('away_score')}")

    # Now Force LeagueGameLog to fail to verify get_nba_games_official_api behavior
    print("\nSimulating LeagueGameLog Failure...")

    # Duck punch the module to break _get_games_from_leaguegamelog
    original_func = nba_timezone_utils._get_games_from_leaguegamelog
    nba_timezone_utils._get_games_from_leaguegamelog = lambda d: (_ for _ in ()).throw(
        Exception("Mock Failure")
    )

    try:
        games = get_nba_games_official_api(target_date)
        print(f"Result after failure: {len(games)} games")
        for g in games:
            print(f"  Status: {g.get('status')}")
            if g.get("status") == "Scheduled":
                print(
                    "  Confirmed: Fallback returns 'Scheduled' for past games -> BUG CONFIRMED"
                )
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Restore
        nba_timezone_utils._get_games_from_leaguegamelog = original_func


if __name__ == "__main__":
    test_fallback_behavior()
