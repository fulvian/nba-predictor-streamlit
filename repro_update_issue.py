import sys
import logging
from datetime import date, timedelta
import time

# Add src to path
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro_update_issue")

from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api
from nba_predictor.api.nba_official_client import NBAOfficialClient


def test_fetch_yesterday():
    # Target date: Yesterday (2025-12-09 in this context)
    target_date = date(2025, 12, 9)
    print(f"Testing fetch for {target_date}...")

    # 1. Test get_nba_games_official_api (The one used in dashboard)
    try:
        games = get_nba_games_official_api(target_date)
        print(f"Found {len(games)} games from get_nba_games_official_api")
        for g in games:
            print(f"Game: {g.get('home_team')} vs {g.get('away_team')}")
            print(f"  Status: {g.get('status')}")
            print(f"  Score: {g.get('home_score')} - {g.get('away_score')}")
            print(f"  Source: {g.get('source')}")

            if g.get("status") == "Scheduled" and target_date < date.today():
                print(
                    "  CRITICAL: Scheduled status for past date! This confirms the bug if LeagueGameLog failed."
                )
    except Exception as e:
        print(f"get_nba_games_official_api failed: {e}")


if __name__ == "__main__":
    test_fetch_yesterday()
