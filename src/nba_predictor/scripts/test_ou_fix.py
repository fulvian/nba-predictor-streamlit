#!/usr/bin/env python3
"""Test the fixed O/U extraction on a single game."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.nba_predictor.intelligence.odds_portal_scraper import OddsPortalScraper


def test_single_game():
    """Test O/U extraction on a known game."""
    # Known game with O/U data from 2020-2021
    test_url = "https://www.centroquote.it/basketball/usa/nba-2020-2021/los-angeles-lakers-detroit-pistons-fTv0BrDJ/"

    print(f"🧪 Testing O/U extraction on: {test_url}\n")

    scraper = OddsPortalScraper(headless=False)
    try:
        data = scraper.scrape_game_data(test_url)

        print(f"\n✅ Extraction Results:")
        print(f"   Home Team: {data.get('home_team')}")
        print(f"   Away Team: {data.get('away_team')}")
        print(f"   Score: {data.get('score_home')}-{data.get('score_away')}")
        print(f"   Date: {data.get('game_date')}")
        print(f"   Closing Lines Found: {len(data.get('closing_lines', []))}")

        for i, line in enumerate(data.get("closing_lines", []), 1):
            print(f"\n   Line {i}:")
            print(f"      Total: {line['line']}")
            print(f"      Over: {line['over_odds']}")
            print(f"      Under: {line['under_odds']}")

        if data.get("closing_lines"):
            print(f"\n🎉 SUCCESS! O/U extraction is working!")
            return True
        else:
            print(f"\n❌ FAIL! No O/U lines extracted.")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        scraper.close()


if __name__ == "__main__":
    success = test_single_game()
    sys.exit(0 if success else 1)
