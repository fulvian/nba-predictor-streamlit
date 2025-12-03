import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from nba_predictor.api.data_provider import NBADataProvider

logging.basicConfig(level=logging.INFO)


def test_provider():
    print("Initializing NBADataProvider...")
    provider = NBADataProvider()

    print("\nFetching games for next 3 days...")
    games = provider.get_scheduled_games(days_ahead=3)

    print(f"\nFound {len(games)} games.")
    for game in games:
        print(
            f"- {game['date']} {game['time']}: {game['away_team']} @ {game['home_team']} ({game['source']})"
        )


if __name__ == "__main__":
    test_provider()
