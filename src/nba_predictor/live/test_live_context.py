import sys
import os

sys.path.append(os.getcwd())
from src.nba_predictor.live.context_loader import LiveContextLoader
from datetime import datetime


def test_loader():
    print("Initializing Context Loader...")
    loader = LiveContextLoader()

    # Mock Scheduled Games (Using IDs that exist in our dataset)
    # Denver Nuggets (1610612743) vs LA Lakers (1610612747)
    # Use real IDs from your dataset to ensure hits.

    mock_games = [
        {
            "gameId": "00999999",
            "homeTeam": {"teamId": 1610612743, "teamName": "Denver Nuggets"},  # Denver
            "awayTeam": {"teamId": 1610612747, "teamName": "Los Angeles Lakers"},
        }
    ]

    print(f"Testing for Mock Game: DEN vs LAL on {datetime.now().date()}")

    df_context = loader.load_todays_context(mock_games)

    print("\nCalculated Context:")
    print(df_context)

    # Basic Validation
    denver = df_context.filter(pl.col("team_id") == 1610612743)
    if not denver.is_empty():
        print("\nDenver Altitude Check:", denver["is_high_altitude"][0] == 1)
        print("Denver Rest:", denver["rest_days"][0])
        print("Denver Density:", denver["density_4d"][0])


if __name__ == "__main__":
    import polars as pl  # Ensure import for script

    test_loader()
