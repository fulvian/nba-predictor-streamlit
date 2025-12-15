import polars as pl
from datetime import date
from nba_predictor.utils.nba_timezone_utils import get_nba_games_official_api
from nba_predictor.core.data_store import UnifiedDataStore
import logging

logging.basicConfig(level=logging.INFO)


def restore_12_08():
    try:
        target_date = date(2025, 12, 8)
        print(f"Fetching games for {target_date}...")

        games = get_nba_games_official_api(target_date)

        if not games:
            print("No games found from Official API!")
            return

        print(f"Found {len(games)} games.")

        # Transform for storage
        games_data = []
        for g in games:
            # SANITIZE TIME HERE TOO just in case
            g_time = g.get("game_time")
            if g_time == "TBD":
                g_time = None

            games_data.append(
                {
                    "game_id": g.get("game_id"),
                    "game_date": target_date.strftime("%Y-%m-%d"),
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "season": "2025-26",
                    "game_time": g_time,
                    "status": g.get("status", "Scheduled"),
                    "home_score": g.get("home_score", 0),
                    "away_score": g.get("away_score", 0),
                    "match_id": g.get("match_id"),
                    "source": "Manual Restore",
                }
            )
            print(f"- {g.get('home_team')} vs {g.get('away_team')}")

        df = pl.DataFrame(games_data)

        ds = UnifiedDataStore(base_path="data")
        ds.store_games_data(df, target_date.strftime("%Y-%m-%d"))
        print("Successfully stored games parquet.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    restore_12_08()
