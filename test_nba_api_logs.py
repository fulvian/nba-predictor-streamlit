import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog


def get_season_logs():
    print("Fetching logs using nba_api...")
    try:
        # season='2024-25'
        log = leaguegamelog.LeagueGameLog(
            season="2024-25", player_or_team_abbreviation="T"
        )
        df = log.get_data_frames()[0]

        print(f"Successfully fetched {len(df)} rows.")
        print("Columns:", df.columns.tolist())
        print("First 5 rows:")
        print(df[["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]].head())

        return df

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_season_logs()
