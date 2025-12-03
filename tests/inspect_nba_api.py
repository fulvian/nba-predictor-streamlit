
from nba_api.stats.endpoints import ScoreboardV2
import pandas as pd

def inspect():
    target_date = '2023-01-01'
    print(f"Fetching ScoreboardV2 for {target_date}...")
    scoreboard = ScoreboardV2(game_date=target_date)
    dfs = scoreboard.get_data_frames()
    
    print(f"Number of DataFrames: {len(dfs)}")
    
    if len(dfs) > 1:
        df = dfs[1]
        print(f"\n--- DataFrame 1 (LineScore) ---")
        print(f"Columns: {df.columns.tolist()}")
        if not df.empty:
            print(df.head(2))

if __name__ == "__main__":
    inspect()
