import pandas as pd
import numpy as np


def inspect():
    print("--- Inspecting Simple Dataset ---")
    try:
        df = pd.read_csv("data/nba_simple_complete_dataset.csv")
        print(df[["AVG_PACE", "TOTAL_SCORE", "GAME_PACE"]].describe())
        print("Head:")
        print(df[["AVG_PACE", "TOTAL_SCORE", "GAME_PACE"]].head())
    except Exception as e:
        print(e)

    print("\n--- Inspecting Bets DB ---")
    try:
        import duckdb
        import json

        conn = duckdb.connect("data/nba_betting.duckdb", read_only=True)
        rows = conn.execute(
            "SELECT prediction FROM bets WHERE status IN ('WON','LOST','SETTLED') LIMIT 5"
        ).fetchall()
        for r in rows:
            print(r[0])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    inspect()
