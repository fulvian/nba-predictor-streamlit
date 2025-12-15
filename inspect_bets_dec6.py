import duckdb
import pandas as pd


def inspect_bets():
    db_path = "data/nba_betting.duckdb"
    try:
        conn = duckdb.connect(db_path)
        # Using game_id to filter for the BDL ones or Dec 6
        # Date format in DB might be text?
        query = "SELECT bet_id, game_id, bet_type, home_team, away_team FROM bets WHERE game_id LIKE 'BDL_%'"
        df = conn.execute(query).fetchdf()
        print(df)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_bets()
