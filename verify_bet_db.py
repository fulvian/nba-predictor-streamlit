import duckdb
import pandas as pd
from pathlib import Path

db_path = "data/nba_betting.duckdb"

try:
    conn = duckdb.connect(db_path)
    # Query the most recent bet
    query = """
        SELECT bet_id, user_id, game_id, bet_type, amount, odds, status, created_at, prediction
        FROM bets
        ORDER BY created_at DESC
        LIMIT 1
    """
    result = conn.execute(query).fetchone()

    if result:
        columns = [
            "bet_id",
            "user_id",
            "game_id",
            "bet_type",
            "amount",
            "odds",
            "status",
            "created_at",
            "prediction",
        ]
        bet_data = dict(zip(columns, result))
        print("\n✅ MOST RECENT BET FOUND:")
        for key, value in bet_data.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ NO BETS FOUND IN DATABASE")

    conn.close()

except Exception as e:
    print(f"\n❌ ERROR QUERYING DATABASE: {e}")
