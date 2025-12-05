import duckdb
import pandas as pd
import json


def inspect_bets():
    db_path = "temp_debug.duckdb"
    try:
        conn = duckdb.connect(db_path)
        print(f"Connected to {db_path}")

        # Check Pending Bets
        print("\n--- PENDING BETS ---")
        pending_query = "SELECT bet_id, game_id, prediction, status, created_at FROM bets WHERE status = 'PENDING' LIMIT 10"
        pending_bets = conn.execute(pending_query).fetchdf()

        if not pending_bets.empty:
            print(f"Found {len(pending_bets)} pending bets.")
            # Print raw JSON for the first bet to understand structure
            first_pred = pending_bets.iloc[0]["prediction"]
            print(f"Raw Prediction JSON (First Bet): {first_pred}")
            for index, row in pending_bets.iterrows():
                try:
                    pred_data = json.loads(row["prediction"])
                    home_team = pred_data.get("home_team", "Unknown")
                    away_team = pred_data.get("away_team", "Unknown")
                    print(
                        f"Bet {row['bet_id'][:8]} | Game {row['game_id']} | {home_team} vs {away_team} | {row['created_at']}"
                    )
                except Exception as e:
                    print(f"Error parsing prediction for bet {row['bet_id']}: {e}")
        else:
            print("No pending bets found.")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_bets()
