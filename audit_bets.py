from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
import pandas as pd

db = BettingDatabaseManager()
try:
    print("--- User ID Distribution ---")
    query_dist = "SELECT user_id, COUNT(*) as count, SUM(profit_loss) as pl FROM bets GROUP BY user_id"
    dist = db.safe_execute_query(query_dist, fetch_all=True)
    print(pd.DataFrame(dist))

    print("\n--- All Bets for test_user_001 ---")
    query_user = "SELECT bet_id, game_id, amount, status, profit_loss FROM bets WHERE user_id = 'test_user_001'"
    user_bets = db.safe_execute_query(query_user, fetch_all=True)
    print(f"Total rows for test_user_001: {len(user_bets)}")
    # print(pd.DataFrame(user_bets)) # Optional: print all if small

    print("\n--- Global Stats from DB Manager Logic ---")
    # Simulate exactly what the dashboard calls
    summary = db.safe_get_user_summary("test_user_001")
    print(summary)

except Exception as e:
    print(f"Error: {e}")
finally:
    db.secure_close()
