from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
import pandas as pd

db = BettingDatabaseManager()
try:
    # 1. Find Test Bets
    # Looking for bets with game_id containing 'game_123' or 'REF_TEST'
    # or bet_id matching known test patterns

    query = """
    SELECT * FROM bets 
    WHERE game_id LIKE '%game_123%' 
       OR game_id LIKE '%GAME_123%'
       OR bet_id LIKE '%REF_TEST%'
       OR bet_id = 'ec549328-8735-46fa-8494-bbcca43d58b7'
    """

    test_bets = db.safe_execute_query(query, fetch_all=True)

    if test_bets:
        print(f"Found {len(test_bets)} test bets to remove:")
        for bet in test_bets:
            print(
                f"- ID: {bet['bet_id']}, Game: {bet['game_id']}, Amount: {bet['amount']}, Status: {bet['status']}, PL: {bet['profit_loss']}"
            )

            # Delete the bet
            # Note: We need user_id for safe_delete_bet
            # db.safe_delete_bet(bet['bet_id'], bet['user_id'])
            # safe_delete_bet might be restricted or logging oriented.
            # Direct delete via query is cleaner for this admin cleanup.

            delete_q = "DELETE FROM bets WHERE bet_id = ?"
            db.safe_execute_query(delete_q, (bet["bet_id"],), fetch_all=False)
            print(f"  -> DELETED")

    else:
        print("No test bets found matching criteria.")

    # 2. Verify Stats after cleanup
    print("\nRe-checking Stats Summary:")
    stats_query = """
        SELECT
            COUNT(*) as total_bets,
            SUM(CASE WHEN result = 'WON' THEN 1 ELSE 0 END) as won_bets,
            SUM(CASE WHEN result = 'LOST' THEN 1 ELSE 0 END) as lost_bets,
            SUM(profit_loss) as net_pl
        FROM bets
        WHERE status != 'PENDING'
    """
    stats = db.safe_execute_query(stats_query, fetch_one=True)
    print(stats)

except Exception as e:
    print(f"Error: {e}")
finally:
    if db:
        db.secure_close()
