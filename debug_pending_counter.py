import sys
import os

sys.path.append(os.path.abspath("src"))
from nba_predictor.utils.betting_database_manager import get_secure_database_manager


def check_pending():
    db = get_secure_database_manager()
    user_id = "test_user_001"
    print(f"Checking pending bets for {user_id}...")
    counts = db.safe_get_pending_bets_by_game(user_id)
    print(f"Pending Counts: {counts}")

    # Also check raw bets
    bets = db.safe_get_user_bets(user_id, status="PENDING")
    print(f"Total Pending Bets Logged: {len(bets)}")
    for b in bets:
        print(f" - Game {b.get('game_id')}: {b.get('amount')}")


if __name__ == "__main__":
    check_pending()
