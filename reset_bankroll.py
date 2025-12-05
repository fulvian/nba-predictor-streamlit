import json
from pathlib import Path
from src.nba_predictor.utils.betting_database_manager import (
    SecureBettingDatabaseManager,
)


def reset_bankroll():
    db = SecureBettingDatabaseManager()

    # 1. User Defined Constants
    INITIAL_CAPITAL = 84.75

    # 2. Get Real P&L from History
    query = """
        SELECT 
            SUM(CASE WHEN status != 'PENDING' THEN profit_loss ELSE 0 END) as net_pl,
            SUM(CASE WHEN status = 'PENDING' THEN amount ELSE 0 END) as pending_staked
        FROM bets
    """
    result = db.safe_execute_query(query, fetch_one=True)

    net_pl = result.get("net_pl") or 0.0
    pending_staked = result.get("pending_staked") or 0.0

    # 3. Calculate Target Bankroll (Total Net Worth)
    # User Formula: Start + P&L
    total_net_worth = INITIAL_CAPITAL + float(net_pl)

    # 4. Calculate Free Bankroll (Cash available to bet)
    # Free = Total - Pending Stakes
    free_bankroll = total_net_worth - pending_staked

    print(f"--- BANKROLL RECALCULATION ---")
    print(f"1. Initial Capital:       € {INITIAL_CAPITAL:.2f}")
    print(f"2. Database Net P&L:      € {net_pl:.2f}")
    print(f"------------------------------")
    print(f"3. TOTAL NET WORTH:       € {total_net_worth:.2f} (Should be ~97.54)")
    print(f"4. Less Pending Stakes:   € {pending_staked:.2f}")
    print(f"------------------------------")
    print(f"5. NEW FREE BANKROLL:     € {free_bankroll:.2f}")

    # 5. Update Bankroll File
    bankroll_path = Path("data/bankroll.json")

    # Create valid JSON structure
    bankroll_data = {
        "current_bankroll": free_bankroll,
        "initial_bankroll": INITIAL_CAPITAL,  # Store this for reference!
        "total_net_worth": total_net_worth,
    }

    with open(bankroll_path, "w") as f:
        json.dump(bankroll_data, f, indent=4)

    print(f"\n✅ Bankroll file updated at {bankroll_path}")
    print(f"   Value written: {free_bankroll:.2f}")


if __name__ == "__main__":
    reset_bankroll()
