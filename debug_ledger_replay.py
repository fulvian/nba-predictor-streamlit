import duckdb
import pandas as pd

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75


def trace_bankroll():
    print(f"--- BANKROLL TRACE START ---")
    print(f"INITIAL DEPOSIT: {INITIAL_DEPOSIT:.2f}")

    conn = duckdb.connect(DB_PATH)
    bets = conn.execute("SELECT * FROM bets ORDER BY created_at ASC").df()

    current_free = INITIAL_DEPOSIT
    current_locked = 0.0

    print(
        f"\n{'DATE':<20} | {'ACTION':<10} | {'STATUS':<8} | {'AMOUNT':<6} | {'ODDS':<5} | {'PAYOUT':<6} | {'FREE':<8} | {'LOCKED':<8} | {'TOTAL':<8}"
    )
    print("-" * 110)

    for _, bet in bets.iterrows():
        amount = float(bet["amount"])
        status = bet["status"]
        odds = float(bet["odds"])
        created_at = str(bet["created_at"])[:19]

        # 1. Place Bet
        current_free -= amount

        # Track locked funds
        is_pending = False
        if status == "PENDING":
            current_locked += amount
            is_pending = True

        # 2. Result
        payout = 0.0
        if status == "WON":
            payout = amount * odds
            current_free += payout
        elif status in ["VOID", "PUSH", "void"]:
            payout = amount
            current_free += payout

        # Total Equity check
        # Note: If pending, Total = Free + Locked.
        # If settled, Locked is 0 for this bet (it's gone), money is either lost or in Free.
        # But 'current_locked' is a global accumulator? No, we should track active bets.
        # Simplification for trace:
        # Total Equity = Current Free + (Sum of ALL CURRENTLY PENDING bets).
        # We need to calculate 'Total Equity' *at this moment*.
        # But 'status' in DB is the *current* status, not status *at that time*.
        # So we can simply sum up the 'value' of the account.

        # Better visual approach:
        # Ledger Balance (Free) changes.
        # Locked Balance is just a bucket for Pending.

        # We will separate the steps to show the flow
        # Step A: Deduction
        print(
            f"{created_at:<20} | PLACE BET  | {status:<8} | -{amount:<5.2f} | {odds:<5.2f} | {'-':<6} | {current_free:<8.2f} | {'+' + str(amount):<8} | {current_free + current_locked if is_pending else '?'}"
        )

        # Step B: Payout (if any)
        if payout > 0:
            print(
                f"{'':<20} | SETTLEMENT | {status:<8} | {'-':<6} | {'-':<5} | +{payout:<5.2f} | {current_free:<8.2f} | {'-':<8} | {current_free + (current_locked if is_pending else 0)}"
            )

    # Final Calculation of current state
    real_locked = (
        conn.execute("SELECT SUM(amount) FROM bets WHERE status='PENDING'").fetchone()[
            0
        ]
        or 0.0
    )
    print("-" * 110)
    print(f"FINAL CALCULATED STATE:")
    print(f"Free Bankroll: {current_free:.2f}")
    print(f"Locked Funds:  {real_locked:.2f}")
    print(f"Total Equity:  {current_free + float(real_locked):.2f}")

    # Analyze Net P&L
    won_bets = bets[bets["status"] == "WON"]
    lost_bets = bets[bets["status"] == "LOST"]
    print(f"\nStats:")
    print(
        f"Won Bets: {len(won_bets)} (Payouts: {sum(won_bets['amount'] * won_bets['odds']):.2f})"
    )
    print(f"Lost Bets: {len(lost_bets)} (Losses: {sum(lost_bets['amount']):.2f})")
    conn.close()


if __name__ == "__main__":
    trace_bankroll()
