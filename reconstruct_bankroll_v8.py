import duckdb
import uuid
import time
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V8 - VERBOSE CHECK")

    DB_PATH = "data/nba_bankroll_v3.duckdb"

    try:
        bk_conn = duckdb.connect(DB_PATH)
        bet_conn = duckdb.connect("data/nba_betting.duckdb")
    except Exception as e:
        print(f"❌ LOCKED: {e}")
        return

    # 1. Wipe
    print("🧹 Wiping...")
    bk_conn.execute("DELETE FROM transactions")
    bk_conn.execute("DELETE FROM bankroll_snapshots")

    # 2. Init
    start_time = datetime.now() - timedelta(days=7)
    init_val = 84.75
    bk_conn.execute(
        f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('txn_init', '{start_time}', 'DEPOSIT', {init_val}, {init_val}, 'Init', NULL, 'V8', '{start_time}')"
    )

    # 3. Fetch bets
    bets = bet_conn.execute(
        "SELECT bet_id, result, amount, profit_loss, created_at, status, prediction FROM bets ORDER BY created_at ASC"
    ).fetchall()

    running_bal = init_val
    python_sum = init_val

    count = 0
    for b in bets:
        bid, res, amt, profit, date, status, pred = b

        stake = float(amt) if amt is not None else 0.0
        prof = float(profit) if profit is not None else 0.0

        # PLACEMENT
        running_bal -= stake
        python_sum -= stake

        # Use F-String to avoid Binder issues
        bk_conn.execute(
            f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_PLACEMENT', {stake}, {running_bal}, 'Place {pred}', '{bid}', 'V8', '{date}')"
        )

        # SETTLEMENT
        if status in ["Settled", "WON", "LOST", "VOID", "PUSH"]:
            payout = 0.0
            if res == "WON":
                payout = stake + prof
                print(f"✅ WON: Stake {stake} + Prof {prof} = {payout}")
            elif res == "VOID" or res == "PUSH":
                payout = stake

            running_bal += payout
            python_sum += payout

            if payout > 0:
                # INSERT
                bk_conn.execute(
                    f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_SETTLEMENT', {payout}, {running_bal}, 'Settle {res}', '{bid}', 'V8', '{date}')"
                )

                # VERIFY IMMEDIATELY
                check_val = bk_conn.execute(
                    f"SELECT amount FROM transactions WHERE bet_id='{bid}' AND transaction_type='BET_SETTLEMENT'"
                ).fetchone()[0]
                if float(check_val) != payout:
                    print(f"❌ CRITICAL ERROR: Inserted {payout}, Read {check_val}")
                else:
                    if count < 3:  # Print first 3 valid checks
                        print(f"   MATCH: DB {check_val} == Py {payout}")
                        count += 1

    # 4. Verify Final Balance INTERNAL
    final_query = """
        SELECT SUM(
            CASE 
                WHEN transaction_type = 'DEPOSIT' THEN amount 
                WHEN transaction_type = 'BET_SETTLEMENT' THEN amount
                WHEN transaction_type = 'BET_PLACEMENT' THEN -amount
                ELSE 0 
            END
        ) as final_balance
        FROM transactions
    """
    final_val = bk_conn.execute(final_query).fetchone()[0]
    print(f"\n🐍 Python Sum: {python_sum:.2f}")
    print(f"🦆 DuckDB Sum: {final_val}")

    if abs(float(final_val) - python_sum) < 0.01:
        print("✅ SUCCESS: Python and DuckDB match.")
    else:
        print("❌ FAILURE: Mismatch.")

    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
