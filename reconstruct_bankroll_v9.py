import duckdb
import uuid
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V9 - RESULT ONLY")

    DB_PATH = "data/nba_bankroll_v3.duckdb"

    # 1. Connect
    try:
        bk_conn = duckdb.connect(DB_PATH)
        bet_conn = duckdb.connect("data/nba_betting.duckdb")
    except:
        print("❌ DB Locked")
        return

    # 2. Wipe
    bk_conn.execute("DELETE FROM transactions")
    bk_conn.execute("DELETE FROM bankroll_snapshots")

    # 3. Init
    start_time = datetime.now() - timedelta(days=7)
    init_val = 84.75
    bk_conn.execute(
        f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('txn_init', '{start_time}', 'DEPOSIT', {init_val}, {init_val}, 'Init', NULL, 'V9', '{start_time}')"
    )

    # 4. Fetch bets
    bets = bet_conn.execute(
        "SELECT bet_id, result, amount, profit_loss, created_at, status, prediction FROM bets ORDER BY created_at ASC"
    ).fetchall()

    running_bal = init_val
    python_sum = init_val

    for b in bets:
        bid, res, amt, profit, date, status, pred = b

        stake = float(amt) if amt is not None else 0.0
        prof = float(profit) if profit is not None else 0.0

        # PLACEMENT
        running_bal -= stake
        python_sum -= stake
        bk_conn.execute(
            f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_PLACEMENT', {stake}, {running_bal}, 'Place {pred}', '{bid}', 'V9', '{date}')"
        )

        # SETTLEMENT - RELY ON RESULT ONLY
        payout = 0.0
        should_settle = False

        if res == "WON":
            payout = stake + prof
            should_settle = True
        elif res == "VOID" or res == "PUSH":
            payout = stake
            should_settle = True
        elif res == "LOST":
            payout = 0.0
            should_settle = True

        if should_settle:
            running_bal += payout
            python_sum += payout

            if payout > 0:
                bk_conn.execute(
                    f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_SETTLEMENT', {payout}, {running_bal}, 'Settle {res}', '{bid}', 'V9', '{date}')"
                )

    # 5. Verify
    print(f"💰 Python Sum V9: {python_sum:.2f}")

    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
