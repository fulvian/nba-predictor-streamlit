import duckdb
import uuid
import os
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V10 - SCHEMA FIX")

    DB_PATH = "data/nba_bankroll_v3.duckdb"
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("🗑️ Deleted existing DB file.")
        except Exception as e:
            print(f"⚠️ Could not delete file: {e}")

    bk_conn = duckdb.connect(DB_PATH)
    bet_conn = duckdb.connect("data/nba_betting.duckdb")

    # --- 1. SCHEMA CREATION (MATCHING engine.py) ---
    print("🏗️ Creating Tables with Correct Schema...")

    # Transactions
    bk_conn.execute("""
        CREATE TABLE transactions (
            transaction_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            transaction_type VARCHAR NOT NULL,
            amount DECIMAL(15,2) NOT NULL,
            balance_after DECIMAL(15,2) NOT NULL,
            description VARCHAR NOT NULL,
            bet_id VARCHAR,
            game_id VARCHAR,
            metadata JSON,
            checksum VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # SNAPSHOTS (The culprit) - Matching engine.py
    bk_conn.execute("""
        CREATE TABLE bankroll_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            current_balance DECIMAL(15,2) NOT NULL,
            total_deposits DECIMAL(15,2) NOT NULL,
            total_withdrawals DECIMAL(15,2) NOT NULL,
            total_bets_placed DECIMAL(15,2) NOT NULL,
            total_bets_won DECIMAL(15,2) NOT NULL,
            total_bets_lost DECIMAL(15,2) NOT NULL,
            total_fees DECIMAL(15,2) NOT NULL,
            active_bets_count INTEGER NOT NULL,
            pending_bets_count INTEGER NOT NULL,
            last_transaction_timestamp TIMESTAMP,
            last_updated TIMESTAMP NOT NULL,
            checksum VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Bet Records (Engine expects this too, usually in the same DB if it initializes it?
    # Engine.py lines 90-107 creates bet_records in the SAME DB context.
    # So we should create it to avoid Engine trying to create it and failing if partially exists?
    # Or Engine uses IF NOT EXISTS.
    # Let's create it to be safe.
    bk_conn.execute("""
        CREATE TABLE bet_records (
            bet_id VARCHAR PRIMARY KEY,
            game_id VARCHAR NOT NULL,
            bet_type VARCHAR NOT NULL,
            selection VARCHAR NOT NULL,
            odds DECIMAL(10,4) NOT NULL,
            stake DECIMAL(15,2) NOT NULL,
            result VARCHAR NOT NULL,
            payout DECIMAL(15,2) NOT NULL,
            profit_loss DECIMAL(15,2) NOT NULL,
            placed_timestamp TIMESTAMP NOT NULL,
            settled_timestamp TIMESTAMP,
            metadata JSON,
            checksum VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # --- 2. POPULATE TRANSACTIONS (V9 LOGIC) ---
    print("▶️ Populating Transactions...")

    start_time = datetime.now() - timedelta(days=7)
    init_val = 84.75

    # Init Deposit
    bk_conn.execute(
        f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('txn_init_{uuid.uuid4().hex}', '{start_time}', 'DEPOSIT', {init_val}, {init_val}, 'Init', NULL, 'REC_V10', '{start_time}')"
    )

    # Fetch bets
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
            f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_PLACEMENT', {stake}, {running_bal}, 'Place {pred}', '{bid}', 'REC_V10', '{date}')"
        )

        # SETTLEMENT (Result Only Logic)
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
                    f"INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, checksum, created_at) VALUES ('{uuid.uuid4().hex}', '{date}', 'BET_SETTLEMENT', {payout}, {running_bal}, 'Settle {res}', '{bid}', 'REC_V10', '{date}')"
                )

    # --- 3. CREATE INITIAL SNAPSHOT (OPTIONAL BUT GOOD FOR ENGINE) ---
    # We won't compute the full snapshot stats here to keep it simple,
    # as the Engine falls back to transactions if latest snapshot is missing or old.
    # But to avoid empty table issues if any view depends on it:
    # We will let the Engine handle the first snapshot creation on next run if it does?
    # Engine.py doesn't seem to crash on empty snapshot table.
    # The error was "Table does not have column".
    # So creating empty table with correct schema is enough.

    # 4. Verify
    print(f"💰 Python Sum V10: {python_sum:.2f}")

    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
