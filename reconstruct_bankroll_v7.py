import duckdb
import uuid
import os
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V7 - NUCLEAR OPTION")

    DB_PATH = "data/nba_bankroll_v3.duckdb"
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("🗑️ Deleted existing DB file.")
        except Exception as e:
            print(f"⚠️ Could not delete file: {e}")

    bk_conn = duckdb.connect(DB_PATH)
    bet_conn = duckdb.connect("data/nba_betting.duckdb")

    # 2. Init
    start_time = datetime.now() - timedelta(days=7)
    init_val = 84.75
    bk_conn.execute("""
        CREATE TABLE transactions (
            transaction_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            transaction_type VARCHAR,
            amount DECIMAL(15,2),
            balance_after DECIMAL(15,2),
            description VARCHAR,
            bet_id VARCHAR,
            game_id VARCHAR,
            metadata JSON,
            checksum VARCHAR,
            created_at TIMESTAMP
        );
    """)
    bk_conn.execute("""
        CREATE TABLE bankroll_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            total_equity DECIMAL(15,2),
            cash_balance DECIMAL(15,2),
            reserved_funds DECIMAL(15,2),
            pending_exposure DECIMAL(15,2),
            metadata JSON
        );
    """)

    bk_conn.execute(
        """
        INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            f"txn_init_{uuid.uuid4().hex}",
            start_time,
            "DEPOSIT",
            init_val,
            init_val,
            "Init",
            None,
            None,
            "{}",
            "REC_V7",
            start_time,
        ),
    )

    # 3. Fetch bets
    bets = bet_conn.execute(
        "SELECT bet_id, result, amount, profit_loss, created_at, status, prediction FROM bets ORDER BY created_at ASC"
    ).fetchall()

    running_bal = init_val

    for b in bets:
        bid, res, amt, profit, date, status, pred = b

        stake = float(amt) if amt is not None else 0.0
        prof = float(profit) if profit is not None else 0.0

        # PLACEMENT
        running_bal -= stake
        bk_conn.execute(
            """
            INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                f"txn_p_{uuid.uuid4().hex}",
                date,
                "BET_PLACEMENT",
                stake,
                running_bal,
                f"Place {pred}",
                bid,
                None,
                "{}",
                "REC_V7",
                date,
            ),
        )

        # SETTLEMENT
        if status in ["Settled", "WON", "LOST", "VOID", "PUSH"]:
            payout = 0.0
            if res == "WON":
                payout = stake + prof
            elif res == "VOID" or res == "PUSH":
                payout = stake

            running_bal += payout

            bk_conn.execute(
                """
                INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    f"txn_s_{uuid.uuid4().hex}",
                    date,
                    "BET_SETTLEMENT",
                    payout,
                    running_bal,
                    f"Settle {res}",
                    bid,
                    None,
                    "{}",
                    "REC_V7",
                    date,
                ),
            )

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
    print(f"\n💰 V7 INTERNAL CHECK: {final_val}")

    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
