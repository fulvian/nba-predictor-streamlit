import duckdb
import uuid
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V5")
    bk_conn = duckdb.connect("data/nba_bankroll_v3.duckdb")
    bet_conn = duckdb.connect("data/nba_betting.duckdb")

    # 1. Wipe
    bk_conn.execute("DELETE FROM transactions")
    bk_conn.execute("DELETE FROM bankroll_snapshots")

    # 2. Init
    start_time = datetime.now() - timedelta(days=7)  # Fallback
    init_val = 84.75
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
            "REC_V5",
            start_time,
        ),
    )

    # 3. Fetch
    bets = bet_conn.execute(
        "SELECT bet_id, result, amount, profit_loss, created_at, status, prediction FROM bets ORDER BY created_at ASC"
    ).fetchall()

    running_bal = init_val
    for b in bets:
        bid, res, amt, profit, date, status, pred = b
        stake = float(amt) if amt else 0.0
        prof = float(profit) if profit else 0.0

        # Placement
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
                "REC_V5",
                date,
            ),
        )

        # Settlement
        if status in ["Settled", "WON", "LOST", "VOID", "PUSH"]:
            payout = 0.0
            if res == "WON":
                payout = stake + prof
                print(
                    f"WON: Amt={amt} (Type {type(amt)}) -> Stake={stake} -> Prof={prof} -> PAYOUT={payout}"
                )
            elif res == "VOID" or res == "PUSH":
                payout = stake

            running_bal += payout

            # FORCE INSERT correct payout
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
                    "REC_V5",
                    date,
                ),
            )

    # 4. Verify
    final = bk_conn.execute(
        "SELECT SUM(CASE WHEN transaction_type='DEPOSIT' THEN amount WHEN transaction_type='BET_SETTLEMENT' THEN amount WHEN transaction_type='BET_PLACEMENT' THEN -amount ELSE 0 END) FROM transactions"
    ).fetchone()[0]
    print(f"💰 Final V5 Balance: {final}")

    bk_conn.commit()  # Important? DuckDB usually auto-commits but explicit check.
    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
