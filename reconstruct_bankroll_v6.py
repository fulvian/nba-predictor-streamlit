import duckdb
import uuid
from datetime import datetime, timedelta


def reconstruct():
    print("🚀 Starting Reconstruction V6 - FINAL FIX")
    bk_conn = duckdb.connect("data/nba_bankroll_v3.duckdb")
    bet_conn = duckdb.connect("data/nba_betting.duckdb")

    # 1. Wipe
    bk_conn.execute("DELETE FROM transactions")
    bk_conn.execute("DELETE FROM bankroll_snapshots")

    # 2. Init
    start_time = datetime.now() - timedelta(days=7)
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
            "REC_V6",
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

        # Ensure conversion from Decimal/String
        stake = float(amt) if amt is not None else 0.0
        prof = float(profit) if profit is not None else 0.0

        # --- PLACEMENT ---
        running_bal -= stake
        # DEBUG: Verify Stake is not 0
        if stake == 0:
            print(f"⚠️ WARNING: Zero stake for bet {bid}")

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
                "REC_V6",
                date,
            ),
        )

        # --- SETTLEMENT ---
        if status in ["Settled", "WON", "LOST", "VOID", "PUSH"]:
            payout = 0.0

            if res == "WON":
                payout = stake + prof
                # CRITICAL DEBUG
                print(f"✅ WON: Stake({stake}) + Prof({prof}) = Payout({payout})")

            elif res == "VOID" or res == "PUSH":
                payout = stake
                print(f"↩️  VOID/PUSH: Returning Stake({stake})")

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
                    "REC_V6",
                    date,
                ),
            )

    # 4. Verify Final Balance
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

    print(f"\n💰 FINAL CHECK: {final_val}")

    # Commit implies close
    bk_conn.close()
    bet_conn.close()


if __name__ == "__main__":
    reconstruct()
