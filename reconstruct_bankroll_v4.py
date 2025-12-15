import duckdb
import shutil
import os
import uuid
from datetime import datetime, timedelta

BANKROLL_DB = "data/nba_bankroll_v3.duckdb"
BETTING_DB = "data/nba_betting.duckdb"
BACKUP_PATH = "data/nba_bankroll_v3.duckdb.recalc_backup"

INITIAL_BANKROLL = 84.75


def backup_database():
    print(f"📦 Backing up bankroll database to {BACKUP_PATH}...")
    if os.path.exists(BANKROLL_DB):
        shutil.copy2(BANKROLL_DB, BACKUP_PATH)
        print("✅ Backup complete.")
    else:
        print("❌ Database file not found!")
        exit(1)


def reconstruct():
    backup_database()

    try:
        # Connect to databases
        bk_conn = duckdb.connect(BANKROLL_DB)
        bet_conn = duckdb.connect(BETTING_DB)

        # 1. Fetch Bet History
        print("\n📜 Fetching complete bet history...")
        # Columns based on DESCRIBE bets:
        # bet_id, user_id, game_id, bet_type, amount, odds, status, result, profit_loss, prediction, confidence_interval, created_at, updated_at, settled_at, ...

        bets = bet_conn.execute("""
            SELECT 
                bet_id, 
                user_id, 
                game_id, 
                prediction, 
                amount, 
                odds, 
                status, 
                result, 
                profit_loss, 
                created_at, 
                settled_at, 
                home_team, 
                away_team
            FROM bets
            ORDER BY created_at ASC
        """).fetchall()
        print(f"   Found {len(bets)} bets.")

        if not bets:
            print("❌ No bets found! Aborting.")
            return

        # Determine start time (1 hour before first bet)
        first_bet_item = bets[0][9]  # created_at

        # Handle string vs datetime
        if isinstance(first_bet_item, str):
            first_bet_time = datetime.fromisoformat(first_bet_item)
        elif isinstance(first_bet_item, datetime):
            first_bet_time = first_bet_item
        else:
            print(
                f"Warning: Unknown date format {type(first_bet_item)}, defaulting to now-7days"
            )
            first_bet_time = datetime.now() - timedelta(days=7)

        start_time = first_bet_time - timedelta(hours=1)

        print(f"   Start Time: {start_time}")

        # 2. Wipe Bankroll Ledger
        print("\n🧹 Wiping Transactions Ledger...")
        bk_conn.execute("DELETE FROM transactions")
        bk_conn.execute("DELETE FROM bankroll_snapshots")  # Force recalculation

        # 3. Insert Initial Deposit
        print(f"\n💰 Inserting Initial Deposit: €{INITIAL_BANKROLL}")
        init_tx_id = f"txn_init_{uuid.uuid4().hex[:12]}"
        running_balance = INITIAL_BANKROLL

        bk_conn.execute(
            """
            INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                init_tx_id,
                start_time,
                "DEPOSIT",
                INITIAL_BANKROLL,
                INITIAL_BANKROLL,
                "Initial Bankroll Reset (User Requested)",
                None,
                None,
                '{"reason": "reconstruction"}',
                "RECONSTRUCTED",
                start_time,
            ),
        )

        # 4. Replay Transactions
        print("\n▶️ Replaying History...")

        for bet in bets:
            (
                bet_id,
                user_id,
                game_id,
                prediction,
                stake,
                odds,
                status,
                result,
                profit_loss,
                placed_at,
                settled_at,
                home,
                away,
            ) = bet

            stake = float(stake) if stake else 0.0
            profit_loss_val = float(profit_loss) if profit_loss else 0.0

            # Payout Logic:
            payout_val = 0.0
            if result == "WON":
                payout_val = stake + profit_loss_val
                print(
                    f"DEBUG: Bet {bet_id} WON. Stake: {stake}, Profit: {profit_loss_val}, Payout: {payout_val}"
                )
            elif result == "VOID" or result == "PUSH":
                payout_val = stake

            # --- Transaction 1: Placement ---
            running_balance -= stake
            place_tx_id = f"txn_place_{uuid.uuid4().hex[:8]}"

            bk_conn.execute(
                """
                INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    place_tx_id,
                    placed_at,
                    "BET_PLACEMENT",
                    stake,
                    running_balance,
                    f"Bet placed: {prediction} ({home} vs {away})",
                    bet_id,
                    game_id,
                    "{}",
                    "RECONSTRUCTED",
                    placed_at,
                ),
            )

            # --- Transaction 2: Settlement ---
            if status in ["Settled", "WON", "LOST", "VOID", "PUSH"]:
                settle_time = (
                    settled_at if settled_at else (placed_at + timedelta(hours=2.5))
                )

                # Always record settlement for closed bets
                running_balance += payout_val
                settle_tx_id = f"txn_settle_{uuid.uuid4().hex[:8]}"
                bk_conn.execute(
                    """
                    INSERT INTO transactions (transaction_id, timestamp, transaction_type, amount, balance_after, description, bet_id, game_id, metadata, checksum, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        settle_tx_id,
                        settle_time,
                        "BET_SETTLEMENT",
                        payout_val,
                        running_balance,
                        f"Bet settled: {result}",
                        bet_id,
                        game_id,
                        "{}",
                        "RECONSTRUCTED",
                        settle_time,
                    ),
                )
        # 5. Final Verification
        final_balance_query = """
             SELECT 
                SUM(CASE 
                    WHEN transaction_type = 'DEPOSIT' THEN amount 
                    WHEN transaction_type = 'BET_SETTLEMENT' THEN amount
                    WHEN transaction_type = 'BET_PLACEMENT' THEN -amount
                    ELSE 0 
                END) as theoretical_balance
            FROM transactions
        """
        final_bal = bk_conn.execute(final_balance_query).fetchone()[0]

        print(f"\n✅ Reconstruction Complete.")
        print(f"💰 Calculated Balance: €{final_bal:.2f}")

        bk_conn.close()
        bet_conn.close()

    except Exception as e:
        print(f"\n❌ Error during reconstruction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    reconstruct()
