import duckdb
import shutil
import os
import json
from datetime import datetime

DB_PATH = "data/nba_bankroll_v3.duckdb"
BACKUP_PATH = "data/nba_bankroll_v3.duckdb.bak"


def backup_database():
    print(f"📦 Backing up database to {BACKUP_PATH}...")
    if os.path.exists(DB_PATH):
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print("✅ Backup complete.")
    else:
        print("❌ Database file not found!")
        exit(1)


def repair_bankroll():
    backup_database()

    try:
        conn = duckdb.connect(DB_PATH)

        # 1. Inspect Initial/Inflation Transactions
        print("\n🔍 Analyzing Deposit Transactions...")
        deposits = conn.execute("""
            SELECT transaction_id, amount, description, created_at 
            FROM transactions 
            WHERE transaction_type = 'DEPOSIT'
            ORDER BY created_at
        """).fetchall()

        print(f"Found {len(deposits)} deposits.")
        for d in deposits:
            print(f" - {d}")

        # 2. Logic to identify duplicates
        # Group by description (if it contains bet_id) or timing
        print("\n🧹 Cleaning up Duplicate Legacy Adjustments...")

        # Identify duplicates: Same description/amount, different IDs
        # We keep the one with the earliest timestamp
        duplicates = conn.execute("""
            WITH RankedDeposits AS (
                SELECT 
                    transaction_id,
                    description,
                    amount,
                    ROW_NUMBER() OVER (
                        PARTITION BY description, amount 
                        ORDER BY created_at ASC
                    ) as rn
                FROM transactions
                WHERE transaction_type = 'DEPOSIT'
                  AND description LIKE '%Legacy Settlement Adjustment%'
            )
            SELECT transaction_id, description, amount
            FROM RankedDeposits
            WHERE rn > 1
        """).fetchall()

        if duplicates:
            print(f"⚠️ Found {len(duplicates)} duplicate legacy adjustments to delete:")
            ids_to_delete = []
            for dup in duplicates:
                print(f"   DELETE: {dup[0]} - {dup[1]} (€{dup[2]})")
                ids_to_delete.append(dup[0])

            # Execute Deletion
            if ids_to_delete:
                placeholders = ", ".join(["?"] * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM transactions WHERE transaction_id IN ({placeholders})",
                    ids_to_delete,
                )
                print(f"✅ Deleted {len(ids_to_delete)} duplicate transactions.")
        else:
            print("✅ No duplicate legacy adjustments found.")

        # 3. Consolidate Initial Deposits
        # If there are multiple "Initial Deposit" or unnamed deposits around the same time
        print("\n🧹 Consolidating Initial Deposits...")
        initial_deposits = conn.execute("""
             SELECT transaction_id, amount, description, created_at
             FROM transactions
             WHERE transaction_type = 'DEPOSIT'
               AND (description IS NULL OR description = 'Deposit' OR description LIKE '%Initial%')
             ORDER BY created_at
        """).fetchall()

        if len(initial_deposits) > 1:
            print(
                f"⚠️ Found {len(initial_deposits)} generic/initial deposits. Reducing to ONE."
            )
            # Keep only the very first one
            base_deposit = initial_deposits[0]
            print(f"   KEEP: {base_deposit}")

            ids_to_delete = [d[0] for d in initial_deposits[1:]]
            if ids_to_delete:
                placeholders = ", ".join(["?"] * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM transactions WHERE transaction_id IN ({placeholders})",
                    ids_to_delete,
                )
                print(f"✅ Deleted {len(ids_to_delete)} redundant initial deposits.")

        # 3.5 Check for Duplicate Settlements
        print("\n🔍 Checking for Duplicate Bet Settlements...")
        duplicate_settlements = conn.execute("""
            SELECT description, COUNT(*) as cnt, SUM(amount) as total_payout
            FROM transactions
            WHERE transaction_type = 'BET_SETTLEMENT'
            GROUP BY description
            HAVING COUNT(*) > 1
        """).fetchall()

        if duplicate_settlements:
            print(
                f"⚠️ Found duplicate settlements for {len(duplicate_settlements)} bets!"
            )
            for ds in duplicate_settlements:
                print(f"   - {ds[0]}: {ds[1]} times, Total: €{ds[2]:.2f}")

            # Logic to delete duplicates (keep earliest)
            print("   -> Deleting duplicate settlements...")
            settlements_to_fix = conn.execute("""
                WITH RankedSettlements AS (
                    SELECT 
                        transaction_id,
                        description,
                        ROW_NUMBER() OVER (
                            PARTITION BY description 
                            ORDER BY created_at ASC
                        ) as rn
                    FROM transactions
                    WHERE transaction_type = 'BET_SETTLEMENT'
                )
                SELECT transaction_id FROM RankedSettlements WHERE rn > 1
            """).fetchall()

            ids_to_del_settle = [x[0] for x in settlements_to_fix]
            if ids_to_del_settle:
                placeholders_s = ", ".join(["?"] * len(ids_to_del_settle))
                conn.execute(
                    f"DELETE FROM transactions WHERE transaction_id IN ({placeholders_s})",
                    ids_to_del_settle,
                )
                print(
                    f"✅ Deleted {len(ids_to_del_settle)} duplicate settlement transactions."
                )
        else:
            print("✅ No duplicate settlements found.")

        # 4. Force Recalculation (or just let it be if Engine is event-sourced)
        # We should delete bankroll_snapshots to force recalculation if possible,
        # or we rely on the engine explicitly.
        # Since we modified the ledger (transactions), any derived view should be updated.
        # But 'bankroll_snapshots' table might hold stale state.

        print("\n🔄 Resetting Snapshots Table to force recalculation...")
        conn.execute(
            "DELETE FROM bankroll_snapshots"
        )  # Engine logic usually writes new snapshot on next action

        # Verify New Balance
        # Since we can't easily run the python engine logic here without dependencies,
        # we sum up the transactions manually to verify.

        print("\n💰 Verifying New Calculated Balance from Ledger:")
        balance_summary = conn.execute("""
            SELECT 
                SUM(CASE 
                    WHEN transaction_type = 'DEPOSIT' THEN amount 
                    WHEN transaction_type = 'BET_SETTLEMENT' THEN amount
                    WHEN transaction_type = 'BET_PLACEMENT' THEN -amount
                    ELSE 0 
                END) as theoretical_balance
            FROM transactions
        """).fetchone()

        print(f"✅ Theoretical Balance after repair: €{balance_summary[0]:.2f}")

        conn.close()
        print("\n🎉 Repair Complete. Restart Dashboard to verify.")

    except Exception as e:
        print(f"\n❌ Error during repair: {e}")
        # Restore backup?
        # shutil.copy2(BACKUP_PATH, DB_PATH)


if __name__ == "__main__":
    repair_bankroll()
