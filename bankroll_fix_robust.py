#!/usr/bin/env python3
"""
ROBUST BANKROLL FIX - Definitive Solution
Fixes all bankroll calculation issues with proper validation and reconciliation
"""

import duckdb
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75
USER_ID = "test_user_001"


class RobustBankrollManager:
    """
    Robust bankroll management system that fixes all calculation issues
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def validate_database_consistency(self):
        """Validate database consistency and identify issues"""
        logger.info("🔍 Validating database consistency...")

        issues = []

        # Check 1: Verify all bets have proper profit_loss
        invalid_pl_bets = self.conn.execute("""
            SELECT bet_id, amount, odds, status, result, profit_loss 
            FROM bets 
            WHERE status IN ('WON', 'LOST', 'PUSH', 'VOID') 
            AND (profit_loss IS NULL OR profit_loss = 0)
        """).fetchall()

        if invalid_pl_bets:
            issues.append(f"Found {len(invalid_pl_bets)} bets with missing/zero P/L")
            logger.warning(f"⚠️  Bets with missing P/L: {len(invalid_pl_bets)}")

        # Check 2: Verify pending bets don't have profit_loss
        pending_with_pl = self.conn.execute("""
            SELECT bet_id, amount, odds, status, profit_loss 
            FROM bets 
            WHERE status = 'PENDING' AND profit_loss IS NOT NULL
        """).fetchall()

        if pending_with_pl:
            issues.append(f"Found {len(pending_with_pl)} pending bets with P/L set")
            logger.warning(f"⚠️  Pending bets with P/L: {len(pending_with_pl)}")

        # Check 3: Verify transactions consistency
        tx_count = self.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        logger.info(f"📊 Total transactions: {tx_count}")

        return issues

    def recalculate_profit_loss(self):
        """Recalculate profit/loss for all settled bets correctly"""
        logger.info("🔄 Recalculating profit/loss for all settled bets...")

        # Get all settled bets
        settled_bets = self.conn.execute("""
            SELECT bet_id, amount, odds, status, result, profit_loss, created_at
            FROM bets 
            WHERE status IN ('WON', 'LOST', 'PUSH', 'VOID', 'SETTLED')
            ORDER BY created_at ASC
        """).fetchall()

        updated_count = 0
        for bet in settled_bets:
            bet_id, amount, odds, status, result, current_pl, created_at = bet
            amount = float(amount)
            odds = float(odds)

            # Calculate correct profit/loss
            if status == "WON" or result == "WON":
                # Net profit = (stake * odds) - stake = stake * (odds - 1)
                correct_pl = amount * (odds - 1)
            elif status in ["PUSH", "VOID", "void"] or result in [
                "PUSH",
                "VOID",
                "void",
            ]:
                correct_pl = 0.0  # Refund
            elif status == "LOST" or result == "LOST":
                correct_pl = -amount
            else:
                correct_pl = 0.0

            # Update if different
            if current_pl is None or abs(float(current_pl) - correct_pl) > 0.01:
                self.conn.execute(
                    """
                    UPDATE bets 
                    SET profit_loss = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE bet_id = ?
                """,
                    (correct_pl, bet_id),
                )
                updated_count += 1
                logger.info(
                    f"✅ Updated P/L for bet {bet_id}: {current_pl} → {correct_pl}"
                )

        logger.info(f"📝 Updated profit/loss for {updated_count} bets")
        return updated_count

    def rebuild_transactions_ledger(self):
        """Rebuild transactions ledger from scratch"""
        logger.info("🏗️  Rebuilding transactions ledger...")

        # Clear existing transactions
        self.conn.execute("DELETE FROM transactions")

        # Add initial deposit
        deposit_id = str(uuid.uuid4())
        first_bet_time = self.conn.execute(
            "SELECT MIN(created_at) FROM bets"
        ).fetchone()[0]

        if first_bet_time:
            deposit_time = first_bet_time - timedelta(seconds=1)
        else:
            deposit_time = datetime.utcnow()

        self.conn.execute(
            """
            INSERT INTO transactions (transaction_id, user_id, amount, type, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                deposit_id,
                USER_ID,
                INITIAL_DEPOSIT,
                "DEPOSIT",
                "Initial Bankroll Deposit",
                deposit_time,
            ),
        )

        # Process all bets in chronological order
        bets = self.conn.execute("""
            SELECT bet_id, amount, odds, status, result, profit_loss, created_at, settled_at
            FROM bets 
            ORDER BY created_at ASC
        """).fetchall()

        tx_count = 1  # Start with deposit
        for bet in bets:
            (
                bet_id,
                amount,
                odds,
                status,
                result,
                profit_loss,
                created_at,
                settled_at,
            ) = bet
            amount = float(amount)
            odds = float(odds)

            # Debit for bet placement
            debit_id = str(uuid.uuid4())
            self.conn.execute(
                """
                INSERT INTO transactions (transaction_id, user_id, amount, type, description, reference_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    debit_id,
                    USER_ID,
                    -amount,
                    "BET_PLACEMENT",
                    f"Bet {bet_id} placed",
                    bet_id,
                    created_at,
                ),
            )
            tx_count += 1

            # Credit for wins/refunds
            if status in ["WON", "PUSH", "VOID", "void", "SETTLED"]:
                if status == "WON" or result == "WON":
                    payout = amount * odds
                    desc = f"Bet Won Payout for {bet_id}"
                    tx_type = "BET_PAYOUT"
                else:
                    payout = amount
                    desc = f"Bet Refund for {bet_id}"
                    tx_type = "BET_REFUND"

                # Use settlement time or estimate
                settlement_time = (
                    settled_at if settled_at else created_at + timedelta(hours=2.5)
                )

                credit_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO transactions (transaction_id, user_id, amount, type, description, reference_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        credit_id,
                        USER_ID,
                        payout,
                        tx_type,
                        desc,
                        bet_id,
                        settlement_time,
                    ),
                )
                tx_count += 1

        logger.info(f"📊 Rebuilt ledger with {tx_count} transactions")
        return tx_count

    def calculate_correct_bankroll(self):
        """Calculate correct bankroll from rebuilt ledger"""
        logger.info("💰 Calculating correct bankroll...")

        # Free bankroll = sum of all transactions
        free_bankroll_result = self.conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE user_id = ?
        """,
            (USER_ID,),
        ).fetchone()[0]
        free_bankroll = (
            float(free_bankroll_result) if free_bankroll_result is not None else 0.0
        )

        # Locked bankroll = sum of pending bets
        locked_result = self.conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) FROM bets
            WHERE user_id = ? AND status = 'PENDING'
        """,
            (USER_ID,),
        ).fetchone()[0]
        locked_bankroll = float(locked_result) if locked_result is not None else 0.0

        total_equity = free_bankroll + locked_bankroll

        result = {
            "free_bankroll": free_bankroll,
            "locked_bankroll": locked_bankroll,
            "total_equity": total_equity,
            "initial_deposit": INITIAL_DEPOSIT,
            "total_profit_loss": total_equity - INITIAL_DEPOSIT,
        }

        logger.info(f"✅ Correct bankroll calculated:")
        logger.info(f"   Free: €{free_bankroll:.2f}")
        logger.info(f"   Locked: €{locked_bankroll:.2f}")
        logger.info(f"   Total: €{total_equity:.2f}")

        return result

    def update_bankroll_file(self, bankroll_data):
        """Update bankroll file with correct data"""
        bankroll_path = Path("data/bankroll.json")
        bankroll_path.parent.mkdir(parents=True, exist_ok=True)

        file_data = {
            "current_bankroll": bankroll_data["free_bankroll"],
            "locked_bankroll": bankroll_data["locked_bankroll"],
            "total_equity": bankroll_data["total_equity"],
            "initial_deposit": bankroll_data["initial_deposit"],
            "total_profit_loss": bankroll_data["total_profit_loss"],
            "last_updated": datetime.utcnow().isoformat(),
            "calculation_method": "ROBUST_LEDGER_V2",
        }

        with open(bankroll_path, "w") as f:
            json.dump(file_data, f, indent=2)

        logger.info(f"💾 Updated bankroll file: €{bankroll_data['free_bankroll']:.2f}")

    def fix_all_issues(self):
        """Execute complete bankroll fix process"""
        logger.info("🚀 Starting ROBUST BANKROLL FIX...")

        # Step 1: Validate and identify issues
        issues = self.validate_database_consistency()
        if issues:
            logger.warning(f"⚠️  Issues found: {issues}")

        # Step 2: Recalculate profit/loss
        updated_bets = self.recalculate_profit_loss()

        # Step 3: Rebuild transactions ledger
        tx_count = self.rebuild_transactions_ledger()

        # Step 4: Calculate correct bankroll
        bankroll_data = self.calculate_correct_bankroll()

        # Step 5: Update bankroll file
        self.update_bankroll_file(bankroll_data)

        # Summary
        logger.info("🎯 BANKROLL FIX COMPLETED!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Fixed {updated_bets} bet P/L calculations")
        logger.info(f"   - Rebuilt {tx_count} transactions")
        logger.info(f"   - Free bankroll: €{bankroll_data['free_bankroll']:.2f}")
        logger.info(f"   - Total equity: €{bankroll_data['total_equity']:.2f}")
        logger.info(
            f"   - Expected vs actual: Should be ~€97, calculated €{bankroll_data['total_equity']:.2f}"
        )

        return bankroll_data


def main():
    """Main execution function"""
    print("=" * 60)
    print("🔧 ROBUST BANKROLL FIX - DEFINITIVE SOLUTION")
    print("=" * 60)

    try:
        with RobustBankrollManager(DB_PATH) as manager:
            bankroll_data = manager.fix_all_issues()

            print("\n" + "=" * 60)
            print("✅ BANKROLL FIX COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Final Results:")
            print(f"   Free Bankroll: €{bankroll_data['free_bankroll']:.2f}")
            print(f"   Locked Stakes: €{bankroll_data['locked_bankroll']:.2f}")
            print(f"   Total Equity: €{bankroll_data['total_equity']:.2f}")
            print(f"   Net P/L: €{bankroll_data['total_profit_loss']:.2f}")
            print(f"   Expected: ~€97.00 (based on user calculation)")
            print("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error during bankroll fix: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
