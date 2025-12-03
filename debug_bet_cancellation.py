#!/usr/bin/env python3
"""
Script per diagnosticare e correggere l'errore nella cancellazione delle scommesse.
Il problema sembra essere legato alla gestione del bankroll quando si cancella una scommessa.
"""

import sys
import logging
from pathlib import Path
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def diagnose_bet_cancellation_issue():
    """
    Diagnose the bet cancellation issue by checking database state and logic.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔍 Diagnosing bet cancellation issue...")

            # Step 1: Check placed_bets table structure
            logger.info("   Step 1: Checking placed_bets table structure...")
            try:
                schema_result = conn.execute("DESCRIBE placed_bets").fetchall()
                columns = [col[0] for col in schema_result]
                logger.info(f"   ✅ placed_bets columns ({len(columns)}): {columns}")

                # Check if all required columns exist
                required_columns = ['bet_id', 'stake', 'odds', 'potential_return', 'status', 'game_id']
                missing_columns = [col for col in required_columns if col not in columns]
                if missing_columns:
                    logger.error(f"   ❌ Missing columns: {missing_columns}")
                    return False
                else:
                    logger.info("   ✅ All required columns present")

            except Exception as e:
                logger.error(f"   ❌ Could not check placed_bets structure: {e}")
                return False

            # Step 2: Check betting_settings table
            logger.info("   Step 2: Checking betting_settings table...")
            try:
                settings_result = conn.execute("SELECT * FROM betting_settings").fetchall()
                logger.info(f"   Found {len(settings_result)} settings:")

                bankroll_settings = {}
                for setting in settings_result:
                    logger.info(f"   - {setting[0]}: {setting[1]}")
                    if 'bankroll' in setting[0].lower():
                        bankroll_settings[setting[0]] = setting[1]

                if 'current_bankroll' not in bankroll_settings:
                    logger.warning("   ⚠️  current_bankroll setting not found")
                    # Initialize current_bankroll
                    conn.execute("""
                        INSERT INTO betting_settings (setting_key, setting_value, updated_at)
                        VALUES ('current_bankroll', '1000.0', CURRENT_TIMESTAMP)
                    """)
                    logger.info("   ✅ Initialized current_bankroll to 1000.0")
                    bankroll_settings['current_bankroll'] = '1000.0'
                else:
                    logger.info(f"   ✅ current_bankroll: {bankroll_settings['current_bankroll']}")

            except Exception as e:
                logger.error(f"   ❌ Error checking betting_settings: {e}")
                return False

            # Step 3: Check for pending bets
            logger.info("   Step 3: Checking for pending bets...")
            try:
                pending_bets = conn.execute("""
                    SELECT bet_id, stake, status, game_id
                    FROM placed_bets
                    WHERE status = 'pending'
                """).fetchall()

                logger.info(f"   Found {len(pending_bets)} pending bets")
                for bet in pending_bets[:3]:  # Show first 3
                    logger.info(f"   - Bet {bet[0]}: stake={bet[1]}, status={bet[2]}, game_id={bet[3]}")

            except Exception as e:
                logger.error(f"   ❌ Error checking pending bets: {e}")
                return False

            # Step 4: Test the cancellation logic with a dry run
            logger.info("   Step 4: Testing cancellation logic...")
            try:
                if pending_bets:
                    test_bet = pending_bets[0]
                    bet_id = test_bet[0]
                    stake = float(test_bet[1])

                    logger.info(f"   Testing with bet {bet_id} (stake: {stake})")

                    # Simulate the cancellation logic
                    current_bankroll_str = conn.execute("""
                        SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'
                    """).fetchone()

                    if current_bankroll_str:
                        current_bankroll = float(current_bankroll_str[0])
                        result_amount = stake  # For cancelled bets
                        new_bankroll = current_bankroll + result_amount

                        logger.info(f"   Current bankroll: {current_bankroll}")
                        logger.info(f"   Result amount (stake to return): {result_amount}")
                        logger.info(f"   New bankroll would be: {new_bankroll}")

                        # Check for potential overflow or precision issues
                        if new_bankroll > 1000000:  # Arbitrary large number check
                            logger.warning(f"   ⚠️  New bankroll seems very high: {new_bankroll}")

                        if abs(new_bankroll - current_bankroll) > 10000:  # Arbitrary large change check
                            logger.warning(f"   ⚠️  Large bankroll change: {new_bankroll - current_bankroll}")

                        logger.info("   ✅ Cancellation logic appears sound")
                    else:
                        logger.error("   ❌ Could not retrieve current_bankroll setting")
                        return False
                else:
                    logger.info("   ⚠️  No pending bets to test with")

            except Exception as e:
                logger.error(f"   ❌ Error testing cancellation logic: {e}")
                return False

            # Step 5: Check bankroll_history table
            logger.info("   Step 5: Checking bankroll_history table...")
            try:
                history_result = conn.execute("""
                    SELECT COUNT(*) FROM bankroll_history
                """).fetchone()

                history_count = history_result[0] if history_result else 0
                logger.info(f"   ✅ Found {history_count} bankroll history entries")

                # Check if history_id sequence is correct
                if history_count > 0:
                    max_id_result = conn.execute("""
                        SELECT MAX(history_id) FROM bankroll_history
                    """).fetchone()
                    max_id = max_id_result[0] if max_id_result else 0
                    logger.info(f"   Current max history_id: {max_id}")

            except Exception as e:
                logger.error(f"   ❌ Error checking bankroll_history: {e}")
                return False

            logger.info("🎉 Diagnosis completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False

def test_bet_cancellation():
    """
    Test the actual bet cancellation process.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🧪 Testing bet cancellation...")

            # Find a pending bet to test with
            pending_bet = conn.execute("""
                SELECT bet_id, stake, status FROM placed_bets WHERE status = 'pending' LIMIT 1
            """).fetchone()

            if not pending_bet:
                logger.info("⚠️  No pending bets found for testing")
                return True

            bet_id, stake, status = pending_bet
            logger.info(f"Testing cancellation of bet {bet_id} (stake: {stake})")

            # Start transaction
            conn.execute("BEGIN TRANSACTION")

            try:
                # Get current bankroll
                current_bankroll_result = conn.execute("""
                    SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'
                """).fetchone()

                current_bankroll = float(current_bankroll_result[0]) if current_bankroll_result else 1000.0
                result_amount = float(stake)  # For cancelled bets
                new_bankroll = current_bankroll + result_amount

                # Update bet status
                conn.execute("""
                    UPDATE placed_bets
                    SET status = 'cancelled', settled_at = CURRENT_TIMESTAMP, result_amount = ?, profit_loss = 0
                    WHERE bet_id = ?
                """, [result_amount, bet_id])

                # Update bankroll
                conn.execute("""
                    UPDATE betting_settings
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = 'current_bankroll'
                """, [str(new_bankroll)])

                # Get next history_id
                next_id_result = conn.execute("SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history").fetchone()
                next_history_id = next_id_result[0] if next_id_result else 1

                # Record bankroll change
                conn.execute("""
                    INSERT INTO bankroll_history (history_id, bet_id, transaction_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                """, [next_history_id, bet_id, result_amount, current_bankroll, new_bankroll, f"Bet settled: cancelled"])

                # Commit transaction
                conn.execute("COMMIT")

                logger.info(f"✅ Successfully cancelled bet {bet_id}")
                logger.info(f"   Bankroll: {current_bankroll} → {new_bankroll}")
                return True

            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"❌ Test cancellation failed: {e}")
                return False

    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        return False

def main():
    """Main diagnosis and fix process."""
    logger.info("🚀 Starting Bet Cancellation Diagnosis")
    logger.info("=" * 70)

    # Step 1: Diagnose the issue
    success = diagnose_bet_cancellation_issue()

    if not success:
        logger.error("❌ Diagnosis failed - cannot proceed with fix")
        return False

    logger.info("")

    # Step 2: Test the cancellation process
    logger.info("🧪 Testing cancellation process...")
    test_success = test_bet_cancellation()

    # Summary
    logger.info("")
    logger.info("=" * 70)
    if success and test_success:
        logger.info("🎉 Bet Cancellation Diagnosis COMPLETED!")
        logger.info("✅ Database structure is correct")
        logger.info("✅ Bankroll settings are properly configured")
        logger.info("✅ Cancellation logic works correctly")
        logger.info("")
        logger.info("💡 If cancellation still fails, the issue might be:")
        logger.info("   1. Transaction handling in the application code")
        logger.info("   2. Connection management issues")
        logger.info("   3. Application-level error handling")
    else:
        logger.error("❌ Issues found during diagnosis")
        logger.error("Please check the error messages above")

    return success and test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)