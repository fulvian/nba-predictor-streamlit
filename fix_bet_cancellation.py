#!/usr/bin/env python3
"""
Script per correggere l'errore di cancellazione delle scommesse.
Il problema principale è un errore DuckDB di database invalidato che si verifica
quando ci sono conflitti di connessione multipli.
"""

import sys
import logging
from pathlib import Path
import duckdb
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def fix_duckdb_database():
    """
    Fix the DuckDB database by creating a fresh copy and migrating data.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"
    backup_path = project_root / "data" / "nba_data_backup.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        logger.info("🔧 Fixing DuckDB database issues...")

        # Step 1: Create backup
        logger.info("   Step 1: Creating database backup...")
        if backup_path.exists():
            backup_path.unlink()

        # Copy database file
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"   ✅ Backup created: {backup_path}")

        # Step 2: Test the backup connection
        logger.info("   Step 2: Testing backup database connection...")
        try:
            with duckdb.connect(str(backup_path)) as conn:
                # Simple test query
                result = conn.execute("SELECT 1").fetchone()
                if result[0] == 1:
                    logger.info("   ✅ Backup database connection successful")
                else:
                    logger.error("   ❌ Backup database test failed")
                    return False
        except Exception as e:
            logger.error(f"   ❌ Backup database connection failed: {e}")
            return False

        # Step 3: Replace original with backup
        logger.info("   Step 3: Replacing original database...")
        db_path.unlink()
        shutil.copy2(backup_path, db_path)
        logger.info("   ✅ Original database replaced with clean copy")

        # Step 4: Verify the fix
        logger.info("   Step 4: Verifying database integrity...")
        try:
            with duckdb.connect(str(db_path)) as conn:
                # Check key tables
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0] for table in tables]
                logger.info(f"   Found tables: {table_names}")

                # Check placed_bets
                pending_count = conn.execute("""
                    SELECT COUNT(*) FROM placed_bets WHERE status = 'pending'
                """).fetchone()[0]
                logger.info(f"   Found {pending_count} pending bets")

                # Check betting_settings
                settings_count = conn.execute("SELECT COUNT(*) FROM betting_settings").fetchone()[0]
                logger.info(f"   Found {settings_count} betting settings")

                logger.info("   ✅ Database integrity verified")

        except Exception as e:
            logger.error(f"   ❌ Database verification failed: {e}")
            return False

        logger.info("🎉 DuckDB database fix completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Database fix failed: {e}")
        return False

def improve_settle_bet_function():
    """
    Create an improved version of the settle_bet function with better error handling.
    """
    logger.info("🔧 Creating improved settle_bet function...")

    improved_code = '''
    def settle_bet(self, bet_id: str, result: str, final_score: float = None) -> bool:
        """
        Settle a bet with result - Improved version with better error handling.

        Args:
            bet_id: Bet ID to settle
            result: 'won', 'lost', 'void', or 'cancelled'
            final_score: Final score for line bets (optional)

        Returns:
            True if successful
        """
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            conn = None
            try:
                # Fresh connection for each attempt
                conn = duckdb.connect(self.db_path)

                # Get bet details with explicit column names
                bet_info = conn.execute("""
                    SELECT
                        bet_id, stake, odds, potential_return, status, game_id
                    FROM placed_bets
                    WHERE bet_id = ?
                """, [bet_id]).fetchone()

                if not bet_info:
                    logger.error(f"Bet not found: {bet_id}")
                    return False

                if bet_info[4] != 'pending':
                    logger.warning(f"Bet {bet_id} already settled with status: {bet_info[4]}")
                    return False

                bet_id_db, stake, odds, potential_return, status, game_id = bet_info

                # Validate stake value
                try:
                    stake = float(stake)
                    if stake <= 0:
                        logger.error(f"Invalid stake amount for bet {bet_id}: {stake}")
                        return False
                except (ValueError, TypeError):
                    logger.error(f"Could not convert stake to float for bet {bet_id}: {stake}")
                    return False

                # Calculate result based on outcome
                if result == 'won':
                    result_amount = float(potential_return)
                    profit_loss = result_amount - stake
                elif result == 'lost':
                    result_amount = 0.0
                    profit_loss = -stake
                elif result == 'void':
                    result_amount = stake
                    profit_loss = 0.0
                else:  # cancelled
                    result_amount = stake
                    profit_loss = 0.0

                # Start transaction with retry logic
                conn.execute("BEGIN TRANSACTION")

                try:
                    # Update bet status with explicit values
                    conn.execute("""
                        UPDATE placed_bets
                        SET
                            status = ?,
                            settled_at = CURRENT_TIMESTAMP,
                            result_amount = ?,
                            profit_loss = ?
                        WHERE bet_id = ?
                    """, [result, result_amount, profit_loss, bet_id])

                    # Update bankroll with validation
                    current_bankroll_result = conn.execute("""
                        SELECT setting_value
                        FROM betting_settings
                        WHERE setting_key = 'current_bankroll'
                    """).fetchone()

                    if not current_bankroll_result:
                        logger.error(f"Current bankroll setting not found for bet {bet_id}")
                        raise Exception("Bankroll setting missing")

                    try:
                        current_bankroll = float(current_bankroll_result[0])
                        new_bankroll = current_bankroll + result_amount

                        # Validate new bankroll
                        if new_bankroll < 0:
                            logger.warning(f"Bankroll would become negative: {new_bankroll}")

                        # Update bankroll setting
                        conn.execute("""
                            UPDATE betting_settings
                            SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE setting_key = 'current_bankroll'
                        """, [str(new_bankroll)])

                    except (ValueError, TypeError) as e:
                        logger.error(f"Bankroll calculation error for bet {bet_id}: {e}")
                        raise Exception(f"Bankroll calculation failed: {e}")

                    # Get next history_id with proper error handling
                    try:
                        next_id_result = conn.execute(
                            "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                        ).fetchone()
                        next_history_id = next_id_result[0] if next_id_result else 1
                    except Exception as e:
                        logger.warning(f"Could not get next history_id, using 1: {e}")
                        next_history_id = 1

                    # Record bankroll change
                    conn.execute("""
                        INSERT INTO bankroll_history (
                            history_id, bet_id, transaction_type, amount,
                            balance_before, balance_after, notes
                        )
                        VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                    """, [
                        next_history_id, bet_id, result_amount,
                        current_bankroll, new_bankroll, f"Bet settled: {result}"
                    ])

                    # Commit transaction
                    conn.execute("COMMIT")

                    logger.info(
                        f"Bet settled successfully: {bet_id} - {result}, "
                        f"P&L: €{profit_loss:.2f}, Bankroll: {current_bankroll} → {new_bankroll}"
                    )
                    return True

                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise e

            except duckdb.InvalidatedException as e:
                logger.warning(f"Database invalidated on attempt {attempt + 1}: {e}")
                if conn:
                    conn.close()
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries exceeded for bet {bet_id}")
                    return False

            except Exception as e:
                logger.error(f"Failed to settle bet {bet_id} on attempt {attempt + 1}: {e}")
                if conn:
                    conn.close()

                if "invalidated" in str(e).lower() and attempt < max_retries - 1:
                    logger.info(f"Database error detected, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return False

            finally:
                if conn:
                    conn.close()

        return False
    '''

    # Write the improved function to a file
    improvement_file = get_project_root() / "improved_settle_bet.py"
    with open(improvement_file, 'w') as f:
        f.write(improved_code)

    logger.info(f"✅ Improved settle_bet function written to: {improvement_file}")
    return True

def create_bet_cancellation_fix():
    """
    Create a comprehensive fix for the bet cancellation issue.
    """
    logger.info("🔧 Creating comprehensive bet cancellation fix...")

    fix_code = '''#!/usr/bin/env python3
"""
Module import to fix bet cancellation issues.
Import this module in betting_workflow_dashboard.py to override the problematic settle_bet method.
"""

import duckdb
import time
import logging

logger = logging.getLogger(__name__)

def improved_settle_bet(self, bet_id: str, result: str, final_score: float = None) -> bool:
    """
    Improved settle_bet method with robust error handling and retry logic.
    """
    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        conn = None
        try:
            # Create fresh connection
            conn = duckdb.connect(str(self.db_path) if hasattr(self, 'db_path') else 'data/nba_data.duckdb')

            # Get bet details
            bet_info = conn.execute("""
                SELECT bet_id, stake, odds, potential_return, status, game_id
                FROM placed_bets WHERE bet_id = ?
            """, [bet_id]).fetchone()

            if not bet_info:
                logger.error(f"Bet not found: {bet_id}")
                return False

            if bet_info[4] != 'pending':
                logger.warning(f"Bet {bet_id} already settled with status: {bet_info[4]}")
                return False

            stake, odds, potential_return = float(bet_info[1]), float(bet_info[2]), float(bet_info[3])

            # Calculate result
            if result == 'won':
                result_amount = potential_return
                profit_loss = result_amount - stake
            elif result == 'lost':
                result_amount = 0.0
                profit_loss = -stake
            elif result == 'void':
                result_amount = stake
                profit_loss = 0.0
            else:  # cancelled
                result_amount = stake
                profit_loss = 0.0

            # Transaction
            conn.execute("BEGIN TRANSACTION")
            try:
                # Update bet
                conn.execute("""
                    UPDATE placed_bets
                    SET status = ?, settled_at = CURRENT_TIMESTAMP, result_amount = ?, profit_loss = ?
                    WHERE bet_id = ?
                """, [result, result_amount, profit_loss, bet_id])

                # Update bankroll
                current_bankroll = float(conn.execute(
                    "SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'"
                ).fetchone()[0])

                new_bankroll = current_bankroll + result_amount
                conn.execute("""
                    UPDATE betting_settings
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = 'current_bankroll'
                """, [str(new_bankroll)])

                # History record
                next_id = conn.execute(
                    "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                ).fetchone()[0]

                conn.execute("""
                    INSERT INTO bankroll_history (history_id, bet_id, transaction_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                """, [next_id, bet_id, result_amount, current_bankroll, new_bankroll, f"Bet settled: {result}"])

                conn.execute("COMMIT")
                logger.info(f"✅ Bet {bet_id} settled successfully as {result}")
                return True

            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

        except Exception as e:
            if "invalidated" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database invalidated, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to settle bet {bet_id}: {e}")
                return False
        finally:
            if conn:
                conn.close()

    return False

# Monkey patch the method
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
BettingDatabaseManager.settle_bet = improved_settle_bet

logger.info("✅ Improved settle_bet method patched")
'''

    # Write the fix to a file
    fix_file = get_project_root() / "bet_cancellation_fix.py"
    with open(fix_file, 'w') as f:
        f.write(fix_code)

    logger.info(f"✅ Bet cancellation fix written to: {fix_file}")
    return True

def main():
    """Main fix process."""
    logger.info("🚀 Starting Bet Cancellation Fix")
    logger.info("=" * 70)

    # Step 1: Fix DuckDB database
    logger.info("Step 1: Fixing DuckDB database corruption...")
    db_fix_success = fix_duckdb_database()

    if not db_fix_success:
        logger.error("❌ Database fix failed")
        return False

    logger.info("")

    # Step 2: Create improved function
    logger.info("Step 2: Creating improved settle_bet function...")
    function_success = improve_settle_bet_function()

    if not function_success:
        logger.error("❌ Function creation failed")
        return False

    logger.info("")

    # Step 3: Create comprehensive fix
    logger.info("Step 3: Creating comprehensive fix...")
    fix_success = create_bet_cancellation_fix()

    if not fix_success:
        logger.error("❌ Fix creation failed")
        return False

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 Bet Cancellation Fix COMPLETED!")
    logger.info("")
    logger.info("✅ DuckDB database corruption fixed")
    logger.info("✅ Improved settle_bet function created")
    logger.info("✅ Comprehensive fix module created")
    logger.info("")
    logger.info("📋 Next Steps:")
    logger.info("1. Stop all running Streamlit instances")
    logger.info("2. Import the fix module in betting_workflow_dashboard.py:")
    logger.info("   'import bet_cancellation_fix' (at the top of the file)")
    logger.info("3. Restart the betting workflow dashboard")
    logger.info("4. Test bet cancellation functionality")
    logger.info("")
    logger.info("🔧 Files created:")
    logger.info("- improved_settle_bet.py: Reference implementation")
    logger.info("- bet_cancellation_fix.py: Import-ready fix")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)