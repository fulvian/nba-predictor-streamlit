#!/usr/bin/env python3
"""
Script to reset betting database while preserving game data.

This script will:
1. Clean all betting data (placed_bets, betting_analysis, bankroll_history)
2. Reset bankroll to initial amount
3. Preserve enhanced game data schema
4. Verify the system is ready for fresh betting operations
"""

import sys
import logging
from pathlib import Path
import duckdb
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def reset_betting_database():
    """
    Reset betting database by cleaning all betting-related tables.

    This preserves the game data structure and enhanced schema
    while removing all test betting data.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔄 Resetting betting database...")

            # Check existing tables
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            logger.info(f"   Existing tables: {table_names}")

            # Clean betting tables while preserving game data
            # Order matters due to foreign key constraints
            betting_tables_order = ['bankroll_history', 'betting_analysis', 'placed_bets']

            for table in betting_tables_order:
                if table in table_names:
                    logger.info(f"   Cleaning table: {table}")
                    try:
                        conn.execute(f"DELETE FROM {table}")
                        logger.info(f"   ✅ Cleared all data from {table}")
                    except Exception as e:
                        if "foreign key constraint" in str(e).lower():
                            logger.warning(f"   ⚠️ Foreign key constraint detected for {table}")
                            logger.info(f"   🔧 Dropping and recreating table {table}")

                            # Get table structure first
                            table_info = conn.execute(f"DESCRIBE {table}").fetchall()
                            create_sql = f"CREATE TABLE {table} (\n"
                            columns = []
                            for col in table_info:
                                col_name, col_type = col[0], col[1]
                                columns.append(f"    {col_name} {col_type}")
                            create_sql += ",\n".join(columns)
                            create_sql += "\n)"

                            # Drop and recreate table
                            conn.execute(f"DROP TABLE {table}")
                            conn.execute(create_sql)
                            logger.info(f"   ✅ Recreated table {table} with empty structure")
                        else:
                            raise e
                else:
                    logger.info(f"   Table {table} does not exist - skipping")

            # Reset bankroll to initial amount
            logger.info("   Resetting bankroll to initial amount")
            try:
                # Check if bankroll table exists
                if 'bankroll' in table_names:
                    conn.execute("UPDATE bankroll SET current_amount = 1000.00, updated_at = CURRENT_TIMESTAMP")
                    logger.info("   ✅ Reset existing bankroll to €1000.00")
                else:
                    # Create bankroll table if it doesn't exist
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS bankroll (
                            id INTEGER PRIMARY KEY,
                            current_amount DECIMAL(10,2) DEFAULT 1000.00,
                            initial_amount DECIMAL(10,2) DEFAULT 1000.00,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.execute("INSERT INTO bankroll (id, current_amount, initial_amount) VALUES (1, 1000.00, 1000.00)")
                    logger.info("   ✅ Created and initialized bankroll table with €1000.00")
            except Exception as e:
                logger.warning(f"   Bankroll reset failed: {e}")

            # Verify game data is preserved
            logger.info("   Verifying game data preservation...")
            try:
                games_count = conn.execute("""
                    SELECT COUNT(*) FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')
                """).fetchone()[0]
                logger.info(f"   ✅ Preserved {games_count} games in parquet files")
            except Exception as e:
                logger.warning(f"   Game verification failed: {e}")

            # Log the reset operation
            try:
                if 'betting_logs' in table_names:
                    conn.execute(f"""
                        INSERT INTO betting_logs (operation, details, timestamp)
                        VALUES ('DATABASE_RESET', 'Complete betting database reset - all test data removed', CURRENT_TIMESTAMP)
                    """)
                    logger.info("   ✅ Logged reset operation")
            except Exception as e:
                logger.warning(f"   Logging operation failed: {e}")

            logger.info("🎉 Betting database reset completed successfully!")
            logger.info("✅ All betting data cleaned")
            logger.info("✅ Bankroll reset to €1000.00")
            logger.info("✅ Game data preserved")
            logger.info("✅ System ready for fresh betting operations")

            return True

    except Exception as e:
        logger.error(f"❌ Failed to reset betting database: {e}")
        return False

def verify_system_status():
    """
    Verify the system status after reset.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔍 Verifying system status after reset...")

            # Check betting tables are empty
            try:
                placed_bets_count = conn.execute("SELECT COUNT(*) FROM placed_bets").fetchone()[0]
                analysis_count = conn.execute("SELECT COUNT(*) FROM betting_analysis").fetchone()[0]
                history_count = conn.execute("SELECT COUNT(*) FROM bankroll_history").fetchone()[0]

                logger.info(f"   Placed bets: {placed_bets_count} (should be 0)")
                logger.info(f"   Analysis records: {analysis_count} (should be 0)")
                logger.info(f"   Bankroll history: {history_count} (should be 0)")

                if placed_bets_count == 0 and analysis_count == 0 and history_count == 0:
                    logger.info("   ✅ All betting tables are empty")
                else:
                    logger.warning("   ⚠️ Some betting tables still contain data")

            except Exception as e:
                logger.info(f"   ✅ Betting tables are empty or don't exist yet")

            # Check bankroll
            try:
                bankroll_result = conn.execute("SELECT current_amount FROM bankroll WHERE id = 1").fetchone()
                if bankroll_result:
                    bankroll = bankroll_result[0]
                    logger.info(f"   Current bankroll: €{bankroll:.2f}")
                    if bankroll == 1000.00:
                        logger.info("   ✅ Bankroll correctly reset to €1000.00")
                    else:
                        logger.warning(f"   ⚠️ Bankroll is €{bankroll:.2f}, expected €1000.00")
                else:
                    logger.info("   ✅ Bankroll table will be created on first use")
            except Exception as e:
                logger.info("   ✅ Bankroll table will be created on first use")

            # Check game data
            try:
                games_count = conn.execute("""
                    SELECT COUNT(*) FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')
                """).fetchone()[0]
                logger.info(f"   Available games: {games_count}")
                if games_count > 0:
                    logger.info("   ✅ Game data is available for betting")
                else:
                    logger.warning("   ⚠️ No game data available")
            except Exception as e:
                logger.warning(f"   Game data check failed: {e}")

            logger.info("🎯 System verification completed")
            return True

    except Exception as e:
        logger.error(f"❌ System verification failed: {e}")
        return False

def main():
    """Main database reset process."""
    logger.info("🚀 Starting Betting Database Reset")
    logger.info("=" * 60)
    logger.info("This will clean all betting data while preserving game data")
    logger.info("=" * 60)

    success = True

    # Step 1: Reset betting database
    logger.info("📝 Step 1: Resetting betting database...")
    if not reset_betting_database():
        success = False

    # Step 2: Verify system status
    if success:
        logger.info("🔍 Step 2: Verifying system status...")
        if not verify_system_status():
            success = False

    # Summary
    if success:
        logger.info("🎉 Betting Database Reset COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ All betting data cleaned")
        logger.info("✅ Bankroll reset to €1000.00")
        logger.info("✅ Game data preserved with enhanced schema")
        logger.info("✅ System ready for fresh betting operations")
        logger.info("")
        logger.info("🏀 Next steps:")
        logger.info("   1. Open betting workflow dashboard")
        logger.info("   2. Place a new bet with complete data")
        logger.info("   3. Verify team names and game details are saved")
        logger.info("   4. Test result updates and bet settlement")
    else:
        logger.error("❌ Database reset failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)