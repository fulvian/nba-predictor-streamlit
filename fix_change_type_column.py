#!/usr/bin/env python3
"""
Script per aggiungere la colonna change_type mancante alla tabella bankroll_history.
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

def add_change_type_to_bankroll_history():
    """
    Add the missing change_type column to bankroll_history table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding change_type column to bankroll_history table...")

            # Check if table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            if 'bankroll_history' not in table_names:
                logger.error("❌ bankroll_history table does not exist")
                return False

            # Check current columns
            columns = conn.execute("DESCRIBE bankroll_history").fetchall()
            column_names = [col[0] for col in columns]
            logger.info(f"   Current columns: {column_names}")

            # Check if change_type already exists
            if 'change_type' in column_names:
                logger.info("   ✅ change_type column already exists")
                return True

            # Add change_type column
            logger.info("   Adding change_type column...")
            conn.execute("ALTER TABLE bankroll_history ADD COLUMN change_type VARCHAR")
            logger.info("   ✅ Added change_type column to bankroll_history table")

            # Update existing records to have default change_type values based on transaction_type
            if 'transaction_type' in column_names:
                logger.info("   Updating existing records with change_type values...")
                conn.execute("""
                    UPDATE bankroll_history
                    SET change_type = transaction_type
                    WHERE change_type IS NULL AND transaction_type IS NOT NULL
                """)
                logger.info("   ✅ Updated existing records with change_type values")

            # Verify the change
            updated_columns = conn.execute("DESCRIBE bankroll_history").fetchall()
            updated_column_names = [col[0] for col in updated_columns]
            logger.info(f"   Updated columns: {updated_column_names}")

            if 'change_type' in updated_column_names:
                logger.info("🎉 Successfully added change_type column to bankroll_history table!")
                return True
            else:
                logger.error("❌ Failed to add change_type column")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to add change_type column: {e}")
        return False

def main():
    """Main schema fix process."""
    logger.info("🚀 Starting Change Type Column Fix")
    logger.info("=" * 60)
    logger.info("Adding missing change_type column to bankroll_history table")

    success = add_change_type_to_bankroll_history()

    # Summary
    if success:
        logger.info("🎉 Change Type Column Fix COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ change_type column added to bankroll_history table")
        logger.info("✅ Database schema is now complete")
        logger.info("✅ Betting dashboard should work properly")
    else:
        logger.error("❌ Schema fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)