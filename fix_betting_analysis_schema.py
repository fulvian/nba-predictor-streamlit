#!/usr/bin/env python3
"""
Script per aggiungere la colonna game_id mancante alla tabella betting_analysis.
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

def add_game_id_to_betting_analysis():
    """
    Add the missing game_id column to betting_analysis table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding game_id column to betting_analysis table...")

            # Check if table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            if 'betting_analysis' not in table_names:
                logger.error("❌ betting_analysis table does not exist")
                return False

            # Check current columns
            columns = conn.execute("DESCRIBE betting_analysis").fetchall()
            column_names = [col[0] for col in columns]
            logger.info(f"   Current columns: {column_names}")

            # Check if game_id already exists
            if 'game_id' in column_names:
                logger.info("   ✅ game_id column already exists")
                return True

            # Add game_id column
            logger.info("   Adding game_id column...")
            conn.execute("ALTER TABLE betting_analysis ADD COLUMN game_id VARCHAR")
            logger.info("   ✅ Added game_id column to betting_analysis table")

            # Verify the change
            updated_columns = conn.execute("DESCRIBE betting_analysis").fetchall()
            updated_column_names = [col[0] for col in updated_columns]
            logger.info(f"   Updated columns: {updated_column_names}")

            if 'game_id' in updated_column_names:
                logger.info("🎉 Successfully added game_id column to betting_analysis table!")
                return True
            else:
                logger.error("❌ Failed to add game_id column")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to add game_id column: {e}")
        return False

def main():
    """Main schema fix process."""
    logger.info("🚀 Starting Betting Analysis Schema Fix")
    logger.info("=" * 60)
    logger.info("Adding missing game_id column to betting_analysis table")

    success = add_game_id_to_betting_analysis()

    # Summary
    if success:
        logger.info("🎉 Betting Analysis Schema Fix COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ game_id column added to betting_analysis table")
        logger.info("✅ Database schema is now complete")
        logger.info("✅ Betting management should work properly")
    else:
        logger.error("❌ Schema fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)