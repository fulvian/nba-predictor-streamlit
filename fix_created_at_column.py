#!/usr/bin/env python3
"""
Script per aggiungere la colonna created_at mancante alla tabella betting_analysis.
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

def add_created_at_to_betting_analysis():
    """
    Add the missing created_at column to betting_analysis table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding created_at column to betting_analysis table...")

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

            # Check if created_at already exists
            if 'created_at' in column_names:
                logger.info("   ✅ created_at column already exists")
                return True

            # Add created_at column
            logger.info("   Adding created_at column...")
            conn.execute("ALTER TABLE betting_analysis ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            logger.info("   ✅ Added created_at column to betting_analysis table")

            # Verify the change
            updated_columns = conn.execute("DESCRIBE betting_analysis").fetchall()
            updated_column_names = [col[0] for col in updated_columns]
            logger.info(f"   Updated columns: {updated_column_names}")

            if 'created_at' in updated_column_names:
                logger.info("🎉 Successfully added created_at column to betting_analysis table!")
                return True
            else:
                logger.error("❌ Failed to add created_at column")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to add created_at column: {e}")
        return False

def main():
    """Main schema fix process."""
    logger.info("🚀 Starting Created At Column Fix")
    logger.info("=" * 60)
    logger.info("Adding missing created_at column to betting_analysis table")

    success = add_created_at_to_betting_analysis()

    # Summary
    if success:
        logger.info("🎉 Created At Column Fix COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ created_at column added to betting_analysis table")
        logger.info("✅ Database schema is now complete")
        logger.info("✅ Betting database manager should work properly")
    else:
        logger.error("❌ Schema fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)