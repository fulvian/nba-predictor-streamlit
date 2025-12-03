#!/usr/bin/env python3
"""
Script per fixare la tabella betting_analysis con lo schema corretto.
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

def fix_betting_analysis_table():
    """
    Drop and recreate the betting_analysis table with correct schema.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Fixing betting_analysis table schema...")

            # Drop existing table
            logger.info("   Dropping existing betting_analysis table...")
            conn.execute("DROP TABLE IF EXISTS betting_analysis")
            logger.info("   ✅ Dropped existing table")

            # Let the _create_schema method recreate it with correct schema
            logger.info("   Recreating table with correct schema...")

            # Import and use the manager to recreate schema
            sys.path.append(str(project_root / "src"))
            from nba_predictor.utils.betting_database_manager import BettingDatabaseManager

            # Create a temporary manager to trigger schema creation
            temp_manager = BettingDatabaseManager()
            temp_manager._create_schema()
            temp_manager.close()

            logger.info("   ✅ Recreated betting_analysis table with correct schema")

            # Verify the new schema
            schema = conn.execute("DESCRIBE betting_analysis").fetchall()
            logger.info("   New table schema:")
            for col in schema:
                logger.info(f"      - {col[0]}: {col[1]} (nullable: {col[2]})")

            # Check if bet_type column exists now
            column_names = [col[0] for col in schema]
            if 'bet_type' in column_names:
                logger.info("🎉 Successfully fixed betting_analysis table schema!")
                return True
            else:
                logger.error("❌ bet_type column still missing after fix")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to fix betting_analysis table: {e}")
        return False

def main():
    """Main schema fix process."""
    logger.info("🚀 Starting Betting Analysis Table Fix")
    logger.info("=" * 60)
    logger.info("Dropping and recreating betting_analysis table with correct schema")

    success = fix_betting_analysis_table()

    # Summary
    if success:
        logger.info("🎉 Betting Analysis Table Fix COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ betting_analysis table recreated with correct schema")
        logger.info("✅ Bet placement should work properly now")
        logger.info("✅ All database operations restored")
    else:
        logger.error("❌ Schema fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)