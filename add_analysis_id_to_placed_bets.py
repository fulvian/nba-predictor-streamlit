#!/usr/bin/env python3
"""
Script per aggiungere la colonna analysis_id mancante alla tabella placed_bets.
Questa colonna è necessaria per collegare le scommesse con le analisi.
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

def add_analysis_id_to_placed_bets():
    """
    Add the missing analysis_id column to placed_bets table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding analysis_id column to placed_bets table...")

            # Step 1: Check current table structure
            logger.info("   Step 1: Checking current placed_bets table structure...")
            try:
                current_schema = conn.execute("DESCRIBE placed_bets").fetchall()
                current_columns = [col[0] for col in current_schema]
                logger.info(f"   Current columns ({len(current_columns)}): {current_columns}")

                # Check if analysis_id already exists
                if 'analysis_id' in current_columns:
                    logger.info("   ✅ analysis_id column already exists")
                    return True

            except Exception as e:
                logger.error(f"   Could not check placed_bets structure: {e}")
                return False

            # Step 2: Add analysis_id column
            logger.info("   Step 2: Adding analysis_id column...")
            try:
                conn.execute("ALTER TABLE placed_bets ADD COLUMN analysis_id VARCHAR")
                logger.info("   ✅ Added analysis_id column to placed_bets table")

            except Exception as e:
                logger.error(f"   ❌ Failed to add analysis_id column: {e}")
                return False

            # Step 3: Verify the change
            logger.info("   Step 3: Verifying table schema...")
            try:
                final_schema = conn.execute("DESCRIBE placed_bets").fetchall()
                final_columns = [col[0] for col in final_schema]
                logger.info(f"   Final columns ({len(final_columns)}): {final_columns}")

                if 'analysis_id' in final_columns:
                    logger.info("   ✅ analysis_id column successfully added")
                else:
                    logger.error("   ❌ analysis_id column still missing")
                    return False

            except Exception as e:
                logger.error(f"   Could not verify final schema: {e}")
                return False

            logger.info("🎉 Successfully added analysis_id column to placed_bets table!")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to add analysis_id column: {e}")
        return False

def main():
    """Main column addition process."""
    logger.info("🚀 Starting Analysis ID Column Addition")
    logger.info("=" * 70)
    logger.info("Adding missing analysis_id column to placed_bets table")

    success = add_analysis_id_to_placed_bets()

    # Summary
    if success:
        logger.info("🎉 Analysis ID Column Addition COMPLETED!")
        logger.info("=" * 70)
        logger.info("✅ placed_bets table now has analysis_id column")
        logger.info("✅ Bet placement should now work properly")
        logger.info("✅ Bets can be linked to their analyses")
    else:
        logger.error("❌ Column addition failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)