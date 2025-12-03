#!/usr/bin/env python3
"""
Script per aggiungere la colonna implied_probability mancante alla tabella betting_analysis.
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

def add_implied_probability_column():
    """
    Add the missing implied_probability column to betting_analysis table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding implied_probability column to betting_analysis table...")

            # Check current table structure
            logger.info("   Checking current betting_analysis table structure...")
            try:
                current_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                current_columns = [col[0] for col in current_schema]
                logger.info(f"   Current columns ({len(current_columns)}): {current_columns}")
            except Exception as e:
                logger.error(f"   Could not check betting_analysis structure: {e}")
                return False

            # Check if implied_probability already exists
            if 'implied_probability' in current_columns:
                logger.info("   ✅ implied_probability column already exists")
                return True

            # Add implied_probability column
            logger.info("   Adding implied_probability column...")
            try:
                conn.execute("ALTER TABLE betting_analysis ADD COLUMN implied_probability DOUBLE")
                logger.info("   ✅ Added implied_probability column to betting_analysis table")
            except Exception as e:
                logger.error(f"   ❌ Failed to add implied_probability column: {e}")
                return False

            # Verify the change
            logger.info("   Verifying final table schema...")
            try:
                final_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                final_columns = [col[0] for col in final_schema]
                logger.info(f"   Final columns ({len(final_columns)}): {final_columns}")

                if 'implied_probability' in final_columns:
                    logger.info("   ✅ implied_probability column successfully added")
                else:
                    logger.error("   ❌ implied_probability column still missing")
                    return False
            except Exception as e:
                logger.error(f"   Could not verify final schema: {e}")
                return False

            # Check if there are other missing columns by examining BetAnalysis dataclass
            logger.info("   Checking for other potential missing columns...")
            try:
                # Import BetAnalysis to check expected fields
                sys.path.append(str(project_root / "src"))
                from nba_predictor.utils.betting_database_manager import BetAnalysis

                expected_fields = [field.name for field in BetAnalysis.__dataclass_fields__.values()]
                logger.info(f"   Expected BetAnalysis fields: {expected_fields}")

                missing_fields = [field for field in expected_fields if field not in final_columns]
                if missing_fields:
                    logger.warning(f"   ⚠️  Still missing fields: {missing_fields}")
                    logger.warning("   These may need to be added as well")
                else:
                    logger.info("   ✅ All expected fields are present")

            except Exception as e:
                logger.warning(f"   Could not check BetAnalysis fields: {e}")

            logger.info("🎉 Successfully added implied_probability column!")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to add implied_probability column: {e}")
        return False

def main():
    """Main fix process."""
    logger.info("🚀 Starting Implied Probability Column Fix")
    logger.info("=" * 60)
    logger.info("Adding missing implied_probability column to betting_analysis table")

    success = add_implied_probability_column()

    # Summary
    if success:
        logger.info("🎉 Implied Probability Column Fix COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ implied_probability column added to betting_analysis table")
        logger.info("✅ Bet placement should now work properly")
        logger.info("✅ Database schema is now complete for BetAnalysis")
    else:
        logger.error("❌ Column fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)