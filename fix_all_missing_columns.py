#!/usr/bin/env python3
"""
Script per aggiungere TUTTE le colonne mancanti alla tabella betting_analysis.
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

def add_all_missing_columns():
    """
    Add all missing columns to betting_analysis table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        # Get expected fields from BetAnalysis
        sys.path.append(str(project_root / "src"))
        from nba_predictor.utils.betting_database_manager import BetAnalysis

        expected_fields = [field.name for field in BetAnalysis.__dataclass_fields__.values()]
        logger.info(f"Expected BetAnalysis fields: {expected_fields}")

        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Adding all missing columns to betting_analysis table...")

            # Check current table structure
            logger.info("   Checking current betting_analysis table structure...")
            try:
                current_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                current_columns = [col[0] for col in current_schema]
                logger.info(f"   Current columns ({len(current_columns)}): {current_columns}")
            except Exception as e:
                logger.error(f"   Could not check betting_analysis structure: {e}")
                return False

            # Define all missing columns with their types
            missing_columns = {
                'true_probability': 'DOUBLE',
                'edge_score': 'DOUBLE',
                'risk_score': 'DOUBLE',
                'consistency_score': 'DOUBLE',
                'stake': 'DOUBLE',
                'roi': 'DOUBLE',
                'is_value': 'BOOLEAN',
                'central_line': 'DOUBLE',
                'timestamp': 'TIMESTAMP'
            }

            added_columns = []

            for col_name, col_type in missing_columns.items():
                if col_name not in current_columns:
                    try:
                        logger.info(f"   Adding column: {col_name} ({col_type})")
                        conn.execute(f"ALTER TABLE betting_analysis ADD COLUMN {col_name} {col_type}")
                        added_columns.append(col_name)
                        logger.info(f"   ✅ Added {col_name}")
                    except Exception as e:
                        logger.error(f"   ❌ Failed to add {col_name}: {e}")
                        return False
                else:
                    logger.info(f"   ✅ Column {col_name} already exists")

            # Verify the final schema
            logger.info("   Verifying final table schema...")
            try:
                final_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                final_columns = [col[0] for col in final_schema]
                logger.info(f"   Final columns ({len(final_columns)}): {final_columns}")

                # Check all expected fields are present
                still_missing = [field for field in expected_fields if field not in final_columns]
                if still_missing:
                    logger.error(f"   ❌ Still missing expected fields: {still_missing}")
                    return False
                else:
                    logger.info("   ✅ All expected BetAnalysis fields are present")

            except Exception as e:
                logger.error(f"   Could not verify final schema: {e}")
                return False

            logger.info("🎉 Successfully added all missing columns!")
            logger.info(f"   Added {len(added_columns)} missing columns: {added_columns}")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to add missing columns: {e}")
        return False

def main():
    """Main fix process."""
    logger.info("🚀 Starting All Missing Columns Fix")
    logger.info("=" * 70)
    logger.info("Adding all missing columns to betting_analysis table")

    success = add_all_missing_columns()

    # Summary
    if success:
        logger.info("🎉 All Missing Columns Fix COMPLETED!")
        logger.info("=" * 70)
        logger.info("✅ All missing BetAnalysis fields added to betting_analysis table")
        logger.info("✅ Database schema now matches BetAnalysis dataclass")
        logger.info("✅ Bet placement should now work properly")
    else:
        logger.error("❌ Column fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)