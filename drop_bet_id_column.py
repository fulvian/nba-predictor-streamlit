#!/usr/bin/env python3
"""
Script per rimuovere la colonna bet_id dalla tabella betting_analysis.
Questa colonna non dovrebbe esistere e sta causando errori NOT NULL constraint.
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

def drop_bet_id_column():
    """
    Remove the incorrect bet_id column from betting_analysis table.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Removing bet_id column from betting_analysis table...")

            # Step 1: Check current table structure
            logger.info("   Step 1: Checking current betting_analysis table structure...")
            try:
                current_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                current_columns = [col[0] for col in current_schema]
                logger.info(f"   Current columns ({len(current_columns)}): {current_columns}")

                # Check if bet_id column exists
                if 'bet_id' not in current_columns:
                    logger.info("   ✅ bet_id column does not exist - nothing to do")
                    return True

                logger.info("   ❌ bet_id column found and needs to be removed")

            except Exception as e:
                logger.error(f"   Could not check betting_analysis structure: {e}")
                return False

            # Step 2: Check if there's any data in the table
            logger.info("   Step 2: Checking for existing data...")
            try:
                count_result = conn.execute("SELECT COUNT(*) FROM betting_analysis").fetchone()
                record_count = count_result[0]
                logger.info(f"   Found {record_count} records in betting_analysis")

            except Exception as e:
                logger.warning(f"   Could not check record count: {e}")

            # Step 3: Try to drop the column directly
            logger.info("   Step 3: Attempting to drop bet_id column...")
            try:
                # DuckDB supports ALTER TABLE DROP COLUMN
                conn.execute("ALTER TABLE betting_analysis DROP COLUMN bet_id")
                logger.info("   ✅ Successfully dropped bet_id column")

            except Exception as e:
                logger.error(f"   ❌ Failed to drop bet_id column: {e}")

                # If direct drop fails, try the backup and recreate approach
                logger.info("   Trying backup and recreate approach...")

                try:
                    # Create backup
                    conn.execute("CREATE TABLE betting_analysis_backup AS SELECT * FROM betting_analysis")
                    logger.info("   ✅ Created backup table")

                    # Drop original table
                    conn.execute("DROP TABLE betting_analysis")
                    logger.info("   ✅ Dropped original table")

                    # Recreate table without bet_id column by selecting from backup
                    conn.execute("""
                        CREATE TABLE betting_analysis AS
                        SELECT
                            analysis_id, analysis_date, model_prediction, confidence_score,
                            market_efficiency, expected_value, kelly_fraction, recommended_stake,
                            risk_assessment, notes, game_id, created_at, bet_type, line, odds,
                            edge, probability, quality_score, risk_level, recommendation,
                            confidence_level, model_predictions, home_team, away_team,
                            implied_probability, true_probability, edge_score, risk_score,
                            consistency_score, stake, roi, is_value, central_line, timestamp
                        FROM betting_analysis_backup
                    """)
                    logger.info("   ✅ Recreated table without bet_id column")

                    # Clean up backup
                    conn.execute("DROP TABLE betting_analysis_backup")
                    logger.info("   ✅ Cleaned up backup table")

                except Exception as recreate_error:
                    logger.error(f"   ❌ Backup and recreate approach also failed: {recreate_error}")
                    return False

            # Step 4: Verify the fix
            logger.info("   Step 4: Verifying the fix...")
            try:
                final_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                final_columns = [col[0] for col in final_schema]
                logger.info(f"   Final columns ({len(final_columns)}): {final_columns}")

                if 'bet_id' in final_columns:
                    logger.error("   ❌ bet_id column still exists!")
                    return False
                else:
                    logger.info("   ✅ bet_id column successfully removed")

                if 'analysis_id' not in final_columns:
                    logger.error("   ❌ analysis_id column missing!")
                    return False
                else:
                    logger.info("   ✅ analysis_id column present")

            except Exception as e:
                logger.error(f"   Could not verify final schema: {e}")
                return False

            logger.info("🎉 Successfully removed bet_id column from betting_analysis table!")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to remove bet_id column: {e}")
        return False

def main():
    """Main column removal process."""
    logger.info("🚀 Starting Bet ID Column Removal")
    logger.info("=" * 70)
    logger.info("Removing incorrect bet_id column from betting_analysis table")

    success = drop_bet_id_column()

    # Summary
    if success:
        logger.info("🎉 Bet ID Column Removal COMPLETED!")
        logger.info("=" * 70)
        logger.info("✅ betting_analysis table now has correct schema")
        logger.info("✅ Removed incorrect bet_id column")
        logger.info("✅ Kept analysis_id as primary identifier")
        logger.info("✅ Bet placement should now work properly")
    else:
        logger.error("❌ Column removal failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)