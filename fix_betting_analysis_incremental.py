#!/usr/bin/env python3
"""
Script incrementale per fixare la tabella betting_analysis aggiungendo le colonne mancanti.
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

def fix_betting_analysis_incremental():
    """
    Fix the betting_analysis table by adding missing columns incrementally.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Incremental Betting Analysis Table Fix Starting...")

            # Step 1: Check current table structure
            logger.info("   Step 1: Checking current betting_analysis table structure...")
            try:
                current_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                current_columns = [col[0] for col in current_schema]
                logger.info(f"   Current columns: {current_columns}")
            except Exception as e:
                logger.error(f"   Could not check betting_analysis structure: {e}")
                return False

            # Step 2: Define required columns and their types
            required_columns = {
                'analysis_id': 'VARCHAR',
                'game_id': 'VARCHAR',
                'bet_type': 'VARCHAR',
                'line': 'DOUBLE',
                'odds': 'DOUBLE',
                'edge': 'DOUBLE',
                'probability': 'DOUBLE',
                'quality_score': 'DOUBLE',
                'risk_level': 'VARCHAR',
                'recommendation': 'VARCHAR',
                'confidence_level': 'DOUBLE',
                'model_predictions': 'VARCHAR',
                'created_at': 'TIMESTAMP',
                'home_team': 'VARCHAR',
                'away_team': 'VARCHAR'
            }

            # Step 3: Add missing columns
            logger.info("   Step 3: Adding missing columns...")
            added_columns = []

            for col_name, col_type in required_columns.items():
                if col_name not in current_columns:
                    try:
                        logger.info(f"      Adding column: {col_name} ({col_type})")
                        conn.execute(f"ALTER TABLE betting_analysis ADD COLUMN {col_name} {col_type}")
                        added_columns.append(col_name)
                        logger.info(f"      ✅ Added {col_name}")
                    except Exception as e:
                        logger.error(f"      ❌ Failed to add {col_name}: {e}")
                        return False
                else:
                    logger.info(f"      ✅ Column {col_name} already exists")

            # Step 4: Verify the final schema
            logger.info("   Step 4: Verifying final table schema...")
            try:
                final_schema = conn.execute("DESCRIBE betting_analysis").fetchall()
                final_columns = [col[0] for col in final_schema]
                logger.info(f"   Final columns: {final_columns}")

                # Check for all required columns
                missing_final = [col for col in required_columns.keys() if col not in final_columns]
                if missing_final:
                    logger.error(f"   Still missing required columns: {missing_final}")
                    return False
                else:
                    logger.info("   ✅ All required columns present")
            except Exception as e:
                logger.error(f"   Could not verify final schema: {e}")
                return False

            # Step 5: Test that we can now insert a record
            logger.info("   Step 5: Testing insert operation...")
            try:
                test_analysis_id = "test_analysis_fix"
                test_values = [
                    test_analysis_id,  # analysis_id
                    "TEST_GAME_001",   # game_id
                    "Over",            # bet_type
                    225.5,             # line
                    -110,              # odds
                    2.5,               # edge
                    0.55,              # probability
                    0.8,               # quality_score
                    "Low",             # risk_level
                    "Strong Bet",      # recommendation
                    0.9,               # confidence_level
                    '{"model1": "Over"}', # model_predictions
                    datetime.now(),    # created_at
                    "Test Home",       # home_team
                    "Test Away"        # away_team
                ]

                # Clean up any existing test record
                conn.execute("DELETE FROM betting_analysis WHERE analysis_id = ?", [test_analysis_id])

                # Insert test record
                conn.execute("""
                    INSERT INTO betting_analysis (
                        analysis_id, game_id, bet_type, line, odds, edge, probability,
                        quality_score, risk_level, recommendation, confidence_level,
                        model_predictions, created_at, home_team, away_team
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, test_values)

                # Verify insertion
                test_result = conn.execute("SELECT COUNT(*) FROM betting_analysis WHERE analysis_id = ?", [test_analysis_id]).fetchone()
                if test_result[0] == 1:
                    logger.info("   ✅ Test insert successful")
                    # Clean up test record
                    conn.execute("DELETE FROM betting_analysis WHERE analysis_id = ?", [test_analysis_id])
                else:
                    logger.error("   ❌ Test insert failed")
                    return False

            except Exception as e:
                logger.error(f"   ❌ Test insert failed: {e}")
                return False

            logger.info("🎉 Successfully fixed betting_analysis table incrementally!")
            logger.info(f"   Added {len(added_columns)} missing columns: {added_columns}")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to fix betting_analysis table: {e}")
        return False

def main():
    """Main schema fix process."""
    logger.info("🚀 Starting Incremental Betting Analysis Table Fix")
    logger.info("=" * 70)
    logger.info("Fixing betting_analysis table by adding missing columns incrementally")

    success = fix_betting_analysis_incremental()

    # Summary
    if success:
        logger.info("🎉 Incremental Betting Analysis Table Fix COMPLETED!")
        logger.info("=" * 70)
        logger.info("✅ betting_analysis table now has all required columns")
        logger.info("✅ Betting system should work properly now")
        logger.info("✅ No table recreation needed - preserved existing data")
    else:
        logger.error("❌ Incremental schema fix failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    from datetime import datetime
    success = main()
    sys.exit(0 if success else 1)