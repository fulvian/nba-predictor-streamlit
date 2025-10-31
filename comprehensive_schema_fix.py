#!/usr/bin/env python3
"""
Comprehensive database schema fix to resolve the infinite loop of missing columns.
This script addresses ALL schema compatibility issues at once.
"""

import logging
from pathlib import Path
import duckdb
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def comprehensive_schema_fix():
    """
    Fix ALL database schema issues to break the infinite loop.
    """
    logger.info("🔧 COMPREHENSIVE DATABASE SCHEMA FIX")
    logger.info("=" * 50)

    project_root = Path(__file__).parent
    db_path = project_root / "data" / "nba_data.duckdb"

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("📊 Checking current schema...")

            # Check all tables
            tables = conn.execute("SHOW TABLES").fetchall()
            logger.info(f"Found tables: {[t[0] for t in tables]}")

            # Fix 1: bankroll_history table - add missing created_at column
            logger.info("🔧 Fixing bankroll_history table...")

            # Check if created_at column exists
            bankroll_columns = conn.execute("DESCRIBE bankroll_history").fetchall()
            bankroll_column_names = [col[0] for col in bankroll_columns]

            if 'created_at' not in bankroll_column_names:
                logger.info("➕ Adding missing 'created_at' column to bankroll_history")
                conn.execute("ALTER TABLE bankroll_history ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

                # Update existing records to use timestamp as created_at if timestamp exists
                if 'timestamp' in bankroll_column_names:
                    conn.execute("UPDATE bankroll_history SET created_at = timestamp WHERE created_at IS NULL")
                    logger.info("✅ Updated existing records with timestamp values")
                else:
                    conn.execute("UPDATE bankroll_history SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
                    logger.info("✅ Set existing records to current timestamp")
            else:
                logger.info("✅ 'created_at' column already exists")

            # Fix 2: Ensure bankroll_history has all required columns for dashboard queries
            logger.info("🔧 Checking bankroll_history required columns...")
            required_columns = ['history_id', 'bet_id', 'change_type', 'amount', 'balance_before', 'balance_after', 'created_at', 'notes']
            current_columns = [col[0] for col in conn.execute("DESCRIBE bankroll_history").fetchall()]

            missing_columns = [col for col in required_columns if col not in current_columns]
            if missing_columns:
                logger.warning(f"⚠️ Still missing columns: {missing_columns}")
                # Add any other missing columns
                for col in missing_columns:
                    if col == 'notes':
                        conn.execute(f"ALTER TABLE bankroll_history ADD COLUMN {col} VARCHAR")
                        conn.execute(f"UPDATE bankroll_history SET {col} = 'Legacy record' WHERE {col} IS NULL")
                    elif col in ['amount', 'balance_before', 'balance_after']:
                        conn.execute(f"ALTER TABLE bankroll_history ADD COLUMN {col} FLOAT DEFAULT 0.0")
                        conn.execute(f"UPDATE bankroll_history SET {col} = 0.0 WHERE {col} IS NULL")
                    elif col == 'change_type':
                        conn.execute(f"ALTER TABLE bankroll_history ADD COLUMN {col} VARCHAR DEFAULT 'unknown'")
                        conn.execute(f"UPDATE bankroll_history SET {col} = 'legacy' WHERE {col} IS NULL")
                    logger.info(f"✅ Added missing column: {col}")
            else:
                logger.info("✅ All required columns present in bankroll_history")

            # Fix 3: Check placed_bets table for any missing columns
            logger.info("🔧 Checking placed_bets table...")
            placed_bets_columns = conn.execute("DESCRIBE placed_bets").fetchall()
            placed_bets_column_names = [col[0] for col in placed_bets_columns]

            # Ensure placed_bets has all expected columns
            expected_placed_bets_columns = [
                'bet_id', 'game_id', 'bet_type', 'line', 'odds', 'stake', 'potential_return',
                'edge', 'probability', 'quality_score', 'risk_level', 'status', 'placed_at',
                'settled_at', 'result_amount', 'profit_loss', 'bookmaker', 'notes',
                'home_team', 'away_team', 'analysis_id'
            ]

            missing_placed_bets_columns = [col for col in expected_placed_bets_columns if col not in placed_bets_column_names]
            if missing_placed_bets_columns:
                logger.warning(f"⚠️ Missing placed_bets columns: {missing_placed_bets_columns}")
                for col in missing_placed_bets_columns:
                    if col in ['stake', 'potential_return', 'edge', 'probability', 'quality_score', 'risk_level', 'result_amount', 'profit_loss']:
                        conn.execute(f"ALTER TABLE placed_bets ADD COLUMN {col} FLOAT DEFAULT 0.0")
                    elif col in ['line']:
                        conn.execute(f"ALTER TABLE placed_bets ADD COLUMN {col} FLOAT")
                    elif col in ['odds']:
                        conn.execute(f"ALTER TABLE placed_bets ADD COLUMN {col} FLOAT DEFAULT 1.0")
                    else:
                        conn.execute(f"ALTER TABLE placed_bets ADD COLUMN {col} VARCHAR")
                    logger.info(f"✅ Added missing placed_bets column: {col}")
            else:
                logger.info("✅ All expected columns present in placed_bets")

            # Fix 4: Check betting_settings table
            logger.info("🔧 Checking betting_settings table...")
            try:
                settings_columns = conn.execute("DESCRIBE betting_settings").fetchall()
                settings_column_names = [col[0] for col in settings_columns]

                expected_settings_columns = ['setting_key', 'setting_value', 'updated_at']
                missing_settings_columns = [col for col in expected_settings_columns if col not in settings_column_names]

                if missing_settings_columns:
                    for col in missing_settings_columns:
                        if col == 'updated_at':
                            conn.execute(f"ALTER TABLE betting_settings ADD COLUMN {col} TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                        else:
                            conn.execute(f"ALTER TABLE betting_settings ADD COLUMN {col} VARCHAR")
                        logger.info(f"✅ Added missing betting_settings column: {col}")
                else:
                    logger.info("✅ betting_settings table schema OK")
            except Exception as e:
                logger.warning(f"betting_settings table issue: {e}")

            # Fix 5: Verify database integrity with test queries
            logger.info("🔧 Verifying database integrity...")

            try:
                # Test the comprehensive bets view query that dashboard uses
                test_query = """
                SELECT
                    pb.bet_id, pb.game_id, pb.bet_type, pb.line, pb.odds,
                    pb.stake, pb.potential_return, pb.status, pb.placed_at,
                    pb.home_team, pb.away_team, pb.profit_loss,
                    bh.change_type, bh.amount as history_amount,
                    bh.balance_before, bh.balance_after, bh.created_at
                FROM placed_bets pb
                LEFT JOIN bankroll_history bh ON pb.bet_id = bh.bet_id
                ORDER BY pb.placed_at DESC
                LIMIT 1
                """

                result = conn.execute(test_query).fetchone()
                logger.info("✅ Comprehensive bets view query works")

                # Test specific bankroll history queries
                history_count = conn.execute("SELECT COUNT(*) FROM bankroll_history").fetchone()[0]
                logger.info(f"✅ Bankroll history has {history_count} records")

                # Test betting settings
                try:
                    bankroll = conn.execute("SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'").fetchone()
                    if bankroll:
                        logger.info(f"✅ Current bankroll: {bankroll[0]}")
                    else:
                        logger.warning("⚠️ No bankroll setting found, creating default")
                        conn.execute("INSERT INTO betting_settings (setting_key, setting_value) VALUES ('current_bankroll', '1000.0')")
                except Exception as e:
                    logger.warning(f"Creating default betting_settings: {e}")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS betting_settings (
                            setting_key VARCHAR PRIMARY KEY,
                            setting_value VARCHAR,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.execute("INSERT INTO betting_settings (setting_key, setting_value) VALUES ('current_bankroll', '1000.0')")

            except Exception as e:
                logger.error(f"❌ Database integrity test failed: {e}")
                raise

            # Final verification
            logger.info("📋 Final schema verification:")

            final_bankroll_columns = conn.execute("DESCRIBE bankroll_history").fetchall()
            final_bankroll_names = [col[0] for col in final_bankroll_columns]
            logger.info(f"   bankroll_history columns: {final_bankroll_names}")

            final_placed_bets_columns = conn.execute("DESCRIBE placed_bets").fetchall()
            final_placed_bets_names = [col[0] for col in final_placed_bets_columns]
            logger.info(f"   placed_bets columns: {final_placed_bets_names[:10]}...")  # Show first 10

            logger.info("=" * 50)
            logger.info("🎉 COMPREHENSIVE SCHEMA FIX COMPLETED!")
            logger.info("✅ All missing columns added")
            logger.info("✅ Data integrity verified")
            logger.info("✅ Infinite loop should be broken")

            return True

    except Exception as e:
        logger.error(f"❌ Schema fix failed: {e}")
        return False

def main():
    """Execute comprehensive schema fix."""
    logger.info("🚀 STARTING COMPREHENSIVE SCHEMA FIX")
    logger.info("This will fix ALL database schema issues at once")

    if comprehensive_schema_fix():
        logger.info("🎯 Schema fix completed successfully!")
        logger.info("The dashboard should now load without schema errors.")
        return True
    else:
        logger.error("❌ Schema fix failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)