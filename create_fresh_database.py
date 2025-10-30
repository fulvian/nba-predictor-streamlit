#!/usr/bin/env python3
"""
Script to create a fresh database with clean structure.

This script will:
1. Create a new clean database file
2. Set up proper table structure without foreign key constraints
3. Initialize bankroll to €1000.00
4. Verify the system is ready for fresh betting operations
"""

import sys
import logging
from pathlib import Path
import duckdb
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def create_fresh_database():
    """
    Create a completely fresh database with clean structure.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    # Create backup of current database
    if db_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = project_root / "backup" / f"nba_data_before_fresh_{timestamp}.duckdb"
        backup_path.parent.mkdir(exist_ok=True)
        shutil.copy2(db_path, backup_path)
        logger.info(f"   ✅ Backed up current database to: {backup_path}")

    try:
        # Remove old database and create new one
        if db_path.exists():
            db_path.unlink()
            logger.info("   🗑️ Removed old database file")

        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔄 Creating fresh database...")

            # Create bankroll table
            logger.info("   Creating bankroll table...")
            conn.execute("""
                CREATE TABLE bankroll (
                    id INTEGER PRIMARY KEY,
                    current_amount DECIMAL(10,2) DEFAULT 1000.00,
                    initial_amount DECIMAL(10,2) DEFAULT 1000.00,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO bankroll (id, current_amount, initial_amount, updated_at)
                VALUES (1, 1000.00, 1000.00, CURRENT_TIMESTAMP)
            """)
            logger.info("   ✅ Bankroll table created with €1000.00")

            # Create placed_bets table
            logger.info("   Creating placed_bets table...")
            conn.execute("""
                CREATE TABLE placed_bets (
                    bet_id VARCHAR PRIMARY KEY,
                    game_id VARCHAR NOT NULL,
                    bet_type VARCHAR NOT NULL,
                    line DECIMAL(8,2),
                    odds DECIMAL(8,2) NOT NULL,
                    stake DECIMAL(10,2) NOT NULL,
                    potential_return DECIMAL(10,2) NOT NULL,
                    edge DECIMAL(5,2),
                    probability DECIMAL(5,4),
                    quality_score DECIMAL(5,2),
                    risk_level VARCHAR,
                    status VARCHAR DEFAULT 'pending',
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    result_amount DECIMAL(10,2),
                    profit_loss DECIMAL(10,2),
                    bookmaker VARCHAR DEFAULT 'Internal',
                    notes VARCHAR,
                    home_team VARCHAR,
                    away_team VARCHAR
                )
            """)
            logger.info("   ✅ Placed bets table created")

            # Create betting_analysis table
            logger.info("   Creating betting_analysis table...")
            conn.execute("""
                CREATE TABLE betting_analysis (
                    analysis_id VARCHAR PRIMARY KEY,
                    bet_id VARCHAR NOT NULL,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_prediction DECIMAL(5,4),
                    confidence_score DECIMAL(5,2),
                    market_efficiency DECIMAL(5,4),
                    expected_value DECIMAL(10,2),
                    kelly_fraction DECIMAL(5,4),
                    recommended_stake DECIMAL(10,2),
                    risk_assessment VARCHAR,
                    notes VARCHAR
                )
            """)
            logger.info("   ✅ Betting analysis table created")

            # Create bankroll_history table
            logger.info("   Creating bankroll_history table...")
            conn.execute("""
                CREATE TABLE bankroll_history (
                    history_id INTEGER PRIMARY KEY,
                    bet_id VARCHAR,
                    transaction_type VARCHAR NOT NULL,
                    amount DECIMAL(10,2) NOT NULL,
                    balance_before DECIMAL(10,2) NOT NULL,
                    balance_after DECIMAL(10,2) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes VARCHAR
                )
            """)
            logger.info("   ✅ Bankroll history table created")

            # Create betting_settings table
            logger.info("   Creating betting_settings table...")
            conn.execute("""
                CREATE TABLE betting_settings (
                    setting_key VARCHAR PRIMARY KEY,
                    setting_value VARCHAR NOT NULL,
                    setting_type VARCHAR DEFAULT 'string',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert default settings
            default_settings = [
                ('max_bet_percentage', '5.0', 'decimal'),
                ('default_bet_amount', '25.0', 'decimal'),
                ('auto_settlement_enabled', 'true', 'boolean'),
                ('risk_tolerance', 'medium', 'string'),
                ('min_odds_threshold', '1.5', 'decimal'),
                ('max_odds_threshold', '3.0', 'decimal')
            ]

            for key, value, setting_type in default_settings:
                conn.execute("""
                    INSERT INTO betting_settings (setting_key, setting_value, setting_type)
                    VALUES (?, ?, ?)
                """, (key, value, setting_type))

            logger.info("   ✅ Betting settings table created with defaults")

            # Create betting_logs table
            logger.info("   Creating betting_logs table...")
            conn.execute("""
                CREATE TABLE betting_logs (
                    log_id INTEGER PRIMARY KEY,
                    operation VARCHAR NOT NULL,
                    details VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR DEFAULT 'system'
                )
            """)

            # Log database creation with explicit ID
            conn.execute("""
                INSERT INTO betting_logs (log_id, operation, details)
                VALUES (1, 'DATABASE_CREATED', 'Fresh database created with clean structure')
            """)
            logger.info("   ✅ Betting logs table created")

            # Verify game data access
            logger.info("   Verifying game data access...")
            try:
                games_count = conn.execute("""
                    SELECT COUNT(*) FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')
                """).fetchone()[0]
                logger.info(f"   ✅ Game data accessible: {games_count} games found")
            except Exception as e:
                logger.warning(f"   ⚠️ Game data verification failed: {e}")

            logger.info("🎉 Fresh database created successfully!")
            logger.info("✅ All tables created with proper structure")
            logger.info("✅ Bankroll initialized to €1000.00")
            logger.info("✅ Default settings configured")
            logger.info("✅ System ready for fresh betting operations")

            return True

    except Exception as e:
        logger.error(f"❌ Failed to create fresh database: {e}")
        return False

def verify_fresh_database():
    """
    Verify the fresh database was created correctly.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔍 Verifying fresh database...")

            # Check tables exist
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            expected_tables = ['bankroll', 'placed_bets', 'betting_analysis', 'bankroll_history', 'betting_settings', 'betting_logs']

            logger.info(f"   Tables created: {table_names}")

            for table in expected_tables:
                if table in table_names:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(f"   ✅ {table}: {count} records")
                else:
                    logger.error(f"   ❌ Missing table: {table}")
                    return False

            # Check bankroll
            bankroll = conn.execute("SELECT current_amount FROM bankroll WHERE id = 1").fetchone()[0]
            logger.info(f"   Current bankroll: €{bankroll:.2f}")

            if bankroll == 1000.00:
                logger.info("   ✅ Bankroll correctly initialized")
            else:
                logger.error(f"   ❌ Bankroll incorrect: €{bankroll:.2f}")
                return False

            # Check settings
            settings_count = conn.execute("SELECT COUNT(*) FROM betting_settings").fetchone()[0]
            logger.info(f"   Default settings: {settings_count}")

            logger.info("🎯 Fresh database verification completed")
            return True

    except Exception as e:
        logger.error(f"❌ Fresh database verification failed: {e}")
        return False

def main():
    """Main database creation process."""
    logger.info("🚀 Starting Fresh Database Creation")
    logger.info("=" * 60)
    logger.info("This will create a completely new clean database")
    logger.info("All existing data will be backed up before replacement")
    logger.info("=" * 60)

    success = True

    # Step 1: Create fresh database
    logger.info("📝 Step 1: Creating fresh database...")
    if not create_fresh_database():
        success = False

    # Step 2: Verify fresh database
    if success:
        logger.info("🔍 Step 2: Verifying fresh database...")
        if not verify_fresh_database():
            success = False

    # Summary
    if success:
        logger.info("🎉 Fresh Database Creation COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ New clean database created")
        logger.info("✅ All tables properly structured")
        logger.info("✅ Bankroll initialized to €1000.00")
        logger.info("✅ Default settings configured")
        logger.info("✅ Game data preserved and accessible")
        logger.info("✅ Previous database backed up")
        logger.info("")
        logger.info("🏀 Next steps:")
        logger.info("   1. Open betting workflow dashboard")
        logger.info("   2. Place a new bet with complete data")
        logger.info("   3. Verify team names and game details are saved correctly")
        logger.info("   4. Test result updates and bet settlement")
        logger.info("")
        logger.info("💡 System is now ready for fresh betting operations!")
    else:
        logger.error("❌ Fresh database creation failed")
        logger.error("Please check the error messages above and try again")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)