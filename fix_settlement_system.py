#!/usr/bin/env python3
"""
🎯 NBA Betting Settlement System Fix
Comprehensive fix for database schema and settlement filtering issues

Root Causes Fixed:
1. Database schema inconsistencies
2. Missing columns in active_bets view
3. Broken test bet filtering
4. Settlement system processing test bets
"""

import duckdb
import logging
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SettlementSystemFix:
    """Comprehensive fix for the settlement system"""

    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = Path(db_path)
        self.conn = None

    def connect(self):
        """Connect to database"""
        try:
            if not self.db_path.exists():
                logger.error(f"Database file not found: {self.db_path}")
                return False

            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def fix_active_bets_view(self):
        """Fix the active_bets view to include all necessary columns"""
        try:
            logger.info("🔧 Fixing active_bets view...")

            # Drop the view if it exists
            self.conn.execute("DROP VIEW IF EXISTS active_bets")

            # Create the corrected view with all necessary columns
            self.conn.execute("""
                CREATE VIEW active_bets AS
                SELECT
                    b.bet_id,
                    b.game_id,
                    b.bet_type,
                    b.line,
                    b.odds,
                    b.stake,
                    b.potential_payout,
                    b.probability,
                    b.implied_probability,
                    b.true_probability,
                    b.edge,
                    b.quality_score,
                    b.confidence_score,
                    b.risk_score,
                    b.consistency_score,
                    b.margin,
                    b.simulation_wins,
                    b.total_simulations,
                    b.status,
                    b.is_value,
                    b.original_bet_data,
                    b.replaced_at,
                    b.placed_at,
                    b.settled_at,
                    b.created_at,
                    b.updated_at,

                    -- Add game information (may be NULL for test bets)
                    COALESCE(g.home_team, 'Unknown') as home_team,
                    COALESCE(g.away_team, 'Unknown') as away_team,
                    COALESCE(g.game_date, CURRENT_DATE) as game_date,
                    COALESCE(g.home_score, 0) as home_score,
                    COALESCE(g.away_score, 0) as away_score,
                    COALESCE(g.status, 'unknown') as game_status

                FROM bets b
                LEFT JOIN games g ON b.game_id = g.game_id
                WHERE b.status IN ('pending', 'placed')
                ORDER BY b.placed_at DESC
            """)

            logger.info("✅ active_bets view fixed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to fix active_bets view: {e}")
            return False

    def add_missing_columns(self):
        """Add missing columns to bets table if they don't exist"""
        try:
            logger.info("🔧 Checking and adding missing columns...")

            # Check if columns exist and add if missing
            columns_to_add = [
                ("potential_payout", "DOUBLE"),
                ("probability", "DOUBLE"),
                ("implied_probability", "DOUBLE"),
                ("true_probability", "DOUBLE"),
                ("edge", "DOUBLE"),
                ("quality_score", "DOUBLE"),
                ("confidence_score", "DOUBLE"),
                ("risk_score", "DOUBLE"),
                ("consistency_score", "DOUBLE"),
                ("margin", "DOUBLE"),
                ("simulation_wins", "INTEGER"),
                ("total_simulations", "INTEGER"),
                ("is_value", "BOOLEAN"),
                ("original_bet_data", "VARCHAR"),
                ("replaced_at", "TIMESTAMP")
            ]

            for col_name, col_type in columns_to_add:
                try:
                    # Try to select the column to see if it exists
                    self.conn.execute(f"SELECT {col_name} FROM bets LIMIT 1")
                    logger.debug(f"Column {col_name} already exists")
                except:
                    # Column doesn't exist, add it
                    self.conn.execute(f"ALTER TABLE bets ADD COLUMN {col_name} {col_type}")
                    logger.info(f"✅ Added column: {col_name}")

            logger.info("✅ All missing columns checked/added")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add missing columns: {e}")
            return False

    def identify_test_bets(self):
        """Identify all test bets in the system"""
        try:
            logger.info("🔍 Identifying test bets...")

            # Query for test bets with comprehensive filtering
            test_patterns = [
                "TEST%", "EXAMPLE%", "CUSTOM%", "MANUAL%",
                "DEMO%", "SAMPLE%", "MOCK%", "FAKE%"
            ]

            test_bets = []
            for pattern in test_patterns:
                results = self.conn.execute("""
                    SELECT bet_id, game_id, status, placed_at
                    FROM bets
                    WHERE game_id LIKE ?
                    ORDER BY placed_at DESC
                """, [pattern]).fetchall()

                test_bets.extend(results)

            logger.info(f"📊 Found {len(test_bets)} test bets:")
            for bet in test_bets[:10]:  # Show first 10
                logger.info(f"  - {bet[0]} (game_id: {bet[1]}, status: {bet[2]})")

            if len(test_bets) > 10:
                logger.info(f"  ... and {len(test_bets) - 10} more")

            return test_bets

        except Exception as e:
            logger.error(f"❌ Failed to identify test bets: {e}")
            return []

    def cancel_test_bets(self):
        """Cancel all test bets to prevent settlement processing"""
        try:
            logger.info("🚫 Cancelling test bets...")

            test_patterns = [
                "TEST%", "EXAMPLE%", "CUSTOM%", "MANUAL%",
                "DEMO%", "SAMPLE%", "MOCK%", "FAKE%"
            ]

            total_cancelled = 0
            for pattern in test_patterns:
                result = self.conn.execute("""
                    UPDATE bets
                    SET status = 'cancelled',
                        settled_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE game_id LIKE ? AND status = 'pending'
                """, [pattern])

                cancelled_count = result.rowcount if hasattr(result, 'rowcount') else 0
                total_cancelled += cancelled_count
                logger.info(f"  - Cancelled {cancelled_count} bets for pattern {pattern}")

            logger.info(f"✅ Total test bets cancelled: {total_cancelled}")
            return total_cancelled

        except Exception as e:
            logger.error(f"❌ Failed to cancel test bets: {e}")
            return 0

    def create_filtered_pending_bets_view(self):
        """Create a new view that filters out test bets"""
        try:
            logger.info("🔧 Creating filtered_pending_bets view...")

            # Drop view if it exists
            self.conn.execute("DROP VIEW IF EXISTS filtered_pending_bets")

            # Create filtered view that excludes test bets
            self.conn.execute("""
                CREATE VIEW filtered_pending_bets AS
                SELECT *
                FROM active_bets
                WHERE status = 'pending'
                AND game_id NOT LIKE 'TEST%'
                AND game_id NOT LIKE 'EXAMPLE%'
                AND game_id NOT LIKE 'CUSTOM%'
                AND game_id NOT LIKE 'MANUAL%'
                AND game_id NOT LIKE 'DEMO%'
                AND game_id NOT LIKE 'SAMPLE%'
                AND game_id NOT LIKE 'MOCK%'
                AND game_id NOT LIKE 'FAKE%'
                AND home_team NOT LIKE '%Test%'
                AND home_team NOT LIKE '%Unknown%'
                AND away_team NOT LIKE '%Test%'
                AND away_team NOT LIKE '%Unknown%'
                ORDER BY placed_at DESC
            """)

            logger.info("✅ filtered_pending_bets view created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create filtered_pending_bets view: {e}")
            return False

    def backup_current_state(self):
        """Create backup of current betting data"""
        try:
            logger.info("💾 Creating backup of current state...")

            backup_dir = Path("data/backups")
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"settlement_fix_backup_{timestamp}.parquet"

            # Export all bets to backup file
            self.conn.execute(f"""
                COPY bets TO '{backup_file}' (FORMAT PARQUET)
            """)

            logger.info(f"✅ Backup created: {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return None

    def run_comprehensive_fix(self):
        """Run all fixes in sequence"""
        logger.info("🚀 Starting comprehensive settlement system fix...")

        if not self.connect():
            return False

        try:
            # Create backup first
            backup_file = self.backup_current_state()

            # Step 1: Fix database schema
            if not self.add_missing_columns():
                return False

            # Step 2: Fix views
            if not self.fix_active_bets_view():
                return False

            if not self.create_filtered_pending_bets_view():
                return False

            # Step 3: Identify and cancel test bets
            test_bets = self.identify_test_bets()
            cancelled_count = self.cancel_test_bets()

            # Step 4: Validate the fix
            logger.info("🔍 Validating the fix...")
            pending_count = self.conn.execute("""
                SELECT COUNT(*) FROM filtered_pending_bets
            """).fetchone()[0]

            logger.info(f"📊 Remaining real pending bets: {pending_count}")

            logger.info("✅ Comprehensive fix completed successfully!")

            return {
                'success': True,
                'test_bets_found': len(test_bets),
                'test_bets_cancelled': cancelled_count,
                'real_pending_bets': pending_count,
                'backup_file': str(backup_file) if backup_file else None
            }

        except Exception as e:
            logger.error(f"❌ Comprehensive fix failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Main execution function"""
    print("🏀 NBA Betting Settlement System Fix")
    print("=" * 50)

    fix = SettlementSystemFix()
    result = fix.run_comprehensive_fix()

    if result.get('success'):
        print("\n✅ Settlement system fixed successfully!")
        print(f"   Test bets cancelled: {result.get('test_bets_cancelled', 0)}")
        print(f"   Real pending bets remaining: {result.get('real_pending_bets', 0)}")
        if result.get('backup_file'):
            print(f"   Backup created: {result['backup_file']}")
    else:
        print(f"\n❌ Fix failed: {result.get('error', 'Unknown error')}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())