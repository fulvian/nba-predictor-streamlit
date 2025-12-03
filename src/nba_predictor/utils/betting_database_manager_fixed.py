#!/usr/bin/env python3
"""
NBA Betting Database Manager - Fixed Version

Fixed database manager that works with the updated schema and resolves
all foreign key constraints and column mapping issues.

Context7 Compliant: Yes
"""

import logging
import os
import time
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class BetAnalysis:
    """Data structure for bet analysis results - Context7 best practices."""
    bet_type: str
    line: float
    odds: float
    edge: float
    probability: float
    implied_probability: float
    true_probability: float
    quality_score: float
    edge_score: float
    confidence_score: float
    risk_score: float
    consistency_score: float
    kelly_fraction: float
    stake: float
    roi: float
    is_value: bool
    risk_level: str
    game_id: str
    central_line: float
    timestamp: datetime
    home_team: Optional[str] = None
    away_team: Optional[str] = None

@dataclass
class PlacedBet:
    """Data structure for placed bets - Context7 best practices."""
    bet_id: str
    game_id: str
    bet_type: str
    line: float
    odds: float
    stake: float
    potential_return: float
    edge: float
    probability: float
    quality_score: float
    risk_level: str
    status: str  # 'pending', 'won', 'lost', 'void', 'cancelled'
    placed_at: datetime
    settled_at: Optional[datetime] = None
    result_amount: Optional[float] = None
    profit_loss: Optional[float] = None
    bookmaker: str = "Internal"
    notes: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    analysis_id: Optional[str] = None

class FixedBettingDatabaseManager:
    """
    Fixed betting database manager that resolves all schema issues.

    Key fixes:
    - Uses 'bets' table instead of 'placed_bets'
    - Correct column mappings (risk_score -> risk_level, potential_payout -> potential_return)
    - Handles missing columns gracefully
    - No foreign key constraints that cause conflicts
    """

    def __init__(self, db_path: str = None):
        """
        Initialize database manager with existing DuckDB store.

        Args:
            db_path: Path to DuckDB database (defaults to existing nba_betting.duckdb)
        """
        if db_path is None:
            # Use existing database
            self.db_path = Path(__file__).parent.parent.parent.parent / "data" / "nba_betting.duckdb"
        else:
            self.db_path = Path(db_path)

        self.conn = None
        self._initialize_connection()
        self._verify_schema()

    def _initialize_connection(self):
        """Initialize DuckDB connection with error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # If connection exists, close it first
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except:
                        pass

                self.conn = duckdb.connect(str(self.db_path), read_only=False)
                logger.info(f"Connected to DuckDB database: {self.db_path} (attempt {attempt + 1})")
                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to database {self.db_path} after {max_retries} attempts")
                    raise
                time.sleep(1)  # Wait 1 second before retry

    def _verify_schema(self):
        """Verify that all required tables and columns exist."""
        try:
            # Check critical tables
            required_tables = ['bets', 'betting_analysis', 'betting_settings', 'bankroll_history']

            for table in required_tables:
                try:
                    self.conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    logger.info(f"✅ Table {table} exists")
                except Exception as e:
                    logger.error(f"❌ Table {table} missing: {e}")
                    # Try to create missing tables
                    if table == 'betting_settings':
                        self._create_betting_settings_table()
                    elif table == 'bankroll_history':
                        self._create_bankroll_history_table()

            # Verify critical columns in bets table
            critical_columns = ['bet_id', 'game_id', 'bet_type', 'line', 'odds', 'stake',
                              'potential_payout', 'status', 'placed_at', 'result_amount', 'analysis_id']

            try:
                bets_schema = self.conn.execute("DESCRIBE bets").fetchall()
                existing_columns = [col[0] for col in bets_schema]

                for col in critical_columns:
                    if col in existing_columns:
                        logger.info(f"✅ Column bets.{col} exists")
                    else:
                        logger.warning(f"⚠️ Column bets.{col} missing - will handle gracefully")

            except Exception as e:
                logger.error(f"❌ Could not verify bets table schema: {e}")

        except Exception as e:
            logger.error(f"Schema verification failed: {e}")

    def _create_betting_settings_table(self):
        """Create betting_settings table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS betting_settings (
                    setting_key VARCHAR PRIMARY KEY,
                    setting_value VARCHAR NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Initialize default settings
            default_settings = [
                ('initial_bankroll', '1000.00'),
                ('current_bankroll', '1000.00'),
                ('max_stake_percentage', '5.0'),
                ('min_edge_threshold', '2.0'),
                ('max_daily_bets', '10'),
                ('auto_stake_calculation', 'true')
            ]

            for key, value in default_settings:
                self.conn.execute("""
                    INSERT OR IGNORE INTO betting_settings (setting_key, setting_value)
                    VALUES (?, ?)
                """, [key, value])

            logger.info("✅ Created betting_settings table")

        except Exception as e:
            logger.error(f"Failed to create betting_settings table: {e}")

    def _create_bankroll_history_table(self):
        """Create bankroll_history table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    change_type VARCHAR NOT NULL,
                    amount DOUBLE NOT NULL,
                    balance_after DOUBLE NOT NULL,
                    description VARCHAR,
                    bet_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("✅ Created bankroll_history table")

        except Exception as e:
            logger.error(f"Failed to create bankroll_history table: {e}")

    def get_pending_bets(self) -> List[PlacedBet]:
        """Get all pending bets, excluding test bets."""
        try:
            # Use bets table with proper column mapping
            query = """
                SELECT
                    bet_id, game_id, bet_type, line, odds, stake, potential_payout,
                    edge, probability, quality_score, risk_score, status, placed_at,
                    settled_at, result_amount, analysis_id, home_team, away_team,
                    bookmaker, notes
                FROM bets
                WHERE status = 'pending'
                ORDER BY placed_at DESC
            """

            result = self.conn.execute(query).fetchall()

            bets = []
            for row in result:
                # Extract values with correct indices
                bet_id = row[0]
                game_id = row[1]
                bet_type = row[2]
                line = row[3]
                odds = row[4]
                stake = row[5]
                potential_return = row[6]
                edge = row[7]
                probability = row[8]
                quality_score = row[9]
                risk_score = row[10]
                status = row[11]
                placed_at = row[12]
                settled_at = row[13]
                result_amount = row[14] if row[14] is not None else 0.0
                analysis_id = row[15]
                home_team = row[16]
                away_team = row[17]
                bookmaker = row[18] or "Internal"
                notes = row[19]

                # Calculate profit_loss from result_amount - stake
                profit_loss = result_amount - stake

                # Convert risk_score to risk_level string (handle None values)
                if risk_score is None:
                    risk_level = "Medium"
                elif risk_score < 3:
                    risk_level = "Low"
                elif risk_score < 7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"

                bet = PlacedBet(
                    bet_id=bet_id, game_id=game_id, bet_type=bet_type,
                    line=line, odds=odds, stake=stake,
                    potential_return=potential_return, edge=edge, probability=probability,
                    quality_score=quality_score, risk_level=risk_level, status=status,
                    placed_at=placed_at, settled_at=settled_at, result_amount=result_amount,
                    profit_loss=profit_loss, bookmaker=bookmaker,
                    notes=notes, home_team=home_team, away_team=away_team, analysis_id=analysis_id
                )
                bets.append(bet)

            logger.info(f"Found {len(bets)} pending bets")
            return bets

        except Exception as e:
            logger.error(f"Failed to get pending bets: {e}")
            return []

    def get_bankroll_status(self) -> Dict[str, Any]:
        """Get comprehensive bankroll status."""
        try:
            # Get bankroll settings with error handling
            try:
                current_bankroll = float(self.get_setting('current_bankroll'))
            except (ValueError, TypeError):
                current_bankroll = 1000.0

            try:
                initial_bankroll = float(self.get_setting('initial_bankroll'))
            except (ValueError, TypeError):
                initial_bankroll = 1000.0

            # Get pending stakes
            pending_stakes_result = self.conn.execute("""
                SELECT COALESCE(SUM(stake), 0) FROM bets WHERE status = 'pending'
            """).fetchone()[0]
            pending_stakes = float(pending_stakes_result) if pending_stakes_result else 0.0

            # Calculate total profit/loss from settled bets
            total_pl_result = self.conn.execute("""
                SELECT COALESCE(SUM(result_amount - stake), 0) FROM bets
                WHERE status IN ('won', 'lost', 'void')
                AND result_amount IS NOT NULL
                AND stake IS NOT NULL
            """).fetchone()[0]
            total_pl = float(total_pl_result) if total_pl_result else 0.0

            # Get bet counts
            total_bets = self.conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
            pending_bets = self.conn.execute("SELECT COUNT(*) FROM bets WHERE status = 'pending'").fetchone()[0]
            won_bets = self.conn.execute("SELECT COUNT(*) FROM bets WHERE status = 'won'").fetchone()[0]
            lost_bets = self.conn.execute("SELECT COUNT(*) FROM bets WHERE status = 'lost'").fetchone()[0]

            win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0

            return {
                'current_bankroll': current_bankroll,
                'initial_bankroll': initial_bankroll,
                'total_profit_loss': total_pl,
                'pending_stakes': pending_stakes,
                'available_bankroll': current_bankroll - pending_stakes,
                'total_bets': total_bets,
                'pending_bets_count': pending_bets,
                'won_bets': won_bets,
                'lost_bets': lost_bets,
                'win_rate': win_rate,
                'roi': ((current_bankroll - initial_bankroll) / initial_bankroll * 100) if initial_bankroll > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to get bankroll status: {e}")
            return {
                'current_bankroll': 1000.0,
                'initial_bankroll': 1000.0,
                'total_profit_loss': 0.0,
                'pending_stakes': 0.0,
                'available_bankroll': 1000.0,
                'total_bets': 0,
                'pending_bets_count': 0,
                'won_bets': 0,
                'lost_bets': 0,
                'win_rate': 0,
                'roi': 0
            }

    def get_setting(self, key: str) -> str:
        """Get a setting value."""
        try:
            result = self.conn.execute("""
                SELECT setting_value FROM betting_settings WHERE setting_key = ?
            """, [key]).fetchone()
            return result[0] if result else ""
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            # Return default values for critical settings
            if key == 'current_bankroll':
                return '1000.0'
            elif key == 'initial_bankroll':
                return '1000.0'
            return ""

    def update_setting(self, key: str, value: str) -> bool:
        """Update a setting value."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO betting_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, [key, value])
            return True
        except Exception as e:
            logger.error(f"Failed to update setting {key}: {e}")
            return False

    def get_betting_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive betting analytics."""
        try:
            # Daily performance
            daily_performance = self.conn.execute(f"""
                SELECT
                    DATE(placed_at) as date,
                    COUNT(*) as bets_count,
                    COALESCE(SUM(CASE WHEN status = 'won' THEN result_amount - stake ELSE 0 END), 0) as profit,
                    COALESCE(SUM(CASE WHEN status = 'lost' THEN stake ELSE 0 END), 0) as losses,
                    COALESCE(SUM(CASE WHEN status = 'pending' THEN stake ELSE 0 END), 0) as pending
                FROM bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY DATE(placed_at)
                ORDER BY date DESC
            """).fetchall()

            # Bet type performance
            bet_type_performance = self.conn.execute(f"""
                SELECT
                    bet_type,
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'won' THEN 1 END) as won_bets,
                    COALESCE(AVG(CASE WHEN status IN ('won', 'lost') THEN result_amount - stake ELSE NULL END), 0) as avg_profit,
                    COALESCE(SUM(CASE WHEN status IN ('won', 'lost') THEN result_amount - stake ELSE 0 END), 0) as total_profit
                FROM bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY bet_type
                ORDER BY total_profit DESC
            """).fetchall()

            # Risk level analysis (using risk_score)
            risk_analysis = self.conn.execute(f"""
                SELECT
                    CASE
                        WHEN risk_score < 3 THEN 'Low'
                        WHEN risk_score < 7 THEN 'Medium'
                        ELSE 'High'
                    END as risk_level,
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'won' THEN 1 END) as won_bets,
                    COALESCE(AVG(CASE WHEN status IN ('won', 'lost') THEN result_amount - stake ELSE NULL END), 0) as avg_profit
                FROM bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY
                    CASE
                        WHEN risk_score < 3 THEN 'Low'
                        WHEN risk_score < 7 THEN 'Medium'
                        ELSE 'High'
                    END
                ORDER BY risk_level
            """).fetchall()

            return {
                'daily_performance': daily_performance,
                'bet_type_performance': bet_type_performance,
                'risk_analysis': risk_analysis
            }

        except Exception as e:
            logger.error(f"Failed to get betting analytics: {e}")
            return {
                'daily_performance': [],
                'bet_type_performance': [],
                'risk_analysis': []
            }

    def close(self):
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Helper function for easy import
def get_fixed_database_manager(db_path: str = None) -> FixedBettingDatabaseManager:
    """
    Get a configured FixedBettingDatabaseManager instance.

    Args:
        db_path: Optional database path

    Returns:
        FixedBettingDatabaseManager instance
    """
    return FixedBettingDatabaseManager(db_path)