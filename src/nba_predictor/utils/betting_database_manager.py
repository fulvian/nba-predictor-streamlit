#!/usr/bin/env python3
"""
NBA Betting Database Manager - Context7 Best Practices Implementation

Sistema professionale di persistenza scommesse integrato con DuckDB esistente.
Follows Context7 best practices for database design and error handling.
"""

import logging
import os
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

class BettingDatabaseManager:
    """
    Professional betting database manager using DuckDB - Context7 compliant.

    Features:
    - ACID transactions
    - Full bet lifecycle tracking
    - Analytics and reporting
    - Integration with existing data store
    - Error handling and logging
    - Type safety and validation
    """

    def __init__(self, db_path: str = None):
        """
        Initialize database manager with existing DuckDB store.

        Args:
            db_path: Path to DuckDB database (defaults to existing nba_data.duckdb)
        """
        if db_path is None:
            # Use existing database
            self.db_path = Path(__file__).parent.parent.parent.parent / "data" / "nba_data.duckdb"
        else:
            self.db_path = Path(db_path)

        self.conn = None
        self._initialize_connection()
        self._create_schema()

    def _initialize_connection(self):
        """Initialize DuckDB connection with error handling."""
        try:
            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to DuckDB database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database {self.db_path}: {e}")
            raise

    def _create_schema(self):
        """Create betting tables schema following Context7 best practices."""
        try:
            # Create betting_analysis table - stores all analysis results
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS betting_analysis (
                    analysis_id VARCHAR PRIMARY KEY,
                    game_id VARCHAR NOT NULL,
                    bet_type VARCHAR NOT NULL,
                    line DOUBLE NOT NULL,
                    odds DOUBLE NOT NULL,
                    edge DOUBLE NOT NULL,
                    probability DOUBLE NOT NULL,
                    implied_probability DOUBLE NOT NULL,
                    true_probability DOUBLE NOT NULL,
                    quality_score DOUBLE NOT NULL,
                    edge_score DOUBLE NOT NULL,
                    confidence_score DOUBLE NOT NULL,
                    risk_score DOUBLE NOT NULL,
                    consistency_score DOUBLE NOT NULL,
                    kelly_fraction DOUBLE NOT NULL,
                    stake DOUBLE NOT NULL,
                    roi DOUBLE NOT NULL,
                    is_value BOOLEAN NOT NULL,
                    risk_level VARCHAR NOT NULL,
                    central_line DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for betting_analysis table
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_game_created ON betting_analysis(game_id, created_at)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_value_edge ON betting_analysis(is_value, edge)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_risk ON betting_analysis(risk_level)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_created ON betting_analysis(created_at)")

            # Create placed_bets table - tracks actual placed bets
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS placed_bets (
                    bet_id VARCHAR PRIMARY KEY,
                    analysis_id VARCHAR,
                    game_id VARCHAR NOT NULL,
                    bet_type VARCHAR NOT NULL,
                    line DOUBLE NOT NULL,
                    odds DOUBLE NOT NULL,
                    stake DOUBLE NOT NULL,
                    potential_return DOUBLE NOT NULL,
                    edge DOUBLE NOT NULL,
                    probability DOUBLE NOT NULL,
                    quality_score DOUBLE NOT NULL,
                    risk_level VARCHAR NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    result_amount DOUBLE,
                    profit_loss DOUBLE,
                    bookmaker VARCHAR DEFAULT 'Internal',
                    notes VARCHAR,
                    FOREIGN KEY (analysis_id) REFERENCES betting_analysis(analysis_id)
                )
            """)

            # Create indexes for placed_bets table
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_placed_bets_game_status ON placed_bets(game_id, status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_placed_bets_placed_at ON placed_bets(placed_at)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_placed_bets_status ON placed_bets(status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_placed_bets_bookmaker ON placed_bets(bookmaker)")

            # Create bankroll_history table - tracks bankroll changes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_history (
                    history_id INTEGER PRIMARY KEY,
                    bet_id VARCHAR,
                    change_type VARCHAR NOT NULL, -- 'bet_placed', 'bet_settled', 'deposit', 'withdrawal'
                    amount DOUBLE NOT NULL,
                    balance_before DOUBLE NOT NULL,
                    balance_after DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes VARCHAR,
                    FOREIGN KEY (bet_id) REFERENCES placed_bets(bet_id)
                )
            """)

            # Create indexes for bankroll_history table
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bankroll_history_type_created ON bankroll_history(change_type, created_at)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bankroll_history_created ON bankroll_history(created_at)")

            # Create betting_settings table - system configuration
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS betting_settings (
                    setting_key VARCHAR PRIMARY KEY,
                    setting_value VARCHAR NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Initialize default settings if not exists
            self._initialize_default_settings()

            logger.info("Betting database schema created/verified successfully")

        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise

    def _initialize_default_settings(self):
        """Initialize default betting settings."""
        default_settings = {
            'initial_bankroll': '1000.00',
            'current_bankroll': '1000.00',
            'max_stake_percentage': '5.0',
            'min_edge_threshold': '2.0',
            'max_daily_bets': '10',
            'auto_stake_calculation': 'true'
        }

        for key, value in default_settings.items():
            self.conn.execute("""
                INSERT OR IGNORE INTO betting_settings (setting_key, setting_value)
                VALUES (?, ?)
            """, [key, value])

    def save_bet_analysis(self, analysis_data) -> Optional[str]:
        """
        Save bet analysis results to database.

        Args:
            analysis_data: Single BetAnalysis object or List of BetAnalysis objects

        Returns:
            analysis_id if successful (for single object), True if successful (for list), None otherwise
        """
        try:
            # Handle single object or list
            if isinstance(analysis_data, BetAnalysis):
                analysis_list = [analysis_data]
                return_single = True
            else:
                analysis_list = analysis_data
                return_single = False

            # Convert to DataFrame for efficient bulk insert
            df_data = []
            analysis_ids = []

            for analysis in analysis_list:
                analysis_dict = asdict(analysis)
                analysis_id = f"{analysis.game_id}_{analysis.bet_type}_{analysis.line}_{analysis.timestamp.strftime('%Y%m%d_%H%M%S')}"
                analysis_dict['analysis_id'] = analysis_id
                df_data.append(analysis_dict)
                analysis_ids.append(analysis_id)

            df = pd.DataFrame(df_data)

            # Ensure all numeric columns have correct types
            numeric_columns = ['line', 'odds', 'edge', 'probability', 'implied_probability',
                             'true_probability', 'quality_score', 'edge_score', 'confidence_score',
                             'risk_score', 'consistency_score', 'kelly_fraction', 'stake', 'roi']

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Use DuckDB's efficient bulk insert with explicit column mapping
            insert_query = """
                INSERT OR REPLACE INTO betting_analysis (
                    analysis_id, game_id, bet_type, line, odds, edge, probability,
                    implied_probability, true_probability, quality_score, edge_score,
                    confidence_score, risk_score, consistency_score, kelly_fraction,
                    stake, roi, is_value, risk_level, central_line, created_at
                ) SELECT
                    analysis_id, game_id, bet_type, line, odds, edge, probability,
                    implied_probability, true_probability, quality_score, edge_score,
                    confidence_score, risk_score, consistency_score, kelly_fraction,
                    stake, roi, is_value, risk_level, central_line, timestamp
                FROM df
            """
            self.conn.execute(insert_query)

            logger.info(f"Saved {len(analysis_list)} bet analysis records")

            if return_single:
                return analysis_ids[0] if analysis_ids else None
            else:
                return True

        except Exception as e:
            logger.error(f"Failed to save bet analysis: {e}")
            return None

    def get_bet_analysis(self, analysis_id: str) -> Optional[dict]:
        """
        Retrieve bet analysis by ID.

        Args:
            analysis_id: Analysis ID to retrieve

        Returns:
            Dictionary with analysis data or None if not found
        """
        try:
            result = self.conn.execute("""
                SELECT * FROM betting_analysis WHERE analysis_id = ?
            """, [analysis_id]).fetchone()

            if result:
                # Convert Row object to dictionary
                columns = [desc[0] for desc in self.conn.description]
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Failed to get bet analysis: {e}")
            return None

    def place_bet(self, analysis: BetAnalysis, selected_stake: float = None, notes: str = None) -> Optional[str]:
        """
        Place a bet based on analysis.

        Args:
            analysis: BetAnalysis object
            selected_stake: Optional override stake
            notes: Optional notes for the bet

        Returns:
            Bet ID if successful, None otherwise
        """
        try:
            # Generate unique bet ID
            bet_id = f"bet_{analysis.game_id}_{analysis.bet_type}_{analysis.line}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Use analysis stake or override
            stake = selected_stake if selected_stake is not None else analysis.stake

            # Calculate potential return
            potential_return = stake * analysis.odds

            # Get current bankroll
            current_bankroll = float(self.get_setting('current_bankroll'))

            # Validate stake against bankroll
            max_stake = current_bankroll * (float(self.get_setting('max_stake_percentage')) / 100)
            if stake > max_stake:
                stake = max_stake
                logger.warning(f"Stake reduced to maximum allowed: €{stake:.2f}")

            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")

            try:
                # Insert bet
                self.conn.execute("""
                    INSERT INTO placed_bets (
                        bet_id, analysis_id, game_id, bet_type, line, odds, stake,
                        potential_return, edge, probability, quality_score, risk_level, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    bet_id,
                    f"{analysis.game_id}_{analysis.bet_type}_{analysis.line}_{analysis.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    analysis.game_id, analysis.bet_type, analysis.line, analysis.odds, stake,
                    potential_return, analysis.edge, analysis.probability, analysis.quality_score,
                    analysis.risk_level, notes
                ])

                # Update bankroll
                new_bankroll = current_bankroll - stake
                self.update_setting('current_bankroll', str(new_bankroll))

                # Get next history_id
                next_id_result = self.conn.execute("SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history").fetchone()
                next_history_id = next_id_result[0] if next_id_result else 1

                # Record bankroll change
                self.conn.execute("""
                    INSERT INTO bankroll_history (history_id, bet_id, change_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_placed', ?, ?, ?, ?)
                """, [next_history_id, bet_id, -stake, current_bankroll, new_bankroll, f"Bet placed: {analysis.bet_type} {analysis.line}"])

                # Commit transaction
                self.conn.execute("COMMIT")

                logger.info(f"Bet placed successfully: {bet_id} - €{stake:.2f} on {analysis.bet_type} {analysis.line}")
                return bet_id

            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            return None

    def settle_bet(self, bet_id: str, result: str, final_score: float = None) -> bool:
        """
        Settle a bet with result.

        Args:
            bet_id: Bet ID to settle
            result: 'won', 'lost', 'void', or 'cancelled'
            final_score: Final score for line bets (optional)

        Returns:
            True if successful
        """
        try:
            # Get bet details
            bet_info = self.conn.execute("""
                SELECT bet_id, stake, odds, potential_return, status, game_id
                FROM placed_bets WHERE bet_id = ?
            """, [bet_id]).fetchone()

            if not bet_info:
                logger.error(f"Bet not found: {bet_id}")
                return False

            if bet_info[4] != 'pending':
                logger.warning(f"Bet {bet_id} already settled with status: {bet_info[4]}")
                return False

            stake, odds, potential_return = bet_info[1], bet_info[2], bet_info[3]

            # Calculate result based on outcome
            if result == 'won':
                result_amount = potential_return
                profit_loss = result_amount - stake
            elif result == 'lost':
                result_amount = 0
                profit_loss = -stake
            elif result == 'void':
                result_amount = stake
                profit_loss = 0
            else:  # cancelled
                result_amount = stake
                profit_loss = 0

            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")

            try:
                # Update bet status
                self.conn.execute("""
                    UPDATE placed_bets
                    SET status = ?, settled_at = CURRENT_TIMESTAMP, result_amount = ?, profit_loss = ?
                    WHERE bet_id = ?
                """, [result, result_amount, profit_loss, bet_id])

                # Update bankroll
                current_bankroll = float(self.get_setting('current_bankroll'))
                new_bankroll = current_bankroll + result_amount
                self.update_setting('current_bankroll', str(new_bankroll))

                # Get next history_id
                next_id_result = self.conn.execute("SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history").fetchone()
                next_history_id = next_id_result[0] if next_id_result else 1

                # Record bankroll change
                self.conn.execute("""
                    INSERT INTO bankroll_history (history_id, bet_id, change_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                """, [next_history_id, bet_id, result_amount, current_bankroll, new_bankroll, f"Bet settled: {result}"])

                # Commit transaction
                self.conn.execute("COMMIT")

                logger.info(f"Bet settled successfully: {bet_id} - {result}, P&L: €{profit_loss:.2f}")
                return True

            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

        except Exception as e:
            logger.error(f"Failed to settle bet {bet_id}: {e}")
            return False

    def get_pending_bets(self) -> List[PlacedBet]:
        """Get all pending bets."""
        try:
            result = self.conn.execute("""
                SELECT * FROM placed_bets WHERE status = 'pending' ORDER BY placed_at DESC
            """).fetchall()

            bets = []
            for row in result:
                bet = PlacedBet(
                    bet_id=row[0], game_id=row[2], bet_type=row[3],
                    line=row[4], odds=row[5], stake=row[6], potential_return=row[7],
                    edge=row[8], probability=row[9], quality_score=row[10],
                    risk_level=row[11], status=row[12], placed_at=row[13],
                    settled_at=row[14], result_amount=row[15], profit_loss=row[16],
                    bookmaker=row[17], notes=row[18]
                )
                bets.append(bet)

            return bets

        except Exception as e:
            logger.error(f"Failed to get pending bets: {e}")
            return []

    def get_bet_history(self, days: int = 30) -> List[PlacedBet]:
        """Get bet history for last N days."""
        try:
            result = self.conn.execute("""
                SELECT * FROM placed_bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY placed_at DESC
            """.format(days=days)).fetchall()

            bets = []
            for row in result:
                bet = PlacedBet(
                    bet_id=row[0], game_id=row[2], bet_type=row[3],
                    line=row[4], odds=row[5], stake=row[6], potential_return=row[7],
                    edge=row[8], probability=row[9], quality_score=row[10],
                    risk_level=row[11], status=row[12], placed_at=row[13],
                    settled_at=row[14], result_amount=row[15], profit_loss=row[16],
                    bookmaker=row[17], notes=row[18]
                )
                bets.append(bet)

            return bets

        except Exception as e:
            logger.error(f"Failed to get bet history: {e}")
            return []

    def get_bankroll_status(self) -> Dict[str, Any]:
        """Get comprehensive bankroll status."""
        try:
            current_bankroll = float(self.get_setting('current_bankroll'))
            initial_bankroll = float(self.get_setting('initial_bankroll'))

            # Get pending stakes
            pending_stakes = self.conn.execute("""
                SELECT COALESCE(SUM(stake), 0) FROM placed_bets WHERE status = 'pending'
            """).fetchone()[0]

            # Get total profit/loss
            total_pl = self.conn.execute("""
                SELECT COALESCE(SUM(profit_loss), 0) FROM placed_bets
                WHERE status IN ('won', 'lost', 'void') AND profit_loss IS NOT NULL
            """).fetchone()[0]

            # Get bet counts
            total_bets = self.conn.execute("""
                SELECT COUNT(*) FROM placed_bets
            """).fetchone()[0]

            pending_bets = self.conn.execute("""
                SELECT COUNT(*) FROM placed_bets WHERE status = 'pending'
            """).fetchone()[0]

            won_bets = self.conn.execute("""
                SELECT COUNT(*) FROM placed_bets WHERE status = 'won'
            """).fetchone()[0]

            lost_bets = self.conn.execute("""
                SELECT COUNT(*) FROM placed_bets WHERE status = 'lost'
            """).fetchone()[0]

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
            return {}

    def get_setting(self, key: str) -> str:
        """Get a setting value."""
        try:
            result = self.conn.execute("""
                SELECT setting_value FROM betting_settings WHERE setting_key = ?
            """, [key]).fetchone()
            return result[0] if result else ""
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
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
                    SUM(CASE WHEN status = 'won' THEN profit_loss ELSE 0 END) as profit,
                    SUM(CASE WHEN status = 'lost' THEN stake ELSE 0 END) as losses,
                    SUM(CASE WHEN status = 'pending' THEN stake ELSE 0 END) as pending
                FROM placed_bets
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
                    AVG(CASE WHEN status IN ('won', 'lost') THEN profit_loss ELSE NULL END) as avg_profit,
                    SUM(CASE WHEN status IN ('won', 'lost') THEN profit_loss ELSE 0 END) as total_profit
                FROM placed_bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY bet_type
                ORDER BY total_profit DESC
            """).fetchall()

            # Risk level analysis
            risk_analysis = self.conn.execute(f"""
                SELECT
                    risk_level,
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'won' THEN 1 END) as won_bets,
                    AVG(CASE WHEN status IN ('won', 'lost') THEN profit_loss ELSE NULL END) as avg_profit
                FROM placed_bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                GROUP BY risk_level
                ORDER BY risk_level
            """).fetchall()

            return {
                'daily_performance': daily_performance,
                'bet_type_performance': bet_type_performance,
                'risk_analysis': risk_analysis
            }

        except Exception as e:
            logger.error(f"Failed to get betting analytics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()