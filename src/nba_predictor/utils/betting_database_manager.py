#!/usr/bin/env python3
"""
NBA Betting Database Manager - Context7 Best Practices Implementation

Sistema professionale di persistenza scommesse integrato con DuckDB esistente.
Follows Context7 best practices for database design and error handling.
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
            # self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_value_edge ON betting_analysis(is_value, edge)")
            # self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_risk ON betting_analysis(risk_level)")
            # self.conn.execute("CREATE INDEX IF NOT EXISTS idx_betting_analysis_created ON betting_analysis(created_at)")

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
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    change_type VARCHAR NOT NULL, -- 'bet_placed', 'bet_settled', 'deposit', 'withdrawal'
                    amount DOUBLE NOT NULL,
                    balance_after DOUBLE NOT NULL,
                    description VARCHAR,
                    bet_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

            # Add created_at column for compatibility
            df['created_at'] = datetime.now()

            # Use DuckDB's efficient bulk insert with explicit column mapping
            insert_query = """
                INSERT OR REPLACE INTO betting_analysis (
                    analysis_id, game_id, bet_type, line, odds, edge, probability,
                    implied_probability, true_probability, quality_score, edge_score,
                    confidence_score, risk_score, consistency_score, kelly_fraction,
                    stake, roi, is_value, risk_level, central_line, timestamp,
                    home_team, away_team, created_at
                ) SELECT
                    analysis_id, game_id, bet_type, line, odds, edge, probability,
                    implied_probability, true_probability, quality_score, edge_score,
                    confidence_score, risk_score, consistency_score, kelly_fraction,
                    stake, roi, is_value, risk_level, central_line, timestamp,
                    home_team, away_team, created_at
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
            # First, save the analysis to get the analysis_id
            analysis_id = self.save_bet_analysis(analysis)
            if not analysis_id:
                logger.error("Failed to save bet analysis")
                return None

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
                # Extract team names from game_id if available
                # game_id format: "MANUAL_Home_Team_Away_Team" or actual game ID
                home_team = None
                away_team = None

                if hasattr(analysis, 'home_team') and hasattr(analysis, 'away_team'):
                    home_team = analysis.home_team
                    away_team = analysis.away_team
                else:
                    # Try to extract from game_id
                    if '_' in analysis.game_id:
                        parts = analysis.game_id.split('_')
                        if len(parts) >= 4 and parts[0] == 'MANUAL':
                            # Format: MANUAL_Home_Team_Away_Team_...
                            home_team = parts[1].replace('_', ' ')
                            away_team = parts[2].replace('_', ' ')

                # Insert bet with complete schema
                self.conn.execute("""
                    INSERT INTO placed_bets (
                        bet_id, analysis_id, game_id, bet_type, line, odds, stake,
                        potential_return, risk_level, notes,
                        home_team, away_team, status, placed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    bet_id,
                    analysis_id,  # Use the analysis_id from save_bet_analysis
                    analysis.game_id, analysis.bet_type, analysis.line, analysis.odds, stake,
                    potential_return,
                    analysis.risk_level, notes,
                    home_team, away_team, 'pending', datetime.now()
                ])

                # Update bankroll
                new_bankroll = current_bankroll - stake
                self.update_setting('current_bankroll', str(new_bankroll))

                # Get next id
                next_id_result = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM bankroll_history").fetchone()
                next_id = next_id_result[0] if next_id_result else 1

                # Record bankroll change
                self.conn.execute("""
                    INSERT INTO bankroll_history (id, bet_id, change_type, amount, balance_after, description)
                    VALUES (?, ?, 'bet_placed', ?, ?, ?)
                """, [next_id, bet_id, -stake, new_bankroll, f"Bet placed: {analysis.bet_type} {analysis.line}"])

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

                # Get next id
                next_id_result = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM bankroll_history").fetchone()
                next_id = next_id_result[0] if next_id_result else 1

                # Record bankroll change
                self.conn.execute("""
                    INSERT INTO bankroll_history (id, bet_id, change_type, amount, balance_after, description)
                    VALUES (?, ?, 'bet_settled', ?, ?, ?)
                """, [next_id, bet_id, result_amount, new_bankroll, f"Bet settled: {result}"])

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
                SELECT bet_id, game_id, bet_type, line, odds, stake, potential_return,
                       edge, probability, quality_score, risk_level, status, placed_at,
                       settled_at, result_amount, profit_loss, bookmaker, notes,
                       home_team, away_team, analysis_id
                FROM placed_bets WHERE status = 'pending' ORDER BY placed_at DESC
            """).fetchall()

            bets = []
            for row in result:
                bet = PlacedBet(
                    bet_id=row[0], game_id=row[1], bet_type=row[2],
                    line=row[3], odds=row[4], stake=row[5], potential_return=row[6],
                    edge=row[7], probability=row[8], quality_score=row[9],
                    risk_level=row[10], status=row[11], placed_at=row[12],
                    settled_at=row[13], result_amount=row[14], profit_loss=row[15],
                    bookmaker=row[16], notes=row[17]
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
                SELECT bet_id, game_id, bet_type, line, odds, stake, potential_return,
                       edge, probability, quality_score, risk_level, status, placed_at,
                       settled_at, result_amount, profit_loss, bookmaker, notes,
                       home_team, away_team, analysis_id
                FROM placed_bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY placed_at DESC
            """.format(days=days)).fetchall()

            bets = []
            for row in result:
                bet = PlacedBet(
                    bet_id=row[0], game_id=row[1], bet_type=row[2],
                    line=row[3], odds=row[4], stake=row[5], potential_return=row[6],
                    edge=row[7], probability=row[8], quality_score=row[9],
                    risk_level=row[10], status=row[11], placed_at=row[12],
                    settled_at=row[13], result_amount=row[14], profit_loss=row[15],
                    bookmaker=row[16], notes=row[17]
                )
                bets.append(bet)

            return bets

        except Exception as e:
            logger.error(f"Failed to get bet history: {e}")
            return []

    def get_bankroll_status(self) -> Dict[str, Any]:
        """Get comprehensive bankroll status."""
        try:
            # Handle Decimal to float conversion with error handling
            try:
                current_bankroll = float(self.get_setting('current_bankroll'))
            except (ValueError, TypeError):
                current_bankroll = 1000.0  # Default value

            try:
                initial_bankroll = float(self.get_setting('initial_bankroll'))
            except (ValueError, TypeError):
                initial_bankroll = 1000.0  # Default value

            # Get pending stakes (ensure float)
            pending_stakes_result = self.conn.execute("""
                SELECT COALESCE(SUM(stake), 0) FROM placed_bets WHERE status = 'pending'
            """).fetchone()[0]
            pending_stakes = float(pending_stakes_result) if pending_stakes_result else 0.0

            # Get total profit/loss (ensure float)
            total_pl_result = self.conn.execute("""
                SELECT COALESCE(SUM(profit_loss), 0) FROM placed_bets
                WHERE status IN ('won', 'lost', 'void') AND profit_loss IS NOT NULL
            """).fetchone()[0]
            total_pl = float(total_pl_result) if total_pl_result else 0.0

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

    def check_existing_bets_for_game(self, game_id: str) -> List[PlacedBet]:
        """
        Check for existing bets on a specific game.

        Args:
            game_id: Game ID to check

        Returns:
            List of existing bets for this game
        """
        try:
            result = self.conn.execute("""
                SELECT pb.bet_id, pb.game_id, pb.bet_type, pb.line, pb.odds, pb.stake, pb.potential_return,
                       ba.edge, ba.probability, ba.quality_score, pb.risk_level, pb.status, pb.placed_at,
                       pb.settled_at, pb.result_amount, pb.profit_loss, pb.bookmaker, pb.notes,
                       pb.home_team, pb.away_team, pb.analysis_id
                FROM placed_bets pb
                LEFT JOIN betting_analysis ba ON pb.analysis_id = ba.analysis_id
                WHERE pb.game_id = ?
                ORDER BY pb.placed_at DESC
            """, [game_id]).fetchall()

            bets = []
            for row in result:
                bet = PlacedBet(
                    bet_id=row[0], game_id=row[1], bet_type=row[2],
                    line=row[3], odds=row[4], stake=row[5], potential_return=row[6],
                    edge=row[7], probability=row[8], quality_score=row[9],
                    risk_level=row[10], status=row[11], placed_at=row[12],
                    settled_at=row[13], result_amount=row[14], profit_loss=row[15],
                    bookmaker=row[16], notes=row[17], home_team=row[18],
                    away_team=row[19], analysis_id=row[20]
                )
                bets.append(bet)

            return bets

        except Exception as e:
            logger.error(f"Failed to check existing bets for game {game_id}: {e}")
            return []

    def get_all_bets_comprehensive(self) -> Dict[str, List[PlacedBet]]:
        """
        Get comprehensive view of all bets separated by status.

        Returns:
            Dictionary with 'pending', 'settled', and 'all' bet lists
        """
        try:
            # Get pending bets with JOIN to betting_analysis
            pending_result = self.conn.execute("""
                SELECT pb.bet_id, pb.game_id, pb.bet_type, pb.line, pb.odds, pb.stake, pb.potential_return,
                       ba.edge, ba.probability, ba.quality_score, pb.risk_level, pb.status, pb.placed_at,
                       pb.settled_at, pb.result_amount, pb.profit_loss, pb.bookmaker, pb.notes,
                       pb.home_team, pb.away_team, pb.analysis_id
                FROM placed_bets pb
                LEFT JOIN betting_analysis ba ON pb.analysis_id = ba.analysis_id
                WHERE pb.status = 'pending'
                ORDER BY pb.placed_at DESC
            """).fetchall()

            # Get settled bets with JOIN to betting_analysis
            settled_result = self.conn.execute("""
                SELECT pb.bet_id, pb.game_id, pb.bet_type, pb.line, pb.odds, pb.stake, pb.potential_return,
                       ba.edge, ba.probability, ba.quality_score, pb.risk_level, pb.status, pb.placed_at,
                       pb.settled_at, pb.result_amount, pb.profit_loss, pb.bookmaker, pb.notes,
                       pb.home_team, pb.away_team, pb.analysis_id
                FROM placed_bets pb
                LEFT JOIN betting_analysis ba ON pb.analysis_id = ba.analysis_id
                WHERE pb.status IN ('won', 'lost', 'void', 'cancelled')
                ORDER BY pb.placed_at DESC
            """).fetchall()

            # Get all bets with JOIN to betting_analysis
            all_result = self.conn.execute("""
                SELECT pb.bet_id, pb.game_id, pb.bet_type, pb.line, pb.odds, pb.stake, pb.potential_return,
                       ba.edge, ba.probability, ba.quality_score, pb.risk_level, pb.status, pb.placed_at,
                       pb.settled_at, pb.result_amount, pb.profit_loss, pb.bookmaker, pb.notes,
                       pb.home_team, pb.away_team, pb.analysis_id
                FROM placed_bets pb
                LEFT JOIN betting_analysis ba ON pb.analysis_id = ba.analysis_id
                ORDER BY pb.placed_at DESC
            """).fetchall()

            def convert_rows_to_bets(rows):
                bets = []
                for row in rows:
                    bet = PlacedBet(
                        bet_id=row[0], game_id=row[1], bet_type=row[2],
                        line=row[3], odds=row[4], stake=row[5], potential_return=row[6],
                        edge=row[7], probability=row[8], quality_score=row[9],
                        risk_level=row[10], status=row[11], placed_at=row[12],
                        settled_at=row[13], result_amount=row[14], profit_loss=row[15],
                        bookmaker=row[16], notes=row[17], home_team=row[18],
                        away_team=row[19], analysis_id=row[20]
                    )
                    bets.append(bet)
                return bets

            return {
                'pending': convert_rows_to_bets(pending_result),
                'settled': convert_rows_to_bets(settled_result),
                'all': convert_rows_to_bets(all_result)
            }

        except Exception as e:
            logger.error(f"Failed to get comprehensive bets: {e}")
            return {'pending': [], 'settled': [], 'all': []}

    def update_game_results_from_scores(self, game_id: str, final_home_score: int, final_away_score: int) -> int:
        """
        Auto-settle pending bets for a game based on final scores.

        Args:
            game_id: Game ID to update
            final_home_score: Final home team score
            final_away_score: Final away team score

        Returns:
            Number of bets settled
        """
        try:
            # Get pending bets for this game
            pending_bets = self.conn.execute("""
                SELECT bet_id, bet_type, line, odds, stake, potential_return
                FROM placed_bets
                WHERE game_id = ? AND status = 'pending'
            """, [game_id]).fetchall()

            settled_count = 0

            for bet_row in pending_bets:
                bet_id, bet_type, line, odds, stake, potential_return = bet_row

                # Determine bet outcome based on type and scores
                result = self._determine_bet_outcome(bet_type, line, final_home_score, final_away_score)

                if result and result != 'pending':
                    # Settle the bet
                    if self.settle_bet(bet_id, result):
                        settled_count += 1
                        logger.info(f"Auto-settled bet {bet_id}: {result} (Score: {final_home_score}-{final_away_score})")

            return settled_count

        except Exception as e:
            logger.error(f"Failed to update game results for {game_id}: {e}")
            return 0

    def _determine_bet_outcome(self, bet_type: str, line: float, home_score: int, away_score: int) -> Optional[str]:
        """
        Determine the outcome of a bet based on final scores.

        Args:
            bet_type: Type of bet ('Over', 'Under', etc.)
            line: Betting line
            home_score: Final home team score
            away_score: Final away team score

        Returns:
            'won', 'lost', 'void', or None if can't determine
        """
        try:
            total_points = home_score + away_score

            if bet_type.lower() == 'over':
                if total_points > line:
                    return 'won'
                elif total_points < line:
                    return 'lost'
                else:  # Exactly on the line
                    return 'void'
            elif bet_type.lower() == 'under':
                if total_points < line:
                    return 'won'
                elif total_points > line:
                    return 'lost'
                else:  # Exactly on the line
                    return 'void'
            else:
                logger.warning(f"Unsupported bet type for auto-settlement: {bet_type}")
                return None

        except Exception as e:
            logger.error(f"Error determining bet outcome: {e}")
            return None

    def overwrite_game_bets(self, game_id: str, new_analysis: BetAnalysis, stake_override: float = None, notes: str = None) -> Optional[str]:
        """
        Overwrite all existing bets for a game with a new bet.

        Args:
            game_id: Game ID
            new_analysis: New bet analysis
            stake_override: Optional stake override
            notes: Optional notes

        Returns:
            New bet ID if successful
        """
        try:
            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")

            try:
                # Cancel all existing bets for this game
                existing_bets = self.conn.execute("""
                    SELECT bet_id FROM placed_bets
                    WHERE game_id = ? AND status = 'pending'
                """, [game_id]).fetchall()

                for bet_row in existing_bets:
                    bet_id = bet_row[0]
                    # Refund stake for cancelled bets
                    bet_info = self.conn.execute("""
                        SELECT stake FROM placed_bets WHERE bet_id = ?
                    """, [bet_id]).fetchone()

                    if bet_info:
                        stake_to_refund = bet_info[0]
                        current_bankroll = float(self.get_setting('current_bankroll'))
                        new_bankroll = current_bankroll + stake_to_refund

                        # Update bankroll
                        self.update_setting('current_bankroll', str(new_bankroll))

                        # Get next id
                        next_id_result = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM bankroll_history").fetchone()
                        next_id = next_id_result[0] if next_id_result else 1

                        # Record refund
                        self.conn.execute("""
                            INSERT INTO bankroll_history (id, bet_id, change_type, amount, balance_after, description)
                            VALUES (?, ?, 'bet_cancelled', ?, ?, ?)
                        """, [next_id, bet_id, stake_to_refund, new_bankroll, f"Cancelled bet: {bet_id}"])

                    # Update bet status
                    self.conn.execute("""
                        UPDATE placed_bets
                        SET status = 'cancelled', settled_at = CURRENT_TIMESTAMP
                        WHERE bet_id = ?
                    """, [bet_id])

                # Place new bet
                new_bet_id = self.place_bet(new_analysis, stake_override, notes)

                # Commit transaction
                self.conn.execute("COMMIT")

                logger.info(f"Overwrote {len(existing_bets)} bets for game {game_id} with new bet {new_bet_id}")
                return new_bet_id

            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

        except Exception as e:
            logger.error(f"Failed to overwrite bets for game {game_id}: {e}")
            return None

    def get_game_from_database(self, game_id: str) -> Optional[Dict]:
        """
        Get game information from the games table.

        Args:
            game_id: Game ID to search for

        Returns:
            Game information dictionary or None
        """
        try:
            # Try to find game in games table by various identifiers
            game_patterns = [
                f"%{game_id}%",  # Contains game_id
                game_id,         # Exact match
            ]

            for pattern in game_patterns:
                result = self.conn.execute("""
                    SELECT game_date, home_team, away_team, home_score, away_score, status, time
                    FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')
                    WHERE game_id LIKE ? OR
                          (home_team || ' vs ' || away_team) LIKE ? OR
                          (away_team || ' @ ' || home_team) LIKE ?
                    LIMIT 1
                """, [pattern, pattern, pattern]).fetchone()

                if result:
                    columns = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'status', 'time']
                    game_dict = dict(zip(columns, result))

                    # Check if game has been played
                    if (game_dict.get('home_score') is not None and
                        game_dict.get('away_score') is not None and
                        game_dict.get('home_score') > 0 and
                        game_dict.get('away_score') > 0):
                        game_dict['is_played'] = True
                        game_dict['final_home_score'] = game_dict['home_score']
                        game_dict['final_away_score'] = game_dict['away_score']
                    else:
                        game_dict['is_played'] = False

                    return game_dict

            return None

        except Exception as e:
            logger.error(f"Failed to get game from database: {e}")
            # Try connection recovery if database is invalidated
            if "database has been invalidated" in str(e).lower():
                logger.info("Attempting database connection recovery...")
                try:
                    self._initialize_connection()
                    # Retry the query once after recovery
                    result = self.conn.execute("""
                        SELECT game_date, home_team, away_team, home_score, away_score, status, time
                        FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')
                        WHERE game_id LIKE ?
                        LIMIT 1
                    """, [f"%{game_id}%"]).fetchone()

                    if result:
                        columns = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'status', 'time']
                        game_dict = dict(zip(columns, result))
                        game_dict['is_played'] = False
                        return game_dict

                except Exception as retry_e:
                    logger.error(f"Database recovery failed: {retry_e}")
            return None

    def sync_data_store(self) -> Dict[str, Any]:
        """
        Synchronize and backup the unified data store.

        Returns:
            Dictionary with sync status and statistics
        """
        try:
            sync_stats = {
                'timestamp': datetime.now(),
                'games_count': 0,
                'bets_count': 0,
                'analysis_count': 0,
                'history_count': 0,
                'data_integrity_check': True,
                'errors': []
            }

            # Get table counts
            sync_stats['games_count'] = self.conn.execute("SELECT COUNT(*) FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')").fetchone()[0]
            sync_stats['bets_count'] = self.conn.execute("SELECT COUNT(*) FROM placed_bets").fetchone()[0]
            sync_stats['analysis_count'] = self.conn.execute("SELECT COUNT(*) FROM betting_analysis").fetchone()[0]
            sync_stats['history_count'] = self.conn.execute("SELECT COUNT(*) FROM bankroll_history").fetchone()[0]

            # Data integrity checks
            try:
                # Check for orphaned bets (without analysis)
                orphaned_bets = self.conn.execute("""
                    SELECT COUNT(*) FROM placed_bets pb
                    LEFT JOIN betting_analysis ba ON pb.analysis_id = ba.analysis_id
                    WHERE pb.analysis_id IS NOT NULL AND ba.analysis_id IS NULL
                """).fetchone()[0]

                if orphaned_bets > 0:
                    sync_stats['data_integrity_check'] = False
                    sync_stats['errors'].append(f"Found {orphaned_bets} orphaned bets")

                # Check for negative bankroll history
                negative_balance = self.conn.execute("""
                    SELECT COUNT(*) FROM bankroll_history WHERE balance_after < 0
                """).fetchone()[0]

                if negative_balance > 0:
                    sync_stats['data_integrity_check'] = False
                    sync_stats['errors'].append(f"Found {negative_balance} negative balance records")

                # Check for invalid bet statuses
                invalid_status = self.conn.execute("""
                    SELECT COUNT(*) FROM placed_bets
                    WHERE status NOT IN ('pending', 'won', 'lost', 'void', 'cancelled')
                """).fetchone()[0]

                if invalid_status > 0:
                    sync_stats['data_integrity_check'] = False
                    sync_stats['errors'].append(f"Found {invalid_status} invalid bet statuses")

            except Exception as e:
                sync_stats['data_integrity_check'] = False
                sync_stats['errors'].append(f"Integrity check error: {str(e)}")

            # Optional: Create backup timestamp
            try:
                self.conn.execute("CREATE TABLE IF NOT EXISTS sync_log (timestamp TIMESTAMP, stats JSON)")
                import json
                self.conn.execute(
                    "INSERT INTO sync_log (timestamp, stats) VALUES (?, ?)",
                    [datetime.now(), json.dumps(sync_stats, default=str)]
                )
            except Exception as e:
                sync_stats['errors'].append(f"Backup log error: {str(e)}")

            logger.info(f"Data store sync completed: {sync_stats}")
            return sync_stats

        except Exception as e:
            logger.error(f"Data store sync failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'data_integrity_check': False
            }

    def get_data_store_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the unified data store.

        Returns:
            Dictionary with data store statistics and health
        """
        try:
            # Database size and file info
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            # Table statistics
            tables_info = {}
            tables = ['games', 'placed_bets', 'betting_analysis', 'bankroll_history', 'betting_settings']

            for table in tables:
                try:
                    count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    latest_date = None

                    if table == 'games':
                        latest_date = self.conn.execute("SELECT MAX(game_date) FROM read_parquet('/Users/fulvioventura/nba-predictor-streamlit/data/games/*.parquet')").fetchone()[0]
                    elif table in ['placed_bets', 'betting_analysis']:
                        latest_date = self.conn.execute(f"SELECT MAX(created_at) FROM {table}").fetchone()[0]
                    elif table == 'bankroll_history':
                        latest_date = self.conn.execute("SELECT MAX(created_at) FROM bankroll_history").fetchone()[0]

                    tables_info[table] = {
                        'count': count,
                        'latest_record': latest_date
                    }
                except Exception as e:
                    tables_info[table] = {'error': str(e)}

            # Bankroll status
            bankroll_status = self.get_bankroll_status()

            # Recent activity - Fixed for DuckDB compatibility
            recent_bets = self.conn.execute("""
                SELECT COUNT(*) FROM placed_bets
                WHERE placed_at >= CURRENT_DATE - INTERVAL '7 days'
            """).fetchone()[0]

            return {
                'database_path': str(self.db_path),
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'tables': tables_info,
                'bankroll_status': bankroll_status,
                'recent_activity_7_days': recent_bets,
                'last_sync': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"Failed to get data store status: {e}")
            return {'error': str(e)}

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()