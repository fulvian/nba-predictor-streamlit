#!/usr/bin/env python3
"""
Enhanced NBA Betting Database Manager - Context7 Best Practices

Enhanced version of the betting database manager with integrated foreign key fixing
and smart game ID generation capabilities.

Key Features:
- Integrated foreign key constraint resolution
- Smart game ID generation and normalization
- Automatic game record creation for missing references
- Enhanced error handling and recovery
- Backward compatibility with existing code
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import duckdb
import pandas as pd

# Import our foreign key fixer
from foreign_key_fix import ForeignKeyConstraintFixer, SmartGameIDGenerator, GameRecord

# Import original classes
from src.nba_predictor.utils.betting_database_manager import BetAnalysis, PlacedBet

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnhancedBetAnalysis:
    """Enhanced BetAnalysis with smart game ID handling."""
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
    # Enhanced fields
    validated_game_id: Optional[str] = None
    game_record_created: bool = False

class EnhancedBettingDatabaseManager:
    """
    Enhanced betting database manager with integrated foreign key fixing.

    This manager extends the original functionality with:
    - Automatic foreign key constraint resolution
    - Smart game ID generation
    - Enhanced error handling and recovery
    - Comprehensive validation and reporting
    """

    def __init__(self, db_path: str = None):
        """
        Initialize enhanced database manager.

        Args:
            db_path: Path to DuckDB database
        """
        if db_path is None:
            self.db_path = Path(__file__).parent.parent.parent / "data" / "nba_betting.duckdb"
        else:
            self.db_path = Path(db_path)

        self.conn = None
        self.id_generator = SmartGameIDGenerator()
        self.fk_fixer = None

        self._initialize_connection()
        self._initialize_fk_fixer()
        self._verify_and_fix_schema()

    def _initialize_connection(self):
        """Initialize database connection with enhanced error handling."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except:
                        pass

                self.conn = duckdb.connect(str(self.db_path), read_only=False)
                logger.info(f"Connected to database: {self.db_path} (attempt {attempt + 1})")
                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def _initialize_fk_fixer(self):
        """Initialize the foreign key fixer."""
        try:
            self.fk_fixer = ForeignKeyConstraintFixer(str(self.db_path))
            logger.info("Foreign key fixer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FK fixer: {e}")
            # Continue without FK fixer for backward compatibility
            self.fk_fixer = None

    def _verify_and_fix_schema(self):
        """Verify schema and fix any issues."""
        try:
            # Check and fix missing games immediately
            if self.fk_fixer:
                fix_results = self.fk_fixer.check_and_create_missing_games()
                if fix_results['created_games'] > 0:
                    logger.info(f"Fixed {fix_results['created_games']} missing game records on startup")

        except Exception as e:
            logger.error(f"Schema verification failed: {e}")

    def place_bet_with_fk_protection(self, bet_analysis: EnhancedBetAnalysis,
                                   stake_override: Optional[float] = None,
                                   notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Place a bet with full foreign key protection.

        Args:
            bet_analysis: Enhanced bet analysis with game information
            stake_override: Optional stake override
            notes: Optional bet notes

        Returns:
            Result dictionary with success status and details
        """
        try:
            # Start transaction for atomic operation
            self.conn.execute("BEGIN TRANSACTION")

            # Step 1: Validate and normalize game_id
            validated_game_id, game_created = self._validate_and_create_game_record(bet_analysis)

            # Step 2: Create unique bet_id
            bet_id = self._generate_bet_id(validated_game_id, bet_analysis.bet_type, bet_analysis.line)

            # Step 3: Prepare bet data
            stake = stake_override if stake_override is not None else bet_analysis.stake

            # Step 4: Insert bet record
            self._insert_bet_record(bet_id, validated_game_id, bet_analysis, stake, notes)

            # Step 5: Update bankroll
            bankroll_updated = self._update_bankroll_for_bet(bet_id, -stake, "bet_placed")

            # Step 6: Save analysis record
            analysis_id = self._save_enhanced_analysis(bet_analysis, validated_game_id, bet_id)

            # Commit transaction
            self.conn.execute("COMMIT")

            logger.info(f"Successfully placed bet: {bet_id} for game {validated_game_id}")

            return {
                'success': True,
                'bet_id': bet_id,
                'game_id': validated_game_id,
                'analysis_id': analysis_id,
                'stake': stake,
                'game_record_created': game_created,
                'bankroll_updated': bankroll_updated
            }

        except Exception as e:
            # Rollback on error
            try:
                self.conn.execute("ROLLBACK")
            except:
                pass

            logger.error(f"Failed to place bet: {e}")
            return {
                'success': False,
                'error': str(e),
                'bet_id': None,
                'game_id': bet_analysis.game_id
            }

    def _validate_and_create_game_record(self, bet_analysis: EnhancedBetAnalysis) -> Tuple[str, bool]:
        """
        Validate game_id and create game record if needed.

        Args:
            bet_analysis: Bet analysis with game information

        Returns:
            Tuple of (validated_game_id, game_record_created)
        """
        try:
            # Generate normalized game_id
            game_date = bet_analysis.timestamp.date()
            validated_game_id = self.id_generator.generate_game_id(
                bet_analysis.home_team or "Unknown Home",
                bet_analysis.away_team or "Unknown Away",
                game_date,
                bet_analysis.game_id
            )

            # Check if game record exists
            existing_game = self.conn.execute("""
                SELECT game_id FROM games WHERE game_id = ?
            """, [validated_game_id]).fetchone()

            game_created = False

            if not existing_game:
                # Create game record
                game_record = GameRecord(
                    game_id=validated_game_id,
                    home_team=bet_analysis.home_team or "Unknown Home Team",
                    away_team=bet_analysis.away_team or "Unknown Away Team",
                    game_date=game_date,
                    home_team_abbr=self.id_generator.team_mappings.get(bet_analysis.home_team, "UNK"),
                    away_team_abbr=self.id_generator.team_mappings.get(bet_analysis.away_team, "UNK"),
                    league="NBA",
                    season="2024-25",
                    status="scheduled"
                )

                self.conn.execute("""
                    INSERT INTO games (
                        game_id, home_team, away_team, home_team_abbr, away_team_abbr,
                        game_date, league, season, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, [
                    game_record.game_id,
                    game_record.home_team,
                    game_record.away_team,
                    game_record.home_team_abbr,
                    game_record.away_team_abbr,
                    game_record.game_date,
                    game_record.league,
                    game_record.season,
                    game_record.status
                ])

                game_created = True
                logger.info(f"Created game record: {validated_game_id}")

            # Update bet_analysis with validated ID
            bet_analysis.validated_game_id = validated_game_id
            bet_analysis.game_record_created = game_created

            return validated_game_id, game_created

        except Exception as e:
            logger.error(f"Failed to validate/create game record: {e}")
            raise

    def _generate_bet_id(self, game_id: str, bet_type: str, line: float) -> str:
        """Generate unique bet ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        safe_game_id = game_id.replace('-', '_').replace(' ', '_')[:20]
        return f"BET_{safe_game_id}_{bet_type}_{line}_{timestamp}"

    def _insert_bet_record(self, bet_id: str, game_id: str, bet_analysis: EnhancedBetAnalysis,
                          stake: float, notes: Optional[str]):
        """Insert bet record with all required fields."""
        try:
            self.conn.execute("""
                INSERT INTO bets (
                    bet_id, game_id, bet_type, line, odds, stake,
                    edge, probability, implied_probability, true_probability,
                    quality_score, confidence_score, risk_score,
                    simulation_wins, total_simulations,
                    status, placed_at, home_team, away_team,
                    bookmaker, notes, analysis_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                bet_id,
                game_id,
                bet_analysis.bet_type,
                bet_analysis.line,
                bet_analysis.odds,
                stake,
                bet_analysis.edge,
                bet_analysis.probability,
                bet_analysis.implied_probability,
                bet_analysis.true_probability,
                bet_analysis.quality_score,
                bet_analysis.confidence_score,
                bet_analysis.risk_score,
                int(bet_analysis.probability * 100),  # simulation_wins
                100,  # total_simulations
                'pending',
                bet_analysis.timestamp,
                bet_analysis.home_team,
                bet_analysis.away_team,
                "Internal System",
                notes,
                None  # analysis_id will be set later
            ])

        except Exception as e:
            logger.error(f"Failed to insert bet record: {e}")
            raise

    def _update_bankroll_for_bet(self, bet_id: str, amount: float,
                               transaction_type: str) -> bool:
        """Update bankroll for bet transaction."""
        try:
            # Get current bankroll
            current_bankroll = self._get_current_bankroll()

            # Calculate new balance
            new_balance = current_bankroll + amount

            # Insert bankroll transaction
            self.conn.execute("""
                INSERT INTO bankroll (
                    transaction_type, amount, balance_after, bet_id,
                    description, created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                transaction_type,
                amount,
                new_balance,
                bet_id,
                f"{transaction_type.replace('_', ' ').title()}: {bet_id}"
            ])

            # Update current bankroll setting
            self._update_bankroll_setting(new_balance)

            return True

        except Exception as e:
            logger.error(f"Failed to update bankroll: {e}")
            return False

    def _save_enhanced_analysis(self, bet_analysis: EnhancedBetAnalysis,
                              game_id: str, bet_id: str) -> Optional[str]:
        """Save enhanced bet analysis."""
        try:
            # Generate analysis_id
            analysis_id = self._generate_analysis_id(game_id, bet_analysis.bet_type)

            self.conn.execute("""
                INSERT INTO betting_analysis (
                    analysis_id, game_id, predicted_total, prediction_date,
                    bookmaker_line, bookmaker_over_quote, bet_recommendation, bet_stake,
                    model_confidence, edge_percentage, expected_value, analysis_version,
                    model_metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                analysis_id,
                game_id,
                bet_analysis.line,
                bet_analysis.timestamp,
                bet_analysis.central_line,
                bet_analysis.odds,
                f"{bet_analysis.bet_type} {bet_analysis.line}",
                bet_analysis.stake,
                bet_analysis.confidence_score,
                bet_analysis.edge * 100,  # Convert to percentage
                (bet_analysis.odds - 1) * bet_analysis.edge,  # Expected value approximation
                "2.0",  # Enhanced version
                json.dumps({
                    'quality_score': bet_analysis.quality_score,
                    'edge_score': bet_analysis.edge_score,
                    'risk_level': bet_analysis.risk_level,
                    'consistency_score': bet_analysis.consistency_score,
                    'kelly_fraction': bet_analysis.kelly_fraction,
                    'roi': bet_analysis.roi,
                    'is_value': bet_analysis.is_value,
                    'enhanced': True
                }),
                bet_analysis.timestamp,
                datetime.now()
            ])

            # Update bet record with analysis_id
            self.conn.execute("""
                UPDATE bets SET analysis_id = ? WHERE bet_id = ?
            """, [analysis_id, bet_id])

            return analysis_id

        except Exception as e:
            logger.error(f"Failed to save enhanced analysis: {e}")
            return None

    def _generate_analysis_id(self, game_id: str, bet_type: str) -> str:
        """Generate unique analysis ID."""
        timestamp = int(datetime.now().timestamp())
        safe_game_id = game_id.replace('-', '_')[:15]
        return f"ANA_{safe_game_id}_{bet_type}_{timestamp}"

    def _get_current_bankroll(self) -> float:
        """Get current bankroll from settings."""
        try:
            result = self.conn.execute("""
                SELECT CAST(setting_value AS REAL) FROM betting_settings
                WHERE setting_key = 'current_bankroll'
            """).fetchone()

            return result[0] if result else 1000.0

        except Exception:
            return 1000.0

    def _update_bankroll_setting(self, new_balance: float):
        """Update bankroll setting."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO betting_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, ['current_bankroll', str(new_balance)])
        except Exception as e:
            logger.error(f"Failed to update bankroll setting: {e}")

    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including foreign key validation.

        Returns:
            Detailed system status report
        """
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'database_path': str(self.db_path),
                'connection_status': 'connected' if self.conn else 'disconnected',
                'fk_fixer_available': self.fk_fixer is not None
            }

            # Foreign key validation
            if self.fk_fixer:
                status['foreign_key_validation'] = self.fk_fixer.validate_all_foreign_keys()

            # Database statistics
            status['database_stats'] = self._get_database_statistics()

            # System health
            status['system_health'] = self._assess_system_health(status)

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_health': 'critical'
            }

    def _get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            stats = {}

            # Table counts
            tables = ['games', 'bets', 'bankroll', 'game_results', 'betting_analysis']
            for table in tables:
                try:
                    count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    stats[f'{table}_count'] = count
                except Exception:
                    stats[f'{table}_count'] = 0

            # Recent activity
            try:
                recent_bets = self.conn.execute("""
                    SELECT COUNT(*) FROM bets
                    WHERE placed_at >= CURRENT_DATE - INTERVAL '7 days'
                """).fetchone()[0]
                stats['recent_bets_7_days'] = recent_bets
            except Exception:
                stats['recent_bets_7_days'] = 0

            # Manual vs automatic games
            try:
                manual_games = self.conn.execute("""
                    SELECT COUNT(*) FROM games WHERE game_id LIKE 'MANUAL_%'
                """).fetchone()[0]
                api_games = stats.get('games_count', 0) - manual_games
                stats['manual_games'] = manual_games
                stats['api_games'] = api_games
            except Exception:
                stats['manual_games'] = 0
                stats['api_games'] = 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {'error': str(e)}

    def _assess_system_health(self, status: Dict[str, Any]) -> str:
        """Assess overall system health."""
        try:
            if status.get('connection_status') != 'connected':
                return 'critical'

            if not status.get('fk_fixer_available'):
                return 'warning'

            fk_validation = status.get('foreign_key_validation', {})
            if not fk_validation.get('validation_passed', True):
                return 'critical'

            stats = status.get('database_stats', {})
            if stats.get('bets_count', 0) == 0:
                return 'info'  # No bets yet is normal

            return 'healthy'

        except Exception:
            return 'unknown'

    # Enhanced methods with backward compatibility
    def get_pending_bets(self) -> List[PlacedBet]:
        """Get all pending bets with enhanced information."""
        try:
            query = """
                SELECT
                    b.bet_id, b.game_id, b.bet_type, b.line, b.odds, b.stake,
                    b.edge, b.probability, b.quality_score, b.risk_score,
                    b.status, b.placed_at, b.settled_at, b.result_amount,
                    b.analysis_id, b.home_team, b.away_team, b.bookmaker, b.notes,
                    g.game_date
                FROM bets b
                LEFT JOIN games g ON b.game_id = g.game_id
                WHERE b.status = 'pending'
                ORDER BY b.placed_at DESC
            """

            result = self.conn.execute(query).fetchall()

            bets = []
            for row in result:
                # Convert risk_score to risk_level
                risk_score = row[10] or 5
                if risk_score < 3:
                    risk_level = "Low"
                elif risk_score < 7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"

                bet = PlacedBet(
                    bet_id=row[0],
                    game_id=row[1],
                    bet_type=row[2],
                    line=row[3],
                    odds=row[4],
                    stake=row[5],
                    potential_return=row[5] * row[4],  # stake * odds
                    edge=row[6],
                    probability=row[7],
                    quality_score=row[8],
                    risk_level=risk_level,
                    status=row[11],
                    placed_at=row[12],
                    settled_at=row[13],
                    result_amount=row[14] if row[14] is not None else 0.0,
                    profit_loss=(row[14] - row[5]) if row[14] is not None else None,
                    bookmaker=row[18] or "Internal",
                    notes=row[19],
                    home_team=row[16],
                    away_team=row[17],
                    analysis_id=row[15]
                )
                bets.append(bet)

            logger.info(f"Found {len(bets)} pending bets")
            return bets

        except Exception as e:
            logger.error(f"Failed to get pending bets: {e}")
            return []

    def close(self):
        """Close database connection and cleanup."""
        if self.fk_fixer:
            self.fk_fixer.close()

        if self.conn:
            try:
                self.conn.close()
                logger.info("Enhanced database manager closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

# Context7-compliant factory function
def get_enhanced_database_manager(db_path: str = None) -> EnhancedBettingDatabaseManager:
    """
    Get an enhanced betting database manager instance.

    Args:
        db_path: Optional database path

    Returns:
        EnhancedBettingDatabaseManager instance
    """
    return EnhancedBettingDatabaseManager(db_path)

if __name__ == "__main__":
    # Test the enhanced manager
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Enhanced NBA Betting Database Manager...")

    manager = None
    try:
        manager = get_enhanced_database_manager()
        status = manager.get_comprehensive_system_status()

        print("\n" + "="*60)
        print("🎯 ENHANCED DATABASE MANAGER STATUS")
        print("="*60)
        print(f"System Health: {status['system_health']}")
        print(f"Connection: {status['connection_status']}")
        print(f"FK Fixer: {'Available' if status['fk_fixer_available'] else 'Not Available'}")

        if 'database_stats' in status:
            stats = status['database_stats']
            print(f"\n📊 Database Statistics:")
            print(f"   Games: {stats.get('games_count', 0)}")
            print(f"   Bets: {stats.get('bets_count', 0)}")
            print(f"   Manual Games: {stats.get('manual_games', 0)}")
            print(f"   Recent Bets (7d): {stats.get('recent_bets_7_days', 0)}")

        if 'foreign_key_validation' in status:
            fk_val = status['foreign_key_validation']
            print(f"\n🔒 Foreign Key Validation:")
            print(f"   All Valid: {fk_val.get('validation_passed', 'Unknown')}")

        print("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        if manager:
            manager.close()