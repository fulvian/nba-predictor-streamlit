#!/usr/bin/env python3
"""
NBA Betting System Foreign Key Fix - Context7 Best Practices

This module provides comprehensive fixes for the foreign key constraint issues
in the NBA betting system, focusing on automatic game creation and smart ID generation.

Key Features:
- Automatic game record creation for missing game_ids
- Smart game ID generation with consistent patterns
- Robust error handling and transaction management
- Backward compatibility with existing data
"""

import logging
import re
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import hashlib
import json

import duckdb
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GameRecord:
    """Data structure for game records with Context7 best practices."""
    game_id: str
    home_team: str
    away_team: str
    game_date: date
    home_team_abbr: Optional[str] = None
    away_team_abbr: Optional[str] = None
    game_time: Optional[str] = None
    league: str = "NBA"
    season: str = "2024-25"
    status: str = "scheduled"

class SmartGameIDGenerator:
    """
    Smart Game ID Generator for consistent game_id creation.

    This class implements multiple strategies for generating unique,
    consistent game IDs that won't conflict with existing NBA API IDs.
    """

    def __init__(self):
        self.team_mappings = self._load_team_mappings()
        self.id_cache = {}  # Cache for generated IDs

    def _load_team_mappings(self) -> Dict[str, str]:
        """Load standard NBA team name to abbreviation mappings."""
        return {
            # Eastern Conference
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Detroit Pistons": "DET", "Indiana Pacers": "IND", "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL", "New York Knicks": "NYK", "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI", "Toronto Raptors": "TOR", "Washington Wizards": "WAS",

            # Western Conference
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
            "Oklahoma City Thunder": "OKC", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Utah Jazz": "UTA"
        }

    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for consistent processing."""
        if not team_name:
            return "Unknown"

        # Remove extra whitespace and normalize
        normalized = " ".join(team_name.strip().split())

        # Try to find exact match first
        if normalized in self.team_mappings:
            return normalized

        # Try case-insensitive match
        for team, abbr in self.team_mappings.items():
            if normalized.lower() == team.lower():
                return team

        # Try partial match
        for team, abbr in self.team_mappings.items():
            if normalized.lower() in team.lower() or team.lower() in normalized.lower():
                return team

        return normalized  # Return original if no match found

    def _generate_manual_id(self, home_team: str, away_team: str, game_date: date) -> str:
        """
        Generate consistent manual game ID.

        Format: MANUAL_{DATE}_{HOME_ABBR}_{AWAY_ABBR}_{HASH}
        """
        home_norm = self._normalize_team_name(home_team)
        away_norm = self._normalize_team_name(away_team)

        home_abbr = self.team_mappings.get(home_norm, home_norm[:3].upper())
        away_abbr = self.team_mappings.get(away_norm, away_norm[:3].upper())

        # Create consistent hash
        hash_input = f"{home_norm}_{away_norm}_{game_date}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6].upper()

        return f"MANUAL_{game_date.strftime('%Y%m%d')}_{home_abbr}_{away_abbr}_{hash_suffix}"

    def generate_game_id(self, home_team: str, away_team: str, game_date: date,
                        existing_id: Optional[str] = None) -> str:
        """
        Generate or validate game ID.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date
            existing_id: Existing game ID to validate/normalize

        Returns:
            Consistent game ID
        """
        # Normalize team names
        home_norm = self._normalize_team_name(home_team)
        away_norm = self._normalize_team_name(away_team)

        # Check cache first
        cache_key = f"{home_norm}_{away_norm}_{game_date}"
        if cache_key in self.id_cache:
            return self.id_cache[cache_key]

        # If existing ID provided, normalize it if it's a manual ID
        if existing_id:
            if existing_id.startswith("MANUAL_"):
                # Normalize existing manual ID
                new_id = self._generate_manual_id(home_norm, away_norm, game_date)
                self.id_cache[cache_key] = new_id
                return new_id
            elif existing_id.startswith("CUSTOM_"):
                # Convert CUSTOM to MANUAL format
                new_id = self._generate_manual_id(home_norm, away_norm, game_date)
                self.id_cache[cache_key] = new_id
                return new_id
            else:
                # Keep existing non-manual ID (likely from NBA API)
                self.id_cache[cache_key] = existing_id
                return existing_id

        # Generate new manual ID
        new_id = self._generate_manual_id(home_norm, away_norm, game_date)
        self.id_cache[cache_key] = new_id
        return new_id

class ForeignKeyConstraintFixer:
    """
    Comprehensive fixer for foreign key constraint issues in NBA betting system.

    This class provides methods to resolve foreign key violations by automatically
    creating missing game records and ensuring data consistency.
    """

    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.id_generator = SmartGameIDGenerator()
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize database connection with error handling."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=False)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def check_and_create_missing_games(self) -> Dict[str, Any]:
        """
        Check for bets with missing game references and create them.

        Returns:
            Dictionary with results of the operation
        """
        try:
            logger.info("Checking for bets with missing game references...")

            # Find bets with game_ids that don't exist in games table
            missing_games_query = """
                SELECT DISTINCT b.game_id, b.home_team, b.away_team, b.placed_at
                FROM bets b
                LEFT JOIN games g ON b.game_id = g.game_id
                WHERE g.game_id IS NULL
                AND b.game_id IS NOT NULL
            """

            missing_games = self.conn.execute(missing_games_query).fetchall()

            if not missing_games:
                logger.info("✅ No missing games found")
                return {'created_games': 0, 'fixed_bets': 0, 'errors': []}

            logger.info(f"🔧 Found {len(missing_games)} missing game references")

            created_games = []
            errors = []

            for game_info in missing_games:
                game_id, home_team, away_team, placed_at = game_info

                try:
                    # Generate consistent game ID
                    game_date = placed_at.date() if placed_at else date.today()
                    new_game_id = self.id_generator.generate_game_id(
                        home_team, away_team, game_date, game_id
                    )

                    # Create game record
                    game_record = GameRecord(
                        game_id=new_game_id,
                        home_team=home_team or "Unknown Home Team",
                        away_team=away_team or "Unknown Away Team",
                        game_date=game_date,
                        home_team_abbr=self.id_generator.team_mappings.get(home_team, home_team[:3].upper() if home_team else "UNK"),
                        away_team_abbr=self.id_generator.team_mappings.get(away_team, away_team[:3].upper() if away_team else "UNK"),
                        league="NBA",
                        season="2024-25",
                        status="scheduled"
                    )

                    # Insert game record
                    self._create_game_record(game_record)

                    # Update bets to use new game_id if changed
                    if new_game_id != game_id:
                        self._update_bet_game_ids(game_id, new_game_id)

                    created_games.append({
                        'old_game_id': game_id,
                        'new_game_id': new_game_id,
                        'home_team': home_team,
                        'away_team': away_team
                    })

                    logger.info(f"✅ Created game record: {new_game_id} ({home_team} vs {away_team})")

                except Exception as e:
                    error_msg = f"Failed to create game for {game_id}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            fixed_bets = len(created_games)

            return {
                'created_games': len(created_games),
                'fixed_bets': fixed_bets,
                'created_game_details': created_games,
                'errors': errors
            }

        except Exception as e:
            logger.error(f"Failed to check and create missing games: {e}")
            return {'created_games': 0, 'fixed_bets': 0, 'errors': [str(e)]}

    def _create_game_record(self, game_record: GameRecord):
        """Create a game record in the database."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO games (
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

        except Exception as e:
            logger.error(f"Failed to create game record for {game_record.game_id}: {e}")
            raise

    def _update_bet_game_ids(self, old_game_id: str, new_game_id: str):
        """Update all bets to use the new game_id."""
        try:
            self.conn.execute("""
                UPDATE bets
                SET game_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE game_id = ?
            """, [new_game_id, old_game_id])

            logger.info(f"Updated bets: {old_game_id} → {new_game_id}")

        except Exception as e:
            logger.error(f"Failed to update bet game_ids from {old_game_id} to {new_game_id}: {e}")
            raise

    def validate_all_foreign_keys(self) -> Dict[str, Any]:
        """
        Validate all foreign key constraints in the database.

        Returns:
            Validation results
        """
        try:
            validation_results = {
                'bets_games_fk': self._validate_bets_games_fk(),
                'bankroll_bets_fk': self._validate_bankroll_bets_fk(),
                'game_results_games_fk': self._validate_game_results_games_fk(),
                'betting_analysis_games_fk': self._validate_betting_analysis_games_fk()
            }

            validation_results['validation_passed'] = all(
                result['valid'] for result in validation_results.values()
            )

            return validation_results

        except Exception as e:
            logger.error(f"Failed to validate foreign keys: {e}")
            return {'validation_passed': False, 'error': str(e)}

    def _validate_bets_games_fk(self) -> Dict[str, Any]:
        """Validate bets.games_id foreign key constraint."""
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) as invalid_count
                FROM bets b
                LEFT JOIN games g ON b.game_id = g.game_id
                WHERE g.game_id IS NULL
            """).fetchone()

            invalid_count = result[0] if result else 0

            return {
                'valid': invalid_count == 0,
                'invalid_count': invalid_count,
                'description': 'bets.game_id → games.game_id'
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _validate_bankroll_bets_fk(self) -> Dict[str, Any]:
        """Validate bankroll.bet_id foreign key constraint."""
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) as invalid_count
                FROM bankroll b
                LEFT JOIN bets be ON b.bet_id = be.bet_id
                WHERE b.bet_id IS NOT NULL AND be.bet_id IS NULL
            """).fetchone()

            invalid_count = result[0] if result else 0

            return {
                'valid': invalid_count == 0,
                'invalid_count': invalid_count,
                'description': 'bankroll.bet_id → bets.bet_id'
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _validate_game_results_games_fk(self) -> Dict[str, Any]:
        """Validate game_results.game_id foreign key constraint."""
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) as invalid_count
                FROM game_results gr
                LEFT JOIN games g ON gr.game_id = g.game_id
                WHERE g.game_id IS NULL
            """).fetchone()

            invalid_count = result[0] if result else 0

            return {
                'valid': invalid_count == 0,
                'invalid_count': invalid_count,
                'description': 'game_results.game_id → games.game_id'
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _validate_betting_analysis_games_fk(self) -> Dict[str, Any]:
        """Validate betting_analysis.game_id foreign key constraint."""
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) as invalid_count
                FROM betting_analysis ba
                LEFT JOIN games g ON ba.game_id = g.game_id
                WHERE g.game_id IS NULL
            """).fetchone()

            invalid_count = result[0] if result else 0

            return {
                'valid': invalid_count == 0,
                'invalid_count': invalid_count,
                'description': 'betting_analysis.game_id → games.game_id'
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def generate_comprehensive_fix_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the fix process.

        Returns:
            Detailed report with current state and recommendations
        """
        try:
            # Get current state
            validation = self.validate_all_foreign_keys()

            # Fix missing games
            fix_results = self.check_and_create_missing_games()

            # Re-validate after fixes
            post_fix_validation = self.validate_all_foreign_keys()

            # Get database statistics
            stats = self._get_database_stats()

            report = {
                'timestamp': datetime.now().isoformat(),
                'database_path': self.db_path,
                'pre_fix_validation': validation,
                'fix_results': fix_results,
                'post_fix_validation': post_fix_validation,
                'database_stats': stats,
                'recommendations': self._generate_recommendations(post_fix_validation, fix_results)
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate fix report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics."""
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

            # Manual game IDs count
            try:
                manual_games = self.conn.execute("""
                    SELECT COUNT(*) FROM games WHERE game_id LIKE 'MANUAL_%'
                """).fetchone()[0]
                stats['manual_games_count'] = manual_games
            except Exception:
                stats['manual_games_count'] = 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, validation: Dict[str, Any],
                                fix_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation and fix results."""
        recommendations = []

        if not validation['validation_passed']:
            recommendations.append("❌ Foreign key constraints still violated - manual intervention required")

            for constraint_name, result in validation.items():
                if isinstance(result, dict) and not result.get('valid', True):
                    recommendations.append(f"   • Fix {result.get('description', constraint_name)}")

        if fix_results.get('errors'):
            recommendations.append("⚠️ Some errors occurred during automatic fixing:")
            recommendations.extend(f"   • {error}" for error in fix_results['errors'])

        if fix_results.get('created_games', 0) > 0:
            recommendations.append(f"✅ Successfully created {fix_results['created_games']} missing game records")
            recommendations.append("📋 Review created game records for accuracy")

        if validation['validation_passed'] and not fix_results.get('errors'):
            recommendations.append("✅ All foreign key constraints are satisfied")
            recommendations.append("🎯 System is ready for production use")

        return recommendations

    def close(self):
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

# Context7-compliant helper function
def fix_foreign_key_constraints(db_path: str = "data/nba_betting.duckdb") -> Dict[str, Any]:
    """
    Comprehensive foreign key constraint fixing function.

    Args:
        db_path: Path to the DuckDB database

    Returns:
        Comprehensive fix report
    """
    fixer = None
    try:
        fixer = ForeignKeyConstraintFixer(db_path)
        return fixer.generate_comprehensive_fix_report()
    finally:
        if fixer:
            fixer.close()

if __name__ == "__main__":
    # Run the fix when executed directly
    logging.basicConfig(level=logging.INFO)
    logger.info("🔧 Starting NBA Betting Foreign Key Constraint Fix...")

    report = fix_foreign_key_constraints()

    print("\n" + "="*80)
    print("🎯 NBA BETTING FOREIGN KEY FIX REPORT")
    print("="*80)
    print(f"Timestamp: {report.get('timestamp')}")
    print(f"Database: {report.get('database_path')}")

    if 'error' in report:
        print(f"❌ ERROR: {report['error']}")
    else:
        print(f"\n📊 FIX RESULTS:")
        print(f"   Games Created: {report['fix_results']['created_games']}")
        print(f"   Bets Fixed: {report['fix_results']['fixed_bets']}")
        print(f"   Errors: {len(report['fix_results']['errors'])}")

        print(f"\n✅ POST-FIX VALIDATION:")
        print(f"   All Constraints Valid: {report['post_fix_validation']['validation_passed']}")

        print(f"\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")

    print("="*80)