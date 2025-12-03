#!/usr/bin/env python3
"""
Data Migration Script for NBA Betting System
Migrates existing JSON data to robust DuckDB database with ACID transactions
Context7-compliant data migration with validation and error handling
"""

import json
import duckdb
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import shutil
from decimal import Decimal

from .schema import NBADatabaseSchema, BetStatus, GameStatus, TransactionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigrator:
    """
    Handles migration of JSON data to DuckDB with validation and error handling
    Implements Context7 best practices for data integrity
    """

    def __init__(self, db_path: str = "data/nba_betting.duckdb",
                 data_dir: str = "data"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.schema = NBADatabaseSchema(db_path)
        self.con = None

    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.con = self.schema.connect()
            logger.info("Database connection established for migration")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def backup_existing_data(self) -> str:
        """Create backup of existing JSON data before migration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.data_dir / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)

            # Files to backup
            files_to_backup = [
                "pending_bets.json",
                "bankroll.json",
                "pending_results.json"
            ]

            backed_up_files = []
            for file_name in files_to_backup:
                source_file = self.data_dir / file_name
                if source_file.exists():
                    backup_file = backup_dir / file_name
                    shutil.copy2(source_file, backup_file)
                    backed_up_files.append(str(backup_file))
                    logger.info(f"Backed up: {source_file} -> {backup_file}")

            logger.info(f"Backup completed: {backup_dir}")
            return str(backup_dir)

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

    def load_json_data(self, file_name: str) -> Optional[Dict]:
        """Load and validate JSON data from file"""
        try:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data) if isinstance(data, list) else 1} records from {file_name}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {file_name}: {e}")
            raise

    def validate_bet_data(self, bet_data: Dict) -> bool:
        """Validate bet data structure and required fields"""
        required_fields = ['bet_id', 'game_id', 'bet_data', 'timestamp', 'status']

        for field in required_fields:
            if field not in bet_data:
                logger.error(f"Missing required field in bet data: {field}")
                return False

        # Validate bet_data structure
        bet_info = bet_data['bet_data']
        required_bet_fields = ['type', 'odds', 'stake']

        for field in required_bet_fields:
            if field not in bet_info:
                logger.error(f"Missing required bet field: {field}")
                return False

        # Validate data types and ranges
        try:
            if not isinstance(bet_info['odds'], (int, float)) or bet_info['odds'] <= 0:
                logger.error(f"Invalid odds value: {bet_info['odds']}")
                return False

            if not isinstance(bet_info['stake'], (int, float)) or bet_info['stake'] <= 0:
                logger.error(f"Invalid stake value: {bet_info['stake']}")
                return False

        except Exception as e:
            logger.error(f"Bet data validation error: {e}")
            return False

        return True

    def migrate_games(self) -> Tuple[int, int]:
        """Migrate game data from pending_results.json"""
        try:
            games_data = self.load_json_data("pending_results.json")
            if not games_data:
                logger.info("No games data to migrate")
                return 0, 0

            migrated_count = 0
            error_count = 0

            self.con.execute("BEGIN TRANSACTION;")

            for game_record in games_data:
                try:
                    game_info = game_record.get('game_info', {})

                    # Parse game date
                    game_date = game_info.get('date', '')
                    if game_date:
                        try:
                            parsed_date = datetime.strptime(game_date, "%Y-%m-%d").date()
                        except ValueError:
                            # Try alternative date formats
                            parsed_date = datetime.now().date()
                    else:
                        parsed_date = datetime.now().date()

                    # Insert game record
                    self.con.execute("""
                        INSERT INTO games (
                            game_id, api_game_id, home_team, home_team_abbr,
                            away_team, away_team_abbr, game_date, game_time,
                            league, season, status, data_completeness_score,
                            data_sources_summary, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_record.get('game_id'),
                        game_info.get('api_game_id'),
                        game_info.get('home_team'),
                        game_info.get('home_team_abbr'),
                        game_info.get('away_team'),
                        game_info.get('away_team_abbr'),
                        parsed_date,
                        game_info.get('time'),
                        game_info.get('league', 'NBA'),
                        game_info.get('season'),
                        GameStatus.SCHEDULED.value,
                        game_info.get('data_completeness_score'),
                        json.dumps(game_info.get('data_sources_summary', {})),
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc)
                    ))

                    migrated_count += 1

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to migrate game {game_record.get('game_id', 'unknown')}: {e}")
                    continue

            self.con.execute("COMMIT;")
            logger.info(f"Games migration completed: {migrated_count} migrated, {error_count} errors")
            return migrated_count, error_count

        except Exception as e:
            self.con.execute("ROLLBACK;")
            logger.error(f"Games migration failed: {e}")
            raise

    def migrate_bets(self) -> Tuple[int, int]:
        """Migrate bet data from pending_bets.json"""
        try:
            bets_data = self.load_json_data("pending_bets.json")
            if not bets_data:
                logger.info("No bets data to migrate")
                return 0, 0

            migrated_count = 0
            error_count = 0

            # First, get all existing game IDs
            existing_games = set()
            try:
                game_results = self.con.execute("SELECT game_id FROM games").fetchall()
                existing_games = set(row[0] for row in game_results)
                logger.info(f"Found {len(existing_games)} existing games")
            except Exception as e:
                logger.warning(f"Failed to get existing games: {e}")

            self.con.execute("BEGIN TRANSACTION;")

            for bet_record in bets_data:
                try:
                    if not self.validate_bet_data(bet_record):
                        error_count += 1
                        continue

                    # Check if game_id exists, if not create a placeholder game
                    game_id = bet_record['game_id']
                    if game_id not in existing_games:
                        logger.warning(f"Game ID {game_id} not found, creating placeholder game")
                        try:
                            self.con.execute("""
                                INSERT INTO games (
                                    game_id, home_team, home_team_abbr, away_team, away_team_abbr,
                                    game_date, status, created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                game_id,
                                'Unknown Home Team',
                                'UNK',
                                'Unknown Away Team',
                                'UNK',
                                datetime.now().date(),
                                GameStatus.SCHEDULED.value,
                                datetime.now(timezone.utc),
                                datetime.now(timezone.utc)
                            ))
                            existing_games.add(game_id)
                        except Exception as e:
                            logger.error(f"Failed to create placeholder game {game_id}: {e}")
                            error_count += 1
                            continue

                    bet_data = bet_record['bet_data']
                    original_bet = bet_record.get('original_bet', {})

                    # Parse timestamp
                    timestamp_str = bet_record.get('timestamp', datetime.now().isoformat())
                    try:
                        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        parsed_timestamp = datetime.now(timezone.utc)

                    # Parse replaced_at timestamp if exists
                    replaced_at = None
                    if 'replaced_at' in bet_record:
                        try:
                            replaced_at = datetime.fromisoformat(bet_record['replaced_at'].replace('Z', '+00:00'))
                        except ValueError:
                            replaced_at = datetime.now(timezone.utc)

                    # Normalize scores to be within 0-1 range if needed
                    confidence_score = bet_data.get('confidence_score', 0.5)
                    risk_score = bet_data.get('risk_score', 0.5)

                    # Clamp values to valid range
                    confidence_score = max(0.0, min(1.0, float(confidence_score)))
                    risk_score = max(0.0, min(1.0, float(risk_score)))

                    # Insert bet record
                    self.con.execute("""
                        INSERT INTO bets (
                            bet_id, game_id, bet_type, line, odds, stake,
                            probability, implied_probability, true_probability, edge,
                            quality_score, confidence_score, risk_score, consistency_score, margin,
                            simulation_wins, total_simulations, status, is_value,
                            original_bet_data, replaced_at, placed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        bet_record['bet_id'],
                        bet_record['game_id'],
                        bet_data['type'],
                        bet_data.get('line'),
                        bet_data['odds'],
                        bet_data['stake'],
                        bet_data.get('probability'),
                        bet_data.get('implied_probability'),
                        bet_data.get('true_probability'),
                        bet_data.get('edge'),
                        bet_data.get('quality_score'),
                        confidence_score,
                        risk_score,
                        bet_data.get('consistency_score'),
                        bet_data.get('margin'),
                        bet_data.get('simulation_wins'),
                        bet_data.get('total_simulations'),
                        bet_record['status'],
                        bet_data.get('is_value', False),
                        json.dumps(original_bet) if original_bet else None,
                        replaced_at,
                        parsed_timestamp,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc)
                    ))

                    # Create corresponding bankroll transaction
                    # Get next transaction_id
                    max_id_result = self.con.execute("SELECT MAX(transaction_id) FROM bankroll").fetchone()
                    next_id = (max_id_result[0] + 1) if max_id_result[0] else 1

                    self.con.execute("""
                        INSERT INTO bankroll (
                            transaction_id, transaction_type, amount, balance_after, bet_id, description,
                            metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        next_id,
                        TransactionType.BET_PLACED.value,
                        -bet_data['stake'],  # Negative because money is being spent
                        0,  # Will be updated after calculating current balance
                        bet_record['bet_id'],
                        f"Bet placed: {bet_data['type']} {bet_data.get('line', '')}",
                        json.dumps({'bet_data': bet_data}),
                        parsed_timestamp
                    ))

                    migrated_count += 1

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to migrate bet {bet_record.get('bet_id', 'unknown')}: {e}")
                    continue

            self.con.execute("COMMIT;")
            logger.info(f"Bets migration completed: {migrated_count} migrated, {error_count} errors")
            return migrated_count, error_count

        except Exception as e:
            self.con.execute("ROLLBACK;")
            logger.error(f"Bets migration failed: {e}")
            raise

    def migrate_bankroll(self) -> Tuple[int, int]:
        """Migrate bankroll data from bankroll.json"""
        try:
            bankroll_data = self.load_json_data("bankroll.json")
            if not bankroll_data:
                logger.info("No bankroll data to migrate")
                return 0, 0

            migrated_count = 0
            error_count = 0

            self.con.execute("BEGIN TRANSACTION;")

            try:
                current_balance = float(bankroll_data.get('current_bankroll', 0))

                # Get next transaction_id
                max_id_result = self.con.execute("SELECT MAX(transaction_id) FROM bankroll").fetchone()
                next_id = (max_id_result[0] + 1) if max_id_result[0] else 1

                # Create initial bankroll transaction
                self.con.execute("""
                    INSERT INTO bankroll (
                        transaction_id, transaction_type, amount, balance_after, description,
                        metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    next_id,
                    TransactionType.DEPOSIT.value,
                    current_balance,
                    current_balance,
                    "Initial balance migration from JSON",
                    json.dumps({'migration_timestamp': datetime.now().isoformat()}),
                    datetime.now(timezone.utc)
                ))

                migrated_count = 1

                self.con.execute("COMMIT;")
                logger.info(f"Bankroll migration completed: {migrated_count} migrated, {error_count} errors")
                return migrated_count, error_count

            except Exception as e:
                error_count += 1
                self.con.execute("ROLLBACK;")
                logger.error(f"Failed to migrate bankroll: {e}")
                return 0, error_count

        except Exception as e:
            logger.error(f"Bankroll migration failed: {e}")
            raise

    def calculate_balances(self) -> None:
        """Calculate and update running balances for all bankroll transactions"""
        try:
            self.con.execute("BEGIN TRANSACTION;")

            # Get all transactions ordered by date
            transactions = self.con.execute("""
                SELECT transaction_id, transaction_type, amount, created_at
                FROM bankroll
                ORDER BY created_at ASC
            """).fetchall()

            running_balance = 0.0
            for transaction in transactions:
                transaction_id, transaction_type, amount, created_at = transaction

                # Update running balance
                if transaction_type in [TransactionType.BET_WON.value]:
                    running_balance += abs(amount)  # Winnings are positive
                elif transaction_type in [TransactionType.BET_PLACED.value, TransactionType.BET_LOST.value]:
                    running_balance -= abs(amount)  # Bets and losses are negative
                elif transaction_type == TransactionType.DEPOSIT.value:
                    running_balance += abs(amount)
                elif transaction_type == TransactionType.WITHDRAWAL.value:
                    running_balance -= abs(amount)

                # Update the balance_after field
                self.con.execute("""
                    UPDATE bankroll
                    SET balance_after = ?
                    WHERE transaction_id = ?
                """, (running_balance, transaction_id))

            self.con.execute("COMMIT;")
            logger.info("Bankroll balances calculated and updated successfully")

        except Exception as e:
            self.con.execute("ROLLBACK;")
            logger.error(f"Failed to calculate balances: {e}")
            raise

    def run_full_migration(self) -> Dict[str, Any]:
        """
        Run complete migration process with validation and error handling
        Returns migration report
        """
        migration_report = {
            'start_time': datetime.now().isoformat(),
            'backup_location': None,
            'migration_results': {},
            'errors': [],
            'success': False,
            'end_time': None
        }

        try:
            logger.info("Starting full data migration...")

            # Step 1: Create backup
            backup_location = self.backup_existing_data()
            migration_report['backup_location'] = backup_location

            # Step 2: Initialize database schema
            self.connect()
            self.schema.create_tables()
            self.schema.create_views()

            # Step 3: Migrate data
            logger.info("Migrating games data...")
            games_migrated, games_errors = self.migrate_games()
            migration_report['migration_results']['games'] = {
                'migrated': games_migrated,
                'errors': games_errors
            }

            logger.info("Migrating bets data...")
            bets_migrated, bets_errors = self.migrate_bets()
            migration_report['migration_results']['bets'] = {
                'migrated': bets_migrated,
                'errors': bets_errors
            }

            logger.info("Migrating bankroll data...")
            bankroll_migrated, bankroll_errors = self.migrate_bankroll()
            migration_report['migration_results']['bankroll'] = {
                'migrated': bankroll_migrated,
                'errors': bankroll_errors
            }

            # Step 4: Calculate balances
            logger.info("Calculating bankroll balances...")
            self.calculate_balances()

            # Step 5: Validate migration
            logger.info("Validating migration...")
            validation_results = self.schema.validate_schema()
            migration_report['validation'] = validation_results

            # Step 6: Generate statistics
            migration_report['statistics'] = self._generate_migration_stats()

            migration_report['success'] = True
            migration_report['end_time'] = datetime.now().isoformat()

            logger.info("Migration completed successfully!")
            return migration_report

        except Exception as e:
            migration_report['errors'].append(str(e))
            migration_report['end_time'] = datetime.now().isoformat()
            logger.error(f"Migration failed: {e}")
            raise

        finally:
            if self.con:
                self.con.close()

    def _generate_migration_stats(self) -> Dict[str, Any]:
        """Generate statistics for migrated data"""
        try:
            if not self.con:
                self.connect()

            stats = {}

            # Table row counts
            tables = ['games', 'bets', 'bankroll', 'game_results', 'betting_analysis']
            for table in tables:
                try:
                    count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    stats[f"{table}_count"] = count
                except Exception:
                    stats[f"{table}_count"] = 0

            # Bankroll summary
            try:
                bankroll_summary = self.con.execute("""
                    SELECT
                        COUNT(*) as total_transactions,
                        SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_deposits,
                        SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_withdrawals,
                        MAX(balance_after) as current_balance
                    FROM bankroll
                """).fetchone()

                stats['bankroll_summary'] = {
                    'total_transactions': bankroll_summary[0],
                    'total_deposits': bankroll_summary[1],
                    'total_withdrawals': bankroll_summary[2],
                    'current_balance': bankroll_summary[3]
                }
            except Exception as e:
                logger.error(f"Failed to generate bankroll summary: {e}")
                stats['bankroll_summary'] = {}

            # Bet summary
            try:
                bet_summary = self.con.execute("""
                    SELECT
                        COUNT(*) as total_bets,
                        COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_bets,
                        SUM(stake) as total_staked,
                        AVG(quality_score) as avg_quality_score,
                        AVG(confidence_score) as avg_confidence_score
                    FROM bets
                """).fetchone()

                stats['bet_summary'] = {
                    'total_bets': bet_summary[0],
                    'pending_bets': bet_summary[1],
                    'total_staked': bet_summary[2],
                    'avg_quality_score': bet_summary[3],
                    'avg_confidence_score': bet_summary[4]
                }
            except Exception as e:
                logger.error(f"Failed to generate bet summary: {e}")
                stats['bet_summary'] = {}

            return stats

        except Exception as e:
            logger.error(f"Failed to generate migration statistics: {e}")
            return {}

def main():
    """Run migration when script is executed directly"""
    try:
        migrator = DataMigrator()
        migration_report = migrator.run_full_migration()

        # Print migration summary
        print("\n" + "="*60)
        print("NBA BETTING DATABASE MIGRATION REPORT")
        print("="*60)
        print(f"Start Time: {migration_report['start_time']}")
        print(f"End Time: {migration_report['end_time']}")
        print(f"Success: {'✅ YES' if migration_report['success'] else '❌ NO'}")
        print(f"Backup Location: {migration_report['backup_location']}")

        print("\nMigration Results:")
        for entity, results in migration_report['migration_results'].items():
            print(f"  {entity.capitalize()}: {results['migrated']} migrated, {results['errors']} errors")

        if 'statistics' in migration_report:
            print("\nDatabase Statistics:")
            for key, value in migration_report['statistics'].items():
                print(f"  {key}: {value}")

        if migration_report['errors']:
            print("\nErrors:")
            for error in migration_report['errors']:
                print(f"  - {error}")

        print("="*60)

        if migration_report['success']:
            print("🎉 Migration completed successfully!")
        else:
            print("❌ Migration failed. Check errors above.")

    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    main()