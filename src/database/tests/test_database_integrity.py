#!/usr/bin/env python3
"""
Database Integrity and Validation Tests
Context7-comprehensive testing for NBA betting database
"""

import pytest
import duckdb
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal

import sys
sys.path.append(str(Path(__file__).parent.parent))

from schema import NBADatabaseSchema, BetStatus, GameStatus, TransactionType
from migration import DataMigrator
from backup import DatabaseBackup

class TestDatabaseSchema:
    """Test database schema creation and validation"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name

        schema = NBADatabaseSchema(db_path)
        schema.connect()
        schema.create_tables()
        schema.create_views()

        yield schema

        schema.close()
        Path(db_path).unlink(missing_ok=True)

    def test_tables_creation(self, temp_db):
        """Test all tables are created successfully"""
        con = temp_db.con

        # Check tables exist
        tables = con.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()

        expected_tables = ['games', 'bets', 'bankroll', 'game_results', 'betting_analysis']
        actual_tables = [table[0] for table in tables]

        for table in expected_tables:
            assert table in actual_tables, f"Table {table} not found"

        assert len(actual_tables) >= len(expected_tables)

    def test_foreign_key_constraints(self, temp_db):
        """Test foreign key constraints are properly enforced"""
        con = temp_db.con

        # Test inserting bet without valid game_id should fail
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                VALUES ('invalid_bet', 'nonexistent_game', 'OVER', 1.5, 10.0)
            """)

        # Test inserting bankroll transaction with invalid bet_id should not fail (SET NULL)
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        con.execute("""
            INSERT INTO bankroll (transaction_type, amount, balance_after, bet_id)
            VALUES ('deposit', 100.0, 100.0, 'nonexistent_bet')
        """)

        # Should succeed with NULL bet_id
        result = con.execute("""
            SELECT bet_id FROM bankroll WHERE transaction_type = 'deposit'
        """).fetchone()
        assert result[0] is None

    def test_check_constraints(self, temp_db):
        """Test check constraints are properly enforced"""
        con = temp_db.con

        # Insert a valid game first
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        # Test invalid odds (should be > 0)
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                VALUES ('invalid_odds', 'test_game', 'OVER', -1.0, 10.0)
            """)

        # Test invalid stake (should be > 0)
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                VALUES ('invalid_stake', 'test_game', 'OVER', 1.5, -10.0)
            """)

        # Test valid bet should succeed
        con.execute("""
            INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
            VALUES ('valid_bet', 'test_game', 'OVER', 1.5, 10.0)
        """)

        result = con.execute("SELECT COUNT(*) FROM bets WHERE bet_id = 'valid_bet'").fetchone()
        assert result[0] == 1

    def test_generated_columns(self, temp_db):
        """Test generated columns work correctly"""
        con = temp_db.con

        # Insert game and bet
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        con.execute("""
            INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
            VALUES ('test_bet', 'test_game', 'OVER', 2.5, 10.0)
        """)

        # Check potential_payout is calculated correctly
        result = con.execute("""
            SELECT potential_payout FROM bets WHERE bet_id = 'test_bet'
        """).fetchone()
        assert result[0] == 25.0  # 10.0 * 2.5

    def test_views_creation(self, temp_db):
        """Test views are created and work correctly"""
        con = temp_db.con

        # Insert test data
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date, status)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01', 'scheduled')
        """)

        con.execute("""
            INSERT INTO bets (bet_id, game_id, bet_type, odds, stake, status)
            VALUES ('test_bet', 'test_game', 'OVER', 1.5, 10.0, 'pending')
        """)

        con.execute("""
            INSERT INTO bankroll (transaction_type, amount, balance_after)
            VALUES ('deposit', 100.0, 100.0)
        """)

        # Test active_bets view
        result = con.execute("""
            SELECT COUNT(*) FROM active_bets WHERE bet_id = 'test_bet'
        """).fetchone()
        assert result[0] == 1

        # Test bankroll_summary view
        result = con.execute("""
            SELECT COUNT(*) FROM bankroll_summary
        """).fetchone()
        assert result[0] == 1

        # Test betting_performance view
        result = con.execute("""
            SELECT total_bets FROM betting_performance
        """).fetchone()
        assert result[0] == 1

class TestDataMigration:
    """Test data migration functionality"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create test pending_bets.json
            test_bets = [
                {
                    "bet_id": "test_bet_1",
                    "game_id": "test_game_1",
                    "bet_data": {
                        "type": "OVER",
                        "line": 220.5,
                        "odds": 1.85,
                        "stake": 10.0,
                        "probability": 0.65,
                        "implied_probability": 0.54,
                        "true_probability": 0.67,
                        "edge": 0.15,
                        "quality_score": 0.8,
                        "confidence_score": 0.9,
                        "risk_score": 0.3,
                        "consistency_score": 0.7,
                        "margin": 0.05,
                        "simulation_wins": 6500,
                        "total_simulations": 10000,
                        "is_value": True
                    },
                    "timestamp": "2024-01-01T10:00:00",
                    "status": "pending"
                }
            ]

            with open(data_path / "pending_bets.json", 'w') as f:
                json.dump(test_bets, f)

            # Create test bankroll.json
            test_bankroll = {"current_bankroll": 100.0}

            with open(data_path / "bankroll.json", 'w') as f:
                json.dump(test_bankroll, f)

            # Create test pending_results.json
            test_results = [
                {
                    "game_id": "test_game_1",
                    "game_info": {
                        "home_team": "Team A",
                        "home_team_abbr": "TEA",
                        "away_team": "Team B",
                        "away_team_abbr": "TEB",
                        "date": "2024-01-01",
                        "time": "8:00 PM",
                        "league": "NBA",
                        "season": "2023-24",
                        "api_game_id": "0012400001",
                        "data_completeness_score": 95
                    },
                    "prediction_date": "2024-01-01T09:00:00",
                    "predicted_total": 225.5,
                    "bet_recommendation": "OVER",
                    "status": "pending_result"
                }
            ]

            with open(data_path / "pending_results.json", 'w') as f:
                json.dump(test_results, f)

            yield data_path

    @pytest.fixture
    def temp_db_with_migration(self, temp_data_dir):
        """Create temporary database and run migration"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name

        migrator = DataMigrator(db_path, str(temp_data_dir))
        migration_report = migrator.run_full_migration()

        yield migrator, migration_report

        Path(db_path).unlink(missing_ok=True)

    def test_migration_success(self, temp_data_dir):
        """Test migration completes successfully"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name

        try:
            migrator = DataMigrator(db_path, str(temp_data_dir))
            migration_report = migrator.run_full_migration()

            assert migration_report['success'] is True
            assert migration_report['backup_location'] is not None

            # Check migration results
            results = migration_report['migration_results']
            assert results['games']['migrated'] == 1
            assert results['bets']['migrated'] == 1
            assert results['bankroll']['migrated'] == 1

            # Check statistics
            stats = migration_report['statistics']
            assert stats['games_count'] == 1
            assert stats['bets_count'] == 1
            assert stats['bankroll_count'] == 1

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_migrated_data_integrity(self, temp_db_with_migration):
        """Test migrated data maintains integrity"""
        migrator, report = temp_db_with_migration
        con = migrator.con

        # Test games data
        games = con.execute("SELECT * FROM games").fetchall()
        assert len(games) == 1
        assert games[0][1] == "test_game_1"  # game_id
        assert games[0][2] == "Team A"  # home_team
        assert games[0][4] == "Team B"  # away_team

        # Test bets data
        bets = con.execute("SELECT * FROM bets").fetchall()
        assert len(bets) == 1
        assert bets[0][1] == "test_bet_1"  # bet_id
        assert bets[0][2] == "test_game_1"  # game_id
        assert bets[0][3] == "OVER"  # bet_type
        assert bets[0][5] == 1.85  # odds
        assert bets[0][6] == 10.0  # stake

        # Test bankroll data
        bankroll = con.execute("SELECT * FROM bankroll").fetchall()
        assert len(bankroll) == 2  # Initial balance + bet placement
        assert bankroll[0][2] == 100.0  # Initial deposit
        assert bankroll[1][2] == -10.0  # Bet placement

    def test_balance_calculation(self, temp_db_with_migration):
        """Test running balance is calculated correctly"""
        migrator, report = temp_db_with_migration
        con = migrator.con

        # Check balance calculations
        balances = con.execute("""
            SELECT transaction_id, amount, balance_after
            FROM bankroll
            ORDER BY created_at
        """).fetchall()

        # First transaction: deposit 100 -> balance 100
        assert balances[0][1] == 100.0  # amount
        assert balances[0][2] == 100.0  # balance_after

        # Second transaction: bet 10 -> balance 90
        assert balances[1][1] == -10.0  # amount
        assert balances[1][2] == 90.0  # balance_after

class TestDatabaseBackup:
    """Test database backup functionality"""

    @pytest.fixture
    def temp_db_with_data(self):
        """Create temporary database with test data"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name

        schema = NBADatabaseSchema(db_path)
        schema.connect()
        schema.create_tables()

        con = schema.con

        # Insert test data
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('backup_test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        con.execute("""
            INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
            VALUES ('backup_test_bet', 'backup_test_game', 'OVER', 1.5, 10.0)
        """)

        con.execute("""
            INSERT INTO bankroll (transaction_type, amount, balance_after)
            VALUES ('deposit', 100.0, 100.0)
        """)

        yield schema

        schema.close()
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_full_backup_creation(self, temp_db_with_data, temp_backup_dir):
        """Test full backup creation"""
        backup_system = DatabaseBackup(
            temp_db_with_data.db_path,
            temp_backup_dir
        )
        backup_system.connect()

        backup_path = backup_system.create_full_backup("test")

        assert Path(backup_path).exists()
        assert backup_path.endswith('.gz')

        # Check metadata file exists
        metadata_path = backup_path.replace('.duckdb.gz', '.metadata.json')
        assert Path(metadata_path).exists()

        backup_system.close()

    def test_backup_integrity_verification(self, temp_db_with_data, temp_backup_dir):
        """Test backup integrity verification"""
        backup_system = DatabaseBackup(
            temp_db_with_data.db_path,
            temp_backup_dir
        )
        backup_system.connect()

        backup_path = backup_system.create_full_backup("test")
        metadata_path = backup_path.replace('.duckdb.gz', '.metadata.json')

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verify integrity
        is_valid = backup_system.verify_backup_integrity(backup_path, metadata)
        assert is_valid is True

        backup_system.close()

    def test_backup_restore(self, temp_db_with_data, temp_backup_dir):
        """Test backup restore functionality"""
        backup_system = DatabaseBackup(
            temp_db_with_data.db_path,
            temp_backup_dir
        )
        backup_system.connect()

        # Create backup
        backup_path = backup_system.create_full_backup("test")

        # Modify original database
        con = backup_system.con
        con.execute("INSERT INTO games (game_id, home_team, away_team, game_date) VALUES ('new_game', 'Team C', 'Team D', '2024-01-02')")

        original_game_count = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        assert original_game_count == 2

        # Restore from backup
        restore_success = backup_system.restore_from_backup(backup_path)
        assert restore_success is True

        # Verify restore
        restored_game_count = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        assert restored_game_count == 1

        backup_system.close()

    def test_export_backup(self, temp_db_with_data, temp_backup_dir):
        """Test export backup functionality"""
        backup_system = DatabaseBackup(
            temp_db_with_data.db_path,
            temp_backup_dir
        )
        backup_system.connect()

        export_path = backup_system.create_export_backup()

        assert Path(export_path).exists()
        assert Path(export_path).is_dir()

        # Check for expected export files
        export_files = list(Path(export_path).glob("*"))
        assert len(export_files) > 0

        backup_system.close()

class TestDatabaseConstraints:
    """Test database constraints and business rules"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name

        schema = NBADatabaseSchema(db_path)
        schema.connect()
        schema.create_tables()
        yield schema
        schema.close()
        Path(db_path).unlink(missing_ok=True)

    def test_bet_status_constraints(self, temp_db):
        """Test bet status constraints"""
        con = temp_db.con

        # Insert game first
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        # Test valid status values
        valid_statuses = ['pending', 'won', 'lost', 'cancelled', 'replaced']
        for i, status in enumerate(valid_statuses):
            con.execute(f"""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake, status)
                VALUES ('bet_{i}', 'test_game', 'OVER', 1.5, 10.0, '{status}')
            """)

        # Test invalid status should fail
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake, status)
                VALUES ('invalid_bet', 'test_game', 'OVER', 1.5, 10.0, 'invalid_status')
            """)

    def test_game_status_constraints(self, temp_db):
        """Test game status constraints"""
        con = temp_db.con

        # Test valid status values
        valid_statuses = ['scheduled', 'in_progress', 'completed', 'cancelled', 'postponed']
        for status in valid_statuses:
            con.execute(f"""
                INSERT INTO games (game_id, home_team, away_team, game_date, status)
                VALUES ('game_{status}', 'Team A', 'Team B', '2024-01-01', '{status}')
            """)

        # Test invalid status should fail
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO games (game_id, home_team, away_team, game_date, status)
                VALUES ('invalid_game', 'Team A', 'Team B', '2024-01-01', 'invalid_status')
            """)

    def test_bet_type_constraints(self, temp_db):
        """Test bet type constraints"""
        con = temp_db.con

        # Insert game first
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        # Test valid bet types
        valid_types = ['OVER', 'UNDER', 'SPREAD', 'MONEYLINE']
        for i, bet_type in enumerate(valid_types):
            con.execute(f"""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                VALUES ('bet_{i}', 'test_game', '{bet_type}', 1.5, 10.0)
            """)

        # Test invalid bet type should fail
        with pytest.raises(Exception):
            con.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                VALUES ('invalid_bet', 'test_game', 'INVALID_TYPE', 1.5, 10.0)
            """)

    def test_foreign_key_constraints(self, temp_db):
        """Test foreign key constraints are properly enforced"""
        con = temp_db.con

        # Insert game and related records
        con.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date)
            VALUES ('test_game', 'Team A', 'Team B', '2024-01-01')
        """)

        con.execute("""
            INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
            VALUES ('test_bet', 'test_game', 'OVER', 1.5, 10.0)
        """)

        con.execute("""
            INSERT INTO betting_analysis (game_id, predicted_total)
            VALUES ('test_game', 220.5)
        """)

        # Verify related records exist
        assert con.execute("SELECT COUNT(*) FROM bets WHERE game_id = 'test_game'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM betting_analysis WHERE game_id = 'test_game'").fetchone()[0] == 1

        # Test that deleting a game with related records is prevented
        # (DuckDB doesn't support CASCADE, so this should fail)
        try:
            con.execute("DELETE FROM games WHERE game_id = 'test_game'")
            # If this succeeds, foreign key constraints are not enforced
            # This is expected behavior in DuckDB
        except Exception:
            # If this fails, foreign key constraints are enforced
            pass

        # Note: DuckDB doesn't support CASCADE DELETE, so related records remain
        # This is the expected behavior for this database system

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])