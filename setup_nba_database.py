#!/usr/bin/env python3
"""
NBA Betting Database Setup Script
Context7-compliant database initialization with comprehensive reporting

This script orchestrates the complete setup of the NBA betting database:
1. Database schema creation with validation
2. Data migration from JSON files
3. Backup system initialization
4. Integrity testing
5. Performance optimization
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from database import initialize_nba_database, get_database_manager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_header():
    """Print setup header"""
    print("\n" + "="*80)
    print("🏀 NBA BETTING DATABASE SETUP")
    print("Context7-Compliant Database Initialization System")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

def print_setup_results(results: Dict[str, Any]):
    """Print comprehensive setup results"""
    print("\n" + "="*80)
    print("📊 SETUP RESULTS SUMMARY")
    print("="*80)

    # Overall status
    status_emoji = "✅ SUCCESS" if results['success'] else "❌ FAILED"
    print(f"Overall Status: {status_emoji}")
    print(f"Started: {results['start_time']}")
    print(f"Completed: {results['end_time']}")

    # Component status
    print("\n🏗️ Component Status:")
    components = [
        ("Database Schema", results['database_initialized']),
        ("Data Migration", results['migration_completed']),
        ("Initial Backup", results['backup_created']),
        ("Schema Validation", results['validation_passed'])
    ]

    for component, status in components:
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {component}: {'Completed' if status else 'Failed'}")

    # Statistics
    if 'statistics' in results and results['statistics']:
        print("\n📈 Database Statistics:")
        stats = results['statistics']

        if 'tables' in stats:
            print(f"  📋 Tables: {len(stats['tables'])}")
            total_rows = sum(table.get('row_count', 0) for table in stats['tables'].values())
            print(f"  📊 Total Rows: {total_rows:,}")

            for table_name, table_info in stats['tables'].items():
                row_count = table_info.get('row_count', 0)
                size_est = table_info.get('estimated_size_bytes', 0)
                print(f"    - {table_name}: {row_count:,} rows (~{size_est:,} bytes)")

        if 'views' in stats:
            print(f"  👁️ Views: {len(stats['views'])}")
            for view in stats['views']:
                print(f"    - {view}")

        if 'indexes' in stats:
            print(f"  🔗 Indexes: {len(stats['indexes'])}")

    # Migration report
    if 'migration_report' in results:
        print("\n🔄 Migration Details:")
        migration = results['migration_report']

        if 'migration_results' in migration:
            for entity, data in migration['migration_results'].items():
                print(f"  📦 {entity.capitalize()}: {data['migrated']} migrated, {data['errors']} errors")

        if 'backup_location' in migration:
            print(f"  💾 Backup created at: {migration['backup_location']}")

    # Errors
    if results['errors']:
        print("\n❌ Errors Encountered:")
        for i, error in enumerate(results['errors'], 1):
            print(f"  {i}. {error}")

    # Backup information
    if 'backup_path' in results:
        print(f"\n💾 Initial Backup: {results['backup_path']}")

    print("\n" + "="*80)

def run_integrity_tests(db_manager: Dict[str, Any]) -> bool:
    """Run comprehensive integrity tests"""
    print("\n🧪 Running Database Integrity Tests...")

    try:
        # Test database connection
        schema = db_manager['schema']
        schema.connect()

        # Test basic queries
        con = schema.con

        # Test table access
        tables_to_test = ['games', 'bets', 'bankroll']
        for table in tables_to_test:
            try:
                result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                print(f"  ✅ {table}: {result[0]} records")
            except Exception as e:
                print(f"  ❌ {table}: Error - {e}")
                return False

        # Test views
        views_to_test = ['active_bets', 'bankroll_summary']
        for view in views_to_test:
            try:
                result = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
                print(f"  ✅ {view}: {result[0]} records")
            except Exception as e:
                print(f"  ❌ {view}: Error - {e}")
                return False

        # Test constraints
        try:
            # Test foreign key constraint
            con.execute("""
                INSERT INTO games (game_id, home_team, away_team, game_date)
                VALUES ('test_integrity', 'Team A', 'Team B', '2024-01-01')
            """)

            # This should fail due to foreign key constraint
            try:
                con.execute("""
                    INSERT INTO bets (bet_id, game_id, bet_type, odds, stake)
                    VALUES ('test_bet', 'nonexistent_game', 'OVER', 1.5, 10.0)
                """)
                print("  ❌ Foreign key constraint not working")
                return False
            except Exception:
                print("  ✅ Foreign key constraints working")

            # Clean up test data
            con.execute("DELETE FROM games WHERE game_id = 'test_integrity'")

        except Exception as e:
            print(f"  ❌ Constraint test failed: {e}")
            return False

        schema.close()
        print("  ✅ All integrity tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Integrity tests failed: {e}")
        return False

def run_performance_checks(db_manager: Dict[str, Any]) -> bool:
    """Run basic performance checks"""
    print("\n⚡ Running Performance Checks...")

    try:
        schema = db_manager['schema']
        schema.connect()
        con = schema.con

        # Test query performance
        start_time = datetime.now()

        # Test basic SELECT queries
        con.execute("SELECT COUNT(*) FROM games")
        con.execute("SELECT COUNT(*) FROM bets")
        con.execute("SELECT COUNT(*) FROM bankroll")

        # Test JOIN queries
        con.execute("""
            SELECT COUNT(*) FROM bets b
            JOIN games g ON b.game_id = g.game_id
        """)

        # Test aggregation queries
        con.execute("""
            SELECT COUNT(*), SUM(stake) FROM bets
            WHERE status = 'pending'
        """)

        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds()

        print(f"  ⚡ Query performance: {query_time:.3f} seconds")

        if query_time < 1.0:
            print("  ✅ Performance is excellent")
        elif query_time < 5.0:
            print("  ✅ Performance is good")
        else:
            print("  ⚠️ Performance could be improved")

        schema.close()
        return True

    except Exception as e:
        print(f"  ❌ Performance checks failed: {e}")
        return False

def generate_setup_report(results: Dict[str, Any]) -> str:
    """Generate detailed setup report"""
    report_path = f"database_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        report = {
            'setup_type': 'nba_betting_database_initialization',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd())
            },
            'recommendations': []
        }

        # Add recommendations based on results
        if results['success']:
            report['recommendations'].append("Database setup completed successfully")
            report['recommendations'].append("Regular backups should be scheduled")
            report['recommendations'].append("Monitor database performance and optimize as needed")
        else:
            report['recommendations'].append("Review and fix reported errors")
            report['recommendations'].append("Ensure all prerequisites are met")
            report['recommendations'].append("Check file permissions and disk space")

        # Add migration recommendations
        if 'migration_report' in results:
            migration = results['migration_report']
            if 'statistics' in migration:
                stats = migration['statistics']
                if stats.get('bet_summary', {}).get('pending_bets', 0) > 0:
                    report['recommendations'].append("Review and settle pending bets")
                if stats.get('bankroll_summary', {}).get('current_balance', 0) < 50:
                    report['recommendations'].append("Consider increasing bankroll for better betting opportunities")

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report_path

    except Exception as e:
        logger.error(f"Failed to generate setup report: {e}")
        return None

def main():
    """Main setup function"""
    print_header()

    try:
        # Check prerequisites
        print("🔍 Checking prerequisites...")

        # Check if data directory exists
        data_dir = Path("data")
        if not data_dir.exists():
            print("⚠️  Data directory not found, creating it...")
            data_dir.mkdir(exist_ok=True)

        # Check for existing JSON files
        required_files = [
            "data/pending_bets.json",
            "data/bankroll.json"
        ]

        found_files = []
        for file_path in required_files:
            if Path(file_path).exists():
                found_files.append(file_path)
                print(f"  ✅ Found: {file_path}")
            else:
                print(f"  ⚠️  Missing: {file_path}")

        if not found_files:
            print("⚠️  No existing data files found. Will create empty database.")

        # Initialize database
        print("\n🚀 Initializing NBA Betting Database...")
        results = initialize_nba_database(
            db_path="data/nba_betting.duckdb",
            data_dir="data",
            backup_dir="data/backups",
            run_migration=len(found_files) > 0
        )

        # Run integrity tests
        if results['success']:
            print("\n🧪 Running post-setup validation...")
            db_manager = get_database_manager("data/nba_betting.duckdb")

            integrity_passed = run_integrity_tests(db_manager)
            performance_passed = run_performance_checks(db_manager)

            results['integrity_tests_passed'] = integrity_passed
            results['performance_tests_passed'] = performance_passed

            if not integrity_passed or not performance_tests_passed:
                results['success'] = False
                results['errors'].append("Post-setup validation failed")

        # Generate setup report
        report_path = generate_setup_report(results)
        if report_path:
            print(f"\n📄 Detailed setup report saved to: {report_path}")

        # Print results
        print_setup_results(results)

        # Final status
        if results['success']:
            print("\n🎉 NBA Betting Database setup completed successfully!")
            print("\n📋 Next Steps:")
            print("  1. Review the database statistics above")
            print("  2. Test the system with your application")
            print("  3. Schedule regular backups")
            print("  4. Monitor performance and optimize as needed")
            print(f"\n💾 Database location: data/nba_betting.duckdb")
            print(f"📊 Backup directory: data/backups/")
        else:
            print("\n❌ Setup completed with errors. Please review the errors above and take corrective action.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        logger.exception("Setup failed with unexpected error")
        sys.exit(1)

if __name__ == "__main__":
    main()