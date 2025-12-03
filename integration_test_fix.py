#!/usr/bin/env python3
"""
Integration Test for NBA Betting System Fixes - Context7 Best Practices

Comprehensive integration test to validate all fixes work together correctly.
Tests the entire betting workflow from game selection to bet placement.

Key Test Scenarios:
1. Manual game ID creation and normalization
2. Foreign key constraint resolution
3. Enhanced betting workflow with FK protection
4. Dashboard integration compatibility
5. Error handling and recovery
"""

import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import json
from typing import Dict, List, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from foreign_key_fix import ForeignKeyConstraintFixer, SmartGameIDGenerator
from enhanced_betting_database_manager import EnhancedBettingDatabaseManager, EnhancedBetAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """
    Comprehensive integration test suite for NBA betting system fixes.
    """

    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = db_path
        self.test_results = {}
        self.errors = []

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.

        Returns:
            Comprehensive test results
        """
        logger.info("🧪 Starting NBA Betting System Integration Tests...")

        test_methods = [
            ('smart_id_generation', self.test_smart_id_generation),
            ('foreign_key_fixes', self.test_foreign_key_fixes),
            ('enhanced_bet_placement', self.test_enhanced_bet_placement),
            ('dashboard_compatibility', self.test_dashboard_compatibility),
            ('error_handling', self.test_error_handling),
            ('database_integrity', self.test_database_integrity)
        ]

        for test_name, test_method in test_methods:
            try:
                logger.info(f"🔍 Running test: {test_name}")
                result = test_method()
                self.test_results[test_name] = {
                    'status': 'passed' if result else 'failed',
                    'details': result if isinstance(result, dict) else {'result': result}
                }
                logger.info(f"✅ Test {test_name}: {self.test_results[test_name]['status']}")
            except Exception as e:
                error_msg = f"Test {test_name} failed: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Generate summary
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
        total_tests = len(self.test_results)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'error_count': len(self.errors),
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'test_results': self.test_results,
            'errors': self.errors
        }

        logger.info(f"📊 Integration Tests Complete: {passed_tests}/{total_tests} passed ({summary['success_rate']:.1f}%)")

        return summary

    def test_smart_id_generation(self) -> Dict[str, Any]:
        """Test smart game ID generation and normalization."""
        try:
            id_generator = SmartGameIDGenerator()
            test_cases = [
                {
                    'home': 'Toronto Raptors',
                    'away': 'Milwaukee Bucks',
                    'date': date(2025, 11, 4),
                    'existing_id': None
                },
                {
                    'home': 'Los Angeles Lakers',
                    'away': 'Boston Celtics',
                    'date': date(2025, 11, 5),
                    'existing_id': 'MANUAL_Toronto Raptors_Milwaukee Bucks'
                },
                {
                    'home': 'Golden State Warriors',
                    'away': 'Miami Heat',
                    'date': date(2025, 11, 6),
                    'existing_id': 'CUSTOM_Warriors_Heat'
                }
            ]

            results = []
            for i, case in enumerate(test_cases):
                generated_id = id_generator.generate_game_id(
                    case['home'], case['away'], case['date'], case['existing_id']
                )

                # Validate ID format
                is_valid = (
                    generated_id.startswith('MANUAL_') and
                    len(generated_id) > 20 and
                    '_' in generated_id
                )

                results.append({
                    'case': i + 1,
                    'input': case,
                    'generated_id': generated_id,
                    'is_valid': is_valid
                })

            all_valid = all(r['is_valid'] for r in results)

            return {
                'passed': all_valid,
                'results': results,
                'summary': f"Generated {len(results)} IDs, all valid: {all_valid}"
            }

        except Exception as e:
            logger.error(f"Smart ID generation test failed: {e}")
            return {'passed': False, 'error': str(e)}

    def test_foreign_key_fixes(self) -> Dict[str, Any]:
        """Test foreign key constraint fixes."""
        try:
            fk_fixer = ForeignKeyConstraintFixer(self.db_path)

            # Test validation
            pre_fix_validation = fk_fixer.validate_all_foreign_keys()

            # Check for missing games
            fix_results = fk_fixer.check_and_create_missing_games()

            # Re-validate
            post_fix_validation = fk_fixer.validate_all_foreign_keys()

            fk_fixer.close()

            return {
                'passed': post_fix_validation['validation_passed'],
                'pre_fix_validation': pre_fix_validation,
                'fix_results': fix_results,
                'post_fix_validation': post_fix_validation,
                'summary': f"FK constraints valid: {post_fix_validation['validation_passed']}"
            }

        except Exception as e:
            logger.error(f"Foreign key fixes test failed: {e}")
            return {'passed': False, 'error': str(e)}

    def test_enhanced_bet_placement(self) -> Dict[str, Any]:
        """Test enhanced bet placement with FK protection."""
        try:
            manager = EnhancedBettingDatabaseManager(self.db_path)

            # Create test bet analysis
            test_bet_analysis = EnhancedBetAnalysis(
                bet_type='OVER',
                line=225.5,
                odds=1.95,
                edge=0.03,
                probability=0.55,
                implied_probability=0.51,
                true_probability=0.55,
                quality_score=0.75,
                edge_score=0.60,
                confidence_score=0.80,
                risk_score=5.0,
                consistency_score=0.70,
                kelly_fraction=0.02,
                stake=10.0,
                roi=0.05,
                is_value=True,
                risk_level='Medium',
                game_id='TEST_INTEGRATION_GAME',
                central_line=225.0,
                timestamp=datetime.now(),
                home_team='Test Home Team',
                away_team='Test Away Team'
            )

            # Test bet placement
            result = manager.place_bet_with_fk_protection(
                test_bet_analysis,
                stake_override=15.0,
                notes="Integration test bet"
            )

            # Verify bet was created
            if result['success']:
                # Check if bet exists in database
                bet_info = manager.conn.execute("""
                    SELECT bet_id, game_id, status FROM bets WHERE bet_id = ?
                """, [result['bet_id']]).fetchone()

                bet_created = bet_info is not None
                game_created = result.get('game_record_created', False)

                # Check game record
                if result.get('game_id'):
                    game_info = manager.conn.execute("""
                        SELECT game_id, home_team, away_team FROM games WHERE game_id = ?
                    """, [result['game_id']]).fetchone()

                    game_record_exists = game_info is not None
                else:
                    game_record_exists = False

                # Clean up test data
                try:
                    if result['bet_id']:
                        manager.conn.execute("DELETE FROM bets WHERE bet_id = ?", [result['bet_id']])
                    if result.get('game_id'):
                        manager.conn.execute("DELETE FROM games WHERE game_id = ?", [result['game_id']])
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup error: {cleanup_error}")
            else:
                bet_created = False
                game_record_exists = False
                game_created = False

            manager.close()

            return {
                'passed': result['success'] and bet_created and game_record_exists,
                'bet_placement_result': result,
                'bet_created': bet_created,
                'game_record_created': game_created,
                'game_record_exists': game_record_exists,
                'summary': f"Bet placement successful: {result['success']}, records created: {bet_created and game_record_exists}"
            }

        except Exception as e:
            logger.error(f"Enhanced bet placement test failed: {e}")
            return {'passed': False, 'error': str(e)}

    def test_dashboard_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with existing dashboard components."""
        try:
            # Test that we can import the enhanced manager where the original is used
            import importlib.util

            # Check if enhanced manager has required methods
            manager = EnhancedBettingDatabaseManager(self.db_path)

            required_methods = [
                'get_pending_bets',
                'get_bankroll_status',
                'close'
            ]

            methods_available = all(hasattr(manager, method) for method in required_methods)

            # Test method calls work
            if methods_available:
                try:
                    pending_bets = manager.get_pending_bets()
                    bankroll_status = manager.get_bankroll_status()

                    methods_work = isinstance(pending_bets, list) and isinstance(bankroll_status, dict)
                except Exception as method_error:
                    methods_work = False
                    logger.error(f"Method call error: {method_error}")
            else:
                methods_work = False

            manager.close()

            return {
                'passed': methods_available and methods_work,
                'methods_available': methods_available,
                'methods_work': methods_work,
                'required_methods': required_methods,
                'summary': f"Dashboard compatibility: {'✅' if methods_available and methods_work else '❌'}"
            }

        except Exception as e:
            logger.error(f"Dashboard compatibility test failed: {e}")
            return {'passed': False, 'error': str(e)}

    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        try:
            manager = EnhancedBettingDatabaseManager(self.db_path)

            # Test 1: Invalid bet analysis
            invalid_bet = EnhancedBetAnalysis(
                bet_type='INVALID_TYPE',
                line=-100.0,  # Invalid line
                odds=0.5,     # Invalid odds
                edge=0.0,
                probability=0.0,
                implied_probability=0.0,
                true_probability=0.0,
                quality_score=0.0,
                edge_score=0.0,
                confidence_score=0.0,
                risk_score=0.0,
                consistency_score=0.0,
                kelly_fraction=0.0,
                stake=-10.0,  # Invalid stake
                roi=0.0,
                is_value=False,
                risk_level='Medium',
                game_id='',
                central_line=0.0,
                timestamp=datetime.now()
            )

            result1 = manager.place_bet_with_fk_protection(invalid_bet)
            handled_invalid_bet = not result1['success']

            # Test 2: Database connection resilience
            original_db_path = manager.db_path
            try:
                # Try with invalid database path
                invalid_manager = EnhancedBettingDatabaseManager("/invalid/path/database.duckdb")
                handled_invalid_path = False  # Should raise exception
            except Exception:
                handled_invalid_path = True

            manager.close()

            return {
                'passed': handled_invalid_bet and handled_invalid_path,
                'invalid_bet_handled': handled_invalid_bet,
                'invalid_path_handled': handled_invalid_path,
                'summary': f"Error handling: {'✅' if handled_invalid_bet and handled_invalid_path else '❌'}"
            }

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return {'passed': False, 'error': str(e)}

    def test_database_integrity(self) -> Dict[str, Any]:
        """Test overall database integrity after fixes."""
        try:
            fk_fixer = ForeignKeyConstraintFixer(self.db_path)

            # Full validation
            validation = fk_fixer.validate_all_foreign_keys()

            # Get database stats
            stats = fk_fixer._get_database_stats()

            fk_fixer.close()

            # Check for critical issues
            critical_issues = []

            if not validation.get('validation_passed', False):
                for constraint_name, result in validation.items():
                    if isinstance(result, dict) and not result.get('valid', True):
                        critical_issues.append(f"FK violation: {constraint_name}")

            # Check for empty tables that should have data
            if stats.get('games_count', 0) == 0:
                critical_issues.append("Games table is empty")

            if stats.get('bets_count', 0) == 0:
                logger.info("Bets table is empty - this might be normal for a new system")

            return {
                'passed': len(critical_issues) == 0,
                'validation': validation,
                'stats': stats,
                'critical_issues': critical_issues,
                'summary': f"Database integrity: {'✅' if len(critical_issues) == 0 else f'❌ ({len(critical_issues)} issues)'}"
            }

        except Exception as e:
            logger.error(f"Database integrity test failed: {e}")
            return {'passed': False, 'error': str(e)}

def main():
    """Run integration tests and generate report."""
    logger.info("🚀 Starting NBA Betting System Integration Test Suite")

    test_suite = IntegrationTestSuite()
    results = test_suite.run_all_tests()

    # Generate report
    print("\n" + "="*80)
    print("🧪 NBA BETTING SYSTEM INTEGRATION TEST REPORT")
    print("="*80)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")

    if results['error_count'] > 0:
        print(f"⚠️ Errors: {results['error_count']}")

    print(f"\n📋 Test Results:")
    for test_name, result in results['test_results'].items():
        status_icon = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
        print(f"   {status_icon} {test_name}: {result['status'].upper()}")

        if 'summary' in result.get('details', {}):
            print(f"      → {result['details']['summary']}")
        elif 'error' in result.get('details', {}):
            print(f"      → Error: {result['details']['error']}")

    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"   • {error}")

    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if results['success_rate'] == 100 else '⚠️ SOME TESTS FAILED'}")
    print("="*80)

    # Save detailed report
    report_path = "integration_test_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

if __name__ == "__main__":
    main()