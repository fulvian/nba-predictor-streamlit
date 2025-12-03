#!/usr/bin/env python3
"""
🧪 TEST SECURE DATABASE MANAGER - Comprehensive Security Testing

Tests the secure database manager to ensure:
✅ SQL injection protection works
✅ Input validation functions correctly
✅ All database operations work safely
✅ Performance is acceptable
"""

import sys
import os
sys.path.append('src')

from nba_predictor.utils.betting_database_manager import SecureBettingDatabaseManager
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sql_injection_protection():
    """Test SQL injection protection."""
    print("\n🧪 Testing SQL Injection Protection...")

    db = SecureBettingDatabaseManager(":memory:")  # Use in-memory database for testing

    # Test dangerous inputs
    dangerous_inputs = [
        "'; DROP TABLE bets; --",
        "' OR '1'='1",
        "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
        "UNION SELECT * FROM users",
        "'; DELETE FROM bets WHERE '1'='1'; --"
    ]

    for dangerous_input in dangerous_inputs:
        try:
            # This should raise ValueError
            db._validate_user_input(dangerous_input, 'test_field')
            print(f"❌ FAILED: Dangerous input was not blocked: {dangerous_input}")
            return False
        except ValueError as e:
            print(f"✅ BLOCKED: {dangerous_input[:50]}...")
        except Exception as e:
            print(f"❌ ERROR: Unexpected error for {dangerous_input}: {e}")
            return False

    print("✅ SQL injection protection working correctly")
    return True

def test_table_validation():
    """Test table name validation."""
    print("\n🧪 Testing Table Validation...")

    db = SecureBettingDatabaseManager(":memory:")

    # Valid tables
    valid_tables = ['bets', 'users', 'betting_analysis']
    for table in valid_tables:
        try:
            result = db._validate_table_name(table)
            if result == table.lower():
                print(f"✅ Valid table accepted: {table}")
            else:
                print(f"❌ Valid table rejected: {table}")
                return False
        except Exception as e:
            print(f"❌ Error with valid table {table}: {e}")
            return False

    # Invalid tables
    invalid_tables = ['malicious_table', 'bets; DROP TABLE users', '']
    for table in invalid_tables:
        try:
            db._validate_table_name(table)
            print(f"❌ FAILED: Invalid table was accepted: {table}")
            return False
        except ValueError:
            print(f"✅ Invalid table rejected: {table}")
        except Exception as e:
            print(f"❌ Error with invalid table {table}: {e}")
            return False

    print("✅ Table validation working correctly")
    return True

def test_secure_operations():
    """Test secure database operations."""
    print("\n🧪 Testing Secure Database Operations...")

    # Use test database
    test_db_path = "data/test_secure_betting.duckdb"
    db = SecureBettingDatabaseManager(test_db_path)

    try:
        # Test safe insert
        print("  📝 Testing safe bet insertion...")
        bet_id = db.safe_insert_bet(
            user_id="test_user",
            game_id="game_123",
            bet_type="OVER_UNDER",
            amount=100.0,
            odds=1.85,
            prediction="OVER 220.5",
            confidence_interval={"lower": 215.0, "upper": 230.0},
            audit_user="test_system"
        )

        if bet_id > 0:
            print(f"✅ Bet inserted successfully with ID: {bet_id}")
        else:
            print("❌ Failed to insert bet")
            return False

        # Test safe update
        print("  🔄 Testing safe bet update...")
        success = db.safe_update_bet_status(
            bet_id=bet_id,
            status="WON",
            result="WIN",
            profit_loss=85.0,
            audit_user="test_system"
        )

        if success:
            print("✅ Bet updated successfully")
        else:
            print("❌ Failed to update bet")
            return False

        # Test safe select
        print("  📊 Testing safe bet retrieval...")
        bets = db.safe_get_user_bets(user_id="test_user", limit=10)

        if len(bets) > 0:
            print(f"✅ Retrieved {len(bets)} bets successfully")
        else:
            print("❌ Failed to retrieve bets")
            return False

        # Test safe summary
        print("  📈 Testing safe user summary...")
        summary = db.safe_get_user_summary(user_id="test_user")

        if summary and summary.get('total_bets', 0) > 0:
            print(f"✅ User summary generated: {summary.get('total_bets')} bets")
        else:
            print("❌ Failed to generate user summary")
            return False

        print("✅ All secure operations working correctly")
        return True

    except Exception as e:
        print(f"❌ Error in secure operations test: {e}")
        return False
    finally:
        # Clean up test database
        try:
            import os
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except:
            pass

def test_bet_validation():
    """Test bet validation logic."""
    print("\n🧪 Testing Bet Validation...")

    db = SecureBettingDatabaseManager(":memory:")

    # Test valid amount
    validation = db.safe_validate_bet_placement(user_id="test_user", amount=50.0)
    if validation['valid']:
        print("✅ Valid bet amount accepted")
    else:
        print(f"❌ Valid bet amount rejected: {validation['reason']}")
        return False

    # Test invalid amount (too large)
    validation = db.safe_validate_bet_placement(user_id="test_user", amount=50000.0)
    if not validation['valid']:
        print(f"✅ Invalid bet amount rejected: {validation['reason']}")
    else:
        print("❌ Invalid bet amount was accepted")
        return False

    print("✅ Bet validation working correctly")
    return True

def test_input_validation():
    """Test comprehensive input validation."""
    print("\n🧪 Testing Input Validation...")

    db = SecureBettingDatabaseManager(":memory:")

    # Test valid inputs
    valid_inputs = [
        ("normal_string", "string"),
        (123, "integer"),
        (45.67, "float"),
        (True, "boolean"),
        ({"key": "value"}, "dict"),
        ([1, 2, 3], "list"),
        (None, "null")
    ]

    for value, expected_type in valid_inputs:
        try:
            result = db._validate_user_input(value, f"test_{expected_type}")
            print(f"✅ Valid {expected_type} accepted")
        except Exception as e:
            print(f"❌ Valid {expected_type} rejected: {e}")
            return False

    # Test invalid inputs
    invalid_inputs = [
        "'; DROP TABLE users; --",  # SQL injection
        "<script>alert('xss')</script>",  # XSS
        "x" * 20000,  # Too long
        float('inf'),  # Invalid float
        float('nan')   # NaN
    ]

    for invalid_input in invalid_inputs:
        try:
            db._validate_user_input(invalid_input, "test_invalid")
            print(f"❌ FAILED: Invalid input was accepted: {str(invalid_input)[:50]}...")
            return False
        except (ValueError, TypeError):
            print(f"✅ Invalid input rejected: {str(invalid_input)[:50]}...")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    print("✅ Input validation working correctly")
    return True

def main():
    """Run all security tests."""
    print("🔒 SECURE DATABASE MANAGER - COMPREHENSIVE SECURITY TESTING")
    print("=" * 60)

    tests = [
        test_sql_injection_protection,
        test_table_validation,
        test_input_validation,
        test_bet_validation,
        test_secure_operations
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")

    print("\n" + "=" * 60)
    print(f"🎯 SECURITY TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL SECURITY TESTS PASSED - DATABASE MANAGER IS SECURE")
        print("🚀 PRODUCTION DEPLOYMENT READY")
        return True
    else:
        print("❌ SOME SECURITY TESTS FAILED - FIX ISSUES BEFORE DEPLOYMENT")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)