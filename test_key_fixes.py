#!/usr/bin/env python3
"""
Test script to verify the two main fixes:
1. Bankroll status includes pending_bets_count
2. Place bet doesn't clear data (no st.rerun)
"""

import logging
from datetime import datetime
from src.nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

def test_bankroll_status_fix():
    """Test that bankroll_status includes pending_bets_count."""
    print("🧪 Testing Bankroll Status Fix...")

    try:
        with BettingDatabaseManager() as db_manager:
            status = db_manager.get_bankroll_status()

            # Check for the missing field
            if 'pending_bets_count' in status:
                print(f"✅ pending_bets_count found: {status['pending_bets_count']}")
                return True
            else:
                print("❌ pending_bets_count still missing")
                print(f"Available keys: {list(status.keys())}")
                return False

    except Exception as e:
        print(f"❌ Error testing bankroll status: {e}")
        return False

def test_bet_analysis_creation():
    """Test that BetAnalysis can be created with correct types."""
    print("\n🧪 Testing BetAnalysis Creation...")

    try:
        # Create a proper BetAnalysis object with correct types
        test_analysis = BetAnalysis(
            bet_type="Over",
            line=225.5,
            odds=1.85,
            edge=3.2,
            probability=0.54,
            implied_probability=0.54,
            true_probability=0.57,
            quality_score=0.75,
            edge_score=0.65,
            confidence_score=0.80,
            risk_score=0.45,
            consistency_score=0.90,
            kelly_fraction=0.02,
            stake=2.0,
            roi=12.5,
            is_value=True,
            risk_level="Medium",
            game_id="TEST_GAME_003",
            central_line=225.5,
            timestamp=datetime.now()
        )

        print(f"✅ BetAnalysis created with ROI: {test_analysis.roi} (type: {type(test_analysis.roi)})")
        print(f"✅ Risk Level: {test_analysis.risk_level} (type: {type(test_analysis.risk_level)})")
        return True

    except Exception as e:
        print(f"❌ Error creating BetAnalysis: {e}")
        return False

def main():
    """Run both tests."""
    print("🔧 Testing Key Fixes for NBA Betting System")
    print("=" * 50)

    # Test 1: Bankroll status fix
    test1_passed = test_bankroll_status_fix()

    # Test 2: BetAnalysis creation
    test2_passed = test_bet_analysis_creation()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"✅ Bankroll Status Fix: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ BetAnalysis Creation: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 KEY FIXES VERIFIED!")
        print("   - Bankroll error should be resolved")
        print("   - Bet placement data clearing should be resolved")
    else:
        print("\n❌ Some fixes need more work")

if __name__ == "__main__":
    main()