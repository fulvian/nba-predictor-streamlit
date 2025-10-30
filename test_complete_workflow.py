#!/usr/bin/env python3
"""
Test script for the complete betting workflow from analysis to bet placement.
"""

from datetime import datetime
from src.nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

def test_complete_betting_workflow():
    """Test the complete betting workflow."""

    print("🧪 Testing Complete Betting Workflow")

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
        game_id="TEST_GAME_002",
        central_line=225.5,
        timestamp=datetime.now()
    )

    print(f"✅ BetAnalysis created with ROI: {test_analysis.roi} (type: {type(test_analysis.roi)})")
    print(f"✅ Risk Level: {test_analysis.risk_level} (type: {type(test_analysis.risk_level)})")

    try:
        with BettingDatabaseManager() as db_manager:
            # Step 1: Save analysis
            print("📝 Step 1: Saving bet analysis...")
            analysis_id = db_manager.save_bet_analysis(test_analysis)
            print(f"✅ Analysis saved with ID: {analysis_id}")

            # Step 2: Place a bet based on analysis
            print("💰 Step 2: Placing bet...")
            bet_id = db_manager.place_bet(
                analysis=test_analysis,
                selected_stake=2.5,  # Override stake slightly
                notes="Test bet from workflow test"
            )
            print(f"✅ Bet placed with ID: {bet_id}")

            # Step 3: Check current bankroll
            print("💳 Step 3: Checking bankroll...")
            bankroll_status = db_manager.get_bankroll_status()
            bankroll = bankroll_status.get('current_bankroll', 0)
            print(f"✅ Current bankroll: €{bankroll:.2f}")

            # Step 4: Verify bet details
            print("🔍 Step 4: Verifying bet details...")
            bet_details = db_manager.get_bet_details(bet_id)
            if bet_details:
                print(f"✅ Bet confirmed: {bet_details['bet_type']} {bet_details['line']} @ {bet_details['odds']}")
                print(f"✅ Stake: €{bet_details['stake']}, Potential return: €{bet_details['potential_return']}")
            else:
                print("❌ Could not retrieve bet details")
                return False

        print("🎉 COMPLETE BETTING WORKFLOW TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        return False

if __name__ == "__main__":
    test_complete_betting_workflow()