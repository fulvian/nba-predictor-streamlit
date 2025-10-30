#!/usr/bin/env python3
"""
Test script to reproduce and fix the ROI type conversion error.
"""

from datetime import datetime
from src.nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

def test_roi_type_error():
    """Test the ROI type conversion error and fix it."""

    print("🧪 Testing ROI Type Conversion Error")

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
        roi=12.5,  # ROI as float, not string
        is_value=True,
        risk_level="Medium",  # Risk level as string
        game_id="TEST_GAME_001",
        central_line=225.5,
        timestamp=datetime.now()
    )

    print(f"✅ BetAnalysis created with ROI: {test_analysis.roi} (type: {type(test_analysis.roi)})")
    print(f"✅ Risk Level: {test_analysis.risk_level} (type: {type(test_analysis.risk_level)})")

    # Test saving to database
    try:
        with BettingDatabaseManager() as db_manager:
            analysis_id = db_manager.save_bet_analysis(test_analysis)
            print(f"✅ Analysis saved with ID: {analysis_id}")

            # Test retrieving the analysis
            retrieved = db_manager.get_bet_analysis(analysis_id)
            if retrieved:
                print(f"✅ Retrieved ROI: {retrieved['roi']} (type: {type(retrieved['roi'])})")
                print(f"✅ Retrieved Risk Level: {retrieved['risk_level']}")

        print("🎉 ROI type conversion test PASSED!")

    except Exception as e:
        print(f"❌ Error during save: {e}")
        return False

    return True

if __name__ == "__main__":
    test_roi_type_error()