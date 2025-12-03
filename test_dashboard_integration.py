import sys
import os
import pandas as pd
from datetime import date

# Add src to path
sys.path.append(os.path.abspath("src"))

from nba_predictor.streamlit.components.enhanced_prediction_bridge_professional import (
    EnhancedPredictionBridgeProfessional,
)


def test_integration():
    print("🧪 Testing Dashboard Integration (EnhancedPredictionBridgeProfessional)...")

    # Define test case (using teams we know have data)
    home_team = "Atlanta Hawks"
    away_team = "LA Clippers"  # Testing the alias fix
    game_date = date.today()
    betting_line = 225.5

    try:
        print(f"Initializing Bridge...")
        bridge = EnhancedPredictionBridgeProfessional()

        print(f"Requesting prediction for {away_team} @ {home_team}...")
        result = bridge.get_professional_prediction(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            betting_line=betting_line,
            force_refresh=True,
        )

        print("\n✅ Call Successful!")
        print(f"Predicted Total: {result.get('predicted_total')}")
        print(f"Confidence: {result.get('confidence_score')}")
        print(f"Market Adjustment: {result.get('market_adjustment')}")

        # Check if we can see evidence of the new logic in the breakdown text if available
        breakdown = result.get("model_breakdown", {})
        print("\nBreakdown Keys:", breakdown.keys())

        return True

    except Exception as e:
        print(f"\n❌ Integration Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_integration()
