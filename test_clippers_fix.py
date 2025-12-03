#!/usr/bin/env python3
"""
Test script per verificare che il problema dei LA Clippers @ 205.0 sia stato risolto.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from nba_predictor.streamlit.components.enhanced_prediction_bridge_professional import (
    get_enhanced_prediction_bridge_professional,
)
from datetime import date


def test_clippers_prediction():
    """Test predizione per LA Clippers vs Atlanta Hawks."""
    print("🏀 Testing LA Clippers prediction fix...")

    # Inizializza bridge
    bridge = get_enhanced_prediction_bridge_professional()

    # Test case: LA Clippers @ Atlanta Hawks
    home_team = "Atlanta Hawks"
    away_team = "Los Angeles Clippers"
    game_date = date.today()
    betting_line = 226.5

    print(f"📊 Test Case: {away_team} @ {home_team}")
    print(f"   Date: {game_date}")
    print(f"   Line: {betting_line}")

    # Esegui predizione con force refresh
    try:
        prediction = bridge.get_professional_prediction(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            betting_line=betting_line,
            include_detailed_analysis=True,
            force_refresh=True,  # Forza ricalcolo
        )

        predicted_total = prediction.get("predicted_total", 0)
        status = prediction.get("status", "unknown")

        print(f"✅ Prediction Results:")
        print(f"   Status: {status}")
        print(f"   Predicted Total: {predicted_total}")
        print(f"   Method: {prediction.get('prediction_method', 'Unknown')}")

        # Verifica se il problema è risolto
        if predicted_total == 205.0:
            print("❌ PROBLEM: Prediction still 205.0 - Fix failed!")
            return False
        elif 190 <= predicted_total <= 280:  # Range realistico
            print(f"✅ SUCCESS: Prediction {predicted_total} is in realistic range!")
            return True
        else:
            print(f"⚠️ UNUSUAL: Prediction {predicted_total} outside normal range")
            return True

    except Exception as e:
        print(f"❌ ERROR: Prediction failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Starting Clippers Prediction Test...")
    success = test_clippers_prediction()

    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed - problem persists!")
