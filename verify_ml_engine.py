import sys
from pathlib import Path
import logging

# Setup path
project_root = Path("/Users/fulvioventura/nba-predictor-streamlit")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "nba_predictive_system"))

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    from advanced_nba_prediction_engine import AdvancedNBAPredictionEngine

    print("Initializing Engine...")
    engine = AdvancedNBAPredictionEngine()

    if engine.historical_data is not None and not engine.historical_data.empty:
        print(f"✅ Historical Data Loaded: {len(engine.historical_data)} rows")

        print(f"\nLoaded {len(engine.team_metrics)} team metrics.")
        print("Sample Team Metrics:")
        count = 0
        for tid, metrics in engine.team_metrics.items():
            print(f"ID: '{tid}' (type: {type(tid)}) -> Name: '{metrics.team_name}'")
            count += 1
            if count >= 5:
                break

        # Check specific lookup
        print("\nChecking specific teams:")
        bulls_id = "1610612741"
        print(f"Lookup 'Chicago Bulls':")
        found = False
        for tid, metrics in engine.team_metrics.items():
            if metrics.team_name == "Chicago Bulls":
                print(f"  FOUND! ID: {tid}")
                found = True
        if not found:
            print("  NOT FOUND in metrics values.")

    else:
        print("❌ Historical Data is EMPTY or None")

    # Try a prediction
    print("\nAttempting Prediction...")
    result = engine.predict_game_total(
        "Chicago Bulls", "Sacramento Kings", "2025-12-02"
    )
    print(f"Prediction Result: {result.predicted_total}")
    print(f"Prediction Method: {result.prediction_factors}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
