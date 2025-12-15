import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup path to import src
sys.path.append("src")

from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    print("Initializing Pipeline...")
    print("Initializing Pipeline...")
    pipeline = UnifiedHybridPipeline(
        data_path=Path("data"),
        enable_explainability=False,  # SAVE MEMORY: Disable SHAP
        use_stacked_ensemble=True,
    )

    if not pipeline.load_model():
        print("Model not found. Training new model (this may take a minute)...")
        pipeline.train_unified_model()
        pipeline.save_model()
        print("✅ New model trained and saved.")
    else:
        print(
            f"✅ Model loaded from {pipeline.model_path / 'unified_model_latest.pkl'}"
        )

    print("\n--- Test Case 1: Unknown Teams ---")
    try:
        result = pipeline.predict_unified(
            team1="Ghost Ballers",
            team2="Phantom Squad",
            line=220.0,
            validate_prediction=False,  # Disable validation to see raw
        )
        print(f"Prediction for Unknown Teams: {result.predicted_total}")
    except Exception as e:
        print(f"Error predicting unknown teams: {e}")

    # Test case 2: Real Teams but bad mapping (simulated by passing slightly wrong names if fuzzy matching isn't robust)
    print("\n--- Test Case 2: Mismatched Names ---")
    # Assuming "L.A. Lakers" might be mismatched to "Los Angeles Lakers" if not handled
    try:
        result = pipeline.predict_unified(
            team1="L.A. Lakers",
            team2="Golden State",  # valid is likely "Golden State Warriors"
            line=220.0,
            validate_prediction=False,
        )
        print(f"Prediction for Mismatched Names: {result.predicted_total}")
    except Exception as e:
        print(f"Error predicting mismatched teams: {e}")

    # Test case 3: Valid Teams (Verify Success Path & EV Fix)
    print("\n--- Test Case 3: Valid Teams ---")
    try:
        result = pipeline.predict_unified(
            team1="Los Angeles Lakers",
            team2="Golden State Warriors",
            line=225.0,
            validate_prediction=True,
        )
        print(f"Prediction for Valid Teams: {result.predicted_total}")
        print(f"EV Analysis Result (Check log for errors): {result.ev_analysis}")
    except Exception as e:
        print(f"Error predicting valid teams: {e}")


if __name__ == "__main__":
    main()
