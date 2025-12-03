import logging
import pandas as pd
import numpy as np
from pathlib import Path
from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

# Configure logging to see internal decisions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_predictor")
logger.setLevel(logging.INFO)


def debug_prediction():
    print("--- Debugging POR vs CLE Prediction ---")

    pipeline = UnifiedHybridPipeline()

    # Mock data sources if needed, or rely on pipeline loading real data
    # The pipeline loads 'nba_data_with_mu_sigma_for_ml.csv' internally

    team1 = "Portland Trail Blazers"
    team2 = "Cleveland Cavaliers"
    line = 234.5

    print(f"Predicting: {team1} vs {team2}, Line: {line}")

    try:
        # Force training if needed (or load existing)
        if not pipeline.is_trained:
            print("Training model first...")
            pipeline.train_unified_model()

        # Make prediction
        result = pipeline.predict_unified(
            team1=team1,
            team2=team2,
            line=line,
            home_team=team2,  # CLE is usually home in this notation "POR @ CLE"
            validate_prediction=True,
        )

        print("\n--- Prediction Result ---")
        print(f"Predicted Total: {result.predicted_total}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence}")

        # We can't easily see internal variables without modifying the code or using a debugger,
        # but the logs should show "Market Adjustment", "Dynamic Envelope", etc.

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_prediction()
