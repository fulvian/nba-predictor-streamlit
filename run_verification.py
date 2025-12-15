import logging
import sys
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline


def run_verification():
    logger.info("Initializing UnifiedHybridPipeline...")
    pipeline = UnifiedHybridPipeline(data_path="data", model_path="models")

    # Force load data
    pipeline.load_all_integrated_data()

    logger.info("Attempting prediction...")
    # Predict Boston vs Miami (Should have plenty of history)
    result = pipeline.predict_unified(
        team1="Boston Celtics", team2="Miami Heat", line=220.5, validate_prediction=True
    )

    print("\nPrediction Result:")
    print(result)

    # Also log the specific Four Factors to see if they are generic or specific
    print("\nFour Factors Analysis:")
    print(result.four_factors_analysis)


if __name__ == "__main__":
    run_verification()
