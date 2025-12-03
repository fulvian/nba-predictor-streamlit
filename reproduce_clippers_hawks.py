import logging
import sys
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path.cwd()
sys.path.append(str(project_root / "src"))

try:
    from nba_predictor.core.unified_hybrid_pipeline import get_unified_hybrid_pipeline

    logger.info("🚀 Instantiating UnifiedHybridPipeline...")
    pipeline = get_unified_hybrid_pipeline()

    team1 = "Atlanta Hawks"
    team2 = "Los Angeles Clippers"  # Note: User said Clippers @ Hawks, so Hawks is home usually, or user notation might be Away @ Home.
    # "LA Clippers @ Atlanta Hawks" usually means Clippers (Away) at Hawks (Home).

    # Let's try both combinations just in case name resolution is tricky

    logger.info(f"\n🏀 Predicting: {team2} @ {team1} (Assuming Hawks Home)")
    result = pipeline.predict_unified(
        team1=team2,  # Away
        team2=team1,  # Home
        line=225.0,  # Guessing line
        home_team=team1,
        validate_prediction=True,
    )
    logger.info(f"✅ Prediction: {result.predicted_total}")

    if result.predicted_total == 205.0:
        logger.error("❌ REPRODUCED: Prediction is exactly 205.0")
    else:
        logger.info("✅ Prediction is dynamic (not 205.0)")

except Exception as e:
    logger.error(f"❌ Test failed: {e}", exc_info=True)
