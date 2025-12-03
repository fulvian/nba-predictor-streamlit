import logging
import sys
from pathlib import Path

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

    logger.info("🎯 Attempting prediction for POR vs CLE...")
    # Using dummy data for prediction
    result = pipeline.predict_unified(
        team1="Portland Trail Blazers",
        team2="Cleveland Cavaliers",
        line=234.5,
        home_team="Cleveland Cavaliers",
        validate_prediction=True,
    )

    logger.info(f"✅ Prediction successful: {result.predicted_total}")
    print(f"Prediction: {result.predicted_total}")

except Exception as e:
    logger.error(f"❌ Pipeline failure: {e}", exc_info=True)
    print(f"Error: {e}")
