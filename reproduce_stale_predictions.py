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

    # Game 1: POR vs CLE
    logger.info("\n🏀 Predicting Game 1: POR vs CLE (Line: 225.0)")
    result1 = pipeline.predict_unified(
        team1="Portland Trail Blazers",
        team2="Cleveland Cavaliers",
        line=225.0,
        home_team="Cleveland Cavaliers",
        validate_prediction=True,
    )
    logger.info(f"✅ Game 1 Prediction: {result1.predicted_total}")

    # Game 2: LAL vs BOS (Different teams, different line)
    logger.info("\n🏀 Predicting Game 2: LAL vs BOS (Line: 230.0)")
    result2 = pipeline.predict_unified(
        team1="Los Angeles Lakers",
        team2="Boston Celtics",
        line=230.0,
        home_team="Boston Celtics",
        validate_prediction=True,
    )
    logger.info(f"✅ Game 2 Prediction: {result2.predicted_total}")

    if result1.predicted_total == result2.predicted_total:
        logger.error("❌ CRITICAL: Predictions are IDENTICAL! State leakage confirmed.")
    else:
        logger.info("✅ Predictions are different. No obvious state leakage.")

except Exception as e:
    logger.error(f"❌ Test failed: {e}", exc_info=True)
