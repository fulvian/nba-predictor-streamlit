import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from src.nba_predictor.core.unified_hybrid_pipeline import (
    UnifiedHybridPipeline,
    UnifiedPredictionResult,
)


def main():
    logger.info("🚀 Starting Full System Verification")

    # 1. Initialize Pipeline
    try:
        pipeline = UnifiedHybridPipeline()
        logger.info("✅ Pipeline Initialized")
    except Exception as e:
        logger.error(f"❌ Pipeline Initialization Failed: {e}")
        return

    # 2. Check Components
    if hasattr(pipeline, "feature_engine"):
        logger.info("✅ Feature Engine: Attached")
    else:
        logger.error("❌ Feature Engine: Missing")

    if hasattr(pipeline, "calibrator"):
        logger.info("✅ Platt Calibrator: Attached")
    else:
        logger.error("❌ Platt Calibrator: Missing")

    if hasattr(pipeline, "bayesian_validator"):
        logger.info("✅ Bayesian Validator: Attached")
    else:
        logger.error("❌ Bayesian Validator: Missing")

    if hasattr(pipeline, "feedback_loop"):
        logger.info("✅ Feedback Loop: Attached")
    else:
        logger.error("❌ Feedback Loop: Missing")

    # 3. Simulate a Prediction Call (Mocking inputs essentially)
    # Since actually running .predict requires live data fetch/API calls which might fail or cost money/rate limit,
    # we will verify the distinct methods that make up the flow.

    # A. Feature Engine Input Check
    try:
        import pandas as pd

        mock_df = pd.DataFrame(
            [
                {
                    "GAME_ID": "0022300001",
                    "GAME_DATE": datetime.now().strftime("%Y-%m-%d"),
                    "HOME_TEAM_NAME": "Boston Celtics",
                    "AWAY_TEAM_NAME": "Los Angeles Lakers",
                    "home_fga": 85,
                    "away_fga": 85,  # Short
                    "home_fta": 20,
                    "away_fta": 20,
                    "home_orb": 10,
                    "away_orb": 10,
                    "home_tov": 12,
                    "away_tov": 12,
                    "home_score": 110,
                    "away_score": 105,
                    "minutes": 48,
                }
            ]
        )
        enriched = pipeline.feature_engine.add_all_features(mock_df)
        if "PACE_MATCHUP" in enriched.columns or "pace_matchup" in enriched.columns:
            logger.info("✅ Feature Engine Logic: Verified (Pace Calculated)")
        else:
            logger.warning(
                f"⚠️ Feature Engine Logic: Pace column missing (Cols: {enriched.columns.tolist()})"
            )
    except Exception as e:
        logger.error(f"❌ Feature Engine Test Failed: {e}")

    # B. Feedback Loop Prompt Gen
    try:
        bias_prompt = pipeline.feedback_loop.generate_correction_prompt(
            "Boston Celtics", "Los Angeles Lakers"
        )
        if "Boston Celtics" in bias_prompt:
            logger.info("✅ Feedback Loop Logic: Verified")
        else:
            logger.error("❌ Feedback Loop Logic: Output empty/wrong")
    except Exception as e:
        logger.error(f"❌ Feedback Loop Test Failed: {e}")

    # C. Kill Switch Check (Offline)
    try:
        # Mock confidence and bucket stats
        raw_conf = 0.60
        calib_conf = pipeline.calibrator.calibrate(raw_conf)
        stats = pipeline.calibrator.get_bucket_stats(calib_conf)
        allow, reason = pipeline.bayesian_validator.should_allow_bet(calib_conf, stats)

        logger.info(f"✅ Calibration Logic: {raw_conf} -> {calib_conf:.3f}")
        logger.info(f"✅ Kill-Switch Logic: Allow={allow}, Reason='{reason}'")

    except Exception as e:
        logger.error(f"❌ Safety Logic Test Failed: {e}")

    logger.info("🏁 Verification Complete")


if __name__ == "__main__":
    main()
