import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path.cwd()
sys.path.append(str(project_root / "src"))

from nba_predictor.core.unified_hybrid_pipeline import get_unified_hybrid_pipeline


def debug_features():
    pipeline = get_unified_hybrid_pipeline()
    if not pipeline.load_model():
        logger.error(
            "❌ Failed to load model. Ensure reproduce_stale_predictions.py has run."
        )
        return

    team1 = "Los Angeles Lakers"
    team2 = "Boston Celtics"

    logger.info(f"🔍 Debugging features for {team1} vs {team2}")

    # Load data
    data_sources = pipeline.load_all_integrated_data()

    # Create features
    features = pipeline._create_unified_prediction_features(
        team1, team2, is_team1_home=False, data_sources=data_sources
    )

    if not features:
        logger.error("❌ Failed to create features")
        return

    logger.info("\n📊 RAW FEATURES:")
    for k, v in features.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Convert to DataFrame
    features_df = pd.DataFrame([features])

    # Add missing columns
    for col in pipeline.feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0
            logger.warning(f"⚠️ Missing feature added as 0.0: {col}")

    features_df = features_df[pipeline.feature_columns]

    # Scale
    features_scaled = pipeline.feature_scaler.transform(features_df)

    logger.info("\n⚖️ SCALED FEATURES (First 20):")
    for i, col in enumerate(features_df.columns[:20]):
        logger.info(f"  {col}: {features_scaled[0][i]:.4f}")

    # Predict
    logger.info(f"Pipeline trained_model: {pipeline.trained_model}")
    logger.info(f"Pipeline model: {getattr(pipeline, 'model', 'Not Set')}")

    if pipeline.trained_model is None:
        if hasattr(pipeline, "model") and pipeline.model is not None:
            logger.warning(
                "⚠️ pipeline.trained_model is None, but pipeline.model is set. Using pipeline.model."
            )
            pipeline.trained_model = pipeline.model
        else:
            logger.error("❌ Both pipeline.trained_model and pipeline.model are None!")
            return

    raw_prediction = float(pipeline.trained_model.predict(features_scaled)[0])
    logger.info(f"\n🔮 RAW MODEL PREDICTION: {raw_prediction:.4f}")


if __name__ == "__main__":
    debug_features()
