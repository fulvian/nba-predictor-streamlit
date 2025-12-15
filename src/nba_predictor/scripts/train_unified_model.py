#!/usr/bin/env python3
"""
🏀 NBA Unified Model Retrainer
Trigger script to retrain the UnifiedHybridPipeline with new feature optimizations.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from src.nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("🚀 Starting Unified Model Retraining...")

        # Initialize pipeline
        pipeline = UnifiedHybridPipeline(
            use_stacked_ensemble=True, enable_explainability=True, validate_realism=True
        )

        # Train model
        logger.info(
            "📊 Training model with new feature optimizations (EWMA, Interactions)..."
        )
        metrics = pipeline.train_unified_model()

        # Save model
        logger.info("💾 Saving new model version...")
        model_path = pipeline.save_model()

        logger.info(f"✅ Retraining complete! Model saved to: {model_path}")
        logger.info("📈 Training Metrics:")
        for metric, value in metrics.items():
            logger.info(f"   - {metric}: {value}")

    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
