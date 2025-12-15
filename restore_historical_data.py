#!/usr/bin/env python3
"""
restore_historical_data.py

Script to restore missing NBA historical data (2021-2025) required for the ML pipeline.
Uses the internal NBAStatisticsDownloadEngine to fetch data via data_store.
"""

import sys
import os
import logging

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

from nba_predictor.core.statistics_download_engine import NBAStatisticsDownloadEngine
from nba_predictor.core.data_store import UnifiedDataStore

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("🏀 Starting Missing Data Restoration...")

    try:
        # Initialize Data Store pointing to the correct PROJECT ROOT data directory
        # The dashboard uses "data/" relative to root.
        data_store = UnifiedDataStore(base_path="data")
        data_store.initialize()
        logger.info("✅ UnifiedDataStore initialized at 'data/'")

        # Initialize Engine with this store
        engine = NBAStatisticsDownloadEngine(data_store=data_store)

        # Seasons to restore: Current + 3 previous (Total 4)
        seasons_to_restore = ["2024-25", "2023-24", "2022-23", "2021-22"]

        logger.info(f"Target seasons: {seasons_to_restore}")

        # Setup tasks
        engine.setup_complete_statistics_download(seasons_to_restore)

        # Run download cycle
        # We allow unlimited tasks until completion
        results = engine.run_statistics_download_cycle()

        logger.info(f"✅ Restoration completed: {results}")

    except Exception as e:
        logger.error(f"❌ Restoration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
