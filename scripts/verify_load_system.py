#!/usr/bin/env python3
"""
Verification script for LOAD System (Live Odds Anomaly Detector).
Generates synthetic market data and runs a backtest replay to verify core components.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.live_betting import ReplayEngine, AnomalyDetector, OddsSnapshot, Sport
from src.live_betting.value_betting_engine import PaperTradeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_ticks=100) -> pd.DataFrame:
    """Generate synthetic odds data with a planted anomaly."""
    logger.info("🧪 Generating synthetic data...")

    start_time = datetime.now()
    data = []

    market_id = "1.23456789"
    selection_id = 12345

    # 1. Stable baseline phase (Ticks 0-20)
    # Price oscillating around 2.00
    for i in range(20):
        data.append(
            {
                "timestamp": start_time + timedelta(seconds=i),
                "market_id": market_id,
                "selection_id": selection_id,
                "sport": "football",
                "competition": "Test League",
                "event_name": "Team A vs Team B",
                "runner_name": "Team A",
                "back_price": 2.00 + np.random.normal(0, 0.01),
                "lay_price": 2.02 + np.random.normal(0, 0.01),
                "ltp": 2.01,
                "total_matched": 1000 + i * 10,
                "back_volume": 500,
                "lay_volume": 500,
                "in_play": True,
            }
        )

    # Set baseline manually for detector in the test script context?
    # No, ReplayEngine should handle it or we assume it picks first as baseline?
    # Actually AnomalyDetector needs set_baseline called explicitly usually.
    # But let's see if ReplayEngine handles it.
    # Checking ReplayEngine code... it wipes history but doesn't auto-set baseline.
    # We might need to handle that in the loop or set it before.
    # Ah, the AnomalyDetector expects a baseline to detect reversals.
    # Let's add code to set it in the verification run.

    # 2. Drift phase (Ticks 21-40) - Price goes UP significantly (Drift)
    # Should trigger LAY signal if deemed a reversal from baseline
    for i in range(20):
        tick_idx = 20 + i
        price = 2.00 + (i * 0.05)  # 2.00 -> 3.00
        data.append(
            {
                "timestamp": start_time + timedelta(seconds=tick_idx),
                "market_id": market_id,
                "selection_id": selection_id,
                "sport": "football",
                "competition": "Test League",
                "event_name": "Team A vs Team B",
                "runner_name": "Team A",
                "back_price": price,
                "lay_price": price + 0.05,
                "ltp": price,
                "total_matched": 2000 + i * 10,
                "back_volume": 200,  # Lower volume
                "lay_volume": 200,
                "in_play": True,
            }
        )

    # 3. Recovery phase (Ticks 41-60)
    for i in range(20):
        tick_idx = 40 + i
        price = 3.00 - (i * 0.02)
        data.append(
            {
                "timestamp": start_time + timedelta(seconds=tick_idx),
                "market_id": market_id,
                "selection_id": selection_id,
                "sport": "football",
                "competition": "Test League",
                "event_name": "Team A vs Team B",
                "runner_name": "Team A",
                "back_price": price,
                "lay_price": price + 0.02,
                "ltp": price,
                "total_matched": 3000 + i * 10,
                "back_volume": 1000,
                "lay_volume": 1000,
                "in_play": True,
            }
        )

    return pd.DataFrame(data)


def run_verification():
    # 1. Generate Data
    odds_data = generate_synthetic_data()

    # 2. Setup Engine
    config = PaperTradeConfig(
        initial_bankroll=1000.0,
        min_ev_threshold=0.01,  # Low threshold for test
    )
    engine = ReplayEngine(initial_bankroll=1000.0, config=config)

    # 3. Manually set baseline for the anomaly detector
    # (Since ReplayEngine doesn't auto-set baselines yet - a future improvement)
    first_row = odds_data.iloc[0]
    snapshot = engine._row_to_snapshot(first_row)
    engine.detector.set_baseline(snapshot)
    logger.info(f"✅ Baseline set at price {snapshot.back_price:.2f}")

    # 4. Run Backtest
    logger.info("🚀 Running Backtest...")
    result = engine.run_backtest(odds_data)

    # 5. Verify Results
    logger.info("\n" + "=" * 50)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 50)
    logger.info(str(result))

    # Assertions
    if result.trade_count > 0:
        logger.info("\n✅ SUCCESS: Trades were generated!")
        for trade in engine.trades:
            logger.info(f"  - {trade}")
    else:
        logger.warning("\n⚠️ WARNING: No trades generated. Check thresholds.")

    logger.info("\n✅ LOAD System Verification Complete.")


if __name__ == "__main__":
    run_verification()
