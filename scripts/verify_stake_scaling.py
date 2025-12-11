#!/usr/bin/env python3
"""
VERIFICATION SCRIPT: Small Bankroll Stake Scaling
"""

import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_stake_scaling():
    logger.info("🚀 Starting Stake Scaling Verification (Small Bankroll: €82.00)")

    # Initialize Manager (Constructor only takes data_path)
    # We will manually set the bankroll after init
    risk_manager = LegacyRiskManager(data_path="data")

    # Manually Inject State
    risk_manager.current_bankroll = 82.0
    risk_manager.debug_mode = True  # Inject debug flag we added in the method

    # Test Cases: Varying Edge and Risk Levels
    test_cases = [
        # Case 1: High Edge, Low Risk (Should scale significantly)
        {
            "edge": 0.15,
            "prob": 0.65,
            "odds": 1.90,
            "risk": "LOW",
            "desc": "High Edge / Low Risk",
        },
        # Case 2: Moderate Edge, Med Risk (Should scale moderate)
        {
            "edge": 0.08,
            "prob": 0.58,
            "odds": 1.90,
            "risk": "MED",
            "desc": "Med Edge / Med Risk",
        },
        # Case 3: Small Edge, High Risk (Should still be small, maybe €1.00)
        {
            "edge": 0.04,
            "prob": 0.54,
            "odds": 1.90,
            "risk": "HIGH",
            "desc": "Small Edge / High Risk",
        },
    ]

    results = []

    for case in test_cases:
        logger.info(f"\n--- Testing: {case['desc']} ---")
        stake = risk_manager.calculate_advanced_stake(
            edge=case["edge"],
            estimated_prob=case["prob"],
            odds=case["odds"],
            bankroll=82.0,
            risk_level=case["risk"],
        )
        logger.info(f"💰 Final Stake: €{stake:.2f}")
        results.append(stake)

    logger.info("\n=== VERIFICATION RESULTS ===")
    logger.info(f"Stakes: {results}")

    # Validation
    # We want granularity: not all 1.0, not all same value.
    unique_stakes = set(results)
    if len(unique_stakes) > 1:
        logger.info("✅ SUCCESS: Granite Detected (Stakes are different)")
        # Check against pure floor
        if all(s >= 1.0 for s in results):
            logger.info("✅ SUCCESS: All stakes respect €1.00 floor")
    else:
        logger.error("❌ FAILURE: Granularity missing! All stakes are same.")

    # Check Case 1 specifically (should be > 1.0)
    if results[0] > 1.2:
        logger.info(f"✅ SUCCESS: Top Tier bet scaled nicely (€{results[0]:.2f})")
    else:
        logger.warning(f"⚠️ WARNING: Top Tier bet is still small (€{results[0]:.2f})")


if __name__ == "__main__":
    test_stake_scaling()
