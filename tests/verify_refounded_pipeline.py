#!/usr/bin/env python3
"""
🧪 Verification Script for Refounded NBA Pipeline
Tests the interaction between:
1. Dynamic Bias Manager (Momentum)
2. Bayesian Shrinkage (Fail-Safe)
3. Consensus Validator (Sharp Gate)
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nba_predictor.intelligence.bias_corrector import get_bias_corrector
from src.nba_predictor.intelligence.dynamic_bias import get_dynamic_bias_manager
from src.nba_predictor.intelligence.consensus_validator import get_consensus_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("VERIFIER")


def test_pipeline_logic():
    print("\n🧪 STARTING PIPELINE LOGIC VERIFICATION\n")

    # 1. Initialize Components
    bias_corrector = get_bias_corrector()
    dynamic_bias_manager = get_dynamic_bias_manager()
    consensus_validator = get_consensus_validator()

    print("✅ Components Initialized")

    # 2. Simulate Scenario A: Normal Edge (Trust)
    # Market: 220.0, Quant: 228.0 (Over)
    # Dynamic Bias: Team A tends to go Over (+2.0)

    market_line = 220.0
    raw_quant = 226.0  # +6 edge

    # Mocking dynamic bias update for test
    # (In real app, this comes from history)
    # Let's assume net bias is +2.0
    momentum_bias = 2.0

    base_pred = raw_quant + momentum_bias  # 228.0

    print(f"\n--- Scenario A: Valid Edge ---")
    print(f"Market: {market_line}")
    print(f"Raw Quant: {raw_quant}")
    print(f"Momentum Bias: {momentum_bias} -> Base Pred: {base_pred}")

    # Apply Shrinkage
    shrunk_pred, weight, status = bias_corrector.apply_bayesian_shrinkage(
        base_pred, market_line
    )

    print(
        f"📉 Shrinkage Result: Pred={shrunk_pred:.2f} (W={weight:.2f}, Status={status})"
    )

    # Validate
    if weight > 0.9 and status == "TRUST":
        print("✅ PASS: System trusts valid edge (<5% dev)")
    else:
        print("❌ FAIL: System over-shrank valid edge")

    # 3. Simulate Scenario B: Extreme Hallucination (Kill)
    # Market: 220.0, Quant: 250.0 (Over)

    raw_quant_bad = 250.0
    base_pred_bad = raw_quant_bad  # No bias used here

    print(f"\n--- Scenario B: Extreme Hallucination ---")
    print(f"Market: {market_line}")
    print(f"Raw Quant: {raw_quant_bad} -> Deviaton: +30 pts")

    # Apply Shrinkage
    shrunk_pred_bad, weight_bad, status_bad = bias_corrector.apply_bayesian_shrinkage(
        base_pred_bad, market_line
    )

    print(
        f"📉 Shrinkage Result: Pred={shrunk_pred_bad:.2f} (W={weight_bad:.2f}, Status={status_bad})"
    )

    # Validator Check (Hypothetical Model Prob vs Market)
    # Model says 80%, Market says 50%
    is_valid, reason, fair_prob = consensus_validator.validate_bet(
        0.80, 1.91, 1.91, "OVER"
    )

    print(f"⚖️ Validator Result: Valid={is_valid}, Reason={reason}")

    # Validate
    if weight_bad < 0.2 and "EXTREME" in reason:
        print("✅ PASS: System correctly killed/shrank hallucination")
    elif weight_bad < 0.5:
        print("✅ PASS: System significantly shrank hallucination")
    else:
        print("❌ FAIL: System allowed hallucination")


if __name__ == "__main__":
    test_pipeline_logic()
