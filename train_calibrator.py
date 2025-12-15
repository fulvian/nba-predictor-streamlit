"""
Train Probability Calibrator on Historical Data

Fits Platt Scaling calibrator using 59 historical bets from nba_betting.duckdb.
Performs time-series split validation and reports calibration metrics.
"""

import duckdb
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nba_predictor.intelligence.probability_calibrator import PlattCalibrator
from src.nba_predictor.intelligence.bayesian_validator import (
    BayesianConfidenceValidator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_historical_bets(db_path: str = "data/nba_betting.duckdb"):
    """
    Fetch all settled bets with confidence and outcome.

    Returns:
        confidences: np.ndarray of model confidences
        outcomes: np.ndarray of binary outcomes (0=loss, 1=win)
        bet_ids: list of bet IDs for reference
    """
    conn = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT 
            bet_id,
            prediction,
            result,
            created_at
        FROM bets
        WHERE status IN ('SETTLED', 'WON', 'LOST')
        
        UNION ALL
        
        SELECT 
            id as bet_id,
            prediction,
            result,
            created_at
        FROM historical_calibration
        
        ORDER BY created_at ASC
    """

    rows = conn.execute(query).fetchall()
    conn.close()

    confidences = []
    outcomes = []
    bet_ids = []

    for row in rows:
        bet_id, prediction_json, result, created_at = row

        # Parse prediction JSON
        import json

        if isinstance(prediction_json, str):
            pred = json.loads(prediction_json)
        else:
            pred = prediction_json

        # Extract confidence (try multiple keys)
        confidence = (
            pred.get("confidence")
            or pred.get("model_confidence")
            or pred.get("over_probability")
        )

        if confidence is None:
            logger.warning(f"Bet {bet_id}: No confidence found. Skipping.")
            continue

        # Normalize confidence to [0,1] if it's in [0,100]
        if confidence > 1.0:
            confidence = confidence / 100.0

        # Extract outcome (result is simple string: "WON", "LOST", "SETTLED")
        # Use result field to determine outcome
        if result == "WON":
            won = True
        elif result == "LOST":
            won = False
        else:
            # SETTLED or other - skip ambiguous ones
            logger.warning(f"Bet {bet_id}: Ambiguous result '{result}'. Skipping.")
            continue

        confidences.append(confidence)
        outcomes.append(1 if won else 0)
        bet_ids.append(bet_id)

    logger.info(f"Loaded {len(confidences)} historical bets")

    return np.array(confidences), np.array(outcomes), bet_ids


def train_calibrator(
    confidences: np.ndarray, outcomes: np.ndarray, test_size: float = 0.20
):
    """
    Train calibrator with time-series split (Consensus requirement).

    Args:
        confidences: Historical model confidences
        outcomes: Historical outcomes (0/1)
        test_size: Fraction of data for testing (most recent bets)
    """
    # Time-series split: Train on first 80%, test on last 20%
    split_idx = int(len(confidences) * (1 - test_size))

    train_conf = confidences[:split_idx]
    train_outcomes = outcomes[:split_idx]
    test_conf = confidences[split_idx:]
    test_outcomes = outcomes[split_idx:]

    logger.info(f"Split: Train N={len(train_conf)}, Test N={len(test_conf)}")

    # Initialize calibrator with L2 regularization
    calibrator = PlattCalibrator(regularization_strength=1.0)

    # Fit on training data
    logger.info("Fitting Platt Calibrator...")
    calibrator.fit(train_conf, train_outcomes)

    # Evaluate on test set
    logger.info("\n=== Test Set Evaluation ===")
    test_calibrated = calibrator.calibrate_batch(test_conf)

    # Metrics
    test_ece = calibrator._compute_ece(test_calibrated, test_outcomes, n_bins=5)
    test_brier_raw = np.mean((test_conf - test_outcomes) ** 2)
    test_brier_calib = np.mean((test_calibrated - test_outcomes) ** 2)

    logger.info(f"Test ECE: {test_ece:.3f} (Target: <0.10)")
    logger.info(
        f"Test Brier Score: {test_brier_raw:.3f} -> {test_brier_calib:.3f} "
        f"(Improvement: {(1 - test_brier_calib / test_brier_raw) * 100:.1f}%)"
    )

    # Check if meets targets
    if test_ece < 0.10:
        logger.info("✅ ECE target MET (<0.10)")
    else:
        logger.warning(f"⚠️ ECE target MISSED: {test_ece:.3f} >= 0.10")

    brier_improvement = (1 - test_brier_calib / test_brier_raw) * 100
    if brier_improvement > 15:
        logger.info(f"✅ Brier improvement target MET (>{brier_improvement:.1f}%)")
    else:
        logger.warning(
            f"⚠️ Brier improvement target MISSED: {brier_improvement:.1f}% < 15%"
        )

    return calibrator


def analyze_buckets(
    calibrator: PlattCalibrator, validator: BayesianConfidenceValidator
):
    """Analyze which confidence buckets pass Kill-Switch validation."""
    logger.info("\n=== Bucket Analysis (Kill-Switch Check) ===")

    buckets = [
        (0.0, 0.6, "Low"),
        (0.6, 0.7, "Medium-Low"),
        (0.7, 0.8, "Medium"),
        (0.8, 0.9, "Medium-High"),
        (0.9, 1.0, "High"),
    ]

    for lower, upper, label in buckets:
        mid = (lower + upper) / 2
        stats = calibrator.get_bucket_stats(mid, window=upper - lower)

        result = validator.validate_bucket(stats["wins"], stats["losses"])

        status = "✅ APPROVED" if result.is_valid else "❌ KILL-SWITCH"

        logger.info(
            f"{label} ({lower:.1f}-{upper:.1f}): N={stats['n']}, "
            f"WR={stats['win_rate']:.1%}, "
            f"CI=[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}] "
            f"- {status}"
        )

        if not result.is_valid:
            logger.warning(f"  Reason: {result.reason}")


def main():
    """Main training pipeline."""
    logger.info("=== Calibrator Training Pipeline ===\n")

    # 1. Load historical data
    confidences, outcomes, bet_ids = fetch_historical_bets()

    if len(confidences) < 10:
        logger.error("Insufficient data for training. Need at least 10 bets.")
        return

    # 2. Train calibrator
    calibrator = train_calibrator(confidences, outcomes, test_size=0.20)

    # 3. Analyze buckets
    validator = BayesianConfidenceValidator(min_samples=50, max_bucket_ece=0.15)
    analyze_buckets(calibrator, validator)

    # 4. Save calibrator
    save_path = "models/probability_calibrator.pkl"
    Path("models").mkdir(exist_ok=True)
    calibrator.save(save_path)
    logger.info(f"\n✅ Calibrator saved to {save_path}")


if __name__ == "__main__":
    main()
