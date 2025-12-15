import duckdb
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nba_predictor.intelligence.probability_calibrator import PlattCalibrator
from nba_predictor.intelligence.bayesian_validator import BayesianConfidenceValidator
from nba_predictor.ml.feature_engineering import AdvancedFeatureEngine


@dataclass
class BacktestResult:
    strategy_name: str
    bets_placed: int
    wins: int
    losses: int
    net_profit: float
    roi: float
    max_drawdown: float


def fetch_historical_bets(db_path: str = "data/nba_betting.duckdb"):
    """Fetch settled bets for backtesting."""
    if not Path(db_path).exists():
        logger.error(f"DB not found: {db_path}")
        return [], [], []

    conn = duckdb.connect(db_path, read_only=True)
    query = """
        SELECT bet_id, prediction, result, created_at
        FROM bets
        WHERE status IN ('SETTLED', 'WON', 'LOST')
        ORDER BY created_at ASC
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    confidences = []
    outcomes = []
    rows_valid = []

    for row in rows:
        bet_id, pred_json, result, created_at = row
        try:
            pred = json.loads(pred_json) if isinstance(pred_json, str) else pred_json
            confidence = (
                pred.get("confidence")
                or pred.get("model_confidence")
                or pred.get("over_probability")
            )

            if confidence is None:
                continue
            if confidence > 1.0:
                confidence /= 100.0

            won = result == "WON"
            if result not in ["WON", "LOST"]:
                continue

            confidences.append(confidence)
            outcomes.append(1 if won else 0)
            rows_valid.append(row)

        except Exception:
            continue

    return np.array(confidences), np.array(outcomes), rows_valid


def run_roi_backtest(confidences, outcomes, test_size=19):
    """
    Compare Base Strategy vs Redesign Strategy on Test Set.
    Assumptions: Flat 1 unit stake. Odds -110 (1.909).
    """
    split_idx = len(confidences) - test_size
    train_conf, test_conf = confidences[:split_idx], confidences[split_idx:]
    train_out, test_out = outcomes[:split_idx], outcomes[split_idx:]

    logger.info(f"\n=== ROI Backtest (N_Test={len(test_conf)}) ===")

    # Train Calibrator
    calibrator = PlattCalibrator(
        regularization_strength=0.01
    )  # Aggressive C=0.01 per decision
    calibrator.fit(train_conf, train_out)

    # Initialize Validator
    validator = BayesianConfidenceValidator(min_samples=50, max_bucket_ece=0.15)

    # Simulation
    strategies = {
        "Base (Always Bet)": {"pnl": 0.0, "bets": 0, "wins": 0},
        "Redesign (Calib+KS)": {"pnl": 0.0, "bets": 0, "wins": 0},
    }

    ODDS = 1.909  # -110 American

    # Debug
    logger.info(f"Test Confidences: {test_conf}")

    for i in range(len(test_conf)):
        raw_conf = test_conf[i]
        outcome = test_out[i]

        # Debug
        # logger.info(f"Bet {i}: Conf={raw_conf}, Outcome={outcome}")

        # --- Base Strategy ---
        # Bets on everything > 0.5 (which is all of them usually)
        if raw_conf > 0.5:
            strategies["Base (Always Bet)"]["bets"] += 1
            if outcome == 1:
                strategies["Base (Always Bet)"]["wins"] += 1
                strategies["Base (Always Bet)"]["pnl"] += ODDS - 1
            else:
                strategies["Base (Always Bet)"]["pnl"] -= 1.0

        # --- Redesign Strategy ---
        calib_conf = calibrator.calibrate(raw_conf)
        bucket_stats = calibrator.get_bucket_stats(calib_conf)
        allow, reason = validator.should_allow_bet(calib_conf, bucket_stats)

        if allow:
            strategies["Redesign (Calib+KS)"]["bets"] += 1
            if outcome == 1:
                strategies["Redesign (Calib+KS)"]["wins"] += 1
                strategies["Redesign (Calib+KS)"]["pnl"] += ODDS - 1
            else:
                strategies["Redesign (Calib+KS)"]["pnl"] -= 1.0
        else:
            # Check what we missed/saved
            if outcome == 0:
                logger.debug(f"KS SAVED a loss! Conf {raw_conf:.2f}->{calib_conf:.2f}")
            else:
                logger.debug(f"KS MISSED a win. Conf {raw_conf:.2f}->{calib_conf:.2f}")

    # Report
    for name, stats in strategies.items():
        bets = stats["bets"]
        if bets > 0:
            roi = (stats["pnl"] / bets) * 100
            wr = (stats["wins"] / bets) * 100
        else:
            roi = 0.0
            wr = 0.0

        logger.info(
            f"{name}: Bets={bets}, WR={wr:.1f}%, PnL={stats['pnl']:.2f}u, ROI={roi:.1f}%"
        )


def validate_feature_engine():
    """
    Validate Feature Engine correlations (Pillar 1 hypothesis).
    """
    logger.info("\n=== Feature Engine Validation ===")

    # Load dataset that has basic stats
    data_path = "data/nba_simple_complete_dataset.csv"
    if not Path(data_path).exists():
        logger.warning(
            f"Feature dataset not found at {data_path}. Skipping Feature Validation."
        )
        return

    try:
        df = pd.read_csv(data_path)

        # We can't verify 'calculation' logic without raw FGA/FTA,
        # but we can verify the 'value' of the feature (Pace) by correlating existing columns.

        # 1. Pace Correlation using existing metric
        # Use dropna to ensure valid correlation
        valid_df = df.dropna(subset=["GAME_PACE", "TOTAL_SCORE"])

        if not valid_df.empty:
            # Check Variance
            pace_std = valid_df["GAME_PACE"].std()
            score_std = valid_df["TOTAL_SCORE"].std()

            if pace_std == 0 or score_std == 0:
                logger.warning(
                    f"Zero variance detected: Pace_Std={pace_std}, Score_Std={score_std}"
                )
            else:
                corr = valid_df["GAME_PACE"].corr(valid_df["TOTAL_SCORE"])
                logger.info(
                    f"Pace (GAME_PACE) vs Total Score Correlation (N={len(valid_df)}): {corr:.3f}"
                )

                if abs(corr) > 0.1:
                    logger.info(
                        "✅ Pace has significant correlation (predictive signal)"
                    )
                else:
                    logger.warning(f"⚠️ Pace correlation weak ({corr:.3f})")
        else:
            logger.warning("No valid data for Pace correlation")

        # 2. Rest Impact (if date available)
        # Placeholder for simple verify

    except Exception as e:
        logger.error(f"Feature validation failed: {e}")


def main():
    logger.info("🚀 STARTING BACKTEST REDESIGN")

    # 1. ROI Backtest
    conf, out, _ = fetch_historical_bets()
    if len(conf) >= 10:
        run_roi_backtest(conf, out, test_size=19)
    else:
        logger.warning("Not enough bets for ROI backtest.")

    # 2. Feature Validation
    validate_feature_engine()


if __name__ == "__main__":
    main()
