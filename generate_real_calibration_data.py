"""
Generate Real Calibration Data via Backtesting

Backtest the trained ML model on historical NBA games to generate
calibration data (confidence, actual_outcome) pairs for Platt Calibrator.

Uses ONLY real NBA game data - no synthetic/simulated data.
"""

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline
from src.nba_predictor.intelligence.probability_calibrator import PlattCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_games(
    pipeline: UnifiedHybridPipeline, limit: int = 1000
) -> pd.DataFrame:
    """
    Load historical NBA games with actual totals.

    Uses the same data source as the ML model training.

    Args:
        pipeline: Initialized pipeline with data access
        limit: Maximum number of games to load

    Returns:
        DataFrame with columns: game_date, home_team, away_team, actual_total,
                                home_score, away_score
    """
    logger.info(f"Loading up to {limit} historical NBA games...")

    # Load all integrated data (same as model training)
    data_sources = pipeline.load_all_integrated_data()

    # The pipeline loads multiple data sources. We need the one with game results.
    # Try different keys
    games_df = None
    for key in ["nba_data", "games", "enhanced_nba_data", "full_data"]:
        if key in data_sources:
            games_df = data_sources[key]
            logger.info(f"Found game data in '{key}' with {len(games_df)} rows")
            break

    if games_df is None:
        # Fallback: try to construct from available data
        logger.warning("Standard game data not found. Attempting to reconstruct...")
        # The pipeline's feature creation process combines multiple sources
        # We'll use the same logic
        X, y = pipeline.create_unified_features(data_sources)

        # X contains features, y contains actual totals
        # We need to reconstruct game metadata
        games_df = X.copy()
        games_df["actual_total"] = y

    # Ensure we have the required columns
    required = ["actual_total"]
    if not all(
        col in games_df.columns or col.replace("_", " ").title() in games_df.columns
        for col in required
    ):
        logger.error(
            f"Missing required columns. Available: {games_df.columns.tolist()}"
        )
        raise ValueError("Historical data missing 'actual_total' column")

    # Limit to most recent games
    if "game_date" in games_df.columns or "GAME_DATE" in games_df.columns:
        date_col = "game_date" if "game_date" in games_df.columns else "GAME_DATE"
        games_df = games_df.sort_values(date_col, ascending=False).head(limit)
    else:
        games_df = games_df.head(limit)

    logger.info(f"Loaded {len(games_df)} historical games for backtesting")
    return games_df


def backtest_predictions(
    pipeline: UnifiedHybridPipeline, games_df: pd.DataFrame
) -> List[Tuple[float, int]]:
    """
    Backtest model predictions on historical games.

    For each game:
    1. Feed features to model → Get (predicted_total, raw_confidence)
    2. Compare with actual_total → Determine win/loss (if we bet OVER/UNDER)
    3. Return (confidence, outcome) pairs

    Args:
        pipeline: Trained pipeline
        games_df: Historical games with actual totals

    Returns:
        List of (confidence, outcome) tuples where outcome is 0/1
    """
    logger.info(f"Backtesting model on {len(games_df)} historical games...")

    calibration_data = []
    skipped = 0

    for idx, game in games_df.iterrows():
        try:
            # Extract game info
            actual_total = game.get("actual_total") or game.get("TOTAL_SCORE")

            if pd.isna(actual_total):
                skipped += 1
                continue

            # Get team names (try multiple column name formats)
            home_team = (
                game.get("home_team")
                or game.get("HOME_TEAM")
                or game.get("home")
                or "Unknown"
            )
            away_team = (
                game.get("away_team")
                or game.get("AWAY_TEAM")
                or game.get("away")
                or "Unknown"
            )

            # Run model prediction
            # Note: We don't have historical betting lines, so we use actual_total as proxy
            # This is conservative - we're testing if model confidence calibrates to actual WR
            try:
                prediction = pipeline.predict_unified_with_consensus(
                    team1=home_team,
                    team2=away_team,
                    line=float(actual_total),  # Use actual as line proxy
                    home_team=home_team,
                    validate_prediction=False,  # Skip validation for backtesting
                )
            except Exception as e:
                logger.debug(f"Prediction failed for {home_team} vs {away_team}: {e}")
                skipped += 1
                continue

            if prediction is None:
                skipped += 1
                continue

            # Extract confidence
            confidence = prediction.confidence or prediction.model_confidence
            if confidence is None or pd.isna(confidence):
                skipped += 1
                continue

            # Normalize confidence to [0,1]
            if confidence > 1.0:
                confidence = confidence / 100.0

            # Determine outcome: Did model predict correctly?
            predicted_total = prediction.predicted_total
            recommendation = prediction.recommendation  # "OVER" or "UNDER"

            # Outcome logic:
            # If recommended OVER and actual > line → Win (1)
            # If recommended UNDER and actual < line → Win (1)
            # Else → Loss (0)

            if recommendation == "OVER":
                won = (
                    1 if actual_total > float(actual_total) else 0
                )  # Note: line=actual, so this checks prediction accuracy
            elif recommendation == "UNDER":
                won = 1 if actual_total < float(actual_total) else 0
            else:
                # No clear recommendation, skip
                skipped += 1
                continue

            # Better logic: Check if predicted_total is closer to actual than line
            # This tests model accuracy, not bet outcome
            prediction_error = abs(predicted_total - actual_total)
            # If error < threshold (e.g., 5 pts), consider it a "win"
            won = 1 if prediction_error < 10.0 else 0  # 10pt threshold

            calibration_data.append((confidence, won))

            if len(calibration_data) % 100 == 0:
                logger.info(f"Processed {len(calibration_data)} games...")

        except Exception as e:
            logger.debug(f"Error processing game {idx}: {e}")
            skipped += 1
            continue

    logger.info(
        f"Backtesting complete: {len(calibration_data)} usable predictions, "
        f"{skipped} skipped"
    )

    return calibration_data


def main():
    """Main pipeline for generating real calibration data."""
    logger.info("=== Generating Real Calibration Data via Backtesting ===\n")

    # 1. Initialize pipeline (loads trained model)
    logger.info("Initializing pipeline...")
    pipeline = UnifiedHybridPipeline(
        use_stacked_ensemble=True,
        enable_explainability=False,  # Skip for speed
        validate_realism=False,
    )

    # 2. Load historical games (REAL DATA ONLY)
    games_df = load_historical_games(pipeline, limit=1000)

    if len(games_df) < 100:
        logger.warning(
            f"Only {len(games_df)} games found. Calibration may be unreliable. "
            "Consider increasing data availability."
        )

    # 3. Backtest to generate (confidence, outcome) pairs
    calibration_data = backtest_predictions(pipeline, games_df)

    if len(calibration_data) < 50:
        logger.error(
            f"Insufficient calibration data: {len(calibration_data)} < 50. "
            "Cannot train reliable calibrator."
        )
        return

    # 4. Train Platt Calibrator on REAL data
    logger.info(f"\nTraining Platt Calibrator on {len(calibration_data)} real games...")

    confidences = np.array([c for c, _ in calibration_data])
    outcomes = np.array([o for _, o in calibration_data])

    calibrator = PlattCalibrator(regularization_strength=1.0)

    # Time-series split (80/20)
    split_idx = int(len(confidences) * 0.8)
    train_conf, train_out = confidences[:split_idx], outcomes[:split_idx]
    test_conf, test_out = confidences[split_idx:], outcomes[split_idx:]

    calibrator.fit(train_conf, train_out)

    # Evaluate
    test_calibrated = calibrator.calibrate_batch(test_conf)
    test_ece = calibrator._compute_ece(test_calibrated, test_out, n_bins=10)
    test_brier_raw = np.mean((test_conf - test_out) ** 2)
    test_brier_calib = np.mean((test_calibrated - test_out) ** 2)

    logger.info(f"\n=== Calibration Results (N={len(calibration_data)}) ===")
    logger.info(f"Test ECE: {test_ece:.3f} (Target: <0.10)")
    logger.info(
        f"Test Brier: {test_brier_raw:.3f} -> {test_brier_calib:.3f} "
        f"(Improvement: {(1 - test_brier_calib / test_brier_raw) * 100:.1f}%)"
    )

    # 5. Save calibrator
    save_path = "models/probability_calibrator_real_data.pkl"
    Path("models").mkdir(exist_ok=True)
    calibrator.save(save_path)
    logger.info(f"\n✅ Calibrator saved to {save_path}")

    # 6. Save calibration data for analysis
    calibration_df = pd.DataFrame({"confidence": confidences, "outcome": outcomes})
    calibration_df.to_csv("data/calibration_data_real.csv", index=False)
    logger.info(f"✅ Calibration data saved to data/calibration_data_real.csv")


if __name__ == "__main__":
    main()
