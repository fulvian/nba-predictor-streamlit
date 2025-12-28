"""
Train Spread Alpha Model (XGBoost).

Target: 'Margin vs Spread' (Score Diff + Handicap).
Positive = Home Covers. Negative = Away Covers.

Features:
- Alpha Factors: Altitude, Schedule Density Mismatch, Pace Style Clash
- Base Factors: L10 Score Trends, Rest Advantage
"""

import logging
import sys
import os
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)

DATA_PATH = "data/nba_spread_features_v1.parquet"
MODEL_DIR = "models/spread_alpha"


def load_data():
    df = pl.read_parquet(DATA_PATH)

    # Calculate Target: Margin vs Spread
    # result_diff = home_score - away_score
    # target = result_diff + spread_line
    # Example: Home wins by 10 (100-90). Line -6.5.
    # Target = 10 + (-6.5) = +3.5. (Covered logic)

    df = df.with_columns(
        [
            (pl.col("home_score") - pl.col("away_score")).alias("score_diff"),
            (
                (pl.col("home_score") - pl.col("away_score")) + pl.col("spread_line")
            ).alias("target_spread_margin"),
        ]
    )

    return df


def train_and_simulate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    df = load_data()
    # Cast to float for XGBoost (excluding metadata)
    meta_cols = ["game_id", "game_date", "season", "stage", "bookmaker"]
    df = df.with_columns(pl.exclude(meta_cols).cast(pl.Float64, strict=False))

    # Features
    features = [
        # Alpha Features
        "is_high_altitude_home",
        "density_mismatch_4d",
        "pace_mismatch",
        # Base Features
        "home_L10_score",
        "away_L10_score",
        "home_L10_points_allowed",
        "away_L10_points_allowed",
        "home_rest",
        "away_rest",
        "spread_line",  # The line itself
    ]

    target = "target_spread_margin"

    # Convert to Pandas
    pdf = df.to_pandas()

    # Split Train/Test (Test on 2024-2025 season)
    # Season format in data is "202x-202y".
    # We want to test on "2024-2025".

    train_df = pdf[pdf["season"] < "2024-2025"]
    test_df = pdf[pdf["season"] == "2024-2025"]

    if len(test_df) == 0:
        logger.warning("No 2024-2025 data found. Using 2023-2024 as test.")
        train_df = pdf[pdf["season"] < "2023-2024"]
        test_df = pdf[pdf["season"] == "2023-2024"]

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
    )

    # Validation set for early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]

    logger.info("Training Spread Alpha Model...")
    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"Spread RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Betting Simulation
    # Strategy: Bet when Pred Margin differs from 0 significantly?
    # Target is Margin vs Spread. 0 means "Push/Market Correct".
    # Pred > +X => Bet Home Cover.
    # Pred < -X => Bet Away Cover.

    # Construct Context
    sim_df = test_df.copy()
    sim_df["pred_margin"] = preds
    sim_df["actual_margin"] = y_test

    logger.info("\n💰 ROI SIMULATION - Spread Alpha (Home/Away Handicap):")
    # Assume -110 odds -> 1.909 payout. Break-even 52.4%.
    odds = 1.909

    cols = [
        "pred_margin",
        "actual_margin",
        "spread_line",
        "home_team_id",
        "away_team_id",
        "game_date",
    ]

    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]

    for thresh in thresholds:
        # Bets
        # Bet Home if Pred > Thresh
        home_bets = sim_df[sim_df["pred_margin"] > thresh].copy()
        # Bet Away if Pred < -Thresh
        away_bets = sim_df[sim_df["pred_margin"] < -thresh].copy()

        # Outcome Logic
        # Home Bet Wins if actual_margin > 0
        home_bets["pnl"] = np.where(home_bets["actual_margin"] > 0, odds - 1, -1)

        # Away Bet Wins if actual_margin < 0
        # (Negative actual margin = Away Covered)
        away_bets["pnl"] = np.where(away_bets["actual_margin"] < 0, odds - 1, -1)

        # Push logic (margin == 0) -> Refund (PnL 0)
        home_bets.loc[home_bets["actual_margin"] == 0, "pnl"] = 0
        away_bets.loc[away_bets["actual_margin"] == 0, "pnl"] = 0

        total_bets = len(home_bets) + len(away_bets)
        total_pnl = home_bets["pnl"].sum() + away_bets["pnl"].sum()
        roi = (total_pnl / total_bets * 100) if total_bets > 0 else 0.0

        logger.info(
            f"Margin Edge > {thresh} pts: {total_bets} bets | PnL: {total_pnl:.2f}u | ROI: {roi:.2f}%"
        )

    # Feature Importance
    logger.info("\nFeature Importance:")
    fi = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.info(fi.head(10))

    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(f"{MODEL_DIR}/spread_alpha_v1.json")


if __name__ == "__main__":
    train_and_simulate()
