"""
Train Market-Adjusted Model (XGBoost/LightGBM).

This script:
1. Loads 'data/nba_training_features_v2.parquet'.
2. Performs Time-Series Split (Train: < 2024-25, Test: 2024-25).
3. Trains XGBoost Regressor on 'target_margin'.
4. Evaluates RMSE/MAE and Betting ROI.
"""

import logging
import sys
import os
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/nba_training_features_v2.parquet"
MODEL_DIR = "models/market_adjusted"


def load_data():
    df = pl.read_parquet(DATA_PATH)
    # Cast closing_total to float (handle potentially mixed types or decimals loaded as strings)
    df = df.with_columns(pl.col("closing_total").cast(pl.Float64))
    return df


def prepare_features(df):
    """Select features and target."""
    # Features: Rolling Stats + Situational + Market Context
    # We EXCLUDE raw ID columns and Future info

    feature_cols = [
        c
        for c in df.columns
        if ("L5_" in c or "L10_" in c or "STD_" in c or "rest" in c or "matchup" in c)
    ]

    # Add Closing Line as feature (Context)
    feature_cols.append("closing_total")

    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

    # Bulk Cast to Float to ensure XGBoost compatibility
    cols_to_cast = feature_cols + ["target_margin"]
    # Filter only those that exist
    existing_cols = [c for c in cols_to_cast if c in df.columns]

    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in existing_cols])

    return feature_cols, df


def train_and_evaluate():
    # Load raw
    df = load_data()

    # Prepare features & Cast types
    feature_cols, df = prepare_features(df)

    # Sort by time
    df = df.sort("game_date")

    # Split: Test on 2024-2025 ("current" season in dataset terms)
    # Filter season. Note: Our ingestion named it '2024-2025'
    train_df = df.filter(pl.col("season") != "2024-2025")
    test_df = df.filter(pl.col("season") == "2024-2025")

    logger.info(
        f"Train set: {len(train_df)} games (Seasons: {train_df['season'].unique().to_list()})"
    )
    logger.info(f"Test set: {len(test_df)} games (Season: 2024-2025)")

    target_col = "target_margin"  # Actual - Closing

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(target_col).to_pandas()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas()

    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        early_stopping_rounds=50,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    logger.info("Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Predictions (Margin)
    preds_margin = model.predict(X_test)

    # Metrics on MARGIN prediction
    rmse = np.sqrt(mean_squared_error(y_test, preds_margin))
    mae = mean_absolute_error(y_test, preds_margin)
    logger.info(f"Model Performance (Margin): RMSE={rmse:.4f}, MAE={mae:.4f}")

    # --- BETTING SIMULATION ---
    # We predicted Margin = Actual - Line
    # If Pred Margin > 0 => We expect Actual > Line => Bet OVER
    # If Pred Margin < 0 => We expect Actual < Line => Bet UNDER

    # Reconstruct Betting Context
    test_context = test_df.select(
        [
            "game_date",
            "home_team_id",
            "away_team_id",
            "total_score",
            "closing_total",
            "closing_over_odds",
        ]
    ).to_pandas()
    test_context["pred_margin"] = preds_margin
    test_context["actual_margin"] = (
        test_context["total_score"] - test_context["closing_total"]
    )

    # Strategy: Bet if abs(margin) > threshold
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]

    logger.info("\n💰 BETTING SIMULATION (2024-2025):")
    for th in thresholds:
        bets = test_context[np.abs(test_context["pred_margin"]) > th].copy()

        if len(bets) == 0:
            logger.info(f"Threshold {th}: No bets placed.")
            continue

        # Determine Bet Side
        # Pred > th => Bet Over
        # Pred < -th => Bet Under
        bets["bet_side"] = np.where(bets["pred_margin"] > 0, "Over", "Under")

        # Determine Outcome
        # Over wins if Actual > Line (Actual Margin > 0)
        # Under wins if Actual < Line (Actual Margin < 0)
        # Push if Actual Margin == 0

        # Vectorized PnL
        # Win: Profit = Odds - 1 (Assuming flat 1 unit bet)
        # Loss: -1
        # Push: 0

        conditions = [
            (bets["bet_side"] == "Over") & (bets["actual_margin"] > 0),  # Win Over
            (bets["bet_side"] == "Under") & (bets["actual_margin"] < 0),  # Win Under
            (bets["actual_margin"] == 0),  # Push
        ]
        choices = [
            bets["closing_over_odds"] - 1,  # Profit
            bets["closing_over_odds"]
            - 1,  # Profit (Assuming equal odds for O/U for simplicity or using column)
            0,
        ]

        # Note: We only have 'closing_over_odds' in dataset usually?
        # ingest_kaggle_2025 set both to 1.909.
        # Let's assume 1.909 (-110) for all if missing.
        odds = bets["closing_over_odds"].fillna(1.909)

        bets["pnl"] = np.select(conditions, [odds - 1, odds - 1, 0], default=-1)

        # Wins
        wins = len(bets[bets["pnl"] > 0])
        total_bets = len(bets)
        win_rate = wins / total_bets * 100
        total_profit = bets["pnl"].sum()
        roi = (total_profit / total_bets) * 100

        logger.info(
            f"Threshold > {th} pts: {total_bets} Bets | Win Rate: {win_rate:.1f}% | Profit: {total_profit:.2f}u | ROI: {roi:.2f}%"
        )

    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = f"{MODEL_DIR}/xgb_margin_v1.json"
    model.save_model(model_file)
    logger.info(f"\nModel saved to {model_file}")


if __name__ == "__main__":
    train_and_evaluate()
