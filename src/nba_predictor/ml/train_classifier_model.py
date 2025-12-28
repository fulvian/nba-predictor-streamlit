"""
Train Alpha Classifier (XGBoost).

This script shifts focus from Regression (Margin) to Classification (Win Probability).
Goal: Identify +EV bets where Model Prob > Implied Prob (break-even ~52.4%).

Steps:
1. Load `nba_training_features_v2.parquet`.
2. Target: `target_over_hit` (1 if Total > Line, 0 otherwise).
3. Train XGBClassifier with probability calibration (logloss).
4. Simulate Betting ROI based on dynamic probability thresholds.
"""

import logging
import sys
import os
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, precision_score
from datetime import datetime

# Add project root
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/nba_training_features_v2.parquet"
MODEL_DIR = "models/alpha_classifier"


def load_data():
    df = pl.read_parquet(DATA_PATH)
    # Ensure Closing Total is float
    df = df.with_columns(pl.col("closing_total").cast(pl.Float64))
    return df


def prepare_features(df):
    """Select features and binary target."""
    feature_cols = [
        c
        for c in df.columns
        if ("L5_" in c or "L10_" in c or "STD_" in c or "rest" in c or "matchup" in c)
    ]
    feature_cols.append("closing_total")

    # Target: 0 or 1
    # Note: target_over_hit is boolean in parquet, need int for XGB
    df = df.with_columns(pl.col("target_over_hit").cast(pl.Int8))

    # Bulk Cast Features to Float64
    cols_to_cast = feature_cols
    existing_cols = [c for c in cols_to_cast if c in df.columns]

    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in existing_cols])

    return feature_cols, df


def train_and_simulate():
    df = load_data()
    feature_cols, df = prepare_features(df)
    target_col = "target_over_hit"

    # Time Split
    df = df.sort("game_date")
    train_df = df.filter(pl.col("season") != "2024-2025")
    test_df = df.filter(pl.col("season") == "2024-2025")

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(target_col).to_pandas()
    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas()

    # XGBoost Classifier
    # scale_pos_weight=1 (assuming balanced O/U class distribution, usually true for spreads/totals)
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,  # Slightly shallower to prevent overfitting noise
        subsample=0.7,
        colsample_bytree=0.7,
        early_stopping_rounds=50,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )

    logger.info("Training Classifer (hunting for Alpha)...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Predict Probabilities
    # probs[:, 1] is probability of Over (Class 1)
    probs = clf.predict_proba(X_test)[:, 1]

    # Basic Metrics
    acc = accuracy_score(y_test, clf.predict(X_test))
    auc = roc_auc_score(y_test, probs)
    loss = log_loss(y_test, probs)

    logger.info(
        f"Classifier Metrics: Accuracy={acc:.4f} | AUC={auc:.4f} | LogLoss={loss:.4f}"
    )

    # --- ROI SIMULATION ---
    # Break-even for standard -110 odds (1.909) is 52.38% (1.1/2.1)
    # We look for Probs > Threshold

    # Reconstruct Context
    test_context = test_df.select(
        ["game_date", "home_team_id", "away_team_id", "closing_over_odds"]
    ).to_pandas()
    test_context["prob_over"] = probs
    test_context["actual_outcome"] = y_test

    # Default decimal odds ~1.909 if missing
    test_context["odds"] = test_context["closing_over_odds"].fillna(1.909).astype(float)

    # Implied Probability from Odds (1/Odds)
    # Edge = Model Prob - Implied Prob
    test_context["implied_prob"] = 1 / test_context["odds"]
    # Ensure prob_over is also float (it usually is from predict_proba but to be safe)
    test_context["prob_over"] = test_context["prob_over"].astype(float)

    test_context["edge"] = test_context["prob_over"] - test_context["implied_prob"]

    # Thresholds for "Confidence" (Model Probability of Over)
    # We mainly bet OVER if prob > X.
    # But a classifier can also bet UNDER if prob < (1 - X).

    thresholds = [0.525, 0.53, 0.54, 0.55, 0.56, 0.60]

    logger.info("\n💰 ROI SIMULATION - 'Value Betting' (2024-2025):")

    for th in thresholds:
        # Bet OVER if Prob > th
        # Bet UNDER if Prob < (1-th)

        over_bets = test_context[test_context["prob_over"] > th].copy()
        over_bets["bet"] = "Over"

        under_bets = test_context[test_context["prob_over"] < (1 - th)].copy()
        under_bets["bet"] = "Under"

        all_bets = pd.concat([over_bets, under_bets])

        if len(all_bets) == 0:
            logger.info(f"Threshold {th:.1%}: No bets found.")
            continue

        # Calc PnL
        # Over Wins if actual=1. Loss if 0.
        # Under Wins if actual=0. Loss if 1.

        conditions = [
            (all_bets["bet"] == "Over") & (all_bets["actual_outcome"] == 1),
            (all_bets["bet"] == "Under") & (all_bets["actual_outcome"] == 0),
        ]

        # Payout = Odds - 1
        payouts = [all_bets["odds"] - 1, all_bets["odds"] - 1]

        all_bets["pnl"] = np.select(conditions, payouts, default=-1)

        # Stats
        count = len(all_bets)
        profit = all_bets["pnl"].sum()
        roi = (profit / count) * 100
        win_rate = len(all_bets[all_bets["pnl"] > 0]) / count

        logger.info(
            f"Threshold > {th:.1%}: {count} Bets | Win Rate: {win_rate:.1%} | Profit: {profit:.2f}u | ROI: {roi:.2f}%"
        )

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = f"{MODEL_DIR}/xgb_classifier_v1.json"
    clf.save_model(model_file)
    logger.info(f"\nModel saved to {model_file}")


if __name__ == "__main__":
    train_and_simulate()
