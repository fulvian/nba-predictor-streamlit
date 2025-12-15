import duckdb
import pandas as pd
import json
import sys
import re
from pathlib import Path

# Add src to path to import LegacyRiskManager
sys.path.append("src")
from nba_predictor.utils.legacy_risk_manager import LegacyRiskManager


def run_backtest():
    print("🚀 Starting Backtest of Granular Stake Sizing Strategy...")

    # Initialize Risk Manager
    rm = LegacyRiskManager(data_path="data")

    # Connect to DB
    conn = duckdb.connect("data/nba_betting.duckdb")

    # Fetch valid settled bets
    query = """
        SELECT 
            created_at,
            bet_type,
            amount as original_stake,
            odds,
            result,
            profit_loss,
            prediction,
            home_team,
            away_team
        FROM bets 
        WHERE status IN ('WON', 'LOST', 'Settled') 
        ORDER BY created_at ASC
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    print(f"📊 Found {len(df)} settled bets to backtest.\n")

    # Initial Bankroll Simulation
    # We'll use the implied starting bankroll from the first bet or a fixed amount?
    # Let's assume a starting bankroll of €100 for comparison, or trace back.
    # User's current bankroll is €79.14.
    # Let's start both at €100.00 to compare performance relative to same start.
    start_bankroll = 100.0

    bankroll_old = start_bankroll
    bankroll_new = start_bankroll

    results = []

    for idx, row in df.iterrows():
        # Parse Bet Details
        bet_str = row["bet_type"]  # e.g. "UNDER 229.5"
        parts = bet_str.split()
        direction = parts[0]  # OVER/UNDER
        try:
            line_val = float(parts[1])
        except:
            print(f"⚠️ Skipping bet {bet_str}: cannot parse line")
            continue

        # Parse Prediction JSON
        try:
            pred_data = json.loads(row["prediction"])
            predicted_total = pred_data.get("predicted_total", 0)
            raw_sigma = pred_data.get("standard_error", 15.0)
            # Apply FIX: Min sigma 10.0
            sigma_p = max(10.0, raw_sigma)
        except:
            # Fallback if no valid prediction data
            predicted_total = 0
            sigma_p = 15.0

        # Calculate Implied Prob for New Strategy
        # We need prob of winning this specific bet (Over or Under)
        from scipy.stats import norm

        if direction == "OVER":
            prob_win_new = 1 - norm.cdf(line_val, predicted_total, sigma_p)
        else:  # UNDER
            prob_win_new = norm.cdf(line_val, predicted_total, sigma_p)

        # Calculate New Stake
        # Using the same fixed parameters we just committed
        # beta=0.10, sigma_target=0.15 => c = min(1, 0.15 / (sigma_p/100)) ???
        # Wait, inside LegacyRiskManager.calculate_mean_variance_kelly_stake:
        # sigma_metric = sigma_p (which is passed as predicted_sigma/100 IN THE CODE?)
        # Let's check how analyze_betting_opportunities calls it.
        # It calls: sigma_p = max(0.05, min(0.25, predicted_sigma / 100.0))

        # So I need to emulate that normalization
        sigma_p_normalized = max(0.05, min(0.25, sigma_p / 100.0))

        new_stake = rm.calculate_mean_variance_kelly_stake(
            estimated_prob=prob_win_new,
            prob_std_error=sigma_p_normalized,  # This param name is misleading in function, but it is treated as normalized sigma
            odds=row["odds"],
            bankroll=bankroll_new,
            beta=0.10,  # The new conservative beta
            sigma_target=0.15,  # The new target
        )

        # Determine Outcome
        if row["profit_loss"] > 0:
            outcome = "WIN"
            pnl_old = row["profit_loss"]
            pnl_new = new_stake * (row["odds"] - 1)
        else:
            outcome = "LOSS"
            pnl_old = -row["original_stake"]
            pnl_new = -new_stake

        # Update Bankrolls
        bankroll_old += pnl_old
        bankroll_new += pnl_new

        results.append(
            {
                "Date": row["created_at"].strftime("%Y-%m-%d"),
                "Match": f"{row['away_team']}@{row['home_team']}",
                "Bet": bet_str,
                "Prob": f"{prob_win_new:.1%}",
                "Sigma": f"{sigma_p:.1f}",
                "Old Stake": f"€{row['original_stake']:.2f}",
                "New Stake": f"€{new_stake:.2f}",
                "Res": outcome,
                "Old BK": f"€{bankroll_old:.2f}",
                "New BK": f"€{bankroll_new:.2f}",
            }
        )

    # Output Results
    print(
        f"{'Date':<12} {'Match':<20} {'Bet':<15} {'Prob':<6} {'Sigma':<6} {'Old Stake':<10} {'New Stake':<10} {'Res':<4} {'Old BK':<10} {'New BK':<10}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['Date']:<12} {r['Match'][:18]:<20} {r['Bet']:<15} {r['Prob']:<6} {r['Sigma']:<6} {r['Old Stake']:<10} {r['New Stake']:<10} {r['Res']:<4} {r['Old BK']:<10} {r['New BK']:<10}"
        )

    print("\n=== SUMMARY ===")
    print(f"Starting Bankroll: €{start_bankroll:.2f}")
    print(
        f"Final Old Bankroll: €{bankroll_old:.2f} ({(bankroll_old - start_bankroll) / start_bankroll:.1%})"
    )
    print(
        f"Final New Bankroll: €{bankroll_new:.2f} ({(bankroll_new - start_bankroll) / start_bankroll:.1%})"
    )

    diff = bankroll_new - bankroll_old
    print(f"Difference: €{diff:+.2f}")


if __name__ == "__main__":
    run_backtest()
