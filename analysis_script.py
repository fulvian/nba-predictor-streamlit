import duckdb
import json
import pandas as pd
import numpy as np

# Connect
conn = duckdb.connect("data/nba_betting.duckdb", read_only=True)

# Query
query = """
SELECT 
    created_at, home_team, away_team, 
    amount as stake, odds, result, profit_loss, 
    home_score, away_score, 
    prediction, bet_type
FROM bets 
WHERE status IN ('SETTLED', 'WON', 'LOST')
ORDER BY created_at ASC
"""
df = conn.execute(query).df()

print(f"Total Bets Fetched: {len(df)}")

if len(df) == 0:
    print("No settled bets found.")
    exit()


# Parse Prediction JSON
def parse_pred(x):
    try:
        if isinstance(x, str):
            return json.loads(x)
        return x
    except:
        return {}


df["pred_data"] = df["prediction"].apply(parse_pred)
df["predicted_total"] = df["pred_data"].apply(
    lambda x: float(x.get("predicted_total") or 0)
)
df["model_confidence"] = df["pred_data"].apply(
    lambda x: float(x.get("model_confidence") or 0)
)
df["actual_total"] = df["home_score"] + df["away_score"]
df["error"] = (
    df["predicted_total"] - df["actual_total"]
)  # Pos = Overestimate, Neg = Underestimate
df["abs_error"] = df["error"].abs()

# --- 1. General Stats ---
total_bets = len(df)
wins = len(df[df["result"] == "WON"])
losses = len(df[df["result"] == "LOST"])
win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
total_pnl = df["profit_loss"].sum()
total_stake = df["stake"].sum()
roi = (total_pnl / total_stake) * 100 if total_stake > 0 else 0
avg_stake = df["stake"].mean()
avg_odds = df["odds"].mean()

print("\n--- GENERAL PERFORMANCE ---")
print(f"Total Bets: {total_bets}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: {total_pnl:.2f} units")
print(f"Total Stake: {total_stake:.2f} units")
print(f"ROI: {roi:.2f}%")
print(f"Avg Stake: {avg_stake:.2f}")
print(f"Avg Odds: {avg_odds:.2f}")

# --- 2. Score Accuracy Analysis ---
mae = df["abs_error"].mean()
bias = df["error"].mean()
mse = (df["error"] ** 2).mean()

print("\n--- SCORE PREDICTION ACCURACY ---")
print(f"MAE (Mean Absolute Error): {mae:.2f} pts")
print(f"Bias (Mean Error): {bias:.2f} pts (Positive = Overestimation)")
print(f"MSE: {mse:.2f}")

# Distribution of Error
print("\n--- ERROR DISTRIBUTION QUANTILES ---")
print(df["error"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# --- 3. Cluster Analysis (Bias by Range) ---
# Create buckets for predicted totals
df["total_bucket"] = pd.cut(
    df["predicted_total"],
    bins=[0, 210, 220, 230, 240, 300],
    labels=["<210", "210-220", "220-230", "230-240", "240+"],
)
print("\n--- BIAS BY SCORE RANGE ---")
print(df.groupby("total_bucket")["error"].agg(["mean", "count", "std"]))

# --- 4. Team Analysis (Worst Offenders) ---
# Flatten by team involved
team_stats = []
for idx, row in df.iterrows():
    team_stats.append(
        {
            "team": row["home_team"],
            "error": row["error"],
            "pnl": row["profit_loss"],
            "result": row["result"],
        }
    )
    team_stats.append(
        {
            "team": row["away_team"],
            "error": row["error"],
            "pnl": row["profit_loss"],
            "result": row["result"],
        }
    )

team_df = pd.DataFrame(team_stats)
print("\n--- TOP 5 TEAMS UNDERESTIMATED (Bias is Negative) ---")
print(team_df.groupby("team")["error"].mean().sort_values().head(5))

print("\n--- TOP 5 TEAMS OVERESTIMATED (Bias is Positive) ---")
print(team_df.groupby("team")["error"].mean().sort_values(ascending=False).head(5))

print("\n--- TOP 5 LOSING TEAMS (PnL) ---")
print(team_df.groupby("team")["pnl"].sum().sort_values().head(5))

# --- 5. Staking Efficiency ---
# Compare Win Rate of High vs Low Stakes
df["stake_bucket"] = pd.qcut(df["stake"], q=3, labels=["Low", "Med", "High"])
print("\n--- WIN RATE BY STAKE SIZE ---")
print(
    df.groupby("stake_bucket")["result"]
    .value_counts(normalize=True)
    .unstack()
    .get("WON", 0)
    * 100
)

print("\n--- PNL BY STAKE SIZE ---")
print(df.groupby("stake_bucket")["profit_loss"].sum())

# --- 6. Confidence Analysis ---
if df["model_confidence"].sum() > 0:
    df["conf_bucket"] = pd.cut(
        df["model_confidence"], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    print("\n--- WIN RATE BY MODEL CONFIDENCE ---")
    print(
        df.groupby("conf_bucket")["result"]
        .value_counts(normalize=True)
        .unstack()
        .get("WON", 0)
        * 100
    )
