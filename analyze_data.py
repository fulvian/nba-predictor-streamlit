import pandas as pd
import numpy as np
from pathlib import Path

# Path to data
data_path = Path("data/nba_data_with_mu_sigma_for_ml.csv")

if not data_path.exists():
    print(f"File not found: {data_path}")
    exit(1)

df = pd.read_csv(data_path)

print(f"Total games: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

if "TOTAL_SCORE" in df.columns:
    print("\n--- Total Score Stats ---")
    print(df["TOTAL_SCORE"].describe())

    # Check for zero scores (future games)
    zeros = df[df["TOTAL_SCORE"] <= 0]
    print(f"\nGames with score <= 0: {len(zeros)}")

    # Check recent games if date column exists
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        print(f"\nDate columns found: {date_cols}")
        # Try to parse the first one
        try:
            df["parsed_date"] = pd.to_datetime(df[date_cols[0]])
            print(f"Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")

            # Stats for last season (assuming > Oct 2023)
            recent = df[df["parsed_date"] > "2023-10-01"]
            print(f"\n--- Recent Games (since Oct 2023) ---")
            print(f"Count: {len(recent)}")
            if not recent.empty:
                print(recent["TOTAL_SCORE"].describe())
        except Exception as e:
            print(f"Could not parse date: {e}")

else:
    print("TOTAL_SCORE column not found!")

# Check for team stats columns
cols_to_check = ["HOME_eFG_PCT", "HOME_ORtg", "HOME_POSSESSIONS"]
for c in cols_to_check:
    if c in df.columns:
        print(f"\n--- {c} Stats ---")
        print(df[c].describe())
