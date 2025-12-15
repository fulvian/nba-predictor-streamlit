import duckdb
import pandas as pd
from pathlib import Path

# Load one parquet file directly
files = list(
    Path("/Users/fulvioventura/nba-predictor-streamlit/data/games").glob("*.parquet")
)
if not files:
    print("No parquet files found")
    exit()

f = files[0]
print(f"Loading {f}")
df = pd.read_parquet(f)
print("Columns:")
for c in df.columns:
    print(c)

print("\nSample Row:")
print(df.iloc[0].to_dict())
