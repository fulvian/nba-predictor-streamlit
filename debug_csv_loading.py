import pandas as pd
import pathlib

file_path = pathlib.Path("data/nba_data_with_mu_sigma_for_ml.csv")
print(f"Reading {file_path}...")
df = pd.read_csv(file_path)

col = "HOME_eFG_PCT"
if col in df.columns:
    print(f"{col} dtype: {df[col].dtype}")
    print(f"First 5 values:\n{df[col].head()}")
    print(f"Number of NaNs raw: {df[col].isna().sum()}")

    # Attempt conversion
    converted = pd.to_numeric(df[col], errors="coerce")
    print(f"NaNs after coercion: {converted.isna().sum()}")

    # Show values that became NaN
    failures = df[col][converted.isna() & df[col].notna()]
    if not failures.empty:
        print(f"Sample of failed conversions:\n{failures.head()}")
        print(f"Type of failed values: {failures.apply(type).head()}")
else:
    print(f"{col} NOT FOUND in columns: {df.columns.tolist()}")
