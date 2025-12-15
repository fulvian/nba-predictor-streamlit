import duckdb
import json

conn = duckdb.connect("data/nba_betting.duckdb", read_only=True)
# Get all columns
cols = conn.execute("DESCRIBE bets").fetchall()
print("Columns:", [c[0] for c in cols])

# Get one full row to see JSON structure and other fields
row = conn.execute("SELECT * FROM bets LIMIT 1").fetchone()
columns = [c[0] for c in cols]
data = dict(zip(columns, row))

print("\nSample Row Keys:", data.keys())
print(
    "\nPrediction JSON keys:",
    json.loads(data["prediction"]).keys()
    if isinstance(data["prediction"], str) and data["prediction"]
    else "No prediction data",
)
print("\nStake/Odds info check:")
try:
    print(f"Stake column: {data.get('stake')}")
    print(f"Odds column: {data.get('odds')}")
except:
    print("Direct columns not found")

# Check inside prediction JSON for stake/odds if distinct columns don't exist
if isinstance(data["prediction"], str):
    pred = json.loads(data["prediction"])
    print(
        "Inside JSON - Stake:",
        pred.get("stake"),
        "Odds:",
        pred.get("odds"),
        "Moneyline:",
        pred.get("moneyline"),
    )
