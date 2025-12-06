import duckdb

try:
    c = duckdb.connect("data/nba_betting.duckdb")
    print("Finding Unknown Team bets...")
    row = c.execute(
        "SELECT bet_id, game_id, prediction FROM bets WHERE home_team = 'Unknown Team' LIMIT 1"
    ).fetchone()
    if row:
        print(f"Bet ID: {row[0]}")
        print(f"Game ID: {row[1]}")
        # Print first 2000 chars of prediction to see structure
        print("Prediction Data:", str(row[2])[:2000])
    else:
        print("No bets with Unknown Team found.")

except Exception as e:
    print(e)
