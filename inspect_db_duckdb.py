import duckdb
import os

db_files = ["data/bets.duckdb", "data/nba_betting.duckdb", "data/betting_data.db"]

for db_file in db_files:
    if os.path.exists(db_file):
        print(f"\n🔍 Inspecting {db_file}...")
        try:
            con = duckdb.connect(db_file)
            tables = con.execute("SHOW TABLES").fetchall()
            print(f"   Tables: {[t[0] for t in tables]}")

            if any(t[0] == "bets" for t in tables):
                print(f"   ✅ Found 'bets' table!")
                schema = con.execute("DESCRIBE bets").fetchall()
                print(f"   Schema columns: {[col[0] for col in schema]}")

                rows = con.execute(
                    "SELECT bet_id, game_id, home_team, away_team, created_at FROM bets ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
                print(f"   Sample Data (last 5):")
                for row in rows:
                    print(row)
            else:
                print("   ❌ No 'bets' table found.")

            con.close()
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"⚠️ {db_file} not found.")
