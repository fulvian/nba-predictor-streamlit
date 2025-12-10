import sys

sys.path.append("src")
import duckdb
import time
from datetime import date, datetime
from nba_predictor.api.data_provider import NBADataProvider


def debug_mismatch():
    # 1. Fetch Pending Bets directly
    print("\n--- 1. FETCHING PENDING BETS FROM DB ---")
    conn = duckdb.connect("data/nba_betting.duckdb")
    pending_bets = []
    try:
        cols = conn.execute("DESCRIBE bets").fetchall()
        col_names = [c[0] for c in cols]
        rows = conn.execute("SELECT * FROM bets WHERE status='PENDING'").fetchall()
        for row in rows:
            pending_bets.append(dict(zip(col_names, row)))
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        conn.close()

    print(f"Found {len(pending_bets)} pending bets.")
    needed_ids = set()
    for bet in pending_bets:
        gid = bet.get("game_id")
        print(f"  - Bet ID: {bet.get('bet_id')} | Game ID: {gid}")
        if gid:
            needed_ids.add(gid)

    # 2. Fetch Completed Games via NBADataProvider
    target_date = "2025-12-09"
    print(f"\n--- 2. FETCHING COMPLETED GAMES FOR {target_date} ---")

    provider = NBADataProvider()
    start_time = time.time()

    print(f"Calling _get_nba_completed_games for {target_date}...")
    # Call the internal method used by auto_update
    completed_games = provider._get_nba_completed_games(specific_date=target_date)

    duration = time.time() - start_time
    print(f"Fetch took {duration:.2f} seconds")

    print(f"\n--- FOUND {len(completed_games)} COMPLETED GAMES ---")
    for g in completed_games:
        print(
            f"  ID: {g.get('game_id')} | {g.get('home_team')} vs {g.get('away_team')} | {g.get('score')} | {g.get('status')} | Src: {g.get('source')}"
        )

    # 3. Check for matches
    print("\n--- 3. MATCHING RESULTS ---")
    if not pending_bets:
        print("No pending bets in DB to match against.")

    # If no pending bets found in DB, use hardcoded IDs for test if 2025-12-09
    if not needed_ids and target_date == "2025-12-09":
        print("Using hardcoded IDs for testing matching logic...")
        needed_ids = {"0022501201", "0022501202"}

    for needed_id in needed_ids:
        found = False
        for g in completed_games:
            if g.get("game_id") == needed_id:
                found = True
                print(f"  ✅ Bet Game ID {needed_id} MATCHED in API results")
                break
        if not found:
            print(f"  ❌ Bet Game ID {needed_id} NOT FOUND in API results")


if __name__ == "__main__":
    debug_mismatch()
