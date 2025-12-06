import duckdb

DB_PATH = "data/nba_betting.duckdb"
BETS_TO_FIX = [
    "7e863776-759f-4d44-b1b3-09ae7a50d328",
    "3a61910e-2361-4ba1-bf03-b6bbfcb8177c",
    "369708f0-e8e2-4024-a188-4a4e0239943d",
]


def fix_status():
    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)

    # Check current status
    print("Current Status:")
    res = conn.execute(
        f"SELECT bet_id, status, result FROM bets WHERE bet_id IN {tuple(BETS_TO_FIX)}"
    ).fetchall()
    print(res)

    print("Updating...")
    conn.execute(
        f"UPDATE bets SET status='PENDING', result=NULL, profit_loss=NULL, settled_at=NULL WHERE bet_id IN {tuple(BETS_TO_FIX)}"
    )

    # Check new status
    print("New Status:")
    res_new = conn.execute(
        f"SELECT bet_id, status, result FROM bets WHERE bet_id IN {tuple(BETS_TO_FIX)}"
    ).fetchall()
    print(res_new)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    fix_status()
