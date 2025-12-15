import duckdb


def dump_12_08():
    target = "data/games/games_2025-12-08.parquet"
    print(f"Dumping {target}...")
    try:
        conn = duckdb.connect(":memory:")
        result = conn.execute(
            f"SELECT home_team, away_team, game_time, status FROM read_parquet('{target}')"
        ).fetchall()
        for row in result:
            print(row)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    dump_12_08()
