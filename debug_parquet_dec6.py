import duckdb


def check_parquet_dec6():
    target = "data/games/games_2025-12-06.parquet"
    print(f"Checking {target} for team names...")
    try:
        conn = duckdb.connect(":memory:")
        # Check game_time column content
        result = conn.execute(
            f"SELECT game_id, home_team, away_team FROM read_parquet('{target}')"
        ).fetchall()
        for row in result:
            print(row)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_parquet_dec6()
