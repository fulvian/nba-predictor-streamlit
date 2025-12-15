import duckdb


def check_12_09():
    target = "data/games/games_2025-12-09.parquet"
    print(f"Checking {target} for sanitized time...")
    try:
        conn = duckdb.connect(":memory:")
        # Check game_time column content
        result = conn.execute(
            f"SELECT game_time FROM read_parquet('{target}') LIMIT 5"
        ).fetchall()
        for row in result:
            print(f"Time: '{row[0]}'")  # Should be 'None' or valid string, NOT 'TBD'

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_12_09()
