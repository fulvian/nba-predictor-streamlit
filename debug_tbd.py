import duckdb


def check_tbd():
    target = "data/games/games_2025-12-09.parquet"
    print(f"Checking {target} for TBD or Schema issues...")
    try:
        conn = duckdb.connect(":memory:")
        print(f"Schema for {target}:")
        result = conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{target}')"
        ).fetchall()
        for row in result:
            print(row)

        print("\nFirst 5 rows:")
        result = conn.execute(
            f"SELECT * FROM read_parquet('{target}') LIMIT 5"
        ).fetchall()
        for row in result:
            print(row)

    except Exception as e:
        print(f"Error checking file: {e}")


if __name__ == "__main__":
    check_tbd()
