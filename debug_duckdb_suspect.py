import duckdb


def check_suspect():
    target = "data/games/games_2025_10_29.parquet"
    print(f"Checking {target}...")
    try:
        conn = duckdb.connect(":memory:")
        print(f"Schema for {target}:")
        result = conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{target}')"
        ).fetchall()
        for row in result:
            print(row)

        print("\nFirst row:")
        result = conn.execute(
            f"SELECT * FROM read_parquet('{target}') LIMIT 1"
        ).fetchall()
        for row in result:
            print(row)

    except Exception as e:
        print(f"Error checking file: {e}")


if __name__ == "__main__":
    check_suspect()
