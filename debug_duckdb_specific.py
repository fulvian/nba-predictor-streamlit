import duckdb
import os


def check_specific_file():
    target = "data/games/games_2025-12-08.parquet"
    print(f"Checking {target}...")
    try:
        conn = duckdb.connect(":memory:")

        # Check explicit schema of this file
        print(f"Schema for {target}:")
        result = conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{target}')"
        ).fetchall()
        for row in result:
            print(row)

        print("\nContent of first 5 rows:")
        result = conn.execute(
            f"SELECT * FROM read_parquet('{target}') LIMIT 5"
        ).fetchall()
        for row in result:
            print(row)

    except Exception as e:
        print(f"Error checking file: {e}")


if __name__ == "__main__":
    check_specific_file()
