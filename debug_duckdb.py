import duckdb
import os


def debug_duckdb_loading():
    print("Initializing DuckDB...")
    try:
        conn = duckdb.connect(":memory:")  # Use in-memory DB for test

        print("Attempting to load data/games/*.parquet...")
        # Mirroring the query from data_store.py
        query = """
        SELECT * REPLACE (CAST(season AS VARCHAR) AS season) 
        FROM read_parquet('data/games/*.parquet', union_by_name=true)
        """

        try:
            # Just try to fetch one row to trigger schema inference
            result = conn.execute(query + " LIMIT 1").fetchall()
            print("Successfully loaded data (LIMIT 1)!")
            print("Schema:")
            for desc in conn.description:
                print(desc)

            # Now try loading everything to find the bad row
            print("\nAttempting to load ALL data...")
            conn.execute(query).fetchall()
            print("Successfully loaded ALL data!")

        except Exception as e:
            print(f"\nCaught ERROR during query: {e}")

            # If main query fails, let's try to isolate the bad file
            print("\nAttempting to isolate the problematic file...")
            files = [f for f in os.listdir("data/games") if f.endswith(".parquet")]
            for f in sorted(files):
                file_path = os.path.join("data/games", f)
                try:
                    conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 1")
                except Exception as inner_e:
                    print(f"❌ FAILED on file {f}: {inner_e}")

    except Exception as e:
        print(f"DuckDB init failed: {e}")


if __name__ == "__main__":
    debug_duckdb_loading()
