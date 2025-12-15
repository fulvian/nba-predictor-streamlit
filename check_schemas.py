import duckdb
import os
import glob


def check_schemas():
    files = sorted(glob.glob("data/games/*.parquet"))
    conn = duckdb.connect(":memory:")

    print(f"Checking {len(files)} files...")

    schemas = {}

    for f in files:
        try:
            res = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").fetchall()
            # extract (col_name, col_type)
            file_schema = {row[0]: row[1] for row in res}

            # Key signature: home_score type, away_score type
            sig = (
                file_schema.get("home_score", "MISSING"),
                file_schema.get("away_score", "MISSING"),
                file_schema.get("season", "MISSING"),
            )

            if sig not in schemas:
                schemas[sig] = []
            schemas[sig].append(f)

        except Exception as e:
            print(f"Error reading {f}: {e}")

    print("\n--- Summary of Schemas ---")
    for sig, flist in schemas.items():
        print(
            f"\nSignature (Home: {sig[0]}, Away: {sig[1]}, Season: {sig[2]}): {len(flist)} files"
        )
        # Print first few files of this type
        for f in flist[:3]:
            print(f"  - {f}")


if __name__ == "__main__":
    check_schemas()
