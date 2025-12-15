import duckdb
import pandas as pd


def inspect_transactions():
    db_path = "data/nba_bankroll_v3.duckdb"
    try:
        conn = duckdb.connect(db_path)
        # Check tables first
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables: {tables}")

        # Dump transactions
        if ("transactions",) in tables or ("transactions",) in tables:  # duckdb tuples
            query = "SELECT * FROM transactions ORDER BY created_at DESC LIMIT 20"
            df = conn.execute(query).fetchdf()
            print("\nRecent Transactions:")
            print(df)

            # Check for multiple 'INITIAL_DEPOSIT' or similar types
            query_types = "SELECT type, COUNT(*) as count, SUM(amount) as total_amount FROM transactions GROUP BY type"
            df_types = conn.execute(query_types).fetchdf()
            print("\nTransaction Types Summary:")
            print(df_types)

        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_transactions()
