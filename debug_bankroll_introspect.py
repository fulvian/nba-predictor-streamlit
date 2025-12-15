import duckdb
import pandas as pd


def inspect_transactions():
    db_path = "data/nba_bankroll_v3.duckdb"
    try:
        conn = duckdb.connect(db_path)

        # Get column names
        df_cols = conn.execute("PRAGMA table_info(transactions)").fetchdf()
        print("Column Names:")
        print(df_cols["name"].tolist())

        # Determine valid type column
        cols = df_cols["name"].tolist()
        type_col = (
            "type"
            if "type" in cols
            else "transaction_type"
            if "transaction_type" in cols
            else "action"
        )

        print(f"\nUsing '{type_col}' as transaction type column.")

        # Dump summary
        query_types = f"SELECT {type_col}, COUNT(*) as count, SUM(amount) as total_amount FROM transactions GROUP BY {type_col}"
        df_types = conn.execute(query_types).fetchdf()
        print("\nTransaction Types Summary:")
        print(df_types)

        # Dump recent rows with detail
        query_recent = f"SELECT transaction_id, {type_col}, amount, created_at FROM transactions ORDER BY created_at DESC LIMIT 20"
        df_recent = conn.execute(query_recent).fetchdf()
        print("\nRecent Transactions:")
        print(df_recent)

        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_transactions()
