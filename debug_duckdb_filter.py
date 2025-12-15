import duckdb


def debug_filter():
    print("Initializing DuckDB...")
    try:
        conn = duckdb.connect(":memory:")

        # Query with filter, similar to get_games_data(date_range)
        # Targeting the TBD file date
        query = """
        SELECT * REPLACE (CAST(season AS VARCHAR) AS season) 
        FROM read_parquet('data/games/*.parquet', union_by_name=true)
        WHERE game_date = '2025-12-09'
        """

        print("Executing filtered query...")
        result = conn.execute(query).fetchall()
        print("Successfully loaded filtered data!")
        for row in result:
            print(row)

    except Exception as e:
        print(f"\n❌ ERROR during filtered query: {e}")


if __name__ == "__main__":
    debug_filter()
