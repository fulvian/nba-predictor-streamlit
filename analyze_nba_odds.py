import duckdb
import pandas as pd

# Define paths
DATA_DIR = "/Users/fulvioventura/nba-predictor-streamlit/data/nba_odds_csv"
GAMES_PATH = f"{DATA_DIR}/nba_games_all.csv"
MONEY_LINE_PATH = f"{DATA_DIR}/nba_betting_money_line.csv"
SPREAD_PATH = f"{DATA_DIR}/nba_betting_spread.csv"
TOTALS_PATH = f"{DATA_DIR}/nba_betting_totals.csv"


def analyze_dataset():
    con = duckdb.connect(database=":memory:")

    # Load games
    print("Loading games data...")
    con.execute(f"CREATE TABLE games AS SELECT * FROM read_csv_auto('{GAMES_PATH}')")

    # Date Range
    min_date, max_date = con.execute(
        "SELECT MIN(game_date), MAX(game_date) FROM games"
    ).fetchone()
    print(f"Dataset Date Range: {min_date} to {max_date}")

    # Games per Season
    print("\nGames per Season:")
    season_counts = con.execute("""
        SELECT season, COUNT(*) as count 
        FROM games 
        GROUP BY season 
        ORDER BY season DESC
    """).fetchdf()
    print(season_counts)

    # Load Odds
    print("\nLoading odds data...")
    con.execute(
        f"CREATE TABLE moneyline AS SELECT * FROM read_csv_auto('{MONEY_LINE_PATH}')"
    )
    con.execute(f"CREATE TABLE spread AS SELECT * FROM read_csv_auto('{SPREAD_PATH}')")

    # Bookmaker analysis for Moneyline
    print("\nTop Bookmakers by Game Count (Moneyline):")
    book_counts_ml = con.execute("""
        SELECT book_name, COUNT(DISTINCT game_id) as games_covered
        FROM moneyline
        GROUP BY book_name
        ORDER BY games_covered DESC
        LIMIT 10
    """).fetchdf()
    print(book_counts_ml)

    # Bookmaker analysis for Spread
    print("\nTop Bookmakers by Game Count (Spread):")
    book_counts_spread = con.execute("""
        SELECT book_name, COUNT(DISTINCT game_id) as games_covered
        FROM spread
        GROUP BY book_name
        ORDER BY games_covered DESC
        LIMIT 10
    """).fetchdf()
    print(book_counts_spread)

    # Coverage verification
    total_games = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    games_with_ml = con.execute(
        "SELECT COUNT(DISTINCT game_id) FROM moneyline"
    ).fetchone()[0]

    print(f"\nTotal Games in DB: {total_games}")
    print(
        f"Games with Moneyline Odds: {games_with_ml} ({games_with_ml / total_games:.2%})"
    )

    # Pinnacle specific check (as benchmark)
    pinnacle_games = con.execute(
        "SELECT COUNT(DISTINCT game_id) FROM moneyline WHERE book_name = 'Pinnacle Sports'"
    ).fetchone()[0]
    print(
        f"Games with Pinnacle Odds: {pinnacle_games} ({pinnacle_games / total_games:.2%})"
    )


if __name__ == "__main__":
    analyze_dataset()
