import duckdb

DATA_DIR = "/Users/fulvioventura/nba-predictor-streamlit/data/nba_odds_csv"
GAMES_PATH = f"{DATA_DIR}/nba_games_all.csv"
MONEY_LINE_PATH = f"{DATA_DIR}/nba_betting_money_line.csv"


def analyze_odds_dates():
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE games AS SELECT * FROM read_csv_auto('{GAMES_PATH}')")
    con.execute(
        f"CREATE TABLE moneyline AS SELECT * FROM read_csv_auto('{MONEY_LINE_PATH}')"
    )

    print("Joining Moneyline with Games to find date range of ODDS...")
    result = con.execute("""
        SELECT 
            MIN(g.game_date), 
            MAX(g.game_date),
            COUNT(DISTINCT g.season)
        FROM moneyline m
        JOIN games g ON m.game_id = g.game_id
    """).fetchone()

    print(f"Odds Date Range: {result[0]} to {result[1]}")
    print(f"Number of Seasons with Odds: {result[2]}")

    # Check seasons distribution for odds
    print("\nSeasons with Odds data:")
    seasons = con.execute("""
        SELECT g.season, COUNT(DISTINCT m.game_id)
        FROM moneyline m
        JOIN games g ON m.game_id = g.game_id
        GROUP BY g.season
        ORDER BY g.season DESC
    """).fetchdf()
    print(seasons)


if __name__ == "__main__":
    analyze_odds_dates()
