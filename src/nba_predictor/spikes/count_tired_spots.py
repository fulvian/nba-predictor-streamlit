import duckdb
import polars as pl

DB_PATH = "data/nba_betting.duckdb"
FEATURE_PATH = "data/nba_spread_features_v1.parquet"


def count_scenarios():
    print("Loading features...")
    df = pl.read_parquet(FEATURE_PATH)

    # Check for "Tired Legs": Home Team playing 3rd game in 4 days
    # In our feature set, we have 'home_density_4d'.
    # If density_4d >= 3? Assuming the metric counts *previous* games.
    # Actually, let's check the distribution.

    # Correction: "3 games in 4 nights" means playing the 3rd game today.
    # So previous games in last 4 days should be 2.
    # "4 games in 5 nights" (the classic death spot) means 3 previous in 5 days? Or 4 in 5 days window including today?
    # Let's count filter >= 2 for density_4d (3rd game in window).

    tired_spots = df.filter(pl.col("home_density_4d") >= 2)

    # Altitude Spots: High Altitude Home + Opponent is on Back-to-Back (Rest=1 in this dataset)
    # Check distribution of rest_days
    print("\nRest Days Distribution (Away):")
    print(df.group_by("away_rest").len().sort("away_rest"))

    altitude_killer = df.filter(
        (pl.col("is_high_altitude_home") == 1)
        & (pl.col("away_rest") == 1)  # 1 Day Diff = Played Yesterday (B2B)
    )

    print(f"Total Games: {len(df)}")
    print(f"Tired Home Spots (Density >= 2 -> 3in4): {len(tired_spots)}")
    print(f"Altitude Killer Spots (Den/Uta Home vs B2B): {len(altitude_killer)}")

    print("\nTired Spots by Season:")
    print(tired_spots.group_by("season").len().sort("season"))

    # Print sample Game IDs to verifying fetching
    print("\nSample Game IDs (Tired):")
    print(
        tired_spots.head(5).select(
            ["game_id", "game_date", "home_team_id", "home_density_4d"]
        )
    )


if __name__ == "__main__":
    count_scenarios()
