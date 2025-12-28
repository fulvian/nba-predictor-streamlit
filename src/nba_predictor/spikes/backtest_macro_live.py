import duckdb
import polars as pl

DB_PATH = "data/nba_betting.duckdb"
FEATURE_PATH = "data/nba_spread_features_v1.parquet"


def run_backtest():
    print("Loading Data...")

    # 1. Load Features (Parquet)
    features_df = pl.read_parquet(FEATURE_PATH)

    # 2. Load Scores (DuckDB)
    # We join inside DuckDB for efficiency specific to our setup
    # But since features are in parquet, we can register them.

    conn = duckdb.connect(DB_PATH)
    conn.register("features", features_df)

    # Join Query
    # We want: Game Context (Altitude, Rest) + Quarter Scores
    print("Joining Features and Scores...")
    df = conn.execute("""
        SELECT 
            f.game_id, f.season, f.game_date,
            f.home_team_id, f.away_team_id,
            f.is_high_altitude_home,
            f.home_density_4d, f.away_rest,
            s.half_home, s.half_away,
            s.q3_home, s.q3_away,
            s.q4_home, s.q4_away,
            (s.q3_home + s.q4_home) as h2_home,
            (s.q3_away + s.q4_away) as h2_away
        FROM features f
        JOIN nba_quarter_scores s ON f.game_id = s.game_id
    """).pl()

    print(f"Total Games with Quarter Data: {len(df)}")

    # --- STRATEGY 1: DENVER LUNG (Altitude + Opponent B2B) ---
    # Trigger: High Altitude Home, Opponent Rest=1 (B2B)
    # Bet: Home Team to Win 2nd Half (Margin > 0)

    strat1 = df.filter(
        (pl.col("is_high_altitude_home") == 1) & (pl.col("away_rest") == 1)
    )

    print(f"\n--- Strategy 1: The Denver Lung (Altitude Home vs B2B Away) ---")
    print(f"Total Spots: {len(strat1)}")

    if len(strat1) > 0:
        # Outcome: 2nd Half Margin (Home - Away)
        strat1 = strat1.with_columns(
            (pl.col("h2_home") - pl.col("h2_away")).alias("h2_margin")
        )

        wins = strat1.filter(pl.col("h2_margin") > 0)
        win_rate = len(wins) / len(strat1)
        avg_margin = strat1["h2_margin"].mean()

        print(f"2nd Half Win Rate (Home): {win_rate:.2%}")
        print(f"Avg 2nd Half Margin: {avg_margin:.2f}")

        # Check by Season
        print("\nBy Season:")
        print(
            strat1.group_by("season")
            .agg(
                [
                    pl.len().alias("games"),
                    (pl.col("h2_margin") > 0).sum().alias("wins"),
                    pl.col("h2_margin").mean().alias("avg_margin"),
                ]
            )
            .sort("season")
        )

    # --- STRATEGY 2: TIRED LEGS REFINED ---
    # Trigger: Home Density >= 2 (3in4)
    # Filter: Home Team Leading or Close at start of Q4?
    # Logic: Tired teams collapse late if pushed, or struggle to hold leads.

    print(f"\n--- Strategy 2: Tired Legs Refined (Home 3-in-4) ---")

    # Calculate Margin at start of Q4
    # Q1+Q2+Q3
    strat2 = df.filter(pl.col("home_density_4d") >= 2).with_columns(
        [
            (
                (pl.col("half_home") + pl.col("q3_home"))
                - (pl.col("half_away") + pl.col("q3_away"))
            ).alias("margin_start_q4"),
            (pl.col("q4_home") - pl.col("q4_away")).alias("q4_margin"),
        ]
    )

    # Sub-Strategy A: Tired Home Team Leading by 5+ (The "Blow Lead" theory)
    blow_lead = strat2.filter(pl.col("margin_start_q4") > 5)
    if len(blow_lead) > 0:
        wins = blow_lead.filter(pl.col("q4_margin") < 0)  # Away wins Q4
        print(f"Scenario A (Tired Home Leads >5): {len(blow_lead)} games")
        print(f"  -> Q4 Fade Win Rate: {len(wins) / len(blow_lead):.2%}")
        print(f"  -> Q4 Avg Margin: {blow_lead['q4_margin'].mean():.2f}")

    # Sub-Strategy B: Close Game (-5 to +5) (The "Crunch Time Fatigue")
    crunch_time = strat2.filter(pl.col("margin_start_q4").abs() <= 5)
    if len(crunch_time) > 0:
        wins = crunch_time.filter(pl.col("q4_margin") < 0)
        print(f"Scenario B (Close Game +/-5): {len(crunch_time)} games")
        print(f"  -> Q4 Fade Win Rate: {len(wins) / len(crunch_time):.2%}")
        print(f"  -> Q4 Avg Margin: {crunch_time['q4_margin'].mean():.2f}")

    # Sub-Strategy C: Tired Home Team Trailing (The "Give Up")
    give_up = strat2.filter(pl.col("margin_start_q4") < -5)
    if len(give_up) > 0:
        wins = give_up.filter(pl.col("q4_margin") < 0)
        print(f"Scenario C (Tired Home Trails >5 - Give Up): {len(give_up)} games")
        print(f"  -> Q4 Fade Win Rate: {len(wins) / len(give_up):.2%}")
        print(f"  -> Q4 Avg Margin: {give_up['q4_margin'].mean():.2f}")


if __name__ == "__main__":
    run_backtest()
