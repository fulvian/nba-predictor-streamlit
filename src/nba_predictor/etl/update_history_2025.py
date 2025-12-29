import polars as pl
from nba_api.stats.endpoints import leaguegamelog
from datetime import datetime
import time
import shutil
import os


def update_history_with_current_season():
    print("🚀 Starting History Update for 2025-26 Season...")

    # 1. Fetch Official Game Log
    print("📡 Fetching Game Log from NBA API...")
    # Season 2025-26. 'Regular Season'
    log = leaguegamelog.LeagueGameLog(
        season="2025-26", season_type_all_star="Regular Season"
    )
    data = log.get_data_frames()[0]

    if data.empty:
        print("⚠️ No games found for 2025-26 Season yet (or API failure).")
        return

    print(f"✅ Fetched {len(data)} rows (games x 2 teams). Processing...")

    # 2. Process Data to Match Schema
    # NBA API returns one row per team per game. We need 1 row per GAME (Home/Away).
    # Group by GAME_ID.

    # Standardize columns
    # GAME_ID, GAME_DATE, TEAM_ID, MATCHUP (to determine home/away)

    games_list = []

    grouped = data.groupby("GAME_ID")

    for game_id, group in grouped:
        if len(group) != 2:
            continue  # Skip incomplete or malformed games

        # Identify Home and Away
        # MATCHUP format: "DEN vs. LAL" (Home) or "LAL @ DEN" (Away)
        # Usually checking "@" is safest for Away team.

        row1 = group.iloc[0]
        row2 = group.iloc[1]

        if "@" in row1["MATCHUP"]:
            away_team = row1
            home_team = row2
        else:
            home_team = row1
            away_team = row2

        # Parse Date
        game_date = datetime.strptime(home_team["GAME_DATE"], "%Y-%m-%d").date()

        # Altitude Logic
        # Denver: 1610612743, Utah: 1610612762
        high_altitude_ids = [1610612743, 1610612762]
        is_high_alt = 1 if home_team["TEAM_ID"] in high_altitude_ids else 0

        game_record = {
            "game_id": str(game_id),
            "game_date": game_date,
            "season": "2025-26",
            "home_team_id": int(home_team["TEAM_ID"]),
            "away_team_id": int(away_team["TEAM_ID"]),
            "home_score": int(home_team["PTS"]),
            "away_score": int(away_team["PTS"]),
            "is_high_altitude_home": int(is_high_alt),
        }

        # Add dummy cols for schema compatibility (if Polars strict)
        # We only populate what context_loader needs.
        games_list.append(game_record)

    df_new = pl.DataFrame(games_list)
    print(f"✅ Processed {len(df_new)} unique games.")

    # 3. Load Existing History & Append
    parquet_path = "data/nba_spread_features_v1.parquet"
    if not os.path.exists(parquet_path):
        print("❌ Historical Parquet not found!")
        return

    print(f"📂 Loading existing history from {parquet_path}...")
    df_old = pl.read_parquet(parquet_path)

    # Filter out games that might already be in history (avoid dups)
    # Check max date in history
    max_date = df_old["game_date"].max()
    print(f"📅 Current History ends on: {max_date}")

    df_new_filtered = df_new.filter(pl.col("game_date") > max_date)

    if len(df_new_filtered) == 0:
        print("🎉 History is already up to date! No new games to append.")
        return

    print(f"🔄 Appending {len(df_new_filtered)} new games...")

    # Align Schema: Select only columns that exist in both + fill missing in new
    # Or simpler: Just relax schema for the new rows?
    # Polars requires schema match for concat.
    # Let's add missing columns to df_new_filtered with Nulls

    for col in df_old.columns:
        target_dtype = df_old.schema[col]

        if col not in df_new_filtered.columns:
            # Add null column with correct type
            df_new_filtered = df_new_filtered.with_columns(
                pl.lit(None).cast(target_dtype).alias(col)
            )
        else:
            # Cast existing column to match target type (e.g. Int64 -> Int32)
            df_new_filtered = df_new_filtered.with_columns(
                pl.col(col).cast(target_dtype)
            )

    # Reorder columns to match df_old
    df_new_filtered = df_new_filtered.select(df_old.columns)

    # Concat
    df_updated = pl.concat([df_old, df_new_filtered])

    # 4. Save
    backup_path = parquet_path + ".bak"
    shutil.copy(parquet_path, backup_path)
    print(f"💾 Backup created at {backup_path}")

    df_updated.write_parquet(parquet_path)
    print(
        f"✅ Database updated! New Total Games: {len(df_updated)} (Ends: {df_updated['game_date'].max()})"
    )


if __name__ == "__main__":
    update_history_with_current_season()
