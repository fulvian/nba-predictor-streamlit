import pandas as pd
import requests
import time
import os
from pathlib import Path

# Setup directories
DATA_DIR = Path("data/raw/lean_odds")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Seasons to fetch (Year represents the season start year in some contexts,
# or end year. For SportsOddsHistory, usually "2023" = 2023-2024 season)
# Checking URL pattern: https://www.sportsoddshistory.com/nba-odds/
# It typically uses format like "?y=2023&sa=nba&a=total&p=full" or similar.
# Let's target the exact confirmed structure.

# Correct URL Pattern: https://www.sportsoddshistory.com/nba-game-season/?y=2023
# The 'y' parameter takes the START year of the season.
SEASONS = [2023, 2024, 2025]


def fetch_season_odds(year):
    """
    Downloads the full season game log with closing odds.
    URL Pattern: https://www.sportsoddshistory.com/nba-game-season/?y=2023
    """
    season_str = f"{year}-{year + 1}"
    url = f"https://www.sportsoddshistory.com/nba-game-season/?y={year}"
    print(f"📥 Downloading data for season {season_str} from {url}...")

    try:
        # Use pandas read_html directly which handles table parsing
        # Add headers to avoid basic bot blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        dfs = pd.read_html(response.text)

        if not dfs:
            print(f"⚠️ No tables found for {season_str}")
            return None

        # The main game table is usually the first big table
        # It typically contains columns: Date, Day, Visitor, Home, Score, total, etc.
        main_df = None
        for df in dfs:
            if len(df) > 100:  # NBA regular season has >100 games
                main_df = df
                break

        if main_df is None:
            # Fallback to largest table
            main_df = max(dfs, key=len)

        save_path = DATA_DIR / f"nba_odds_{season_str}.csv"
        main_df.to_csv(save_path, index=False)
        print(f"✅ Saved {len(main_df)} games to {save_path}")
        return main_df

    except Exception as e:
        print(f"❌ Error fetching {season_str}: {e}")
        return None


if __name__ == "__main__":
    print("🏀 Starting Lean Odds Fetcher...")
    for season in SEASONS:
        fetch_season_odds(season)
        time.sleep(2)  # Be polite
    print("\n🏁 All downloads completed.")
