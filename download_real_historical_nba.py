#!/usr/bin/env python3
"""
🏀 NBA REAL Historical Data Downloader (Alternative Approach)
Download REAL NBA game results using nba_api for completed seasons with real scores.
"""

import polars as pl
import time
from datetime import datetime, date
from pathlib import Path
import logging
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealHistoricalNBADownloader:
    """Download REAL NBA historical game results using nba_api (reliable source)."""

    def __init__(self):
        self.rate_limit = 0.5  # 500ms between requests
        self.last_request_time = 0

    def _make_request(self, season, season_type):
        """Make rate-limited request to nba_api."""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        try:
            logger.info(f"📡 Requesting {season_type} games for season {season}...")

            # Use LeagueGameLog endpoint
            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation='T',
                timeout=60
            )

            self.last_request_time = time.time()
            return game_log.get_data_frames()[0]

        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def download_2023_24_season_results(self):
        """Download complete 2023-24 season game results with real scores."""
        logger.info("🏀 Starting REAL NBA 2023-24 season results download")

        all_games = []

        # Download Regular Season games (2023-24)
        logger.info("📅 Downloading 2023-24 Regular Season results...")
        regular_games = self._download_and_process_season("2023", "Regular Season")
        all_games.extend(regular_games)
        logger.info(f"✅ Downloaded {len(regular_games):,} regular season games")

        # Download Playoff games (2023-24)
        logger.info("🏆 Downloading 2023-24 Playoff results...")
        playoff_games = self._download_and_process_season("2023", "Playoffs")
        all_games.extend(playoff_games)
        logger.info(f"✅ Downloaded {len(playoff_games):,} playoff games")

        # Convert to DataFrame
        if all_games:
            df = pl.DataFrame(all_games)
            logger.info(f"📊 Total games downloaded: {len(df):,}")

            # Validate data
            if 'PTS' in df.columns:
                avg_score = df['PTS'].mean()
                logger.info(f"🏀 Average score: {avg_score:.1f}")

                if avg_score > 80:
                    logger.info("✅ REAL scores detected - NBA average range!")
                else:
                    logger.warning("⚠️ Scores may still be incomplete")

            # Save to file
            output_dir = Path("data/persistent/game_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "real_nba_results_2023-24_complete.parquet"
            df.write_parquet(output_file)
            logger.info(f"💾 Saved to {output_file}")

            return df
        else:
            logger.error("❌ No games downloaded")
            return None

    def _download_and_process_season(self, season, season_type):
        """Download and process games for a specific season and type."""
        # Get raw data from nba_api
        raw_df = self._make_request(season, season_type)

        if raw_df is None or len(raw_df) == 0:
            logger.error(f"❌ No data returned for {season_type} {season}")
            return []

        logger.info(f"📊 Raw data received: {len(raw_df)} games")

        # Process to our standard format
        processed_games = []

        for _, row in raw_df.iterrows():
            # Extract game information
            game_date = self._parse_nba_date(row['GAME_DATE'])

            # Determine home/away teams from matchup
            matchup = row['MATCHUP']
            if '@' in matchup:
                away_team = row['TEAM_NAME']
                home_team = matchup.split('@ ')[1].strip()
            elif 'vs.' in matchup:
                home_team = row['TEAM_NAME']
                away_team = matchup.split('vs. ')[1].strip()
            else:
                continue  # Skip if we can't determine home/away

            # Skip duplicate games (we'll get both home and away team records)
            game_id = row['GAME_ID']
            if any(g['game_id'] == game_id for g in processed_games):
                continue

            # Create standardized game record
            game = {
                'game_id': str(game_id),
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'season': f"{season}-{int(season)+1}",  # Convert 2023 to 2023-24
                'season_type': season_type,
                'home_score': self._find_team_score(raw_df, game_id, home_team),
                'away_score': self._find_team_score(raw_df, game_id, away_team),
                'result': row['WL'] if row['TEAM_NAME'] == home_team else ('W' if row['WL'] == 'L' else 'L'),
                'minutes': int(row['MIN']) if pd.notna(row['MIN']) else 0,
                'field_goals_made': int(row['FGM']) if pd.notna(row['FGM']) else 0,
                'field_goals_attempted': int(row['FGA']) if pd.notna(row['FGA']) else 0,
                'field_goal_percentage': float(row['FG_PCT']) if pd.notna(row['FG_PCT']) else 0.0,
                'three_points_made': int(row['FG3M']) if pd.notna(row['FG3M']) else 0,
                'three_points_attempted': int(row['FG3A']) if pd.notna(row['FG3A']) else 0,
                'three_point_percentage': float(row['FG3_PCT']) if pd.notna(row['FG3_PCT']) else 0.0,
                'free_throws_made': int(row['FTM']) if pd.notna(row['FTM']) else 0,
                'free_throws_attempted': int(row['FTA']) if pd.notna(row['FTA']) else 0,
                'free_throw_percentage': float(row['FT_PCT']) if pd.notna(row['FT_PCT']) else 0.0,
                'offensive_rebounds': int(row['OREB']) if pd.notna(row['OREB']) else 0,
                'defensive_rebounds': int(row['DREB']) if pd.notna(row['DREB']) else 0,
                'total_rebounds': int(row['REB']) if pd.notna(row['REB']) else 0,
                'assists': int(row['AST']) if pd.notna(row['AST']) else 0,
                'steals': int(row['STL']) if pd.notna(row['STL']) else 0,
                'blocks': int(row['BLK']) if pd.notna(row['BLK']) else 0,
                'turnovers': int(row['TOV']) if pd.notna(row['TOV']) else 0,
                'personal_fouls': int(row['PF']) if pd.notna(row['PF']) else 0,
                'plus_minus': int(row['PLUS_MINUS']) if pd.notna(row['PLUS_MINUS']) else 0,
                'source': 'NBA_Official_API_nba_api',
                'created_at': datetime.now()
            }
            processed_games.append(game)

        logger.info(f"✅ Processed {len(processed_games)} {season_type} games")
        return processed_games

    def _find_team_score(self, df, game_id, team_name):
        """Find the score for a specific team in a game."""
        team_row = df[(df['GAME_ID'] == game_id) & (df['TEAM_NAME'] == team_name)]
        if len(team_row) > 0:
            return int(team_row.iloc[0]['PTS'])
        return 0

    def _parse_nba_date(self, date_str):
        """Parse NBA date format from nba_api."""
        try:
            # nba_api date format is typically "MM/DD/YYYY"
            return datetime.strptime(date_str, "%m/%d/%Y").date()
        except:
            try:
                # Try other formats
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return date(2023, 1, 1)

def main():
    """Main function to download real NBA historical data."""
    print("🏀 NBA REAL HISTORICAL DATA DOWNLOADER (Alternative Approach)")
    print("=" * 80)

    downloader = RealHistoricalNBADownloader()

    # Download 2023-24 season data (completed season with real scores)
    df = downloader.download_2023_24_season_results()

    if df is not None and len(df) > 0:
        print("\n" + "=" * 80)
        print("🎯 DOWNLOAD COMPLETE!")
        print(f"✅ Total games: {len(df):,}")
        print(f"✅ Regular Season: {len(df.filter(df['season_type'] == 'Regular Season')):,}")
        print(f"✅ Playoffs: {len(df.filter(df['season_type'] == 'Playoffs')):,}")

        # Show sample data
        if len(df) > 0:
            sample = df.select(['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']).row(0)
            print(f"\n📝 Sample REAL game result:")
            print(f"   {sample[1]} vs {sample[2]} - {sample[3]}-{sample[4]} ({sample[5]})")
            print(f"   Date: {sample[0]}")

            # Show score statistics
            avg_home = df['home_score'].mean()
            avg_away = df['away_score'].mean()
            max_score = df['home_score'].max()
            print(f"\n📊 Score Statistics:")
            print(f"   Average Home Score: {avg_home:.1f}")
            print(f"   Average Away Score: {avg_away:.1f}")
            print(f"   Highest Score: {max_score}")

            if avg_home > 80:
                print("✅ CONFIRMED: Real NBA scores detected!")
            else:
                print("⚠️  Scores may be incomplete - check API response")

        print("\n🚀 Real NBA historical data download completed!")
        print("💾 Data saved to: data/persistent/game_results/real_nba_results_2023-24_complete.parquet")
    else:
        print("❌ Download failed - no data retrieved")

if __name__ == "__main__":
    # Import pandas for nba_api compatibility
    import pandas as pd
    main()