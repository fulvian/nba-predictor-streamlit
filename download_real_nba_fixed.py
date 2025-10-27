#!/usr/bin/env python3
"""
🏀 NBA REAL Historical Data Downloader (Fixed Version)
Download REAL NBA game results with proper home/away score pairing.
"""

import polars as pl
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path
import logging
from nba_api.stats.endpoints import leaguegamelog

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedRealNBADownloader:
    """Download REAL NBA historical game results with correct score pairing."""

    def __init__(self):
        self.rate_limit = 0.5  # 500ms between requests

    def _make_request(self, season, season_type):
        """Make rate-limited request to nba_api."""
        time.sleep(self.rate_limit)

        try:
            logger.info(f"📡 Requesting {season_type} games for season {season}...")

            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation='T',
                timeout=60
            )

            return game_log.get_data_frames()[0]

        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def download_2023_24_season_results(self):
        """Download complete 2023-24 season with correct score pairing."""
        logger.info("🏀 Starting REAL NBA 2023-24 season results download (FIXED)")

        all_games = []

        # Download Regular Season games
        logger.info("📅 Downloading 2023-24 Regular Season results...")
        regular_games = self._download_and_process_season("2023", "Regular Season")
        all_games.extend(regular_games)
        logger.info(f"✅ Downloaded {len(regular_games):,} regular season games")

        # Download Playoff games
        logger.info("🏆 Downloading 2023-24 Playoff results...")
        playoff_games = self._download_and_process_season("2023", "Playoffs")
        all_games.extend(playoff_games)
        logger.info(f"✅ Downloaded {len(playoff_games):,} playoff games")

        # Convert to DataFrame
        if all_games:
            df = pl.DataFrame(all_games)
            logger.info(f"📊 Total games downloaded: {len(df):,}")

            # Validate scores
            avg_home = df['home_score'].mean()
            avg_away = df['away_score'].mean()
            logger.info(f"🏀 Average scores: Home {avg_home:.1f}, Away {avg_away:.1f}")

            if avg_home > 100 and avg_away > 100:
                logger.info("✅ REAL NBA scores detected!")
            else:
                logger.warning("⚠️ Scores may still be incomplete")

            # Save to file
            output_dir = Path("data/persistent/game_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "real_nba_results_2023-24_fixed.parquet"
            df.write_parquet(output_file)
            logger.info(f"💾 Saved to {output_file}")

            return df
        else:
            logger.error("❌ No games downloaded")
            return None

    def _download_and_process_season(self, season, season_type):
        """Download and process games with correct home/away pairing."""
        raw_df = self._make_request(season, season_type)

        if raw_df is None or len(raw_df) == 0:
            logger.error(f"❌ No data returned for {season_type} {season}")
            return []

        logger.info(f"📊 Raw data received: {len(raw_df)} team records")

        # Group by GAME_ID and process each game
        processed_games = []
        game_ids = raw_df['GAME_ID'].unique()

        for game_id in game_ids:
            # Get both team records for this game
            game_rows = raw_df[raw_df['GAME_ID'] == game_id]

            if len(game_rows) != 2:
                logger.warning(f"Game {game_id} has {len(game_rows)} team records, expected 2")
                continue

            # Extract home and away team info
            home_row = None
            away_row = None

            for _, row in game_rows.iterrows():
                matchup = row['MATCHUP']
                if '@' in matchup:  # Away team
                    away_row = row
                elif 'vs.' in matchup:  # Home team
                    home_row = row

            if home_row is None or away_row is None:
                logger.warning(f"Could not determine home/away for game {game_id}")
                continue

            # Create game record
            game = {
                'game_id': str(game_id),
                'game_date': self._parse_nba_date(home_row['GAME_DATE']),
                'home_team': home_row['TEAM_NAME'],
                'away_team': away_row['TEAM_NAME'],
                'season': f"{season}-{int(season)+1}",
                'season_type': season_type,
                'home_score': int(home_row['PTS']),
                'away_score': int(away_row['PTS']),
                'home_win': home_row['WL'] == 'W',
                'away_win': away_row['WL'] == 'W',
                'home_minutes': int(home_row['MIN']),
                'away_minutes': int(away_row['MIN']),
                'home_fgm': int(home_row['FGM']),
                'home_fga': int(home_row['FGA']),
                'home_fg_pct': float(home_row['FG_PCT']) if pd.notna(home_row['FG_PCT']) else 0.0,
                'home_fg3m': int(home_row['FG3M']),
                'home_fg3a': int(home_row['FG3A']),
                'home_fg3_pct': float(home_row['FG3_PCT']) if pd.notna(home_row['FG3_PCT']) else 0.0,
                'home_ftm': int(home_row['FTM']),
                'home_fta': int(home_row['FTA']),
                'home_ft_pct': float(home_row['FT_PCT']) if pd.notna(home_row['FT_PCT']) else 0.0,
                'home_oreb': int(home_row['OREB']),
                'home_dreb': int(home_row['DREB']),
                'home_reb': int(home_row['REB']),
                'home_ast': int(home_row['AST']),
                'home_stl': int(home_row['STL']),
                'home_blk': int(home_row['BLK']),
                'home_tov': int(home_row['TOV']),
                'home_pf': int(home_row['PF']),
                'home_plus_minus': int(home_row['PLUS_MINUS']),
                'away_fgm': int(away_row['FGM']),
                'away_fga': int(away_row['FGA']),
                'away_fg_pct': float(away_row['FG_PCT']) if pd.notna(away_row['FG_PCT']) else 0.0,
                'away_fg3m': int(away_row['FG3M']),
                'away_fg3a': int(away_row['FG3A']),
                'away_fg3_pct': float(away_row['FG3_PCT']) if pd.notna(away_row['FG3_PCT']) else 0.0,
                'away_ftm': int(away_row['FTM']),
                'away_fta': int(away_row['FTA']),
                'away_ft_pct': float(away_row['FT_PCT']) if pd.notna(away_row['FT_PCT']) else 0.0,
                'away_oreb': int(away_row['OREB']),
                'away_dreb': int(away_row['DREB']),
                'away_reb': int(away_row['REB']),
                'away_ast': int(away_row['AST']),
                'away_stl': int(away_row['STL']),
                'away_blk': int(away_row['BLK']),
                'away_tov': int(away_row['TOV']),
                'away_pf': int(away_row['PF']),
                'away_plus_minus': int(away_row['PLUS_MINUS']),
                'source': 'NBA_Official_API_Fixed',
                'created_at': datetime.now()
            }
            processed_games.append(game)

        logger.info(f"✅ Processed {len(processed_games)} complete {season_type} games")
        return processed_games

    def _parse_nba_date(self, date_str):
        """Parse NBA date format."""
        try:
            return datetime.strptime(date_str, "%m/%d/%Y").date()
        except:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return date(2023, 1, 1)

def main():
    """Main function to download real NBA data with correct scores."""
    print("🏀 NBA REAL HISTORICAL DATA DOWNLOADER (FIXED)")
    print("=" * 80)

    downloader = FixedRealNBADownloader()

    # Download 2023-24 season data
    df = downloader.download_2023_24_season_results()

    if df is not None and len(df) > 0:
        print("\n" + "=" * 80)
        print("🎯 DOWNLOAD COMPLETE!")
        print(f"✅ Total games: {len(df):,}")
        print(f"✅ Regular Season: {len(df.filter(df['season_type'] == 'Regular Season')):,}")
        print(f"✅ Playoffs: {len(df.filter(df['season_type'] == 'Playoffs')):,}")

        # Show sample data
        if len(df) > 0:
            sample = df.select(['game_date', 'home_team', 'away_team', 'home_score', 'away_score']).row(0)
            print(f"\n📝 Sample REAL game result:")
            print(f"   {sample[1]} {sample[3]} - {sample[4]} {sample[2]}")
            print(f"   Date: {sample[0]}")

            # Show score statistics
            avg_home = df['home_score'].mean()
            avg_away = df['away_score'].mean()
            max_score = df['home_score'].max()
            print(f"\n📊 Score Statistics:")
            print(f"   Average Home Score: {avg_home:.1f}")
            print(f"   Average Away Score: {avg_away:.1f}")
            print(f"   Highest Score: {max_score}")

            # Check for zero scores
            zero_home = len(df.filter(df['home_score'] == 0))
            zero_away = len(df.filter(df['away_score'] == 0))
            print(f"   Games with 0 home score: {zero_home}")
            print(f"   Games with 0 away score: {zero_away}")

            if avg_home > 100 and avg_away > 100 and zero_home == 0 and zero_away == 0:
                print("✅ CONFIRMED: Real NBA scores with complete data!")
            else:
                print("⚠️  Data quality issues detected")

        print("\n🚀 Real NBA data download completed!")
        print("💾 Data saved to: data/persistent/game_results/real_nba_results_2023-24_fixed.parquet")
    else:
        print("❌ Download failed - no data retrieved")

if __name__ == "__main__":
    main()