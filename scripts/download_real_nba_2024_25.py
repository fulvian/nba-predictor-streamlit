#!/usr/bin/env python3
"""
🏀 NBA REAL 2024-25 Season Data Downloader
Download REAL NBA game results for 2024-25 season using proven algorithm.
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

class NBA2024_25Downloader:
    """Download REAL NBA 2024-25 season game results."""

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

    def download_2024_25_season_results(self):
        """Download complete 2024-25 season with real scores."""
        logger.info("🏀 Starting REAL NBA 2024-25 season results download")

        all_games = []

        # Download Regular Season games
        logger.info("📅 Downloading 2024-25 Regular Season results...")
        regular_games = self._download_and_process_season("2024", "Regular Season")
        all_games.extend(regular_games)
        logger.info(f"✅ Downloaded {len(regular_games):,} regular season games")

        # Download Playoff games (if available)
        logger.info("🏆 Downloading 2024-25 Playoff results...")
        playoff_games = self._download_and_process_season("2024", "Playoffs")
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

            if avg_home > 80 and avg_away > 80:
                logger.info("✅ REAL NBA scores detected!")
            else:
                logger.warning("⚠️ Scores may be incomplete (season might be ongoing)")

            # Save to file
            output_dir = Path("data/persistent/game_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "real_nba_results_2024-25_fixed.parquet"
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

        processed_game_ids = set()

        for game_id in game_ids:
            # Skip duplicates
            if game_id in processed_game_ids:
                continue

            # Get both team records for this game
            game_rows = raw_df[raw_df['GAME_ID'] == game_id]

            if len(game_rows) != 2:
                if len(game_rows) == 1:
                    # Season might be ongoing, process single team record
                    logger.debug(f"Game {game_id} has 1 team record (season ongoing)")
                    self._process_single_team_game(game_rows.iloc[0], processed_games, season, season_type)
                else:
                    logger.warning(f"Game {game_id} has {len(game_rows)} team records, expected 2")
                processed_game_ids.add(game_id)
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
                processed_game_ids.add(game_id)
                continue

            # Create complete game record
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
                'source': 'NBA_Official_API_2024_25',
                'created_at': datetime.now()
            }
            processed_games.append(game)
            processed_game_ids.add(game_id)

        logger.info(f"✅ Processed {len(processed_games)} complete {season_type} games")
        return processed_games

    def _process_single_team_game(self, row, processed_games, season, season_type):
        """Process games where only one team record is available (ongoing season)."""
        matchup = row['MATCHUP']

        if '@' in matchup:  # Away team record
            home_team = matchup.split('@ ')[1].strip()
            away_team = row['TEAM_NAME']
            home_score = 0  # Not available yet
            away_score = int(row['PTS'])
        elif 'vs.' in matchup:  # Home team record
            home_team = row['TEAM_NAME']
            away_team = matchup.split('vs. ')[1].strip()
            home_score = int(row['PTS'])
            away_score = 0  # Not available yet
        else:
            return

        game = {
            'game_id': str(row['GAME_ID']),
            'game_date': self._parse_nba_date(row['GAME_DATE']),
            'home_team': home_team,
            'away_team': away_team,
            'season': f"{season}-{int(season)+1}",
            'season_type': season_type,
            'home_score': home_score,
            'away_score': away_score,
            'home_win': row['WL'] == 'W' if home_score > 0 else None,
            'away_win': row['WL'] == 'W' if away_score > 0 else None,
            'home_minutes': int(row['MIN']) if home_score > 0 else 0,
            'away_minutes': int(row['MIN']) if away_score > 0 else 0,
            # Add all other stats similarly, checking which team they belong to
            'source': 'NBA_Official_API_2024_25_Partial',
            'created_at': datetime.now()
        }

        # Add team-specific stats
        team_prefix = 'home' if row['TEAM_NAME'] == home_team else 'away'
        for stat in ['fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct',
                     'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus']:
            game[f'{team_prefix}_{stat}'] = row.get(stat.upper(), 0)
            # Set other team stats to 0
            other_prefix = 'away' if team_prefix == 'home' else 'home'
            game[f'{other_prefix}_{stat}'] = 0

        processed_games.append(game)

    def _parse_nba_date(self, date_str):
        """Parse NBA date format."""
        try:
            return datetime.strptime(date_str, "%m/%d/%Y").date()
        except:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return date(2024, 1, 1)

def main():
    """Main function to download real NBA 2024-25 data."""
    print("🏀 NBA REAL 2024-25 SEASON DATA DOWNLOADER")
    print("=" * 80)

    downloader = NBA2024_25Downloader()

    # Download 2024-25 season data
    df = downloader.download_2024_25_season_results()

    if df is not None and len(df) > 0:
        print("\n" + "=" * 80)
        print("🎯 DOWNLOAD COMPLETE!")
        print(f"✅ Total games: {len(df):,}")

        # Separate by season type
        regular = df.filter(df['season_type'] == 'Regular Season')
        playoffs = df.filter(df['season_type'] == 'Playoffs')

        print(f"✅ Regular Season: {len(regular):,}")
        print(f"✅ Playoffs: {len(playoffs):,}")

        # Show sample data
        if len(df) > 0:
            sample = df.select(['game_date', 'home_team', 'away_team', 'home_score', 'away_score']).row(0)
            print(f"\n📝 Sample game:")
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

            if avg_home > 100 and avg_away > 100:
                print("✅ CONFIRMED: Real NBA scores detected!")
            elif avg_home > 80 and avg_away > 80:
                print("✅ Good: NBA scores detected (season might be ongoing)")
            else:
                print("⚠️  Limited data (early season or schedule-only data)")

        print("\n🚀 NBA 2024-25 data download completed!")
        print("💾 Data saved to: data/persistent/game_results/real_nba_results_2024-25_fixed.parquet")
    else:
        print("❌ Download failed - no data retrieved")

if __name__ == "__main__":
    main()