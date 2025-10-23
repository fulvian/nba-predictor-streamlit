#!/usr/bin/env python3
"""
ULTIMATE NBA SOLUTION - Context7-Compliant
Recupera le partite NBA per i PROSSIMI 7 giorni (non quelle già giocate!)

Problem Identified via Context7:
- Live Data API: Solo partite di OGGI (già giocate!)
- ScheduleLeagueV2: Partite future ma con problemi di colonne
- Stats API: Completamente rotta (30s timeouts)

Solution: Smart ScheduleLeagueV2 Implementation
1. Fix column names issue
2. Parse future games correctly
3. Filter next 7 days only
4. Return actionable upcoming games
"""

import sys
import os
import time
from datetime import datetime, date, timedelta
import pandas as pd

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class UltimateNBASolution:
    """Ultimate NBA Solution for Next 7 Days"""

    def __init__(self):
        """Initialize the ultimate solution"""
        print("🚀 ULTIMATE NBA SOLUTION - Next 7 Days")
        print("=" * 60)
        print("🎯 Strategy:")
        print("   ❌ Skip TODAY's games (already played)")
        print("   ✅ Focus on NEXT 7 DAYS")
        print("   📅 Use ScheduleLeagueV2 with proper column parsing")
        print("   ⚡ Fast and reliable future game detection")
        print("=" * 60)

    def debug_schedule_columns(self):
        """Debug ScheduleLeagueV2 to find correct column names"""
        try:
            print("🔍 DEBUG: Finding ScheduleLeagueV2 column names...")

            from nba_api.stats.endpoints import scheduleleaguev2

            # Try current season 2024-25 (should have real data)
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                league_id='00',
                season='2024-25'
            )

            # Get all data frames
            data_frames = schedule.get_data_frames()

            print(f"📊 Found {len(data_frames)} data frames")

            for i, df in enumerate(data_frames):
                print(f"\n📋 DataFrame {i+1}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")

                if len(df) > 0:
                    print(f"   Sample row:")
                    for col in df.columns:
                        value = df[col].iloc[0]
                        print(f"      {col}: {value}")

            return data_frames

        except Exception as e:
            print(f"❌ DEBUG Error: {str(e)}")
            return None

    def get_next_7_days_games(self):
        """Get NBA games for the next 7 days (skipping today)"""
        try:
            print("📅 Getting NBA games for NEXT 7 DAYS...")

            # Define date range
            today = date.today()
            start_date = today + timedelta(days=1)  # Skip today
            end_date = today + timedelta(days=7)

            print(f"📅 Date Range: {start_date} to {end_date}")

            # Determine NBA season
            year = start_date.year
            if start_date.month >= 10:  # NBA season starts in October
                season = f"{year}-{str(year+1)[-2:]}"
            else:
                season = f"{year-1}-{str(year)[-2:]}"

            print(f"🏀 NBA Season: {season}")

            from nba_api.stats.endpoints import scheduleleaguev2

            # Get schedule for the season
            print("🔄 Fetching season schedule...")
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                league_id='00',
                season=season
            )

            # Get data frames
            data_frames = schedule.get_data_frames()

            if not data_frames:
                print("❌ No data frames returned")
                return []

            # Use the first data frame (LeagueSchedule)
            df = data_frames[0]

            print(f"📊 Schedule DataFrame: {df.shape[0]} total games")
            print(f"📋 Columns: {list(df.columns)}")

            # Find the date column - we now know it's 'gameDate'
            date_column = 'gameDate'
            if date_column in df.columns:
                print(f"✅ Found date column: {date_column}")
            else:
                print("❌ gameDate column not found!")
                print("Available columns:", list(df.columns))
                return []

            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column])

            # Filter games for our date range
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())

            filtered_df = df[
                (df[date_column] >= start_datetime) &
                (df[date_column] <= end_datetime)
            ]

            print(f"📊 Found {len(filtered_df)} games in next 7 days")

            if len(filtered_df) == 0:
                print("⚠️  No games found in next 7 days")
                print("💡 This could be because:")
                print("   - NBA offseason")
                print("   - No games scheduled")
                print("   - Wrong season detected")
                return []

            # Convert to our format
            games = []
            for _, row in filtered_df.iterrows():
                try:
                    # Get teams from the correct columns
                    away_team = row.get('awayTeam_teamName', 'Unknown')
                    home_team = row.get('homeTeam_teamName', 'Unknown')

                    # Get team IDs
                    away_team_id = row.get('awayTeam_teamId', 0)
                    home_team_id = row.get('homeTeam_teamId', 0)

                    # Get game ID
                    game_id = row.get('gameId', f"SCHEDULE_{len(games)}")

                    # Format date
                    game_date = row[date_column].strftime('%Y-%m-%d')

                    games.append({
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_team_id': away_team_id,
                        'home_team_id': home_team_id,
                        'game_id': game_id,
                        'date': game_date,
                        'time_utc': row[date_column].isoformat(),
                        'status': 'Scheduled',
                        'score': '',
                        'source': 'NBA ScheduleLeagueV2 (Next 7 Days)',
                        'api_endpoint': 'stats.nba.com/stats/scheduleleaguev2',
                        'season': season
                    })

                except Exception as e:
                    print(f"⚠️  Error processing game: {e}")
                    continue

            print(f"✅ Successfully processed {len(games)} games for next 7 days")
            return games

        except Exception as e:
            print(f"❌ Error getting next 7 days games: {str(e)}")
            import traceback
            print(f"🐛 Stack trace: {traceback.format_exc()}")
            return []

    def display_next_7_days_schedule(self):
        """Display the schedule for next 7 days"""
        games = self.get_next_7_days_games()

        if not games:
            print("❌ No NBA games found for the next 7 days")
            return []

        print(f"\n🏀 NBA GAMES - NEXT 7 DAYS")
        print("=" * 60)

        # Group games by date
        games_by_date = {}
        for game in games:
            date_str = game['date']
            if date_str not in games_by_date:
                games_by_date[date_str] = []
            games_by_date[date_str].append(game)

        # Display games by date
        for date_str in sorted(games_by_date.keys()):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_name = date_obj.strftime('%A')

            print(f"\n📅 {day_name} {date_str}:")
            print("-" * 40)

            for i, game in enumerate(games_by_date[date_str], 1):
                print(f"   {i}. {game['away_team']} @ {game['home_team']}")
                print(f"      🆔 Game ID: {game['game_id']}")
                print(f"      📡 Source: {game['source']}")
                print()

        return games

    def test_solution(self):
        """Test the complete solution"""
        print("🧪 TESTING ULTIMATE NBA SOLUTION")
        print("=" * 60)

        # First, debug columns to understand the data structure
        print("\n🔍 Step 1: Debug ScheduleLeagueV2 Structure")
        debug_info = self.debug_schedule_columns()

        # Then test the actual solution
        print("\n📅 Step 2: Test Next 7 Days Detection")
        games = self.display_next_7_days_schedule()

        # Summary
        print("\n📊 SOLUTION TEST RESULTS")
        print("=" * 60)

        if games:
            print(f"✅ SUCCESS: Found {len(games)} upcoming NBA games")
            print(f"📅 Date Range: Next 7 days")
            print(f"🏀 Teams: {len(set(g['away_team'] for g in games) + set(g['home_team'] for g in games))} unique teams")
            print("🚀 Ready for Streamlit Cloud deployment!")
        else:
            print("❌ No upcoming games detected")
            print("💡 This might be due to NBA offseason or scheduling")

        return len(games) > 0


def main():
    """Run the ultimate solution test"""
    # Import pandas for data manipulation
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not available, installing...")
        os.system(f"{sys.executable} -m pip install pandas")
        import pandas as pd

    solution = UltimateNBASolution()
    success = solution.test_solution()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)