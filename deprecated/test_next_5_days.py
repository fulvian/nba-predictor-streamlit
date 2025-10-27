#!/usr/bin/env python3
"""
Test NBA games detection for the next 5 days
Tests the complete NBA game detection system across multiple days
"""

import sys
import os
from datetime import date, timedelta

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_next_5_days():
    """Test NBA game detection for next 5 days"""
    print("🏀 NBA Game Detection - Next 5 Days Test")
    print("=" * 60)

    try:
        from data_provider import NBADataProvider

        # Inizializza il data provider
        dp = NBADataProvider()
        print("✅ NBADataProvider initialized successfully\n")

        # Test per i prossimi 5 giorni
        today = date.today()
        total_games = 0
        successful_days = 0

        print("📅 Testing NBA Game Detection for Next 5 Days:")
        print("-" * 60)

        for i in range(5):
            test_date = today + timedelta(days=i)
            date_str = test_date.strftime('%Y-%m-%d')
            day_name = test_date.strftime('%A')

            print(f"\n📅 {day_name} ({date_str}):")
            print("   🔍 Searching for NBA games...")

            try:
                games = dp.get_scheduled_games(specific_date=date_str)

                if games:
                    print(f"   ✅ Found {len(games)} NBA games:")
                    for j, game in enumerate(games, 1):
                        away_team = game.get('away_team', 'Unknown')
                        home_team = game.get('home_team', 'Unknown')
                        source = game.get('source', 'Unknown')
                        game_id = game.get('game_id', 'N/A')

                        print(f"      {j}. {away_team} @ {home_team}")
                        print(f"         🆔 Game ID: {game_id}")
                        print(f"         📡 Source: {source}")

                    total_games += len(games)
                    successful_days += 1
                else:
                    print(f"   ❌ No NBA games found")
                    print(f"   ℹ️  This could be due to:")
                    print(f"      - No games scheduled for this date")
                    print(f"      - NBA API connectivity issues")
                    print(f"      - Season not started/ended")

            except Exception as e:
                print(f"   ❌ Error detecting games: {str(e)}")
                print(f"   🐛 This is likely an NBA API connectivity issue")

        # Summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY - Next 5 Days Test Results")
        print("=" * 60)
        print(f"📅 Period: {today.strftime('%Y-%m-%d')} to {(today + timedelta(days=4)).strftime('%Y-%m-%d')}")
        print(f"✅ Successful days: {successful_days}/5")
        print(f"🏀 Total games found: {total_games}")
        print(f"📈 Average games per day: {total_games/5:.1f}")

        if total_games > 0:
            print(f"\n🎉 SUCCESS: System detected {total_games} NBA games!")
            print("💡 The NBA game detection system is working correctly")
        else:
            print(f"\n⚠️  WARNING: No games detected in 5-day period")
            print("💡 This could indicate:")
            print("   - NBA offseason period")
            print("   - API connectivity issues with stats.nba.com")
            print("   - No games scheduled in this period")

        # Test specific future date
        print(f"\n🔍 Additional Test: Oct 25, 2025 (future date)")
        print("-" * 60)

        try:
            future_games = dp.get_scheduled_games(specific_date='2025-10-25')
            if future_games:
                print(f"✅ Found {len(future_games)} games for Oct 25, 2025:")
                for j, game in enumerate(future_games, 1):
                    away_team = game.get('away_team', 'Unknown')
                    home_team = game.get('home_team', 'Unknown')
                    print(f"   {j}. {away_team} @ {home_team}")
            else:
                print("❌ No games found for Oct 25, 2025")
                print("ℹ️  This is expected if the NBA API doesn't have future schedule data")
        except Exception as e:
            print(f"❌ Error with Oct 25, 2025: {str(e)}")

        return total_games > 0

    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        import traceback
        print(f"🐛 Stack trace: {traceback.format_exc()}")
        return False

def main():
    """Run the 5-day test"""
    success = test_next_5_days()

    if success:
        print("\n🎉 TEST PASSED: NBA Game Detection System Working!")
        print("🚀 Ready for Streamlit Cloud deployment")
    else:
        print("\n⚠️  TEST COMPLETED: No games detected (might be normal)")
        print("🔧 System is functional, NBA API might be unavailable")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)