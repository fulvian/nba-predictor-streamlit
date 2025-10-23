#!/usr/bin/env python3
"""
Quick test to check NBA games for next 5 days without extensive retries
"""

import sys
import os
from datetime import date, timedelta

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test for NBA games next 5 days"""
    print("🏀 Quick NBA Games Test - Next 5 Days")
    print("=" * 50)

    try:
        from data_provider import NBADataProvider
        print("✅ NBADataProvider imported")

        # Quick initialization test
        dp = NBADataProvider()
        print("✅ NBADataProvider initialized")

        today = date.today()
        print(f"\n📅 Testing period: {today.strftime('%Y-%m-%d')} to {(today + timedelta(days=4)).strftime('%Y-%m-%d')}")
        print("-" * 50)

        total_games = 0

        for i in range(5):
            test_date = today + timedelta(days=i)
            date_str = test_date.strftime('%Y-%m-%d')
            day_name = test_date.strftime('%A')[:3]  # First 3 letters

            print(f"\n📅 {day_name} {date_str}:")
            print("   🔍 Quick check (single attempt)...")

            try:
                # Use shorter timeout for quick test
                games = dp.get_scheduled_games(specific_date=date_str)

                if games:
                    print(f"   ✅ {len(games)} games found:")
                    for j, game in enumerate(games[:3], 1):  # Show max 3 games
                        away = game.get('away_team', 'Unknown')
                        home = game.get('home_team', 'Unknown')
                        print(f"      {j}. {away} @ {home}")

                    if len(games) > 3:
                        print(f"      ... and {len(games) - 3} more games")

                    total_games += len(games)
                else:
                    print("   ❌ No games found (or API issues)")

            except Exception as e:
                print(f"   ❌ Error: {str(e)[:60]}...")

        print(f"\n📊 QUICK SUMMARY:")
        print(f"   🏀 Total games: {total_games}")
        print(f"   📅 Period: 5 days")
        print(f"   📈 Average: {total_games/5:.1f} games/day")

        if total_games > 0:
            print(f"\n🎉 SUCCESS: Found {total_games} NBA games!")
        else:
            print(f"\n⚠️  No games detected - likely NBA offseason or API issues")

        return total_games > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_test()