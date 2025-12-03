import logging
import pandas as pd
from datetime import datetime, date
from nba_predictor.core.unified_hybrid_pipeline import get_unified_hybrid_pipeline
from nba_predictor.api.data_provider import NBADataProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def debug_all_games():
    """
    Debugs predictions for all games scheduled for today using real DataProvider.
    """
    pipeline = get_unified_hybrid_pipeline()
    provider = NBADataProvider()

    # Fetch games for next 7 days to see what's available
    print(f"\n🔍 Fetching ALL scheduled games...\n")

    # Bypass get_scheduled_games date filtering by calling client directly or using wide range
    # But get_scheduled_games uses specific_date.
    # Let's try to fetch for a range of days.

    all_games = []
    for i in range(5):
        d_str = (date.today() + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"Checking {d_str}...")
        g = provider.get_scheduled_games(specific_date=d_str)
        if g:
            print(f"  Found {len(g)} games for {d_str}")
            all_games.extend(g)

    if not all_games:
        print("❌ No games found in next 5 days.")
        return

    print(f"✅ Found {len(all_games)} total games.\n")

    for i, game in enumerate(all_games):
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        print(f"🏀 Game {i + 1}: {away_team} @ {home_team}")

        try:
            # Run prediction
            result = pipeline.predict_unified(
                team1=away_team,
                team2=home_team,
                line=225.0,  # Dummy line
                home_team=home_team,
                validate_prediction=True,
            )

            print(f"   ✅ Prediction: {result.predicted_total:.2f}")
            print(f"   📊 Confidence: {result.confidence:.1f}%")

            # Check for fallback/cap
            if abs(result.predicted_total - 205.0) < 0.01:
                print("   ⚠️  POSSIBLE FALLBACK DETECTED (205.0)")
            elif abs(result.predicted_total - 245.0) < 0.01:
                print("   ⚠️  POSSIBLE FALLBACK DETECTED (245.0)")

            # Check for missing features (zeros)
            if hasattr(result, "feature_importance"):
                zeros = [k for k, v in result.feature_importance.items() if v == 0.0]
                if len(zeros) > 10:
                    print(f"   ⚠️  High number of zero features: {len(zeros)}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print("-" * 50)


if __name__ == "__main__":
    debug_all_games()
