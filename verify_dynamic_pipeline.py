import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))

from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_dynamic_teams():
    print("🧪 Verifying Dynamic Team Logic in UnifiedHybridPipeline")
    print("=" * 60)

    try:
        # Initialize pipeline
        pipeline = UnifiedHybridPipeline()

        # Load data manually to pass to _get_team_adjustments (simulating pipeline flow)
        data_path = Path("data/nba_data_with_mu_sigma_for_ml.csv")
        if not data_path.exists():
            print("❌ Data file not found!")
            return False

        df = pd.read_csv(data_path)

        # Test with a known high performing team (e.g. Utah Jazz from our prototype test)
        # and a known low performing team (e.g. Boston Celtics from our prototype test)

        print("Running _get_team_adjustments...")
        adjustments = pipeline._get_team_adjustments("Utah Jazz", "Boston Celtics", df)

        print("\n📊 Adjustments Result:")
        print(adjustments)

        # Debug: Print expected logic
        print(f"Team 1 (Utah Jazz) Adjustment: {adjustments['team1_score']}")
        print(f"Team 2 (Boston Celtics) Adjustment: {adjustments['team2_score']}")

        # Note: New Logic is "Advanced Momentum" (Weighted ORtg * Match Pace)
        # The logs revealed:
        # Utah Jazz: ORtg trending UP (108.9 -> 111.3) -> Expect POSITIVE adjustment (Scoring more)
        # Boston Celtics: ORtg trending DOWN (118.4 -> 114.8) -> Expect NEGATIVE adjustment (Scoring less)
        # This highlights the difference between "Net Rating" (Winning) and "ORtg Momentum" (Scoring).
        # For a Totals predictor, capturing the Scoring Trend is exactly what we want.

        is_jazz_scoring_up = adjustments["team1_score"] > 0
        is_celtics_scoring_down = adjustments["team2_score"] < 0

        print(
            f"\nUtah Jazz (Scoring Trend UP): {'✅ Positive Adj' if is_jazz_scoring_up else '❌ Unexpected Adj'}"
        )
        print(
            f"Boston Celtics (Scoring Trend DOWN): {'✅ Negative Adj' if is_celtics_scoring_down else '❌ Unexpected Adj'}"
        )

        if is_jazz_scoring_up:
            print(f"   Adjustment Value: {adjustments['team1_score']:.2f}")
        if is_celtics_scoring_down:
            print(f"   Adjustment Value: {adjustments['team2_score']:.2f}")

        if is_jazz_scoring_up and is_celtics_scoring_down:
            print("\n✅ Advanced Momentum logic integrated successfully!")
            return True
        else:
            print("\n⚠️ Logic verification failed. Check logs.")
            return False

    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    verify_dynamic_teams()
