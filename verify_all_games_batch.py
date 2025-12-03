import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import date

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path.cwd()
sys.path.append(str(project_root / "src"))

try:
    from nba_predictor.core.unified_hybrid_pipeline import get_unified_hybrid_pipeline

    print(
        "\n🏀 BATCH VERIFICATION: Testing New Logic Across Multiple Games\n" + "=" * 60
    )

    logger.info("🚀 Instantiating UnifiedHybridPipeline...")
    pipeline = get_unified_hybrid_pipeline()

    # Ensure data is loaded (using the same source as _create_unified_prediction_features)
    print("   Loading data sources...")
    nba_data_file = pipeline.data_path / "nba_data_with_mu_sigma_for_ml.csv"
    if nba_data_file.exists():
        games_df = pd.read_csv(nba_data_file)
    else:
        print("   ⚠️ CSV data file not found. Momentum checks might fail.")
        games_df = pd.DataFrame()

    # List of diverse matchups to test various scenarios
    games = [
        ("Portland Trail Blazers", "Cleveland Cavaliers", 225.0),
        ("Los Angeles Lakers", "Boston Celtics", 230.0),
        ("Golden State Warriors", "Phoenix Suns", 235.0),
        ("Miami Heat", "Chicago Bulls", 215.0),
        ("New York Knicks", "Brooklyn Nets", 220.0),
        ("Toronto Raptors", "Philadelphia 76ers", 222.0),
        ("Milwaukee Bucks", "Indiana Pacers", 240.0),
        ("Denver Nuggets", "Utah Jazz", 228.0),
        ("Dallas Mavericks", "Houston Rockets", 232.0),
        ("San Antonio Spurs", "Memphis Grizzlies", 224.0),
        ("LA Clippers", "Atlanta Hawks", 225.5),  # The original problematic case
    ]

    results = []

    for i, (team1, team2, line) in enumerate(games):
        print(f"\n🔸 Game {i + 1}: {team1} vs {team2} (Line: {line})")

        # 1. Check Momentum Logic explicitly
        # Note: team1 is usually away, team2 home in this list structure for prediction
        # We need to resolve names to IDs for the internal check if using aliases
        t1_real = team1
        if team1 == "LA Clippers":
            t1_real = "Los Angeles Clippers"

        adj = pipeline._get_team_adjustments(t1_real, team2, games_df)

        mom_t1 = adj.get("team1_score", 0.0)
        mom_t2 = adj.get("team2_score", 0.0)

        print(f"   🔹 Momentum: {team1} ({mom_t1:+.2f}), {team2} ({mom_t2:+.2f})")

        # 2. Run Prediction
        try:
            pred_result = pipeline.predict_unified(
                team1=team1,
                team2=team2,
                line=line,
                home_team=team2,
                validate_prediction=True,
            )
            pred_total = pred_result.predicted_total

            status = "✅ OK"
            if pred_total == 205.0:
                status = "❌ FALLBACK (205.0)"
            elif pred_total == 200.5:
                status = "⚠️ FALLBACK (200.5)"

            print(f"   🔹 Prediction: {pred_total:.2f} [{status}]")

            results.append(
                {
                    "Matchup": f"{team1} vs {team2}",
                    "Prediction": pred_total,
                    "Momentum_T1": mom_t1,
                    "Momentum_T2": mom_t2,
                    "Status": status,
                }
            )

        except Exception as e:
            print(f"   ❌ CRASH: {e}")
            results.append(
                {
                    "Matchup": f"{team1} vs {team2}",
                    "Prediction": 0.0,
                    "Momentum_T1": mom_t1,
                    "Momentum_T2": mom_t2,
                    "Status": f"CRASH: {e}",
                }
            )

    print("\n" + "=" * 60)
    print("📊 SUMMARY REPORT")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("=" * 60 + "\n")

except Exception as e:
    logger.error(f"❌ Test failed: {e}", exc_info=True)
