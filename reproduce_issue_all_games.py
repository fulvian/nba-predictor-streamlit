import logging
import sys
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path.cwd()
sys.path.append(str(project_root / "src"))

try:
    from nba_predictor.core.unified_hybrid_pipeline import get_unified_hybrid_pipeline

    logger.info("🚀 Instantiating UnifiedHybridPipeline...")
    pipeline = get_unified_hybrid_pipeline()

    # Mock list of games to simulate a full day schedule
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
    ]

    for i, (team1, team2, line) in enumerate(games):
        logger.info(f"\n🏀 Predicting Game {i + 1}: {team1} vs {team2} (Line: {line})")
        result = pipeline.predict_unified(
            team1=team1,
            team2=team2,
            line=line,
            home_team=team2,
            validate_prediction=True,
        )
        logger.info(f"✅ Game {i + 1} Prediction: {result.predicted_total}")

        if result.predicted_total == 205.0:
            logger.error(
                f"❌ FALLBACK DETECTED for {team1} vs {team2}: Prediction is exactly 205.0"
            )

except Exception as e:
    logger.error(f"❌ Test failed: {e}", exc_info=True)
