import logging
import sys
from pathlib import Path
import polars as pl

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.nba_predictor.core.data_store import UnifiedDataStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_integration():
    logger.info("Initializing UnifiedDataStore...")

    # Use existing DB
    store = UnifiedDataStore(base_path="data")
    store.initialize()

    try:
        # Check games data
        games = store.get_games_data()
        logger.info(f"Existing games count: {len(games)}")
        if not games.is_empty():
            print(games.head())
        else:
            logger.warning(
                "No games found in DB. Integration will only save processed CLV."
            )

        # Run integration
        logger.info("Running integrate_clv_data()...")
        store.integrate_clv_data()

        # Verify output
        output_file = Path("data/games_clv_enriched.parquet")
        processed_file = Path("data/nba_clv_processed.parquet")

        if output_file.exists():
            df = pl.read_parquet(output_file)
            logger.info(f"Enriched Data Shape: {df.shape}")
            print(df.columns)
            print(
                df.select(
                    ["home_team", "away_team", "clv_total", "clv_over_odds"]
                ).head()
            )
        elif processed_file.exists():
            df = pl.read_parquet(processed_file)
            logger.info(f"Processed CLV (Unmerged) Shape: {df.shape}")
            print(df.columns)
            print(
                df.select(
                    ["home_team", "away_team", "clv_total", "clv_over_odds"]
                ).head()
            )
        else:
            logger.error("No output file generated!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        store.close()


if __name__ == "__main__":
    test_integration()
