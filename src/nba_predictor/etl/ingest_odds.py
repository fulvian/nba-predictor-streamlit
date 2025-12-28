import logging
from pathlib import Path
from src.nba_predictor.services.nba_totals_service import get_totals_service

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/odds")


def ingest_recovered_files():
    """
    Ingest manually recovered scraping files into the database.
    """
    service = get_totals_service()

    # Debug import path
    import src.nba_predictor.etl.odds.normalize_odds_harvester as norm_module

    logger.info(f"Using normalizer from: {norm_module.__file__}")
    files = sorted(list(DATA_DIR.glob("scraped_*.json")))

    logger.info(f"Found {len(files)} files to ingest: {[f.name for f in files]}")

    total_inserted = 0

    for file_path in files:
        if file_path.stat().st_size < 100:  # Skip empty files
            logger.warning(f"Skipping empty file: {file_path.name}")
            continue

        logger.info(f"Processing {file_path.name}...")
        try:
            # Normalize JSON to DataFrame
            df = service.normalizer.normalize_to_dataframe(file_path)

            if df.is_empty():
                logger.warning(f"No valid records found in {file_path.name}")
                continue

            # Insert into DuckDB
            inserted = service.repository.insert_odds(df)
            logger.info(f"✅ Ingested {inserted} records from {file_path.name}")
            total_inserted += inserted

        except Exception as e:
            logger.error(f"❌ Failed to ingest {file_path.name}: {e}")

    logger.info("=" * 50)
    logger.info(f"🎉 Total records ingested: {total_inserted}")

    # Print final DB stats
    stats = service.get_statistics()
    logger.info(f"Final DB Stats: {stats}")

    service.close()


if __name__ == "__main__":
    ingest_recovered_files()
