
import sys
import os
import logging
from datetime import date, timedelta
import pandas as pd

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nba_predictive_system.unified_nba_data_pipeline import UnifiedNBADataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    logger.info("Starting Pipeline Verification...")
    
    try:
        pipeline = UnifiedNBADataPipeline()
        
        # Test fetching data for yesterday
        end_date = date.today() - timedelta(days=1)
        start_date = end_date
        
        logger.info(f"Fetching data for {start_date} to {end_date}")
        
        data = pipeline.fetch_all_data(
            date_range=(start_date, end_date),
            include_boxscores=True
        )
        
        logger.info("Fetch completed.")
        
        if 'games' in data and not data['games'].empty:
            logger.info(f"✅ Games found: {len(data['games'])}")
            print(data['games'].head())
        else:
            logger.warning("⚠️ No games found.")
            
        if 'boxscores' in data and not data['boxscores'].empty:
             logger.info(f"✅ Boxscores found: {len(data['boxscores'])}")
        else:
             logger.warning(f"⚠️ No boxscores found (Data: {type(data.get('boxscores'))})")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_pipeline()
