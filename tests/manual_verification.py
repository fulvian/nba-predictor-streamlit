
import sys
import os
import logging
from datetime import date, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nba_predictive_system.unified_nba_data_pipeline import UnifiedNBADataPipeline
from nba_predictive_system.advanced_predictive_model import AdvancedPredictiveModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_system():
    logger.info("🚀 Starting System Verification...")
    
    try:
        # 1. Initialize Components
        logger.info("1. Initializing Components...")
        pipeline = UnifiedNBADataPipeline()
        model = AdvancedPredictiveModel()
        
        # 2. Fetch Data
        logger.info("2. Fetching Data...")
        # Use a past date range to ensure games have scores for training
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        raw_data = pipeline.fetch_all_data(
            date_range=(start_date, end_date),
            include_boxscores=False
        )
        
        if raw_data.get('games') is None or raw_data['games'].empty:
            logger.error("❌ No games fetched. Cannot proceed.")
            return
            
        games_data = raw_data['games']
        logger.info(f"✅ Fetched {len(games_data)} games.")
        
        # 3. Train Model
        logger.info("3. Training Model...")
        features = pipeline.preprocess_features(raw_data)
        
        logger.info(f"Features columns: {features.columns.tolist()}")
        logger.info(f"Games data columns: {games_data.columns.tolist()}")

        if features.empty:
            logger.error("❌ Preprocessing failed.")
            return
            
        # Create target
        if 'home_score' in features.columns and 'away_score' in features.columns:
            features['target'] = (features['home_score'] > features['away_score']).astype(int)
        else:
            logger.error("❌ Missing score columns.")
            return
            
        training_results = model.train_predictive_models(
            training_data=features,
            target_column='target'
        )
        
        if training_results and training_results.get('status') == 'success':
            logger.info("✅ Model trained successfully.")
            logger.info(f"Metrics: {training_results.get('metrics')}")
        else:
            logger.error("❌ Model training failed.")
            return
            
        # 4. Predict
        logger.info("4. Predicting...")
        # Pick a game to predict
        game_row = games_data.iloc[[0]]
        logger.info(f"Predicting for: {game_row.iloc[0].get('away_team')} @ {game_row.iloc[0].get('home_team')}")
        
        # Preprocess single game
        single_game_features = pipeline.preprocess_features({'games': game_row})
        
        prediction = model.predict_game_outcome(single_game_features)
        
        if not prediction.empty:
            logger.info(f"✅ Prediction: {prediction.iloc[0].to_dict()}")
        else:
            logger.error("❌ Prediction failed.")
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}", exc_info=True)

if __name__ == "__main__":
    verify_system()
