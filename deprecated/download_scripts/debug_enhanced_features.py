#!/usr/bin/env python3
"""
Debug script to check feature generation consistency in enhanced prediction system.
"""

import sys
sys.path.append('src')

from nba_predictor.core.enhanced_prediction_pipeline import EnhancedPredictionPipeline
import pandas as pd
from datetime import datetime

def debug_features():
    print("🔍 DEBUG: Feature Generation Analysis")
    print("=" * 50)

    # Initialize pipeline
    pipeline = EnhancedPredictionPipeline()

    # Load data sources
    data_sources = pipeline._load_all_data_sources()
    print(f"✅ Data sources loaded: {list(data_sources.keys())}")

    # Create mock game for testing (same as prediction)
    mock_game = pd.Series({
        'HOME_TEAM_NAME': "Boston Celtics",
        'AWAY_TEAM_NAME': "New Orleans Pelicans",
        'HOME_TEAM_ID': 1610612738,
        'AWAY_TEAM_ID': 1610612740,
        'HOME_SCORE': 110,
        'AWAY_SCORE': 105,
        'HOME_ORtg_sAvg': 112.0,
        'AWAY_ORtg_sAvg': 110.0,
        'HOME_DRtg_sAvg': 108.0,
        'AWAY_DRtg_sAvg': 110.0,
        'HOME_PACE': 100.0,
        'AWAY_PACE': 98.0,
        'GAME_PACE': 99.0,
        'GAME_DATE': datetime.now().strftime('%Y-%m-%d')
    })

    # Test feature creation
    print("\n🎯 Testing feature creation...")
    features = pipeline._create_comprehensive_game_features(mock_game, data_sources)

    if features:
        print(f"✅ Features created: {len(features)} features")
        print("\n📋 Feature names:")
        for i, (name, value) in enumerate(sorted(features.items()), 1):
            print(f"  {i:2d}. {name:30s} = {value}")
    else:
        print("❌ No features created!")

    # Test training features
    print("\n🎯 Testing training feature structure...")
    try:
        base_games = data_sources.get('base_games')
        if base_games is not None and not base_games.empty:
            sample_game = base_games.iloc[0]
            training_features = pipeline._create_comprehensive_game_features(sample_game, data_sources)

            if training_features:
                print(f"✅ Training features: {len(training_features)} features")
                print("\n📋 Training feature names:")
                for i, (name, value) in enumerate(sorted(training_features.items()), 1):
                    print(f"  {i:2d}. {name:30s} = {value}")

                # Compare features
                if features:
                    print("\n🔍 Feature comparison:")
                    pred_features = set(features.keys())
                    train_features = set(training_features.keys())

                    missing_in_pred = train_features - pred_features
                    extra_in_pred = pred_features - train_features

                    if missing_in_pred:
                        print(f"❌ Missing in prediction: {sorted(missing_in_pred)}")
                    if extra_in_pred:
                        print(f"⚠️  Extra in prediction: {sorted(extra_in_pred)}")
                    if not missing_in_pred and not extra_in_pred:
                        print("✅ Feature sets match perfectly!")
            else:
                print("❌ No training features created!")
    except Exception as e:
        print(f"❌ Error testing training features: {e}")

if __name__ == "__main__":
    debug_features()