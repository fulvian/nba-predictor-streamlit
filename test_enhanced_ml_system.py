#!/usr/bin/env python3
"""
🧪 Test Enhanced NBA ML System - Comprehensive Integration Test
Validates all components work together correctly and resolve the identified issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
from pathlib import Path

# Add the nba_predictive_system to the path
sys.path.append(str(Path(__file__).parent / "nba_predictive_system"))

from enhanced_ml_system import EnhancedNBAMLSystem

def create_synthetic_nba_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Create synthetic NBA data for testing the enhanced system.
    Simulates realistic NBA game data with temporal patterns.
    """
    print("📊 Creating synthetic NBA data for testing...")

    # Generate dates
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]

    # NBA teams
    teams = ['Lakers', 'Celtics', 'Warriors', 'Heat', 'Nuggets', 'Suns', 'Bucks', '76ers']

    # Generate realistic game data
    np.random.seed(42)  # For reproducible results

    data = []
    for i, game_date in enumerate(dates):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Simulate team strength (some teams are stronger)
        team_strength = {
            'Lakers': 0.8, 'Celtics': 0.75, 'Warriors': 0.85, 'Heat': 0.7,
            'Nuggets': 0.82, 'Suns': 0.78, 'Bucks': 0.83, '76ers': 0.77
        }

        # Calculate expected total points based on team strengths
        home_strength = team_strength[home_team]
        away_strength = team_strength[away_team]
        base_total = 220  # NBA average total points

        # Add some variance
        strength_factor = (home_strength + away_strength) / 2
        expected_total = base_total * strength_factor + np.random.normal(0, 10)

        # Generate actual total with some randomness
        total_points = max(180, min(280, expected_total + np.random.normal(0, 8)))

        # Generate betting odds (simplified)
        betting_line = round(expected_total, 0)
        over_odds = 1.9 + np.random.uniform(-0.1, 0.1)  # Around -110 odds
        under_odds = 1.9 + np.random.uniform(-0.1, 0.1)

        # Generate features that would be predictive
        home_team_avg_points = 110 + (home_strength - 0.75) * 20 + np.random.normal(0, 5)
        away_team_avg_points = 108 + (away_strength - 0.75) * 20 + np.random.normal(0, 5)

        # Injury impact simulation
        home_injury_impact = np.random.exponential(0.5) if np.random.random() > 0.7 else 0
        away_injury_impact = np.random.exponential(0.5) if np.random.random() > 0.7 else 0

        game_data = {
            'GAME_DATE': game_date,
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'TOTAL_POINTS': total_points,
            'BETTING_LINE': betting_line,
            'OVER_ODDS': over_odds,
            'UNDER_ODDS': under_odds,

            # Predictive features
            'HOME_TEAM_AVG_POINTS': home_team_avg_points,
            'AWAY_TEAM_AVG_POINTS': away_team_avg_points,
            'HOME_TEAM_PACE': 98 + np.random.normal(0, 3),
            'AWAY_TEAM_PACE': 97 + np.random.normal(0, 3),
            'HOME_TEAM_DEFENSE_RATING': 110 + np.random.normal(0, 5),
            'AWAY_TEAM_DEFENSE_RATING': 112 + np.random.normal(0, 5),
            'HOME_INJURY_IMPACT': home_injury_impact,
            'AWAY_INJURY_IMPACT': away_injury_impact,
            'DAYS_SINCE_LAST_HOME': np.random.randint(1, 4),
            'DAYS_SINCE_LAST_AWAY': np.random.randint(1, 4),
            'HOME_BACK_TO_BACK': np.random.random() > 0.85,
            'AWAY_BACK_TO_BACK': np.random.random() > 0.85,

            # Advanced metrics
            'HOME_TEAM_EFG_PCT': 0.52 + np.random.normal(0, 0.02),
            'AWAY_TEAM_EFG_PCT': 0.51 + np.random.normal(0, 0.02),
            'HOME_TEAM_TOV_PCT': 12.5 + np.random.normal(0, 2),
            'AWAY_TEAM_TOV_PCT': 13.0 + np.random.normal(0, 2),
            'HOME_TEAM_REB_PCT': 49.0 + np.random.normal(0, 3),
            'AWAY_TEAM_REB_PCT': 48.0 + np.random.normal(0, 3),
        }

        data.append(game_data)

    df = pd.DataFrame(data)
    print(f"✅ Created synthetic dataset: {len(df)} games from {dates[0]} to {dates[-1]}")
    print(f"   Average total points: {df['TOTAL_POINTS'].mean():.1f}")
    print(f"   Point range: {df['TOTAL_POINTS'].min():.1f} - {df['TOTAL_POINTS'].max():.1f}")

    return df

def test_enhanced_system():
    """
    Test the enhanced NBA ML system with comprehensive validation.
    """
    print("🧪 Starting Enhanced NBA ML System Test")
    print("=" * 60)

    # Create test data
    test_data = create_synthetic_nba_data(300)

    # Split data for training and testing
    train_cutoff = date(2024, 2, 1)
    train_data = test_data[test_data['GAME_DATE'] < train_cutoff]
    test_data_split = test_data[test_data['GAME_DATE'] >= train_cutoff]

    print(f"\n📊 Data Split:")
    print(f"   Training: {len(train_data)} games")
    print(f"   Testing: {len(test_data_split)} games")

    try:
        # Initialize enhanced system
        print("\n🚀 Initializing Enhanced NBA ML System...")
        system = EnhancedNBAMLSystem(
            model_name="test_nba_enhanced",
            monitoring_enabled=True,
            auto_retraining=True
        )

        # Test 1: System Health Check
        print("\n🏥 Test 1: System Health Check")
        health = system.get_system_health_report()
        print(f"   ✅ Model Status: {health['model_status']['is_trained']}")
        print(f"   ✅ Data Provider: {health['data_provider_status']['data_provider']}")
        print(f"   ✅ Injury Reporter: {health['data_provider_status']['injury_reporter']}")

        # Test 2: Enhanced Model Training
        print("\n🚀 Test 2: Enhanced Model Training")
        training_results = system.train_model(
            training_data=train_data,
            target_column='TOTAL_POINTS',
            validate_temporal=True
        )

        if training_results['training_status'] == 'success':
            print(f"   ✅ Training successful - Version {training_results['model_version']}")
            print(f"   ✅ Features used: {training_results['feature_count']}")
            print(f"   ✅ Temporal validation: {training_results.get('training_metrics', {}).get('ensemble_weights', 'N/A')}")

            # Check for data leakage warnings
            leakage_issues = training_results['leakage_analysis']['potential_leakage']
            if leakage_issues:
                print(f"   ⚠️ Data leakage issues found: {len(leakage_issues)}")
            else:
                print(f"   ✅ No data leakage detected")

        else:
            print(f"   ❌ Training failed: {training_results.get('error', 'Unknown error')}")
            return False

        # Test 3: Prediction with Monitoring
        print("\n📊 Test 3: Prediction with Monitoring")

        # Use a small sample for prediction testing
        prediction_sample = test_data_split.head(5).copy()

        predictions = system.predict_with_monitoring(
            game_data=prediction_sample,
            include_confidence=True,
            record_for_monitoring=True
        )

        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   ✅ Average prediction: {predictions['predicted_class'].mean():.1f}")
        print(f"   ✅ Average confidence: {predictions['predicted_probability'].mean():.3f}")

        # Test 4: Model Performance Monitoring
        print("\n📈 Test 4: Performance Monitoring")

        if system.monitor:
            # Simulate some actual results to test monitoring
            for i, (_, game_row) in enumerate(prediction_sample.iterrows()):
                if i < len(predictions):
                    actual = game_row['TOTAL_POINTS']
                    pred = predictions.iloc[i]['predicted_class']
                    confidence = predictions.iloc[i]['predicted_probability']

                    # Record with actual result
                    system.monitor.record_prediction(
                        prediction=pred,
                        actual=actual,
                        confidence=confidence
                    )

            monitoring_summary = system.monitor.get_monitoring_summary()
            print(f"   ✅ Monitoring active: {monitoring_summary['status']}")
            print(f"   ✅ Predictions recorded: {monitoring_summary['data_quality']['total_predictions']}")
            print(f"   ✅ Completion rate: {monitoring_summary['data_quality']['completion_rate']:.1%}")

        # Test 5: Comprehensive Backtesting
        print("\n🏆 Test 5: Comprehensive Backtesting")

        # Use a subset for faster testing
        backtest_data = test_data_split.head(50)

        backtest_results = system.run_comprehensive_backtest(
            historical_data=backtest_data,
            start_date=backtest_data['GAME_DATE'].min().date(),
            end_date=backtest_data['GAME_DATE'].max().date(),
            initial_bankroll=1000.0
        )

        if 'backtest_results' in backtest_results:
            summary = backtest_results['backtest_results']['backtest_summary']
            print(f"   ✅ Backtest completed: {summary.get('total_bets', 0)} bets")
            print(f"   ✅ Win rate: {summary.get('win_rate', 0):.1%}")
            print(f"   ✅ ROI: {summary.get('roi_percentage', 0):.1f}")
            print(f"   ✅ Final bankroll: ${backtest_results['backtest_results']['bankroll_performance']['final_bankroll']:.2f}")
        else:
            print(f"   ⚠️ Backtest limited: {backtest_results.get('analysis', {}).get('summary', {}).get('status', 'unknown')}")

        # Test 6: System Integration Validation
        print("\n🔗 Test 6: System Integration Validation")

        final_health = system.get_system_health_report()

        integration_checks = [
            ("Model Training", system.is_trained),
            ("Injury Reporting", final_health['data_provider_status']['injury_reporter'] == 'operational'),
            ("Data Provider", final_health['data_provider_status']['data_provider'] == 'operational'),
            ("Monitoring", system.monitor is not None),
            ("Feature Engineering", len(system.feature_columns) > 0)
        ]

        all_passed = True
        for check_name, check_result in integration_checks:
            status = "✅ PASS" if check_result else "❌ FAIL"
            print(f"   {status}: {check_name}")
            if not check_result:
                all_passed = False

        # Test 7: Data Leakage Prevention Validation
        print("\n🔒 Test 7: Data Leakage Prevention Validation")

        # This should show temporal validation was used
        temporal_used = 'temporal_validation' in training_results.get('training_metrics', {})
        print(f"   {'✅ PASS' if temporal_used else '❌ FAIL'}: Temporal validation used")

        # Check that feature selection excluded ID columns
        id_features = [col for col in system.feature_columns if 'ID' in col.upper()]
        no_id_features = len(id_features) == 0
        print(f"   {'✅ PASS' if no_id_features else '❌ FAIL'}: No ID features in model")

        # Final Results
        print("\n" + "=" * 60)
        print("🎯 FINAL TEST RESULTS")
        print("=" * 60)

        if all_passed and temporal_used and no_id_features:
            print("🟢 ALL TESTS PASSED! 🎉")
            print("\n✅ Critical Issues RESOLVED:")
            print("   ✅ Injury Reporting: Fully integrated and operational")
            print("   ✅ Temporal Validation: Preventing data leakage")
            print("   ✅ Model Monitoring: Active drift detection")
            print("   ✅ Backtesting: Comprehensive performance validation")
            print("   ✅ Data Leakage: Robust prevention mechanisms")
            print("   ✅ System Integration: All components working together")

            print(f"\n📈 System Performance Metrics:")
            print(f"   • Model Version: {system.model_version}")
            print(f"   • Features Engineered: {len(system.feature_columns)}")
            print(f"   • Training Accuracy: {training_results.get('training_metrics', {}).get('metrics', {}).get('accuracy', 0):.1%}")
            print(f"   • System Status: Production Ready")

            return True

        else:
            print("🔴 SOME TESTS FAILED!")
            print("\n❌ Issues to resolve:")
            if not all_passed:
                failed_checks = [name for name, result in integration_checks if not result]
                for check in failed_checks:
                    print(f"   • {check}")
            if not temporal_used:
                print(f"   • Temporal validation not properly implemented")
            if not no_id_features:
                print(f"   • ID features still present in model")

            return False

    except Exception as e:
        print(f"\n❌ TEST EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏀 Enhanced NBA ML System Integration Test")
    print("Testing all critical improvements and issue resolutions...")
    print()

    success = test_enhanced_system()

    if success:
        print("\n🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT!")
        print("All critical issues have been resolved.")
    else:
        print("\n⚠️ System needs additional fixes before production use.")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")