#!/usr/bin/env python3
"""
🏀 NBA Prediction System - Main Script
Context7-compliant prediction pipeline using real NBA data.

This script provides:
- Complete NBA Over/Under predictions using real data
- Ensemble ML models with confidence intervals
- Command-line interface for easy testing
- Integration with real NBA datasets

Usage:
    python main_prediction.py --team1 "Los Angeles Lakers" --team2 "Boston Celtics" --line 225.5
    python main_prediction.py --train-model
    python main_prediction.py --test-pipeline
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from nba_predictor.core.prediction_pipeline import NBAPredictionPipeline, PredictionPipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NBAPredictionCLI:
    """Command-line interface for NBA predictions."""

    def __init__(self, data_path: str = "data", models_path: str = "models"):
        """
        Initialize the CLI interface.

        Args:
            data_path: Path to NBA data files
            models_path: Path to model files
        """
        self.pipeline = NBAPredictionPipeline(
            data_path=data_path,
            model_path=models_path
        )
        logger.info("NBA Prediction CLI initialized")

    def train_model(self) -> bool:
        """
        Train the prediction model using real NBA data.

        Returns:
            True if training successful, False otherwise
        """
        try:
            print("\n" + "="*80)
            print("🚀 TRAINING NBA PREDICTION MODEL")
            print("="*80)

            # Try to load existing model first
            if self.pipeline.load_model():
                print("✅ Existing model loaded successfully")
                print(f"📊 Model info: {self.pipeline.get_model_info()}")
                return True

            print("📚 No existing model found - training new model...")
            print("📈 Loading real NBA data...")

            metrics = self.pipeline.train_model()

            print("\n✅ MODEL TRAINING COMPLETED!")
            print(f"📊 Performance Metrics:")
            print(f"   • Mean Absolute Error: {metrics['mae']:.2f} points")
            print(f"   • Root Mean Squared Error: {metrics['rmse']:.2f} points")
            print(f"   • R² Score: {metrics['r2_score']:.3f}")
            print(f"   • Cross-validation MAE: {metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f}")
            print(f"   • Training samples: {metrics['training_samples']}")
            print(f"   • Features used: {metrics['feature_count']}")
            print(f"   • Training date: {metrics['training_date']}")

            return True

        except PredictionPipelineError as e:
            print(f"❌ Model training failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during training: {e}")
            return False

    def predict_game(
        self,
        team1: str,
        team2: str,
        line: float,
        home_team: str = None,
        verbose: bool = True
    ) -> bool:
        """
        Make prediction for NBA game.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line (total points)
            home_team: Which team is home (optional)
            verbose: Whether to print detailed output

        Returns:
            True if prediction successful, False otherwise
        """
        try:
            if verbose:
                print("\n" + "="*80)
                print("🏀 NBA OVER/UNDER PREDICTION")
                print("="*80)
                print(f"Match: {team1} vs {team2}")
                print(f"Line: {line}")
                if home_team:
                    print(f"Home Team: {home_team}")
                print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)

            # Ensure model is trained
            if not self.pipeline.is_trained:
                if not self.train_model():
                    return False

            # Make prediction
            result = self.pipeline.predict_over_under(
                team1=team1,
                team2=team2,
                line=line,
                home_team=home_team
            )

            if verbose:
                self._print_prediction_result(result, team1, team2, line)

            return True

        except PredictionPipelineError as e:
            print(f"❌ Prediction failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during prediction: {e}")
            return False

    def _print_prediction_result(self, result, team1: str, team2: str, line: float):
        """Print detailed prediction results."""
        print(f"\n📊 PREDICTION RESULTS")
        print("=" * 50)
        print(f"Predicted Total: {result.predicted_total:.1f}")
        print(f"Confidence Interval: {result.confidence_interval[0]:.1f} - {result.confidence_interval[1]:.1f}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Under Probability: {result.under_probability:.1%}")
        print(f"Over Probability: {result.over_probability:.1%}")

        print(f"\n📈 TEAM ANALYSIS")
        print("=" * 50)

        home_data = result.team_analysis['home_team']
        away_data = result.team_analysis['away_team']

        print(f"{home_data['name']} (Home):")
        print(f"  Avg Points Scored: {home_data['avg_points_scored']:.1f}")
        print(f"  Avg Points Allowed: {home_data['avg_points_allowed']:.1f}")
        print(f"  Offensive Rating: {home_data['offensive_rating']:.1f}")
        print(f"  Defensive Rating: {home_data['defensive_rating']:.1f}")
        print(f"  Pace: {home_data['pace']:.1f}")

        print(f"{away_data['name']} (Away):")
        print(f"  Avg Points Scored: {away_data['avg_points_scored']:.1f}")
        print(f"  Avg Points Allowed: {away_data['avg_points_allowed']:.1f}")
        print(f"  Offensive Rating: {away_data['offensive_rating']:.1f}")
        print(f"  Defensive Rating: {away_data['defensive_rating']:.1f}")
        print(f"  Pace: {away_data['pace']:.1f}")

        if result.feature_importance:
            print(f"\n🔍 KEY FACTORS")
            print("=" * 50)
            # Show top 5 most important features
            top_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                print(f"  • {feature}: {importance:.3f}")

        print(f"\n🎯 BETTING RECOMMENDATION")
        print("=" * 50)
        if result.recommendation == 'OVER':
            print(f"✅ RECOMMENDATION: OVER {line}")
            print(f"💰 Predicted total: {result.predicted_total:.1f} (+{result.predicted_total - line:.1f})")
        else:
            print(f"✅ RECOMMENDATION: UNDER {line}")
            print(f"💰 Predicted total: {result.predicted_total:.1f} ({result.predicted_total - line:.1f})")

        print(f"📊 Confidence: {result.confidence:.1f}%")
        print(f"🎲 Over Probability: {result.over_probability:.1%}")
        print(f"🎲 Under Probability: {result.under_probability:.1%}")

        print(f"\n📋 MODEL INFORMATION")
        print("=" * 50)
        print(f"Model Type: VotingRegressor (RandomForest + XGBoost + GradientBoosting)")
        print(f"Data Source: Real NBA games dataset")
        print(f"Training Samples: {result.metadata['training_data_size']}")
        print(f"Model Version: {result.metadata['model_version']}")

        print(f"\n⚠️  DISCLAIMER")
        print("=" * 50)
        print("• Predictions are for informational purposes only")
        print("• Always gamble responsibly")
        print("• Past performance does not guarantee future results")
        print("=" * 80)

    def test_pipeline(self) -> bool:
        """
        Test the prediction pipeline with sample data.

        Returns:
            True if test successful, False otherwise
        """
        try:
            print("\n" + "="*80)
            print("🧪 TESTING NBA PREDICTION PIPELINE")
            print("="*80)

            # Test model training
            print("1. Testing model training...")
            if not self.train_model():
                return False

            # Test predictions with sample teams
            print("\n2. Testing sample predictions...")
            test_cases = [
                ("Los Angeles Lakers", "Boston Celtics", 225.5),
                ("Golden State Warriors", "Miami Heat", 230.0),
                ("Denver Nuggets", "Phoenix Suns", 228.5)
            ]

            for i, (team1, team2, line) in enumerate(test_cases, 1):
                print(f"\nTest Case {i}: {team1} vs {team2} (Line: {line})")
                success = self.predict_game(
                    team1=team1,
                    team2=team2,
                    line=line,
                    verbose=False
                )

                if success:
                    result = self.pipeline.predict_over_under(team1, team2, line)
                    print(f"  ✅ Prediction: {result.predicted_total:.1f} -> {result.recommendation}")
                else:
                    print(f"  ❌ Prediction failed")
                    return False

            print("\n✅ ALL TESTS PASSED!")
            print("🎉 Pipeline is ready for use!")
            return True

        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            return False

    def list_available_data(self) -> bool:
        """List available NBA data files."""
        try:
            print("\n" + "="*80)
            print("📚 AVAILABLE NBA DATA")
            print("="*80)

            data_path = Path("data")
            if not data_path.exists():
                print("❌ Data directory not found")
                return False

            # List data files
            csv_files = list(data_path.glob("*.csv"))
            parquet_files = list(data_path.glob("*.parquet"))

            print(f"📁 Data directory: {data_path.absolute()}")
            print(f"📊 CSV files: {len(csv_files)}")
            print(f"📊 Parquet files: {len(parquet_files)}")

            # Check main dataset
            main_dataset = data_path / "nba_simple_complete_dataset.csv"
            if main_dataset.exists():
                import pandas as pd
                df = pd.read_csv(main_dataset)
                print(f"\n✅ Main Dataset:")
                print(f"   • Games: {len(df)}")
                print(f"   • Seasons: {sorted(df['SEASON'].unique())}")
                print(f"   • Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
                print(f"   • Avg total score: {df['TOTAL_SCORE'].mean():.1f}")
            else:
                print("❌ Main dataset not found")

            # Check complete games
            complete_games = data_path / "game_results_2024-25_Regular_Season.parquet"
            if complete_games.exists():
                df_complete = pd.read_parquet(complete_games)
                print(f"\n✅ Complete Games Dataset:")
                print(f"   • Games: {len(df_complete)}")
                if 'game_date' in df_complete.columns:
                    print(f"   • Date range: {df_complete['game_date'].min()} to {df_complete['game_date'].max()}")
            else:
                print("⚠️ Complete games dataset not found")

            return True

        except Exception as e:
            print(f"❌ Error listing data: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Over/Under Prediction System with Real Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --team1 "Los Angeles Lakers" --team2 "Boston Celtics" --line 225.5
  %(prog)s --train-model
  %(prog)s --test-pipeline
  %(prog)s --list-data
  %(prog)s --team1 "Lakers" --team2 "Celtics" --line 225.5 --home "Celtics" --json

This system uses real NBA data and ensemble ML methods for accurate predictions.
        """
    )

    # Prediction arguments
    parser.add_argument('--team1', type=str, help='First team name')
    parser.add_argument('--team2', type=str, help='Second team name')
    parser.add_argument('--line', type=float, help='Betting line (total points)')
    parser.add_argument('--home', type=str, help='Home team name (optional)')

    # Action arguments
    parser.add_argument('--train-model', action='store_true', help='Train the prediction model')
    parser.add_argument('--test-pipeline', action='store_true', help='Test the prediction pipeline')
    parser.add_argument('--list-data', action='store_true', help='List available data files')

    # Output options
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (minimal output)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Configuration
    parser.add_argument('--data-path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models-path', type=str, default='models', help='Path to models directory')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize CLI
        cli = NBAPredictionCLI(
            data_path=args.data_path,
            models_path=args.models_path
        )

        # Handle different commands
        if args.list_data:
            return 0 if cli.list_available_data() else 1

        elif args.train_model:
            return 0 if cli.train_model() else 1

        elif args.test_pipeline:
            return 0 if cli.test_pipeline() else 1

        elif args.team1 and args.team2 and args.line is not None:
            success = cli.predict_game(
                team1=args.team1,
                team2=args.team2,
                line=args.line,
                home_team=args.home,
                verbose=not args.quiet and not args.json
            )

            if success and args.json:
                result = cli.pipeline.predict_over_under(args.team1, args.team2, args.line, args.home)
                # Convert result to JSON-serializable format
                json_result = {
                    'predicted_total': result.predicted_total,
                    'confidence_interval': result.confidence_interval,
                    'recommendation': result.recommendation,
                    'confidence': result.confidence,
                    'over_probability': result.over_probability,
                    'under_probability': result.under_probability,
                    'team_analysis': result.team_analysis,
                    'metadata': result.metadata
                }
                print(json.dumps(json_result, indent=2, default=str))

            return 0 if success else 1

        else:
            parser.print_help()
            print("\n❌ Error: Must provide either --team1, --team2, and --line for prediction")
            print("   Or use --train-model, --test-pipeline, or --list-data")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Prediction cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"CLI error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())