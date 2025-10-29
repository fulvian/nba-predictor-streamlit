#!/usr/bin/env python3
"""
🏀 Enhanced NBA Prediction System - Main Script
COMPLETE integration with ALL UnifiedDataStore data sources.

This script provides:
- Complete NBA Over/Under predictions using ALL available data
- Enhanced ML models with injuries, rosters, player stats, head-to-head
- Advanced feature engineering using full data integration
- Comprehensive analysis and reporting
- Command-line interface for enhanced predictions

Usage:
    python main_enhanced_prediction.py --team1 "Boston Celtics" --team2 "New Orleans Pelicans" --line 233.5
    python main_enhanced_prediction.py --train-enhanced-model
    python main_enhanced_prediction.py --test-enhanced-pipeline
    python main_enhanced_prediction.py --system-status
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from nba_predictor.core.enhanced_prediction_pipeline import EnhancedPredictionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedNBAPredictionCLI:
    """Enhanced CLI interface for NBA predictions using ALL data sources."""

    def __init__(self, data_path: str = "data", models_path: str = "models"):
        """
        Initialize the Enhanced CLI interface.

        Args:
            data_path: Path to NBA data files
            models_path: Path to model files
        """
        self.pipeline = EnhancedPredictionPipeline(
            data_path=data_path,
            model_path=models_path
        )
        logger.info("Enhanced NBA Prediction CLI initialized with FULL data integration")

    def train_enhanced_model(self) -> bool:
        """
        Train the enhanced prediction model using ALL data sources.

        Returns:
            True if training successful, False otherwise
        """
        try:
            print("\n" + "="*80)
            print("🚀 TRAINING ENHANCED NBA PREDICTION MODEL")
            print("="*80)
            print("🔥 Utilizzando TUTTE le fonti dati disponibili:")
            print("   • Statistiche base partite")
            print("   • 🏥 Injury reports completi")
            print("   • 👥 Roster e cambiamenti")
            print("   • 📈 Statistiche giocatori individuali")
            print("   • ⚔️ Head-to-head storico")
            print("   • 🎯 Player momentum e forma")
            print("="*80)

            # Try to load existing enhanced model first
            if self.pipeline.load_enhanced_model():
                print("✅ Enhanced model loaded successfully")
                print(f"📊 Model info: {self.pipeline.metrics}")
                return True

            print("📚 No existing enhanced model found - training new model...")
            print("📈 Loading ALL integrated data sources...")

            metrics = self.pipeline.train_enhanced_model()

            print("\n✅ ENHANCED MODEL TRAINING COMPLETED!")
            print("📊 Enhanced Performance Metrics:")
            print(f"   • Mean Absolute Error: {metrics['mae']:.2f} points")
            print(f"   • Root Mean Squared Error: {metrics['rmse']:.2f} points")
            print(f"   • R² Score: {metrics['r2_score']:.3f}")
            print(f"   • Cross-validation MAE: {metrics['cv_mae_mean']:.2f} ± {metrics['cv_mae_std']:.2f}")
            print(f"   • Training samples: {metrics['training_samples']}")
            print(f"   • Features engineered: {metrics['feature_count']}")
            print(f"   • Data sources used: {len(metrics['data_sources_used'])}")
            print(f"   • Training date: {metrics['training_date']}")

            print("\n📋 Data Sources Integrated:")
            for source in metrics['data_sources_used']:
                print(f"   • ✅ {source}")

            return True

        except Exception as e:
            print(f"❌ Enhanced model training failed: {e}")
            return False

    def predict_with_all_data(
        self,
        team1: str,
        team2: str,
        line: float,
        home_team: str = None,
        verbose: bool = True
    ) -> bool:
        """
        Make enhanced prediction using ALL available data sources.

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
                print("🏀 ENHANCED NBA PREDICTION WITH ALL DATA SOURCES")
                print("="*80)
                print(f"Match: {team1} vs {team2}")
                print(f"Line: {line}")
                if home_team:
                    print(f"Home Team: {home_team}")
                print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("🔥 Using: Injuries, Rosters, Player Stats, Head-to-Head, Momentum")
                print("-" * 80)

            # Ensure enhanced model is trained
            if not self.pipeline.is_trained:
                if not self.train_enhanced_model():
                    return False

            # Make enhanced prediction
            result = self.pipeline.predict_with_all_data(
                team1=team1,
                team2=team2,
                line=line,
                home_team=home_team
            )

            if verbose:
                self._print_enhanced_prediction_result(result, team1, team2, line)

            return True

        except Exception as e:
            print(f"❌ Enhanced prediction failed: {e}")
            return False

    def _print_enhanced_prediction_result(self, result, team1: str, team2: str, line: float):
        """Print detailed enhanced prediction results."""
        print(f"\n📊 ENHANCED PREDICTION RESULTS")
        print("=" * 50)
        print(f"Predicted Total: {result.predicted_total:.1f}")
        print(f"Confidence Interval: {result.confidence_interval[0]:.1f} - {result.confidence_interval[1]:.1f}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Under Probability: {result.under_probability:.1%}")
        print(f"Over Probability: {result.over_probability:.1%}")

        # Enhanced data analysis
        print(f"\n🏥 INJURY IMPACT ANALYSIS")
        print("=" * 50)
        injury = result.injury_impact
        print(f"Team 1 Injuries: {injury['team1_injuries']['count']} players ({injury['team1_injuries']['impact_level']} impact)")
        if injury['team1_injuries']['key_players']:
            print(f"  Key players out: {', '.join(injury['team1_injuries']['key_players'][:3])}")

        print(f"Team 2 Injuries: {injury['team2_injuries']['count']} players ({injury['team2_injuries']['impact_level']} impact)")
        if injury['team2_injuries']['key_players']:
            print(f"  Key players out: {', '.join(injury['team2_injuries']['key_players'][:3])}")

        print(f"Overall Assessment: {injury['overall_assessment']}")

        print(f"\n👥 ROSTER STABILITY")
        print("=" * 50)
        roster = result.roster_changes
        print(f"Team 1: {roster['team1_stability']} (Turnover: {roster['roster_turnover']['team1']})")
        print(f"Team 2: {roster['team2_stability']} (Turnover: {roster['roster_turnover']['team2']})")
        print(f"Overall: {roster['overall_stability']}")

        print(f"\n📈 PLAYER MOMENTUM")
        print("=" * 50)
        momentum = result.player_momentum
        print(f"Team 1 Momentum: {momentum['team1_momentum']['rating']} (Avg: {momentum['team1_momentum']['avg_production']:.1f})")
        if momentum['team1_momentum']['key_performers']:
            print(f"  Key performers: {', '.join(momentum['team1_momentum']['key_performers'][:2])}")

        print(f"Team 2 Momentum: {momentum['team2_momentum']['rating']} (Avg: {momentum['team2_momentum']['avg_production']:.1f})")
        if momentum['team2_momentum']['key_performers']:
            print(f"  Key performers: {', '.join(momentum['team2_momentum']['key_performers'][:2])}")

        print(f"Momentum Edge: {momentum['momentum_edge']}")

        print(f"\n⚔️ HEAD-TO-HEAD ANALYSIS")
        print("=" * 50)
        h2h = result.head_to_head_analysis
        print(f"Recent Meetings: {h2h['recent_meetings']['count']} games")
        if h2h['recent_meetings']['count'] > 0:
            print(f"  Team 1 Wins: {h2h['recent_meetings']['team1_wins']}")
            print(f"  Team 2 Wins: {h2h['recent_meetings']['team2_wins']}")
            print(f"  Avg Total Points: {h2h['avg_total_points']:.1f}")
            print(f"  Recent Trend: {h2h['trend']}")
            print(f"  Scoring Patterns: {h2h['patterns']}")

        # Enhanced feature importance
        if result.feature_importance:
            print(f"\n🔍 TOP ENHANCED FEATURES")
            print("=" * 50)
            # Show top 7 most important features
            top_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:7]
            for feature, importance in top_features:
                print(f"  • {feature}: {importance:.3f}")

        print(f"\n🎯 ENHANCED BETTING RECOMMENDATION")
        print("=" * 50)
        if result.recommendation == 'OVER':
            print(f"✅ RECOMMENDATION: OVER {line}")
            print(f"💰 Predicted total: {result.predicted_total:.1f} (+{result.predicted_total - line:.1f})")
        else:
            print(f"✅ RECOMMENDATION: UNDER {line}")
            print(f"💰 Predicted total: {result.predicted_total:.1f} ({result.predicted_total - line:.1f})")

        print(f"📊 Enhanced Confidence: {result.confidence:.1f}%")
        print(f"🎲 Over Probability: {result.over_probability:.1%}")
        print(f"🎲 Under Probability: {result.under_probability:.1%}")

        print(f"\n📋 ENHANCED MODEL INFORMATION")
        print("=" * 50)
        print(f"Model Type: Enhanced Ensemble (RF + XGB + GB)")
        print(f"Data Sources: {result.metadata['data_sources']} integrated")
        print(f"Features Used: {result.metadata['features_used']} engineered")
        print(f"Training Samples: {result.metadata['training_samples']}")
        print(f"Model Version: {result.metadata['model_version']}")

        print(f"\n⚠️  DISCLAIMER: Enhanced predictions use comprehensive data analysis")
        print("   • Always gamble responsibly")
        print("   • All data sources integrated for maximum accuracy")
        print("=" * 80)

    def test_enhanced_pipeline(self) -> bool:
        """
        Test the enhanced prediction pipeline with sample data.

        Returns:
            True if test successful, False otherwise
        """
        try:
            print("\n" + "="*80)
            print("🧪 TESTING ENHANCED NBA PREDICTION PIPELINE")
            print("="*80)

            # 1. Test system status
            print("1. Testing enhanced system status...")
            status = self.pipeline.get_enhanced_system_status()
            print(f"   System Type: {status['system_type']}")
            print(f"   Data Sources Available: {status['total_sources']}/6")
            print(f"   System Health: {status['system_health']}")

            if status['system_health'] != 'healthy':
                print("   ⚠️  Warning: Some data sources may be missing")

            for source, available in status['data_sources_available'].items():
                status_icon = "✅" if available else "❌"
                print(f"   {status_icon} {source}")

            # 2. Test enhanced model training
            print("\n2. Testing enhanced model training...")
            if not self.train_enhanced_model():
                return False

            # 3. Test enhanced predictions with sample teams
            print("\n3. Testing enhanced sample predictions...")
            test_cases = [
                ("Boston Celtics", "New Orleans Pelicans", 233.5),
                ("Los Angeles Lakers", "Portland Trail Blazers", 225.0),
                ("Golden State Warriors", "Memphis Grizzlies", 230.5)
            ]

            for i, (team1, team2, line) in enumerate(test_cases, 1):
                print(f"\nEnhanced Test Case {i}: {team1} vs {team2} (Line: {line})")
                success = self.predict_with_all_data(
                    team1=team1,
                    team2=team2,
                    line=line,
                    verbose=False
                )

                if success:
                    result = self.pipeline.predict_with_all_data(team1, team2, line)
                    print(f"  ✅ Enhanced Prediction: {result.predicted_total:.1f} -> {result.recommendation}")
                    print(f"  📊 Data sources used: {result.metadata['data_sources']}")
                    print(f"  🔥 Features analyzed: {result.metadata['features_used']}")
                else:
                    print(f"  ❌ Enhanced prediction failed")
                    return False

            print("\n✅ ALL ENHANCED TESTS PASSED!")
            print("🎉 Enhanced Pipeline is ready for production use!")
            return True

        except Exception as e:
            print(f"❌ Enhanced pipeline test failed: {e}")
            return False

    def show_system_status(self) -> bool:
        """Display comprehensive enhanced system status."""
        try:
            print("\n" + "="*80)
            print("📊 ENHANCED NBA PREDICTION SYSTEM STATUS")
            print("="*80)

            status = self.pipeline.get_enhanced_system_status()

            print(f"🔥 System Type: {status['system_type']}")
            print(f"📈 System Health: {status['system_health'].upper()}")
            print(f"🧠 Model Trained: {'✅ Yes' if status['model_trained'] else '❌ No'}")
            print(f"📊 Features Available: {status['feature_count']}")

            if status['model_trained']:
                print(f"📅 Last Training: {status['last_training']}")

            print(f"\n📁 Data Sources Integration Status:")
            print("=" * 50)

            for source, available in status['data_sources_available'].items():
                status_icon = "✅" if available else "❌"
                status_text = "Available" if available else "Missing"
                print(f"   {status_icon} {source}: {status_text}")

            print(f"\n📈 Integration Summary:")
            print(f"   • Total Sources: {status['total_sources']}/6")
            print(f"   • Model Version: {status['model_version']}")

            if status['total_sources'] >= 5:
                print("   🎉 System: FULLY INTEGRATED - All data sources available")
            elif status['total_sources'] >= 3:
                print("   ⚠️  System: Partially integrated - Some data sources missing")
            else:
                print("   ❌ System: Limited integration - Most data sources missing")

            if 'error' in status:
                print(f"\n❌ System Error: {status['error']}")

            print("="*80)
            return True

        except Exception as e:
            print(f"❌ Error showing system status: {e}")
            return False

    def compare_with_basic(self, team1: str, team2: str, line: float) -> bool:
        """Compare enhanced prediction with basic prediction."""
        try:
            print("\n" + "="*80)
            print("🔄 ENHANCED VS BASIC PREDICTION COMPARISON")
            print("="*80)

            # Import basic pipeline for comparison
            from nba_predictor.core.prediction_pipeline import NBAPredictionPipeline

            basic_pipeline = NBAPredictionPipeline()

            # Get basic prediction
            print("1. Getting Basic Prediction (base statistics only)...")
            basic_success = basic_pipeline.train_model()

            if basic_success:
                basic_result = basic_pipeline.predict_over_under(team1, team2, line)
                print(f"   ✅ Basic Prediction: {basic_result.predicted_total:.1f} -> {basic_result.recommendation}")
                print(f"   📊 Basic Confidence: {basic_result.confidence:.1f}%")
            else:
                print("   ❌ Basic prediction failed")
                return False

            # Get enhanced prediction
            print("\n2. Getting Enhanced Prediction (ALL data sources)...")
            enhanced_success = self.predict_with_all_data(team1, team2, line, verbose=False)

            if enhanced_success:
                enhanced_result = self.pipeline.predict_with_all_data(team1, team2, line)
                print(f"   ✅ Enhanced Prediction: {enhanced_result.predicted_total:.1f} -> {enhanced_result.recommendation}")
                print(f"   📊 Enhanced Confidence: {enhanced_result.confidence:.1f}%")
                print(f"   🔥 Data Sources Used: {enhanced_result.metadata['data_sources']}")
            else:
                print("   ❌ Enhanced prediction failed")
                return False

            # Comparison analysis
            if basic_success and enhanced_success:
                print(f"\n📊 COMPARISON ANALYSIS")
                print("=" * 50)

                basic_total = basic_result.predicted_total
                enhanced_total = enhanced_result.predicted_total
                difference = abs(enhanced_total - basic_total)

                print(f"Prediction Difference: {difference:.1f} points")

                if basic_result.recommendation == enhanced_result.recommendation:
                    print(f"Agreement: ✅ Both recommend {enhanced_result.recommendation}")
                else:
                    print(f"Agreement: ❌ Different recommendations")
                    print(f"   • Basic: {basic_result.recommendation}")
                    print(f"   • Enhanced: {enhanced_result.recommendation}")

                print(f"\nEnhanced Model Advantages:")
                print(f"   • 🏥 Injury impact analysis")
                print(f"   • 👥 Roster stability consideration")
                print(f"   • 📈 Player momentum analysis")
                print(f"   • ⚔️ Head-to-head patterns")
                print(f"   • 🎯 Advanced context features")

            return True

        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced NBA Over/Under Prediction System with FULL Data Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --team1 "Boston Celtics" --team2 "New Orleans Pelicans" --line 233.5
  %(prog)s --train-enhanced-model
  %(prog)s --test-enhanced-pipeline
  %(prog)s --system-status
  %(prog)s --team1 "Lakers" --team2 "Celtics" --line 230.5 --home "Celtics" --compare
  %(prog)s --team1 "Warriors" --team2 "Grizzlies" --line 225.0 --json

This enhanced system integrates ALL available data sources:
• 🏥 Injury reports and player availability
• 👥 Team rosters and recent changes
• 📈 Individual player statistics and momentum
• ⚔️ Head-to-head historical patterns
• 🎯 Advanced context and situational factors
• 📊 Comprehensive feature engineering
        """
    )

    # Prediction arguments
    parser.add_argument('--team1', type=str, help='First team name')
    parser.add_argument('--team2', type=str, help='Second team name')
    parser.add_argument('--line', type=float, help='Betting line (total points)')
    parser.add_argument('--home', type=str, help='Home team name (optional)')

    # Enhanced action arguments
    parser.add_argument('--train-enhanced-model', action='store_true', help='Train the enhanced prediction model')
    parser.add_argument('--test-enhanced-pipeline', action='store_true', help='Test the enhanced prediction pipeline')
    parser.add_argument('--system-status', action='store_true', help='Show enhanced system status')
    parser.add_argument('--compare', action='store_true', help='Compare enhanced vs basic prediction')

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
        # Initialize enhanced CLI
        cli = EnhancedNBAPredictionCLI(
            data_path=args.data_path,
            models_path=args.models_path
        )

        # Handle different commands
        if args.system_status:
            return 0 if cli.show_system_status() else 1

        elif args.train_enhanced_model:
            return 0 if cli.train_enhanced_model() else 1

        elif args.test_enhanced_pipeline:
            return 0 if cli.test_enhanced_pipeline() else 1

        elif args.team1 and args.team2 and args.line is not None:
            if args.compare:
                success = cli.compare_with_basic(args.team1, args.team2, args.line)
            else:
                success = cli.predict_with_all_data(
                    team1=args.team1,
                    team2=args.team2,
                    line=args.line,
                    home_team=args.home,
                    verbose=not args.quiet and not args.json
                )

            if success and args.json:
                result = cli.pipeline.predict_with_all_data(args.team1, args.team2, args.line, args.home)
                # Convert enhanced result to JSON-serializable format
                json_result = {
                    'predicted_total': result.predicted_total,
                    'confidence_interval': result.confidence_interval,
                    'recommendation': result.recommendation,
                    'confidence': result.confidence,
                    'over_probability': result.over_probability,
                    'under_probability': result.under_probability,
                    'injury_impact': result.injury_impact,
                    'roster_changes': result.roster_changes,
                    'player_momentum': result.player_momentum,
                    'head_to_head_analysis': result.head_to_head_analysis,
                    'feature_importance': result.feature_importance,
                    'metadata': result.metadata
                }
                print(json.dumps(json_result, indent=2, default=str))

            return 0 if success else 1

        else:
            parser.print_help()
            print("\n❌ Error: Must provide either --team1, --team2, and --line for prediction")
            print("   Or use --train-enhanced-model, --test-enhanced-pipeline, or --system-status")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Enhanced prediction cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Enhanced CLI error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())