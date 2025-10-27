#!/usr/bin/env python3
"""
🏀 Modern NBA Prediction CLI
Context7-compliant command-line interface integrating legacy prediction system with UnifiedDataStore.

This script provides:
- Bridge between deprecated main.py functionality and modern architecture
- Over/Under predictions using ProbabilisticModel with UnifiedDataStore
- Real-time NBA data integration
- Context7-compliant dependency injection and error handling
- Team analysis and betting recommendations
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import date, datetime
from typing import Dict, Any, Optional
import logging

# Context7-compliant imports
from nba_predictor.integration.modern_prediction_system import create_prediction_system_with_data_store
from nba_predictor.core.data_store import UnifiedDataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModernNBACLIPrediction:
    """
    Modern CLI interface for NBA predictions using integrated legacy + modern system.

    Context7-compliant: Bridges deprecated main.py functionality with new architecture.
    """

    def __init__(self, data_store_path: str = None):
        """
        Initialize CLI prediction system.

        Args:
            data_store_path: Path to UnifiedDataStore directory
        """
        try:
            if data_store_path is None:
                data_store_path = "/Users/fulvioventura/nba-predictor-streamlit/data"

            logger.info(f"Initializing prediction system with data store: {data_store_path}")
            self.prediction_system = create_prediction_system_with_data_store(data_store_path)

            # Get system status
            status = self.prediction_system.get_system_status()
            logger.info(f"System status: {status['system_health']}")

            if status['system_health'] != 'healthy':
                logger.warning(f"System health is {status['system_health']}: {status}")

            self.unified_store = self.prediction_system.unified_store
            logger.info("Modern NBA CLI Prediction System initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize prediction system: {e}")
            raise

    def predict_game(self,
                    team1: str,
                    team2: str,
                    line: float,
                    season: str = "2025-26",
                    auto_mode: bool = False,
                    output_format: str = "json") -> Dict[str, Any]:
        """
        Make prediction for NBA game.

        Context7-compliant: Bridge to legacy main.py prediction functionality.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line (total points)
            season: NBA season
            auto_mode: Automatic mode (less verbose output)
            output_format: Output format ('json' or 'table')

        Returns:
            Prediction results
        """
        try:
            logger.info(f"Starting prediction: {team1} vs {team2}, line: {line}")

            # Validate teams exist in data store
            team1_id = self.prediction_system.legacy_bridge.adapter.get_team_id(team1)
            team2_id = self.prediction_system.legacy_bridge.adapter.get_team_id(team2)

            if not team1_id:
                return self._error_response(f"Team '{team1}' not found in data store")
            if not team2_id:
                return self._error_response(f"Team '{team2}' not found in data store")

            # Make prediction
            prediction_result = self.prediction_system.predict_over_under(
                team1=team1,
                team2=team2,
                line=line,
                season=season
            )

            if prediction_result.get('error'):
                return prediction_result

            # Format output based on mode
            if not auto_mode:
                self._print_detailed_output(prediction_result, team1, team2, line)
            else:
                self._print_auto_output(prediction_result)

            return prediction_result

        except Exception as e:
            logger.error(f"Error in predict_game: {e}")
            return self._error_response(f"Prediction failed: {str(e)}")

    def _print_detailed_output(self, result: Dict[str, Any], team1: str, team2: str, line: float):
        """Print detailed prediction output"""
        print(f"\n{'='*60}")
        print(f"🏀 NBA OVER/UNDER PREDICTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Match: {team1} vs {team2}")
        print(f"Line: {line}")
        print(f"Date: {result['metadata']['prediction_date']}")
        print(f"Prediction System: {result['metadata']['system_version']}")
        print(f"\n{'='*60}")
        print(f"📊 PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"Predicted Total: {result['predicted_total']:.1f}")
        print(f"Under Quote: {result['under_quote']:.2f}")
        print(f"Over Quote: {result['over_quote']:.2f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Under Probability: {result['under_probability']:.1%}")
        print(f"Over Probability: {result['over_probability']:.1%}")

        # Team analysis
        print(f"\n{'='*60}")
        print(f"📈 TEAM ANALYSIS")
        print(f"{'='*60}")

        for team_key, team_data in result['team_analysis'].items():
            team_name = team_data['name']
            print(f"\n{team_name}:")
            print(f"  Momentum Score: {team_data['momentum_score']:.3f}")
            print(f"  Win Rate: {team_data['win_rate']:.1%}")
            print(f"  Avg Points Scored: {team_data['avg_points_scored']:.1f}")
            print(f"  Injuries: {team_data['injuries_count']} players")
            if team_data['key_players_injured']:
                print(f"  Key Players Out: {', '.join(team_data['key_players_injured'])}")

        # Historical context
        if 'historical_context' in result:
            print(f"\n{'='*60}")
            print(f"📚 HISTORICAL CONTEXT")
            print(f"{'='*60}")
            ctx = result['historical_context']
            print(f"Recent Games Analyzed: {ctx['recent_games_count']}")
            print(f"Average Total Score: {ctx['avg_total_score']:.1f}")
            print(f"Score Range: {ctx['min_total']:.1f} - {ctx['max_total']:.1f}")
            print(f"Games Over {line}: {ctx['over_line_count']}")
            print(f"Games Under {line}: {ctx['under_line_count']}")

        print(f"\n{'='*60}")
        print("🎯 BETTING RECOMMENDATION")
        print(f"{'='*60}")
        if result['recommendation'] == 'Under':
            print(f"✅ RECOMMENDATION: UNDER {line}")
            print(f"💰 Quote: {result['under_quote']:.2f}")
        else:
            print(f"✅ RECOMMENDATION: OVER {line}")
            print(f"💰 Quote: {result['over_quote']:.2f}")
        print(f"📊 Confidence: {result['confidence']:.1%}")

        print(f"\n{'='*60}")
        print("⚠️  DISCLAIMER: Use predictions responsibly")
        print(f"{'='*60}\n")

    def _print_auto_output(self, result: Dict[str, Any]):
        """Print automatic mode output"""
        if result.get('error'):
            print(f"ERROR: {result['error_message']}")
        else:
            print(f"{result['recommendation']} {result['under_quote']:.2f} / {result['over_quote']:.2f}")

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'error': True,
            'error_message': error_message,
            'prediction': None,
            'under_quote': None,
            'over_quote': None,
            'recommendation': None,
            'confidence': 0.0
        }

    def list_available_teams(self) -> Dict[str, Any]:
        """
        List all available teams in data store.

        Context7-compliant: Data exploration functionality.
        """
        try:
            teams_df = self.unified_store.get_all_teams()
            if teams_df is not None and not teams_df.empty:
                teams = teams_df[['team_name', 'team_abbreviation', 'team_id']].to_dict('records')
                return {
                    'teams': teams,
                    'total_count': len(teams)
                }
            else:
                return {
                    'teams': [],
                    'total_count': 0,
                    'message': 'No teams found in data store'
                }
        except Exception as e:
            logger.error(f"Error listing teams: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'teams': [],
                'total_count': 0
            }

    def get_team_info(self, team_name: str) -> Dict[str, Any]:
        """Get detailed information about a team"""
        try:
            team_info = self.prediction_system.legacy_bridge.adapter._find_team_by_name(team_name)
            if team_info:
                return {
                    'team_info': team_info,
                    'momentum': self.prediction_system.legacy_bridge.adapter.get_team_momentum_metrics(team_name),
                    'recent_games': len(self.prediction_system.legacy_bridge.adapter.get_team_recent_games(team_name)),
                    'injuries': len(self.prediction_system.legacy_bridge.adapter.get_team_injuries(team_name))
                }
            else:
                return {
                    'error': True,
                    'error_message': f"Team '{team_name}' not found"
                }
        except Exception as e:
            logger.error(f"Error getting team info: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }


def main():
    """
    Main CLI entry point.

    Context7-compliant: Bridges deprecated main.py CLI with modern architecture.
    """
    parser = argparse.ArgumentParser(
        description="Modern NBA Over/Under Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --team1 "Utah Jazz" --team2 "Golden State Warriors" --line 225.0
  %(prog)s --team1 "Lakers" --team2 "Celtics" --line 230.5 --auto-mode
  %(prog)s --list-teams
  %(prog)s --team-info "Lakers"

This system integrates the legacy probabilistic model with the modern UnifiedDataStore
architecture, providing Context7-compliant Over/Under predictions for NBA games.
        """
    )

    parser.add_argument('--team1', type=str, help='First team name')
    parser.add_argument('--team2', type=str, help='Second team name')
    parser.add_argument('--line', type=float, help='Betting line (total points)')
    parser.add_argument('--season', type=str, default='2025-26', help='NBA season (default: 2025-26)')
    parser.add_argument('--auto-mode', action='store_true', help='Automatic mode (minimal output)')
    parser.add_argument('--list-teams', action='store_true', help='List all available teams')
    parser.add_argument('--team-info', type=str, help='Get detailed information about a team')
    parser.add_argument('--data-store', type=str, help='Path to data store directory')
    parser.add_argument('--output', type=str, choices=['json', 'table'], default='table',
                       help='Output format (default: table)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize prediction system
        cli = ModernNBACLIPrediction(data_store_path=args.data_store)

        # Handle different commands
        if args.list_teams:
            result = cli.list_available_teams()
            if args.output == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(f"\n📋 Available Teams ({result['total_count']})")
                print("=" * 50)
                for team in result['teams']:
                    print(f"{team['team_name']} ({team['team_abbreviation']}) - ID: {team['team_id']}")

        elif args.team_info:
            result = cli.get_team_info(args.team_info)
            if args.output == 'json':
                print(json.dumps(result, indent=2))
            else:
                if result.get('error'):
                    print(f"❌ Error: {result['error_message']}")
                else:
                    team = result['team_info']
                    momentum = result['momentum']
                    print(f"\n🏀 Team Information: {team['team_name']}")
                    print("=" * 50)
                    print(f"Abbreviation: {team['team_abbreviation']}")
                    print(f"Team ID: {team['team_id']}")
                    print(f"Momentum Score: {momentum['momentum_score']:.3f}")
                    print(f"Win Rate: {momentum['win_rate']:.1%}")
                    print(f"Avg Points Scored: {momentum['avg_points_scored']:.1f}")
                    print(f"Recent Games: {result['recent_games']}")
                    print(f"Current Injuries: {result['injuries']}")

        elif args.team1 and args.team2 and args.line is not None:
            result = cli.predict_game(
                team1=args.team1,
                team2=args.team2,
                line=args.line,
                season=args.season,
                auto_mode=args.auto_mode,
                output_format=args.output
            )

            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))

        else:
            parser.print_help()
            print("\n❌ Error: Must provide --team1, --team2, and --line for prediction")
            print("   Or use --list-teams or --team-info <team_name>")
            return 1

        return 0

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