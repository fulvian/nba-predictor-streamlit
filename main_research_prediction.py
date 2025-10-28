#!/usr/bin/env python3
"""
🏀 NBA Research Prediction CLI - Context7 Compliant
Command-line interface for research-based NBA prediction system.

This module provides:
- Complete research prediction pipeline CLI
- Model training with advanced ensemble methods
- Game predictions with SHAP explanations
- Performance evaluation and metrics
- Research-based feature engineering integration
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")

from src.nba_predictor.core.research_prediction_pipeline import (
    create_research_prediction_pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser with research-specific options.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="NBA Research Prediction System - Advanced analytics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --data-path ./data --models-path ./models
  %(prog)s predict --team1 "Boston Celtics" --team2 "Lakers" --line 225.5
  %(prog)s explain --team1 "Golden State Warriors" --team2 "Phoenix Suns" --line 240.0
  %(prog)s evaluate --data-path ./data --models-path ./models
        """
    )

    # Global options
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to NBA data directory (default: ./data)"
    )

    parser.add_argument(
        "--models-path",
        type=str,
        default="./models",
        help="Path to trained models directory (default: ./models)"
    )

    parser.add_argument(
        "--model-file",
        type=str,
        help="Specific model file to load (default: latest in models-path)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--use-stacked-ensemble",
        action="store_true",
        default=True,
        help="Use stacked ensemble model (default: True)"
    )

    model_group.add_argument(
        "--lightgbm-only",
        action="store_true",
        help="Use LightGBM model instead of stacked ensemble"
    )

    model_group.add_argument(
        "--enable-explainability",
        action="store_true",
        default=True,
        help="Enable SHAP model explanations (default: True)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train research prediction model"
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )
    train_parser.add_argument(
        "--save-model",
        type=str,
        help="Save trained model to specific file"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make NBA game prediction"
    )
    predict_parser.add_argument(
        "--team1",
        type=str,
        required=True,
        help="First team name"
    )
    predict_parser.add_argument(
        "--team2",
        type=str,
        required=True,
        help="Second team name"
    )
    predict_parser.add_argument(
        "--line",
        type=float,
        required=True,
        help="Over/under line for the game"
    )

    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Generate SHAP explanation for prediction"
    )
    explain_parser.add_argument(
        "--team1",
        type=str,
        required=True,
        help="First team name"
    )
    explain_parser.add_argument(
        "--team2",
        type=str,
        required=True,
        help="Second team name"
    )
    explain_parser.add_argument(
        "--line",
        type=float,
        required=True,
        help="Over/under line for the game"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance"
    )
    eval_parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data file (optional)"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display model information"
    )

    return parser


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def create_pipeline(args: argparse.Namespace) -> 'ResearchPredictionPipeline':
    """
    Create research prediction pipeline from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Configured research prediction pipeline
    """
    # Determine model configuration
    use_stacked_ensemble = args.use_stacked_ensemble and not args.lightgbm_only

    # Create pipeline
    pipeline = create_research_prediction_pipeline(
        data_path=args.data_path,
        models_path=args.models_path,
        use_stacked_ensemble=use_stacked_ensemble,
        enable_explainability=args.enable_explainability
    )

    logger.info(
        "Research pipeline created",
        extra={
            "stacked_ensemble": use_stacked_ensemble,
            "explainability": args.enable_explainability,
            "data_path": args.data_path,
            "models_path": args.models_path
        }
    )

    return pipeline


def handle_train_command(pipeline: 'ResearchPredictionPipeline', args: argparse.Namespace) -> None:
    """
    Handle model training command.

    Args:
        pipeline: Research prediction pipeline
        args: Command line arguments
    """
    logger.info("Starting model training...")

    try:
        # Train model
        metrics = pipeline.train_model(validation_split=args.validation_split)

        # Display results
        print("\n🏀 Training Complete!")
        print("=" * 50)
        print(f"✅ Model trained successfully")
        print(f"📊 Training samples: {metrics.get('train_samples', 'N/A')}")
        print(f"🔍 Validation samples: {metrics.get('val_samples', 'N/A')}")
        print(f"🎯 Features used: {metrics.get('features', 'N/A')}")

        # Display metrics
        print("\n📈 Performance Metrics:")
        print(f"   MAE: {metrics.get('mae', 'N/A'):.3f}")
        print(f"   MSE: {metrics.get('mse', 'N/A'):.3f}")
        print(f"   RMSE: {metrics.get('rmse', 'N/A'):.3f}")
        print(f"   MAPE: {metrics.get('mape', 'N/A'):.3f}")

        # Save model if requested
        if args.save_model:
            model_path = pipeline.save_model(args.save_model)
            print(f"💾 Model saved to: {model_path}")
        else:
            model_path = pipeline.save_model("research_model.pkl")
            print(f"💾 Model auto-saved to: {model_path}")

    except Exception as e:
        logger.error("Training failed", extra={"error": str(e)})
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def handle_predict_command(pipeline: 'ResearchPredictionPipeline', args: argparse.Namespace) -> None:
    """
    Handle prediction command.

    Args:
        pipeline: Research prediction pipeline
        args: Command line arguments
    """
    logger.info(
        "Making prediction",
        extra={
            "team1": args.team1,
            "team2": args.team2,
            "line": args.line
        }
    )

    try:
        # Load model if specified
        if args.model_file:
            pipeline.load_model(args.model_file)

        # Make prediction
        result = pipeline.predict(
            team1_name=args.team1,
            team2_name=args.team2,
            line=args.line
        )

        # Display results
        print("\n🏀 NBA Game Prediction")
        print("=" * 50)
        print(f"🆚 {args.team1} vs {args.team2}")
        print(f"📊 Line: {args.line}")

        print(f"\n🎯 Prediction: {result['predicted_total']:.1f}")
        print(f"📈 Recommendation: {result['recommendation']}")
        print(f"💪 Confidence: {result['confidence']:.1%}")
        print(f"📏 Difference: {result['difference']:+.1f}")

        # Display model info
        if 'model_metrics' in result:
            model_metrics = result['model_metrics']
            print(f"\n📊 Model Performance:")
            print(f"   MAE: {model_metrics.get('mae', 'N/A'):.3f}")
            print(f"   RMSE: {model_metrics.get('rmse', 'N/A'):.3f}")

    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)})
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)


def handle_explain_command(pipeline: 'ResearchPredictionPipeline', args: argparse.Namespace) -> None:
    """
    Handle explanation command.

    Args:
        pipeline: Research prediction pipeline
        args: Command line arguments
    """
    logger.info(
        "Generating explanation",
        extra={
            "team1": args.team1,
            "team2": args.team2,
            "line": args.line
        }
    )

    try:
        # Load model if specified
        if args.model_file:
            pipeline.load_model(args.model_file)

        # Generate explanation
        explanation = pipeline.explain_prediction(
            team1_name=args.team1,
            team2_name=args.team2,
            line=args.line
        )

        # Display results
        print("\n🏀 NBA Game Prediction Explanation")
        print("=" * 50)
        print(f"🆚 {args.team1} vs {args.team2}")
        print(f"📊 Line: {args.line}")

        print(f"\n🎯 Prediction: {explanation['predicted_total']:.1f}")
        print(f"📈 Recommendation: {explanation['recommendation']}")
        print(f"💪 Confidence: {explanation['confidence']:.1%}")

        # Display SHAP explanation if available
        if 'shap_explanation' in explanation:
            shap_exp = explanation['shap_explanation']
            print(f"\n🔍 SHAP Explanation:")
            print(f"   Base value: {shap_exp.get('base_value', 'N/A'):.1f}")

            if 'feature_importance' in shap_exp:
                print(f"   Top features:")
                for i, feature in enumerate(shap_exp['feature_importance'][:5], 1):
                    print(f"     {i}. {feature.get('feature', 'N/A')}: {feature.get('impact', 'N/A'):+.3f}")

    except Exception as e:
        logger.error("Explanation failed", extra={"error": str(e)})
        print(f"❌ Explanation failed: {e}")
        sys.exit(1)


def handle_evaluate_command(pipeline: 'ResearchPredictionPipeline', args: argparse.Namespace) -> None:
    """
    Handle model evaluation command.

    Args:
        pipeline: Research prediction pipeline
        args: Command line arguments
    """
    logger.info("Starting model evaluation...")

    try:
        # Load model if specified
        if args.model_file:
            pipeline.load_model(args.model_file)

        # Get model info
        info = pipeline.get_model_info()

        # Display results
        print("\n🏀 Model Evaluation")
        print("=" * 50)
        print(f"🤖 Model Type: {info.get('model_type', 'N/A')}")
        print(f"📊 Is Trained: {info.get('is_trained', 'N/A')}")
        print(f"🔢 Feature Count: {info.get('feature_columns_count', 'N/A')}")
        print(f"🎯 Stacked Ensemble: {info.get('use_stacked_ensemble', 'N/A')}")
        print(f"🔍 Explainability: {info.get('enable_explainability', 'N/A')}")

        # Display base models if available
        if 'base_models' in info:
            print(f"\n🏗️  Base Models:")
            for model in info['base_models']:
                print(f"   • {model}")

        # Display performance metrics
        if 'metrics' in info and info['metrics']:
            metrics = info['metrics']
            print(f"\n📈 Performance Metrics:")
            print(f"   MAE: {metrics.get('mae', 'N/A'):.3f}")
            print(f"   MSE: {metrics.get('mse', 'N/A'):.3f}")
            print(f"   RMSE: {metrics.get('rmse', 'N/A'):.3f}")
            print(f"   MAPE: {metrics.get('mape', 'N/A'):.3f}")

    except Exception as e:
        logger.error("Evaluation failed", extra={"error": str(e)})
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def handle_info_command(pipeline: 'ResearchPredictionPipeline', args: argparse.Namespace) -> None:
    """
    Handle model info command.

    Args:
        pipeline: Research prediction pipeline
        args: Command line arguments
    """
    try:
        # Load model if specified
        if args.model_file:
            pipeline.load_model(args.model_file)

        # Get model info
        info = pipeline.get_model_info()

        # Display JSON formatted info
        print(json.dumps(info, indent=2, default=str))

    except Exception as e:
        logger.error("Info command failed", extra={"error": str(e)})
        print(f"❌ Info command failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create directories if they don't exist
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    Path(args.models_path).mkdir(parents=True, exist_ok=True)

    try:
        # Create pipeline
        pipeline = create_pipeline(args)

        # Handle command
        if args.command == "train":
            handle_train_command(pipeline, args)
        elif args.command == "predict":
            handle_predict_command(pipeline, args)
        elif args.command == "explain":
            handle_explain_command(pipeline, args)
        elif args.command == "evaluate":
            handle_evaluate_command(pipeline, args)
        elif args.command == "info":
            handle_info_command(pipeline, args)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error("CLI error", extra={"error": str(e)})
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()