#!/usr/bin/env python3
"""
🏀 Unified ML Interface - Single Point of Entry for NBA Predictions
This module provides a unified interface for all ML operations in the NBA Predictor system.

Phase 2.1 Implementation: Enforce UnifiedHybridPipeline everywhere
- Single ML interface for all components
- Standardized prediction method
- Unified model management
- Consistent error handling
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .unified_hybrid_pipeline import UnifiedHybridPipeline, UnifiedPredictionResult
from ..ensemble.nba_ensemble_predictor import NBAEnsemblePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedMLInterface:
    """
    Unified ML Interface - Single point of entry for all NBA prediction operations.

    This class implements the unified interface pattern to consolidate all ML operations
    behind a single, consistent API. It enforces the use of UnifiedHybridPipeline
    throughout the system while maintaining compatibility with existing components.

    Key Features:
    - Single prediction interface for all components
    - Automatic model training when needed
    - Unified error handling and logging
    - Consistent result format across all operations
    - Integration with both UnifiedHybridPipeline and NBAEnsemblePredictor
    """

    def __init__(
        self,
        data_path: str = "data",
        model_path: str = "models",
        use_stacked_ensemble: bool = True,
        enable_explainability: bool = True,
        validate_realism: bool = True,
    ) -> None:
        """
        Initialize the unified ML interface.

        Args:
            data_path: Path to NBA data files
            model_path: Path to save/load trained models
            use_stacked_ensemble: Whether to use advanced stacked ensemble
            enable_explainability: Whether to enable SHAP explanations
            validate_realism: Whether to validate prediction realism
        """
        self.data_path = data_path
        self.model_path = model_path

        # Initialize the primary pipeline (UnifiedHybridPipeline)
        self.pipeline = UnifiedHybridPipeline(
            data_path=data_path,
            model_path=model_path,
            use_stacked_ensemble=use_stacked_ensemble,
            enable_explainability=enable_explainability,
            validate_realism=validate_realism,
        )

        # Initialize the ensemble predictor for compatibility
        self.ensemble = NBAEnsemblePredictor()

        # Interface state
        self.is_initialized = True
        self.last_prediction_time = None
        self.prediction_count = 0

        logger.info(
            "🎯 UNIFIED ML INTERFACE INITIALIZED",
            extra={
                "data_path": data_path,
                "model_path": model_path,
                "use_stacked_ensemble": use_stacked_ensemble,
                "enable_explainability": enable_explainability,
                "validate_realism": validate_realism,
            },
        )

    def predict(
        self,
        team1: str,
        team2: str,
        line: float,
        home_team: Optional[str] = None,
        validate_prediction: bool = True,
        use_ensemble_fallback: bool = True,
    ) -> UnifiedPredictionResult:
        """
        Unified prediction method - single entry point for all predictions.

        This method implements the core unified interface pattern, providing
        a single, consistent way to make predictions across all components.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line (total points)
            home_team: Which team is playing at home
            validate_prediction: Whether to validate prediction realism
            use_ensemble_fallback: Whether to use ensemble as fallback

        Returns:
            UnifiedPredictionResult with comprehensive analysis

        Raises:
            ValueError: If prediction fails
        """

    def predict_unified(
        self,
        team1: str,
        team2: str,
        line: float,
        home_team: Optional[str] = None,
        validate_prediction: bool = True,
        use_ensemble_fallback: bool = True,
    ) -> UnifiedPredictionResult:
        """
        Alias for predict method to maintain backward compatibility.
        
        This method provides the same functionality as predict() but with the
        method name expected by the SystemValidator.
        """
        return self.predict(team1, team2, line, home_team, validate_prediction, use_ensemble_fallback)
        """
        Unified prediction method - single entry point for all predictions.

        This method implements the core unified interface pattern, providing
        a single, consistent way to make predictions across all components.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line (total points)
            home_team: Which team is playing at home
            validate_prediction: Whether to validate prediction realism
            use_ensemble_fallback: Whether to use ensemble as fallback

        Returns:
            UnifiedPredictionResult with comprehensive analysis

        Raises:
            ValueError: If prediction fails
        """
        try:
            self.prediction_count += 1
            from datetime import datetime

            self.last_prediction_time = datetime.now()

            logger.info(
                f"🎯 UNIFIED PREDICTION REQUEST #{self.prediction_count}: {team1} vs {team2}, line: {line}",
                extra={
                    "home_team": home_team,
                    "prediction_count": self.prediction_count,
                    "interface": "UnifiedMLInterface",
                },
            )

            # Primary prediction using UnifiedHybridPipeline
            try:
                result = self.pipeline.predict_unified(
                    team1=team1,
                    team2=team2,
                    line=line,
                    home_team=home_team,
                    validate_prediction=validate_prediction,
                )

                logger.info(
                    f"✅ UNIFIED PREDICTION SUCCESS: {result.predicted_total:.1f} vs {line} ({result.recommendation})",
                    extra={
                        "confidence": f"{result.confidence:.1f}%",
                        "over_probability": f"{result.over_probability:.1%}",
                        "under_probability": f"{result.under_probability:.1%}",
                        "pipeline": "UnifiedHybridPipeline",
                    },
                )

                return result

            except Exception as pipeline_error:
                logger.warning(f"⚠️ UnifiedHybridPipeline failed: {pipeline_error}")

                # Fallback to ensemble if enabled
                if use_ensemble_fallback:
                    try:
                        logger.info("🔄 Attempting ensemble fallback prediction...")
                        ensemble_result = self._ensemble_predict_fallback(
                            team1, team2, line, home_team
                        )

                        logger.info(
                            f"✅ ENSEMBLE FALLBACK SUCCESS: {ensemble_result.predicted_total:.1f} vs {line} ({ensemble_result.recommendation})",
                            extra={
                                "confidence": f"{ensemble_result.confidence:.1f}%",
                                "pipeline": "NBAEnsemblePredictor (fallback)",
                            },
                        )

                        return ensemble_result

                    except Exception as ensemble_error:
                        logger.error(
                            f"❌ Ensemble fallback also failed: {ensemble_error}"
                        )
                        raise ValueError(
                            f"All prediction methods failed. Pipeline: {pipeline_error}, Ensemble: {ensemble_error}"
                        )
                else:
                    raise ValueError(
                        f"UnifiedHybridPipeline prediction failed: {pipeline_error}"
                    )

        except Exception as e:
            logger.error(f"❌ Unified prediction failed: {e}")
            raise ValueError(f"Failed to make unified prediction: {e}")

    def _ensemble_predict_fallback(
        self, team1: str, team2: str, line: float, home_team: Optional[str] = None
    ) -> UnifiedPredictionResult:
        """
        Fallback prediction using NBAEnsemblePredictor.

        This method provides compatibility with the existing ensemble predictor
        while maintaining the unified interface contract.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line
            home_team: Home team

        Returns:
            UnifiedPredictionResult converted from ensemble prediction
        """
        try:
            # Use ensemble predictor
            ensemble_prediction = self.ensemble.predict_game(
                team1, team2, line, home_team
            )

            # Convert to unified result format
            unified_result = UnifiedPredictionResult(
                predicted_total=ensemble_prediction.get("predicted_total", line),
                confidence_interval=ensemble_prediction.get(
                    "confidence_interval", (line - 10, line + 10)
                ),
                recommendation=ensemble_prediction.get("recommendation", "HOLD"),
                confidence=ensemble_prediction.get("confidence", 50.0),
                over_probability=ensemble_prediction.get("over_probability", 0.5),
                under_probability=ensemble_prediction.get("under_probability", 0.5),
                # Enhanced analyses (placeholder values for ensemble fallback)
                injury_impact={
                    "status": "ensemble_fallback",
                    "analysis": "Limited injury analysis in fallback mode",
                },
                roster_changes={
                    "status": "ensemble_fallback",
                    "analysis": "Limited roster analysis in fallback mode",
                },
                player_momentum={
                    "status": "ensemble_fallback",
                    "analysis": "Limited momentum analysis in fallback mode",
                },
                head_to_head_analysis={
                    "status": "ensemble_fallback",
                    "analysis": "Limited H2H analysis in fallback mode",
                },
                # Research analyses (placeholder values)
                shap_explanation={
                    "status": "ensemble_fallback",
                    "explanation": "SHAP not available in ensemble fallback",
                },
                feature_importance={
                    "status": "ensemble_fallback",
                    "importance": "Feature importance not available in fallback mode",
                },
                model_performance={
                    "status": "ensemble_fallback",
                    "metrics": "Limited performance metrics in fallback mode",
                },
                four_factors_analysis={
                    "status": "ensemble_fallback",
                    "analysis": "Limited Four Factors analysis in fallback mode",
                },
                # System metadata
                model_weights={"ensemble_predictor": 1.0},
                team_analysis={"team1": team1, "team2": team2, "home_team": home_team},
                prediction_metadata={
                    "prediction_date": self.last_prediction_time.isoformat()
                    if self.last_prediction_time
                    else None,
                    "line": line,
                    "teams": f"{team1} vs {team2}",
                    "home_team": home_team,
                    "system_type": "NBAEnsemblePredictor (fallback)",
                    "fallback_reason": "UnifiedHybridPipeline failed",
                    "prediction_count": self.prediction_count,
                },
            )

            return unified_result

        except Exception as e:
            logger.error(f"❌ Ensemble fallback prediction failed: {e}")
            raise ValueError(f"Ensemble fallback prediction failed: {e}")

    def train_models(self, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the unified model.

        Args:
            validation_split: Fraction of data for validation

        Returns:
            Training metrics
        """
        try:
            logger.info("🚀 TRAINING UNIFIED MODEL...")

            # Train the primary pipeline
            metrics = self.pipeline.train_unified_model(
                validation_split=validation_split
            )

            logger.info(
                "✅ UNIFIED MODEL TRAINING COMPLETED",
                extra={
                    "mae": f"{metrics.get('mae', 0):.2f}",
                    "r2_score": f"{metrics.get('r2_score', 0):.3f}",
                    "train_samples": metrics.get("train_samples", 0),
                    "features": metrics.get("features", 0),
                },
            )

            return metrics

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise ValueError(f"Failed to train unified model: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            System status dictionary
        """
        try:
            # Get pipeline status
            pipeline_status = self.pipeline.get_unified_system_status()

            # Add interface-specific status
            interface_status = {
                "interface_type": "UnifiedMLInterface",
                "interface_version": "1.0",
                "prediction_count": self.prediction_count,
                "last_prediction_time": self.last_prediction_time.isoformat()
                if self.last_prediction_time
                else None,
                "is_initialized": self.is_initialized,
                "ensemble_available": True,
                "fallback_enabled": True,
            }

            # Combine statuses
            combined_status = {
                **pipeline_status,
                **interface_status,
                "overall_status": "healthy"
                if pipeline_status.get("system_health") == "healthy"
                else "degraded",
            }

            return combined_status

        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {
                "interface_type": "UnifiedMLInterface",
                "overall_status": "error",
                "error": str(e),
            }

    def validate_teams(self, team1: str, team2: str) -> bool:
        """
        Validate team names against known teams.

        Args:
            team1: First team name
            team2: Second team name

        Returns:
            True if both teams are valid
        """
        try:
            # Check against pipeline's team mappings
            valid_teams = set(self.pipeline.team_name_to_id.keys())

            # Add common team name variations
            team_variations = {
                "Boston Celtics": ["Celtics", "Boston"],
                "Los Angeles Lakers": ["Lakers", "LA Lakers"],
                "Golden State Warriors": ["Warriors", "GSW"],
                "Brooklyn Nets": ["Nets", "Brooklyn"],
                "New York Knicks": ["Knicks", "NY Knicks"],
                "Philadelphia 76ers": ["76ers", "Sixers", "Philadelphia"],
                "Milwaukee Bucks": ["Bucks", "Milwaukee"],
                "Phoenix Suns": ["Suns", "Phoenix"],
                "Denver Nuggets": ["Nuggets", "Denver"],
                "Miami Heat": ["Heat", "Miami"],
                "Dallas Mavericks": ["Mavericks", "Mavs", "Dallas"],
                "Los Angeles Clippers": ["Clippers", "LA Clippers"],
                "Memphis Grizzlies": ["Grizzlies", "Memphis"],
                "Sacramento Kings": ["Kings", "Sacramento"],
                "Cleveland Cavaliers": ["Cavaliers", "Cavs", "Cleveland"],
                "Atlanta Hawks": ["Hawks", "Atlanta"],
                "Charlotte Hornets": ["Hornets", "Charlotte"],
                "Indiana Pacers": ["Pacers", "Indiana"],
                "Detroit Pistons": ["Pistons", "Detroit"],
                "Orlando Magic": ["Magic", "Orlando"],
                "Washington Wizards": ["Wizards", "Washington"],
                "Toronto Raptors": ["Raptors", "Toronto"],
                "Chicago Bulls": ["Bulls", "Chicago"],
                "Minnesota Timberwolves": ["Timberwolves", "Minnesota"],
                "Oklahoma City Thunder": ["Thunder", "OKC"],
                "Portland Trail Blazers": ["Trail Blazers", "Portland"],
                "Utah Jazz": ["Jazz", "Utah"],
                "New Orleans Pelicans": ["Pelicans", "New Orleans"],
                "San Antonio Spurs": ["Spurs", "San Antonio"],
                "Houston Rockets": ["Rockets", "Houston"],
            }

            # Expand valid teams with variations
            for canonical_name, variations in team_variations.items():
                if canonical_name in valid_teams:
                    valid_teams.update(variations)

            # Normalize team names
            def normalize_team(name: str) -> str:
                return name.strip().title()

            team1_norm = normalize_team(team1)
            team2_norm = normalize_team(team2)

            # Check validity
            team1_valid = team1_norm in valid_teams or any(
                team1_norm.lower() in v.lower() for v in valid_teams
            )
            team2_valid = team2_norm in valid_teams or any(
                team2_norm.lower() in v.lower() for v in valid_teams
            )

            if team1_valid and team2_valid:
                logger.info(f"✅ Team validation passed: {team1} vs {team2}")
                return True
            else:
                logger.warning(
                    f"⚠️ Team validation failed: {team1} (valid: {team1_valid}) vs {team2} (valid: {team2_valid})"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Error validating teams: {e}")
            return False

    def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent prediction history (placeholder for future implementation).

        Args:
            limit: Maximum number of predictions to return

        Returns:
            List of prediction records
        """
        # This would be implemented with actual prediction history storage
        # For now, return placeholder information
        return [
            {
                "prediction_id": i,
                "timestamp": self.last_prediction_time.isoformat()
                if self.last_prediction_time
                else None,
                "teams": "Sample prediction",
                "line": 225.0,
                "predicted_total": 227.5,
                "recommendation": "OVER",
                "confidence": 65.0,
            }
            for i in range(min(limit, self.prediction_count))
        ]


# Global instance for easy access
_unified_ml_interface = None


def get_unified_ml_interface(
    data_path: str = "data",
    model_path: str = "models",
    use_stacked_ensemble: bool = True,
    enable_explainability: bool = True,
    validate_realism: bool = True,
) -> UnifiedMLInterface:
    """
    Get or create the global unified ML interface instance.

    This function implements the singleton pattern for the unified ML interface,
    ensuring consistent usage across the entire application.

    Args:
        data_path: Path to NBA data files
        model_path: Path to save/load trained models
        use_stacked_ensemble: Whether to use advanced stacked ensemble
        enable_explainability: Whether to enable SHAP explanations
        validate_realism: Whether to validate prediction realism

    Returns:
        UnifiedMLInterface instance
    """
    global _unified_ml_interface

    if _unified_ml_interface is None:
        _unified_ml_interface = UnifiedMLInterface(
            data_path=data_path,
            model_path=model_path,
            use_stacked_ensemble=use_stacked_ensemble,
            enable_explainability=enable_explainability,
            validate_realism=validate_realism,
        )

    return _unified_ml_interface


def predict_nba_game(
    team1: str, team2: str, line: float, home_team: Optional[str] = None
) -> UnifiedPredictionResult:
    """
    Convenience function for making NBA predictions.

    This function provides the simplest possible interface for making predictions,
    automatically handling initialization and error handling.

    Args:
        team1: First team name
        team2: Second team name
        line: Betting line (total points)
        home_team: Which team is playing at home

    Returns:
        UnifiedPredictionResult with comprehensive analysis

    Raises:
        ValueError: If prediction fails
    """
    try:
        # Get the unified interface
        interface = get_unified_ml_interface()

        # Make prediction
        result = interface.predict(team1, team2, line, home_team)

        return result

    except Exception as e:
        logger.error(f"❌ Convenience prediction failed: {e}")
        raise ValueError(f"Failed to make NBA prediction: {e}")
