#!/usr/bin/env python3
"""
🏀 ENHANCED PREDICTION BRIDGE PROFESSIONAL - Bridge NBA con Previsioni Avanzate

Integra il motore di previsione avanzato nel sistema Streamlit:
1. ✅ Usa Advanced NBA Prediction Engine con team metrics
2. ✅ Confidence intervals e probability distributions
3. ✅ Situational factors analysis
4. ✅ Professional betting recommendations
5. ✅ Statistical validation e calibration
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)
# Add the predictive system directory to path
project_root = Path(__file__).resolve().parents[4]
predictive_system_path = project_root / "nba_predictive_system"
if str(predictive_system_path) not in sys.path:
    sys.path.append(str(predictive_system_path))

try:
    from advanced_nba_prediction_engine import (
        predict_nba_game_advanced,
        get_advanced_prediction_engine,
        PredictionResult,
    )
except ImportError as e:
    # Fallback se il modulo non è trovato
    logger.warning(f"⚠️ Advanced prediction engine not found: {e}, using fallback")
    predict_nba_game_advanced = None
    get_advanced_prediction_engine = None
    PredictionResult = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPredictionBridgeProfessional:
    """Bridge professionale per previsioni NBA avanzate."""

    def __init__(self):
        """Inizializza il bridge professionale."""
        logger.info("🚀 Initializing Enhanced NBA Prediction Bridge Professional...")

        try:
            if get_advanced_prediction_engine:
                self.prediction_engine = get_advanced_prediction_engine()
            else:
                logger.warning(
                    "⚠️ Prediction engine function is None, using fallback mode"
                )
                self.prediction_engine = None

            self.historical_data = self._load_historical_data()
            self.calibration_data = {}

            logger.info(
                "✅ Enhanced Prediction Bridge Professional initialized successfully"
            )

        except Exception as e:
            logger.error(f"❌ Error initializing professional bridge: {e}")
            self.prediction_engine = None
            self.historical_data = None

    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """Carica dati storici per calibration."""
        try:
            df = pd.read_csv("data/nba_data_with_mu_sigma_for_ml.csv", low_memory=False)

            # Filtra solo partite valide
            valid_games = df[
                df["TOTAL_SCORE"].notna()
                & (df["TOTAL_SCORE"] > 0)
                & (df["TOTAL_SCORE"] < 400)  # Remove outliers
            ].copy()

            if "GAME_DATE_EST" in valid_games.columns:
                valid_games["GAME_DATE_EST"] = pd.to_datetime(
                    valid_games["GAME_DATE_EST"]
                )

            logger.info(f"✅ Loaded {len(valid_games):,} valid games for calibration")
            return valid_games

        except Exception as e:
            logger.error(f"❌ Error loading historical data: {e}")
            return None

    def get_professional_prediction(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        betting_line: Optional[float] = None,
        include_detailed_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Ottieni previsione professionale con analisi dettagliata.

        Args:
            home_team: Team casa
            away_team: Team ospite
            game_date: Data partita
            betting_line: Linea di betting (opzionale)
            include_detailed_analysis: Include analisi dettagliata

        Returns:
            Dict con previsione professionale e analisi
        """

        logger.info(
            f"🎯 Professional prediction: {away_team} @ {home_team} ({game_date})"
        )

        try:
            if not self.prediction_engine:
                return self._get_fallback_response(
                    home_team, away_team, "Engine not available"
                )

            # Esegui previsione avanzata
            prediction_result = predict_nba_game_advanced(
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                betting_line=betting_line,
            )

            # Costruisci response professionale
            response = {
                # Basic prediction
                "status": "success",
                "predicted_total": prediction_result.predicted_total,
                "confidence_interval": prediction_result.confidence_interval,
                "standard_error": prediction_result.standard_error,
                "model_confidence": prediction_result.model_confidence,
                "recommendation": prediction_result.recommendation,
                # Professional analysis
                "professional_analysis": self._build_professional_analysis(
                    prediction_result, home_team, away_team, betting_line
                ),
                # Risk management
                "risk_assessment": self._calculate_risk_assessment(
                    prediction_result, betting_line
                ),
                # Model validation
                "model_validation": self._validate_prediction(prediction_result),
                # Metadata
                "prediction_method": "Advanced NBA Engine",
                "data_quality": self._assess_data_quality(),
                "prediction_timestamp": datetime.now().isoformat(),
                "game_date": game_date.isoformat(),
                "betting_line": betting_line,
            }

            if include_detailed_analysis:
                response.update(
                    {
                        "team_metrics": self._format_team_metrics(
                            prediction_result.team_metrics
                        ),
                        "situational_factors": prediction_result.situational_adjustments,
                        "probability_analysis": prediction_result.probability_over_line,
                        "historical_comparisons": self._find_historical_comparisons(
                            prediction_result.predicted_total
                        ),
                        "prediction_factors": prediction_result.prediction_factors,
                    }
                )

            return response

        except Exception as e:
            logger.error(f"❌ Error in professional prediction: {e}")
            return self._get_fallback_response(home_team, away_team, f"Error: {str(e)}")

    def _build_professional_analysis(
        self,
        prediction_result: PredictionResult,
        home_team: str,
        away_team: str,
        betting_line: Optional[float],
    ) -> Dict[str, Any]:
        """Costruisci analisi professionale dettagliata."""

        analysis = {
            "summary": f"Prediction: {prediction_result.predicted_total} points total",
            "confidence_level": self._interpret_confidence(
                prediction_result.model_confidence
            ),
            "edge_analysis": self._analyze_betting_edge(
                prediction_result, betting_line
            ),
            "value_assessment": self._assess_betting_value(
                prediction_result, betting_line
            ),
        }

        # Aggiungi insights specifici
        insights = []

        # Team matchup analysis
        if prediction_result.team_metrics:
            home_metrics = prediction_result.team_metrics.get("home")
            away_metrics = prediction_result.team_metrics.get("away")

            if home_metrics and away_metrics:
                # Offensive matchup
                if home_metrics.offensive_rating > away_metrics.defensive_rating + 5:
                    insights.append(
                        f"Strong offensive matchup: {home_team} offense vs {away_team} defense"
                    )
                if away_metrics.offensive_rating > home_metrics.defensive_rating + 5:
                    insights.append(
                        f"Strong offensive matchup: {away_team} offense vs {home_team} defense"
                    )

                # Pace analysis
                avg_pace = (home_metrics.pace + away_metrics.pace) / 2
                if avg_pace > 100:
                    insights.append(f"Fast-paced game expected (pace: {avg_pace:.1f})")
                elif avg_pace < 95:
                    insights.append(f"Slow-paced game expected (pace: {avg_pace:.1f})")

        # Situational insights
        total_adjustment = sum(prediction_result.situational_adjustments.values())
        if abs(total_adjustment) > 3:
            direction = "increasing" if total_adjustment > 0 else "decreasing"
            insights.append(
                f"Significant situational factors {direction} total by {abs(total_adjustment):.1f} points"
            )

        analysis["insights"] = insights

        return analysis

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpreta level di confidence."""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def _analyze_betting_edge(
        self, prediction_result: PredictionResult, betting_line: Optional[float]
    ) -> Dict[str, Any]:
        """Analizza betting edge."""
        if not betting_line:
            return {"edge": 0.0, "assessment": "No betting line available"}

        edge = prediction_result.predicted_total - betting_line
        edge_pct = (edge / betting_line) * 100 if betting_line != 0 else 0

        # Determine edge quality
        if abs(edge) > 10:
            quality = "Exceptional"
        elif abs(edge) > 7:
            quality = "Strong"
        elif abs(edge) > 4:
            quality = "Moderate"
        elif abs(edge) > 2:
            quality = "Small"
        else:
            quality = "Minimal"

        direction = "Over" if edge > 0 else "Under"

        return {
            "edge_points": edge,
            "edge_percentage": edge_pct,
            "direction": direction,
            "quality": quality,
            "assessment": f"{quality} {direction} edge of {abs(edge):.1f} points",
        }

    def _assess_betting_value(
        self, prediction_result: PredictionResult, betting_line: Optional[float]
    ) -> Dict[str, Any]:
        """Assess betting value."""
        if not betting_line:
            return {"value_score": 0.0, "recommendation": "No line available"}

        edge = prediction_result.predicted_total - betting_line
        confidence = prediction_result.model_confidence

        # Calculate value score (edge * confidence)
        value_score = abs(edge) * confidence

        # Determine value level
        if value_score > 6:
            value_level = "Excellent"
            recommendation = "Strong Bet"
        elif value_score > 4:
            value_level = "Good"
            recommendation = "Moderate Bet"
        elif value_score > 2:
            value_level = "Fair"
            recommendation = "Small Bet"
        else:
            value_level = "Poor"
            recommendation = "Pass"

        return {
            "value_score": value_score,
            "value_level": value_level,
            "recommendation": recommendation,
            "justification": f"Value Score: {value_score:.2f} (Edge: {abs(edge):.1f}, Confidence: {confidence:.2f})",
        }

    def _calculate_risk_assessment(
        self, prediction_result: PredictionResult, betting_line: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate risk assessment for the prediction."""

        risk_factors = []
        risk_score = 0  # 0-100 scale

        # Confidence risk
        confidence = prediction_result.model_confidence
        if confidence < 0.3:
            risk_factors.append("Low model confidence")
            risk_score += 30
        elif confidence < 0.5:
            risk_factors.append("Moderate model confidence")
            risk_score += 15

        # Standard error risk
        std_error = prediction_result.standard_error
        if std_error > 25:
            risk_factors.append("High prediction uncertainty")
            risk_score += 20
        elif std_error > 20:
            risk_factors.append("Moderate prediction uncertainty")
            risk_score += 10

        # Data quality risk
        data_quality = self._assess_data_quality()
        if data_quality < 0.7:
            risk_factors.append("Limited historical data")
            risk_score += 15

        # Betting line risk
        if betting_line:
            edge = abs(prediction_result.predicted_total - betting_line)
            if edge < 2:
                risk_factors.append("Small betting edge")
                risk_score += 10

        # Determine risk level
        if risk_score > 50:
            risk_level = "High"
        elif risk_score > 30:
            risk_level = "Moderate"
        elif risk_score > 15:
            risk_level = "Low"
        else:
            risk_level = "Very Low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "assessment": f"{risk_level} risk score ({risk_score}/100)",
        }

    def _validate_prediction(
        self, prediction_result: PredictionResult
    ) -> Dict[str, Any]:
        """Validate prediction against historical patterns."""

        validation_checks = []
        validation_score = 0

        # Range validation
        pred_total = prediction_result.predicted_total
        if 180 <= pred_total <= 280:
            validation_checks.append("✅ Prediction in realistic NBA range")
            validation_score += 25
        else:
            validation_checks.append("⚠️ Prediction outside typical NBA range")

        # Confidence interval validation
        ci_width = (
            prediction_result.confidence_interval[1]
            - prediction_result.confidence_interval[0]
        )
        if 20 <= ci_width <= 50:
            validation_checks.append("✅ Appropriate confidence interval width")
            validation_score += 20
        else:
            validation_checks.append("⚠️ Unusual confidence interval width")

        # Standard error validation
        if 10 <= prediction_result.standard_error <= 25:
            validation_checks.append("✅ Reasonable prediction uncertainty")
            validation_score += 20
        else:
            validation_checks.append("⚠️ Unusual prediction uncertainty")

        # Model confidence validation
        if 0.2 <= prediction_result.model_confidence <= 0.9:
            validation_checks.append("✅ Appropriate model confidence")
            validation_score += 15
        else:
            validation_checks.append("⚠️ Extreme model confidence")

        validation_score += 20  # Base score

        return {
            "validation_score": min(100, validation_score),
            "validation_checks": validation_checks,
            "overall_status": "Valid" if validation_score >= 70 else "Needs Review",
        }

    def _assess_data_quality(self) -> float:
        """Assess quality of available data."""
        if self.historical_data is None:
            return 0.0

        # Factors affecting data quality
        data_points = len(self.historical_data)
        recency_factor = self._calculate_recency_factor()

        # Score based on data volume
        if data_points > 1000:
            data_score = 1.0
        elif data_points > 500:
            data_score = 0.8
        elif data_points > 100:
            data_score = 0.6
        else:
            data_score = 0.4

        # Combined quality score
        quality_score = data_score * recency_factor

        return max(0.1, min(1.0, quality_score))

    def _calculate_recency_factor(self) -> float:
        """Calculate recency factor for data quality."""
        if (
            self.historical_data is None
            or "GAME_DATE_EST" not in self.historical_data.columns
        ):
            return 0.5

        latest_date = self.historical_data["GAME_DATE_EST"].max()
        days_old = (datetime.now().date() - latest_date.date()).days

        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.9
        elif days_old <= 180:
            return 0.8
        elif days_old <= 365:
            return 0.7
        else:
            return 0.5

    def _format_team_metrics(self, team_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format team metrics for display."""
        formatted = {}

        for location, metrics in team_metrics.items():
            if metrics:
                formatted[location] = {
                    "team_name": metrics.team_name,
                    "offensive_rating": round(metrics.offensive_rating, 1),
                    "defensive_rating": round(metrics.defensive_rating, 1),
                    "net_rating": round(metrics.net_rating, 1),
                    "pace": round(metrics.pace, 1),
                    "home_offensive_rating": round(metrics.home_offensive_rating, 1),
                    "away_offensive_rating": round(metrics.away_offensive_rating, 1),
                    "home_defensive_rating": round(metrics.home_defensive_rating, 1),
                    "away_defensive_rating": round(metrics.away_defensive_rating, 1),
                }

        return formatted

    def _find_historical_comparisons(
        self, predicted_total: float, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find historical games with similar totals."""
        if self.historical_data is None:
            return []

        # Find games within ±15 points of prediction
        similar_games = self.historical_data[
            (self.historical_data["TOTAL_SCORE"] >= predicted_total - 15)
            & (self.historical_data["TOTAL_SCORE"] <= predicted_total + 15)
        ].copy()

        # Sort by closeness to prediction
        similar_games["difference"] = abs(
            similar_games["TOTAL_SCORE"] - predicted_total
        )
        similar_games = similar_games.sort_values("difference").head(limit)

        comparisons = []
        for _, game in similar_games.iterrows():
            comparisons.append(
                {
                    "date": game.get("GAME_DATE_EST", "N/A"),
                    "total_score": game.get("TOTAL_SCORE", 0),
                    "difference_from_prediction": game.get("difference", 0),
                    "home_team": game.get("HOME_TEAM", "N/A"),
                    "away_team": game.get("AWAY_TEAM", "N/A"),
                }
            )

        return comparisons

    def _get_fallback_response(
        self, home_team: str, away_team: str, reason: str
    ) -> Dict[str, Any]:
        """Response when advanced prediction not available."""
        return {
            "status": "fallback",
            "predicted_total": 226.0,  # NBA average
            "confidence_interval": (200.0, 252.0),
            "standard_error": 15.0,
            "model_confidence": 0.3,
            "recommendation": "Low Confidence",
            "professional_analysis": {
                "summary": f"Using fallback prediction: {reason}",
                "confidence_level": "Low",
                "edge_analysis": {"assessment": "No advanced analysis available"},
                "value_assessment": {"recommendation": "Pass"},
                "insights": ["Limited data available for advanced analysis"],
            },
            "risk_assessment": {
                "risk_level": "High",
                "risk_factors": [f"Fallback mode: {reason}"],
                "assessment": "High uncertainty due to limited data",
            },
            "model_validation": {
                "validation_score": 30,
                "validation_checks": ["⚠️ Using fallback prediction mode"],
                "overall_status": "Limited Validation",
            },
            "prediction_method": "Fallback",
            "data_quality": 0.3,
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "prediction_engine_status": "Operational"
            if self.prediction_engine
            else "Unavailable",
            "historical_data_available": self.historical_data is not None,
            "data_points": len(self.historical_data)
            if self.historical_data is not None
            else 0,
            "data_quality_score": self._assess_data_quality(),
            "last_updated": datetime.now().isoformat(),
            "system_status": "Healthy"
            if self.prediction_engine and self.historical_data is not None
            else "Degraded",
        }


# Singleton instance
_professional_bridge = None


def get_enhanced_prediction_bridge_professional() -> (
    EnhancedPredictionBridgeProfessional
):
    """Get singleton instance of professional prediction bridge."""
    global _professional_bridge
    if _professional_bridge is None:
        _professional_bridge = EnhancedPredictionBridgeProfessional()
    return _professional_bridge


def get_professional_nba_prediction(
    home_team: str,
    away_team: str,
    game_date: date,
    betting_line: Optional[float] = None,
    detailed: bool = True,
) -> Dict[str, Any]:
    """Convenience function for professional NBA prediction."""
    bridge = get_enhanced_prediction_bridge_professional()
    return bridge.get_professional_prediction(
        home_team, away_team, game_date, betting_line, detailed
    )
