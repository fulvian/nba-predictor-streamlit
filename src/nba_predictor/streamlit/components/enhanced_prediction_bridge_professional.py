#!/usr/bin/env python3
"""
🏀 ENHANCED PREDICTION BRIDGE PROFESSIONAL - Bridge NBA con Previsioni Avanzate

Integra il motore di previsione avanzato nel sistema Streamlit:
1. ✅ Usa Unified Hybrid Pipeline (New System)
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
import logging

# Import Unified Pipeline
from nba_predictor.core.unified_hybrid_pipeline import (
    get_unified_hybrid_pipeline,
    UnifiedPredictionResult,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPredictionBridgeProfessional:
    """Bridge professionale per previsioni NBA avanzate (Unified Hybrid System)."""

    def __init__(self):
        """Inizializza il bridge professionale."""
        logger.info(
            "🚀 Initializing Enhanced NBA Prediction Bridge Professional (Unified System)..."
        )

        try:
            self.prediction_engine = get_unified_hybrid_pipeline()
            self.historical_data = self._load_historical_data()

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
            # Use the pipeline's data if possible, or load directly
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
        force_refresh: bool = True,  # Default to True to prevent caching issues
    ) -> Dict[str, Any]:
        """
        Ottieni previsione professionale con analisi dettagliata.
        """

        logger.info(
            f"🎯 Professional prediction (Unified): {away_team} @ {home_team} ({game_date})"
        )

        try:
            if not self.prediction_engine:
                return self._get_fallback_response(
                    home_team, away_team, "Engine not available"
                )

            # Ensure betting line is a float, default to 225.0 if None
            line = float(betting_line) if betting_line else 225.0

            # Esegui previsione unificata
            # Note: predict_unified expects team1, team2. We map away=team1, home=team2 usually,
            # but let's stick to the method signature: predict_unified(team1, team2, line, home_team=...)
            prediction_result = self.prediction_engine.predict_unified(
                team1=away_team,
                team2=home_team,
                line=line,
                home_team=home_team,
                validate_prediction=True,
            )

            # Costruisci response professionale
            response = {
                # Basic prediction
                "status": "success",
                "predicted_total": prediction_result.predicted_total,
                "confidence_interval": prediction_result.confidence_interval,
                "standard_error": (
                    prediction_result.confidence_interval[1]
                    - prediction_result.predicted_total
                )
                / 1.96,  # Approx
                "model_confidence": prediction_result.confidence
                / 100.0,  # Convert 0-100 to 0-1
                "recommendation": prediction_result.recommendation,
                # Professional analysis
                "professional_analysis": self._build_professional_analysis(
                    prediction_result, home_team, away_team, line
                ),
                # Risk management
                "risk_assessment": self._calculate_risk_assessment(
                    prediction_result, line
                ),
                # Model validation
                "model_validation": self._validate_prediction(prediction_result),
                # Metadata
                "prediction_method": "Unified Hybrid Pipeline (Enhanced)",
                "data_quality": self._assess_data_quality(),
                "prediction_timestamp": datetime.now().isoformat(),
                "game_date": game_date.isoformat(),
                "betting_line": line,
            }

            if include_detailed_analysis:
                response.update(
                    {
                        "team_metrics": {
                            "home": {"team_name": home_team},  # Simplified for now
                            "away": {"team_name": away_team},
                        },
                        "situational_factors": {
                            "injury_impact": prediction_result.injury_impact.get(
                                "impact_score", 0
                            ),
                            "momentum": prediction_result.player_momentum.get(
                                "momentum_score", 0
                            ),
                        },
                        "probability_analysis": {
                            "over": prediction_result.over_probability,
                            "under": prediction_result.under_probability,
                        },
                        "historical_comparisons": self._find_historical_comparisons(
                            prediction_result.predicted_total
                        ),
                        "prediction_factors": {
                            "shap_values": getattr(
                                prediction_result, "shap_values", {}
                            ),
                            "feature_importance": getattr(
                                prediction_result, "feature_importance", {}
                            ),
                        },
                    }
                )

            return response

        except Exception as e:
            logger.error(f"❌ Error in professional prediction: {e}")
            return self._get_fallback_response(home_team, away_team, f"Error: {str(e)}")

    def _build_professional_analysis(
        self,
        prediction_result: UnifiedPredictionResult,
        home_team: str,
        away_team: str,
        betting_line: Optional[float],
    ) -> Dict[str, Any]:
        """Costruisci analisi professionale dettagliata."""

        confidence_val = prediction_result.confidence / 100.0

        analysis = {
            "summary": f"Prediction: {prediction_result.predicted_total:.1f} points total",
            "confidence_level": self._interpret_confidence(confidence_val),
            "edge_analysis": self._analyze_betting_edge(
                prediction_result, betting_line
            ),
            "value_assessment": self._assess_betting_value(
                prediction_result, betting_line
            ),
        }

        # Aggiungi insights specifici
        insights = []

        # Add insights from the unified result
        if (
            prediction_result.injury_impact
            and prediction_result.injury_impact.get("impact_score", 0) != 0
        ):
            insights.append(
                f"Injury Impact: {prediction_result.injury_impact.get('summary', 'N/A')}"
            )

        if (
            prediction_result.player_momentum
            and prediction_result.player_momentum.get("momentum_score", 0) != 0
        ):
            insights.append(
                f"Momentum: {prediction_result.player_momentum.get('summary', 'N/A')}"
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
        self, prediction_result: UnifiedPredictionResult, betting_line: Optional[float]
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
        self, prediction_result: UnifiedPredictionResult, betting_line: Optional[float]
    ) -> Dict[str, Any]:
        """Assess betting value."""
        if not betting_line:
            return {"value_score": 0.0, "recommendation": "No line available"}

        edge = prediction_result.predicted_total - betting_line
        confidence = prediction_result.confidence / 100.0

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
        self, prediction_result: UnifiedPredictionResult, betting_line: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate risk assessment for the prediction."""

        risk_factors = []
        risk_score = 0  # 0-100 scale

        # Confidence risk
        confidence = prediction_result.confidence / 100.0
        if confidence < 0.3:
            risk_factors.append("Low model confidence")
            risk_score += 30
        elif confidence < 0.5:
            risk_factors.append("Moderate model confidence")
            risk_score += 15

        # Standard error risk (approximate from CI)
        std_error = (
            prediction_result.confidence_interval[1] - prediction_result.predicted_total
        ) / 1.96
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
        self, prediction_result: UnifiedPredictionResult
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

        # Model confidence validation
        conf = prediction_result.confidence / 100.0
        if 0.2 <= conf <= 0.9:
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
