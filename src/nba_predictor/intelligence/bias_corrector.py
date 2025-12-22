#!/usr/bin/env python3
"""
🔧 Bias Corrector Module for NBA Predictions

Implements asymmetric bias correction based on empirical analysis:
- OVER bets: +22.34 error on losses (massive overestimation)
- UNDER bets: -25.46 error on losses (massive underestimation)

This module applies line-stratified corrections to reduce systematic bias.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BiasCorrection:
    """Result of bias correction calculation."""

    original_prediction: float
    corrected_prediction: float
    correction_applied: float
    correction_type: str
    line_zone: str
    confidence_adjustment: float


class AsymmetricBiasCorrector:
    """
    Asymmetric Bias Corrector for NBA Over/Under Predictions.

    Based on empirical analysis of 97 bets:
    - Overall bias: -2.48 (slight underestimation)
    - OVER losses: +22.34 error (we predict too high → actual << predicted)
    - UNDER losses: -25.46 error (we predict too low → actual >> predicted)

    Line zone analysis:
    - 220-230: WR 41.7%, P&L -€37.07 (WORST - avoid or heavy correction)
    - 230-240: WR 57.5%, P&L +€10.76 (BEST - light correction)
    """

    # Empirical correction constants (from Consensus analysis)
    OVER_BASE_CORRECTION = -4.5  # Reduce prediction for OVER bets
    UNDER_BASE_CORRECTION = +5.2  # Increase prediction for UNDER bets

    # Volatility scaling factor (how much correction varies with line distance from 230)
    VOLATILITY_SCALE = 0.08

    # Line zone thresholds
    DANGER_ZONE = (220, 230)  # Worst performing range
    OPTIMAL_ZONE = (230, 240)  # Best performing range

    # Correction dampening for optimal zone
    OPTIMAL_ZONE_DAMPENER = 0.3  # Apply only 30% of correction in good zone
    DANGER_ZONE_AMPLIFIER = 1.5  # Apply 150% of correction in bad zone

    def __init__(self, enabled: bool = True):
        """
        Initialize the bias corrector.

        Args:
            enabled: Whether to apply corrections (can be disabled for A/B testing)
        """
        self.enabled = enabled
        self._correction_history = []
        logger.info(f"🔧 AsymmetricBiasCorrector initialized (enabled={enabled})")

    def correct_prediction(
        self,
        raw_prediction: float,
        market_line: float,
        bet_direction: Optional[str] = None,
    ) -> BiasCorrection:
        """
        Apply asymmetric bias correction to a prediction.

        Args:
            raw_prediction: Original model prediction (predicted_total)
            market_line: The betting line from the market
            bet_direction: "OVER" or "UNDER" (if known). If None, inferred from prediction vs line.

        Returns:
            BiasCorrection with corrected prediction and metadata
        """
        if not self.enabled:
            return BiasCorrection(
                original_prediction=raw_prediction,
                corrected_prediction=raw_prediction,
                correction_applied=0.0,
                correction_type="DISABLED",
                line_zone="N/A",
                confidence_adjustment=0.0,
            )

        # Infer bet direction if not provided
        if bet_direction is None:
            bet_direction = "OVER" if raw_prediction > market_line else "UNDER"

        # Determine line zone
        line_zone = self._get_line_zone(market_line)

        # Calculate volatility factor (how far from optimal center of 230)
        volatility = (market_line - 230) * self.VOLATILITY_SCALE

        # Calculate base correction
        if bet_direction.upper() == "OVER":
            base_correction = self.OVER_BASE_CORRECTION + (1.2 * volatility)
        else:
            base_correction = self.UNDER_BASE_CORRECTION - (1.0 * volatility)

        # Apply zone-specific modifiers
        zone_multiplier = self._get_zone_multiplier(line_zone)
        final_correction = base_correction * zone_multiplier

        # Calculate corrected prediction
        corrected_prediction = raw_prediction + final_correction

        # Calculate confidence adjustment (reduce confidence in danger zone)
        confidence_adjustment = self._get_confidence_adjustment(line_zone)

        # Log the correction
        result = BiasCorrection(
            original_prediction=raw_prediction,
            corrected_prediction=corrected_prediction,
            correction_applied=final_correction,
            correction_type=f"ASYMMETRIC_{bet_direction}",
            line_zone=line_zone,
            confidence_adjustment=confidence_adjustment,
        )

        self._correction_history.append(result)

        logger.info(
            f"🔧 Bias Correction: {raw_prediction:.1f} → {corrected_prediction:.1f} "
            f"(Δ{final_correction:+.1f}, {bet_direction}, zone={line_zone})"
        )

        return result

    def _get_line_zone(self, market_line: float) -> str:
        """Categorize the market line into a zone."""
        if market_line < self.DANGER_ZONE[0]:
            return "LOW"
        elif self.DANGER_ZONE[0] <= market_line < self.DANGER_ZONE[1]:
            return "DANGER"
        elif self.OPTIMAL_ZONE[0] <= market_line < self.OPTIMAL_ZONE[1]:
            return "OPTIMAL"
        else:
            return "HIGH"

    def _get_zone_multiplier(self, line_zone: str) -> float:
        """Get correction multiplier based on zone."""
        if line_zone == "DANGER":
            return self.DANGER_ZONE_AMPLIFIER
        elif line_zone == "OPTIMAL":
            return self.OPTIMAL_ZONE_DAMPENER
        else:
            return 1.0  # Normal correction for other zones

    def _get_confidence_adjustment(self, line_zone: str) -> float:
        """
        Get confidence adjustment factor.

        Returns a negative number to reduce confidence in dangerous zones.
        """
        if line_zone == "DANGER":
            return -0.15  # Reduce confidence by 15% in danger zone
        elif line_zone == "OPTIMAL":
            return +0.05  # Slight confidence boost in optimal zone
        else:
            return 0.0

    def should_filter_bet(self, market_line: float) -> Tuple[bool, str]:
        """
        Determine if a bet should be filtered out entirely.

        Based on empirical data:
        - Lines 220-230 have 41.7% WR and -€37.07 P&L (dangerous)

        Args:
            market_line: The betting line

        Returns:
            Tuple of (should_filter, reason)
        """
        line_zone = self._get_line_zone(market_line)

        if line_zone == "DANGER":
            return True, f"Line {market_line} in DANGER zone (220-230), WR=41.7%"

        return False, ""

    def get_stats(self) -> dict:
        """Get statistics about applied corrections."""
        if not self._correction_history:
            return {"corrections_applied": 0}

        corrections = [c.correction_applied for c in self._correction_history]
        return {
            "corrections_applied": len(self._correction_history),
            "avg_correction": sum(corrections) / len(corrections),
            "max_correction": max(corrections),
            "min_correction": min(corrections),
            "zone_distribution": {
                zone: len([c for c in self._correction_history if c.line_zone == zone])
                for zone in ["LOW", "DANGER", "OPTIMAL", "HIGH"]
            },
        }


# Singleton instance for use across the pipeline
_default_corrector: Optional[AsymmetricBiasCorrector] = None


def get_bias_corrector(enabled: bool = True) -> AsymmetricBiasCorrector:
    """Get or create the default bias corrector instance."""
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = AsymmetricBiasCorrector(enabled=enabled)
    return _default_corrector
