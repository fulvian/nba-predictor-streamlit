"""
EVCalculator Module
-------------------
Calculates Expected Value (EV) for betting opportunities and recommends stake sizes
using Fractional Kelly Criterion with strict safety filters.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EVResult:
    """Result of an EV calculation."""

    ev_percentage: float
    edge: float
    kelly_stake_percentage: float
    recommended_stake_amount: float
    is_value_bet: bool
    reason: str


class EVCalculator:
    """
    Calculates Expected Value and recommended stakes for bets.

    Implements:
    - Implied Probability calculation from American Odds
    - Expected Value (EV) calculation
    - Fractional Kelly Criterion for stake sizing
    - Safety Filters (Min Edge, Min Confidence)
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.025,
        min_model_prob: float = 0.60,
    ):
        """
        Initialize the EV Calculator.

        Args:
            bankroll: Total betting bankroll.
            kelly_fraction: Fraction of Kelly stake to use (e.g., 0.25 for quarter Kelly).
            min_edge: Minimum edge (EV) required to consider a bet (e.g., 0.025 for 2.5%).
            min_model_prob: Minimum model probability required to consider a bet.
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_model_prob = min_model_prob

    def calculate_implied_probability(self, american_odds: int) -> float:
        """
        Convert American Odds to Implied Probability.

        Args:
            american_odds: Odds in American format (e.g., -110, +150).

        Returns:
            Implied probability as a float (0.0 to 1.0).
        """
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def calculate_decimal_odds(self, american_odds: int) -> float:
        """
        Convert American Odds to Decimal Odds.

        Args:
            american_odds: Odds in American format.

        Returns:
            Decimal odds as a float.
        """
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))

    def calculate_ev(self, model_prob: float, american_odds: int) -> EVResult:
        """
        Calculate EV and recommended stake for a single bet.

        Args:
            model_prob: Probability of winning estimated by the model (0.0 to 1.0).
            american_odds: Bookmaker odds in American format.

        Returns:
            EVResult object containing metrics and recommendation.
        """
        # 1. Calculate Implied Probability and Decimal Odds
        implied_prob = self.calculate_implied_probability(american_odds)
        decimal_odds = self.calculate_decimal_odds(american_odds)

        # 2. Calculate Edge (Difference between Model Prob and Implied Prob)
        edge = model_prob - implied_prob

        # 3. Calculate Expected Value (EV)
        # EV = (Probability * Profit) - (Probability of Loss * Stake)
        # Assuming Stake = 1 unit
        profit_on_win = decimal_odds - 1
        ev = (model_prob * profit_on_win) - (1 - model_prob)

        # 4. Apply Safety Filters
        if edge < self.min_edge:
            return EVResult(
                ev_percentage=ev * 100,
                edge=edge * 100,
                kelly_stake_percentage=0.0,
                recommended_stake_amount=0.0,
                is_value_bet=False,
                reason=f"Edge {edge:.1%} below threshold {self.min_edge:.1%}",
            )

        if model_prob < self.min_model_prob:
            return EVResult(
                ev_percentage=ev * 100,
                edge=edge * 100,
                kelly_stake_percentage=0.0,
                recommended_stake_amount=0.0,
                is_value_bet=False,
                reason=f"Model confidence {model_prob:.1%} below threshold {self.min_model_prob:.1%}",
            )

        # 5. Calculate Kelly Stake
        # Kelly % = (bp - q) / b
        # b = decimal odds - 1 (net odds)
        # p = probability of winning
        # q = probability of losing (1-p)
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p

        kelly_percentage = (b * p - q) / b

        # Apply Fractional Kelly
        fractional_kelly_percentage = max(0.0, kelly_percentage * self.kelly_fraction)

        # Calculate Stake Amount
        stake_amount = self.bankroll * fractional_kelly_percentage

        return EVResult(
            ev_percentage=ev * 100,
            edge=edge * 100,
            kelly_stake_percentage=fractional_kelly_percentage * 100,
            recommended_stake_amount=stake_amount,
            is_value_bet=True,
            reason="Value bet identified",
        )
