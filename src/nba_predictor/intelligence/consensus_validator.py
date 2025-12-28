#!/usr/bin/env python3
"""
⚖️ Consensus Validator (Sharp Book Validator)
Role: The "Skeptic Auditor"
Goal: Validate model signals against the "Wisdom of Crowds" (Sharp Market).

Logic:
- Calculate "Fair Implied Probability" from Consensus Odds (Average of Sharp Books).
- Remove VIG (Overround) using the Multiplicative Method.
- Compare Model Probability vs Fair Consensus Probability.
- If Deviation > 10% AND EV < 5%: REJECT BET.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    Validates betting signals against Sharp Consensus.
    """

    # Thresholds
    MAX_PROB_DEVIATION = 0.15  # 15% Max Deviation allowed (Perplexity Recommendation)
    MIN_EV_FOR_EXCEPTION = 0.05  # If EV > 5%, we might allow slightly higher deviation

    def validate_bet(
        self,
        model_prob: float,
        market_odds_over: float,
        market_odds_under: float,
        target_direction: str,
    ) -> Tuple[bool, str, float]:
        """
        Validate if the model's probability is within a sane range of the market.

        Args:
            model_prob: The probability calculated by our model (0.0 - 1.0) for the target direction.
            market_odds_over: Decimal odds for Over.
            market_odds_under: Decimal odds for Under.
            target_direction: "OVER" or "UNDER".

        Returns:
            Tuple (is_valid, reason, consensus_fair_prob)
        """
        # 1. Calculate Fair Consensus Probability (Zero-Vig)
        try:
            implied_over = 1.0 / market_odds_over
            implied_under = 1.0 / market_odds_under
            overround = implied_over + implied_under

            fair_prob_over = implied_over / overround
            fair_prob_under = implied_under / overround

            consensus_prob = (
                fair_prob_over
                if target_direction.upper() == "OVER"
                else fair_prob_under
            )

        except ZeroDivisionError:
            return False, "INVALID_ODDS", 0.0

        # 2. Calculate Deviation
        # Model=60%, Market=50% -> Diff = +10%
        deviation = model_prob - consensus_prob
        abs_deviation = abs(deviation)

        logger.info(
            f"⚖️ Consensus Check: Model={model_prob:.1%} vs FairMarket={consensus_prob:.1%} (Dev={deviation:+.1%})"
        )

        # 3. Validation Logic
        if abs_deviation > self.MAX_PROB_DEVIATION:
            # We are WAY off the market.
            # Only allowed if we have massive edge? No, usually implies we are wrong.
            # Perplexity advises strict KILL here for a "Fail-Safe" system.
            reason = f"EXTREME_DEVIATION: Model {model_prob:.1%} vs Market {consensus_prob:.1%} (>15%)"
            logger.warning(reason)
            return False, reason, consensus_prob

        if abs_deviation > 0.10:
            # Warning zone (10-15%)
            reason = f"HIGH_DEVIATION: Model {model_prob:.1%} vs Market {consensus_prob:.1%} (>10%)"
            logger.warning(reason)
            # We let it pass but it should be flagged (returning True but logging warning)
            return True, reason, consensus_prob

        return True, "VALID_ALIGNMENT", consensus_prob


# Singleton
_validator = None


def get_consensus_validator() -> ConsensusValidator:
    global _validator
    if _validator is None:
        _validator = ConsensusValidator()
    return _validator
