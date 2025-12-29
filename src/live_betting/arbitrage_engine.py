"""
Live Odds Arbitrage & Value Betting Engine

Implements:
1. Three-way arbitrage detection (back/lay/back on different bookmakers)
2. Two-way arbitrage (single bookmaker back/lay)
3. Value betting with Kelly criterion (fractional)
4. Stake optimization based on liquidity and edge

References:
- Vlastakis et al. (2009): Higher returns in low-liquidity matches
- Constantinou & Fenton (2012): Value-based returns in football betting
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of profitable betting opportunities."""
    ARBITRAGE_2WAY = "arbitrage_2way"  # Back/lay same bookmaker
    ARBITRAGE_3WAY = "arbitrage_3way"  # Multi-bookmaker hedge
    VALUE_BET = "value_bet"  # Positive EV based on fair value estimate
    MIDDLE = "middle"  # Exploit differing odds on same event


@dataclass
class BettingOpportunity:
    """A specific profitable betting scenario."""
    timestamp: datetime
    event_id: str
    opportunity_type: OpportunityType
    
    # Core metrics
    gross_profit_pct: float  # Total return before fees/slippage
    net_profit_pct: float  # After 5% estimated fees
    probability_win: float  # 0-1, estimated win probability
    
    # Betting legs (each leg = one bet to place)
    legs: List[Dict] = field(default_factory=list)  # [{outcome, stake, odds, bookmaker}, ...]
    
    # Risk management
    required_liquidity: float  # Total stake needed
    max_loss: float  # Maximum loss if one leg fails (value bets only)
    execution_urgency: float  # 0-1, how soon before market corrects
    
    # Metadata
    anomaly_source: Optional[str] = None  # Which anomaly triggered this
    confidence: float = 0.7
    notes: str = ""


class ArbitrageEngine:
    """
    Identifies and evaluates profitable opportunities from live odds.
    """

    def __init__(
        self,
        min_profit_pct: float = 1.0,
        estimated_fees_pct: float = 5.0,
        kelly_fraction: float = 0.25,
        min_liquidity_stake: float = 50.0,
    ):
        """
        Args:
            min_profit_pct: Only flag opportunities with >X% gross profit
            estimated_fees_pct: Betfair commission + slippage assumption
            kelly_fraction: Use Kelly * kelly_fraction for stake (conservative)
            min_liquidity_stake: Minimum available at quoted odds
        """
        self.min_profit_pct = min_profit_pct
        self.estimated_fees_pct = estimated_fees_pct
        self.kelly_fraction = kelly_fraction
        self.min_liquidity_stake = min_liquidity_stake

        self.opportunities: List[BettingOpportunity] = []

    def detect_2way_arbitrage(
        self,
        event_id: str,
        backing_odds: float,
        laying_odds: float,
        back_volume: float,
        lay_volume: float,
        bookmaker: str = "betfair",
    ) -> Optional[BettingOpportunity]:
        """
        Detect arbitrage on same bookmaker (back vs lay).

        Example:
            You can back @ 2.50 and lay @ 1.80 on same outcome.
            Stake back: €100 @ 2.50 = +€150 if wins, -€100 if loses
            Stake lay: €x @ 1.80 = hedge loss

        Args:
            backing_odds: Price at which you can back
            laying_odds: Price at which you can lay (reciprocal relationship)
            back_volume: Available liquidity at backing price
            lay_volume: Available liquidity at laying price
        """
        if backing_odds <= 1.0 or laying_odds <= 1.0:
            return None

        # Implied probability check
        prob_back = 1.0 / backing_odds
        prob_lay = 1.0 / laying_odds

        # Two outcomes: the backed outcome or its inverse
        # If we back @ backing_odds and lay @ laying_odds:
        # Stake back = S_back, profit if wins = S_back * (backing_odds - 1)
        # Stake lay = S_lay, profit if backed outcome loses = S_lay * (laying_odds - 1)

        # For guaranteed profit: S_back * (backing_odds - 1) == S_lay
        # (i.e., back profit covers lay stake)
        # Total stake = S_back + S_lay = S_back + S_back * (backing_odds - 1)
        #            = S_back * backing_odds
        # Return = S_back * (backing_odds - 1)  (= S_lay in this case)
        # ROI = S_back * (backing_odds - 1) / (S_back * backing_odds)
        #     = (backing_odds - 1) / backing_odds
        #     = 1 - 1/backing_odds

        # More generally, if backing_odds > 1/prob_lay (i.e., we beat market probability),
        # then there's value.
        implied_profit_pct = (1.0 - prob_back - prob_lay) * 100 if prob_back + prob_lay < 1.0 else 0

        if implied_profit_pct < self.min_profit_pct:
            return None

        # Check liquidity
        if min(back_volume, lay_volume) < self.min_liquidity_stake:
            return None

        # Estimate actual stake based on liquidity
        max_stake = min(back_volume, lay_volume)

        # Net profit after fees
        gross_profit = implied_profit_pct
        net_profit = gross_profit - self.estimated_fees_pct

        if net_profit < self.min_profit_pct:
            return None

        opportunity = BettingOpportunity(
            timestamp=datetime.utcnow(),
            event_id=event_id,
            opportunity_type=OpportunityType.ARBITRAGE_2WAY,
            gross_profit_pct=gross_profit,
            net_profit_pct=net_profit,
            probability_win=1.0,  # Arbitrage = risk-free (in theory)
            legs=[
                {
                    "action": "back",
                    "outcome": "primary",
                    "odds": backing_odds,
                    "bookmaker": bookmaker,
                    "implied_stake_pct": (1 - prob_lay) * 100 / prob_back,
                },
                {
                    "action": "lay",
                    "outcome": "primary",
                    "odds": laying_odds,
                    "bookmaker": bookmaker,
                    "implied_stake_pct": 100.0,
                },
            ],
            required_liquidity=max_stake,
            max_loss=0,  # No loss in true arbitrage
            execution_urgency=0.8,
            confidence=0.95,
            notes=f"2-way arbitrage: back@{backing_odds:.2f} vs lay@{laying_odds:.2f}",
        )

        self.opportunities.append(opportunity)
        return opportunity

    def detect_value_bet(
        self,
        event_id: str,
        fair_probability: float,
        market_odds: float,
        market_probability: float,
        bookmaker: str = "betfair",
        outcome_name: str = "unknown",
    ) -> Optional[BettingOpportunity]:
        """
        Detect value bet (positive expected value).

        Value exists when:
            fair_prob > market_prob  (we think outcome more likely than market)
        EV = fair_prob * (odds - 1) - (1 - fair_prob) = fair_prob * odds - 1

        Args:
            event_id: Match ID
            fair_probability: Your estimated win probability (0-1)
            market_odds: Current market odds
            market_probability: Implied probability from market
            bookmaker: Source
            outcome_name: Description (e.g., "Home Win", "Over 2.5")
        """
        if fair_probability <= 0 or fair_probability >= 1:
            return None
        if market_odds <= 1.0 or market_odds > 100:
            return None

        # Expected value per unit stake
        ev = fair_probability * (market_odds - 1) - (1 - fair_probability)
        ev_pct = ev * 100

        if ev_pct < self.min_profit_pct:
            return None

        # Kelly criterion stake sizing
        # Kelly = (bp - q) / b, where b = odds - 1, p = win prob, q = 1 - p
        b = market_odds - 1
        kelly_frac = (fair_probability * b - (1 - fair_probability)) / b
        kelly_frac = max(0, min(kelly_frac, 1.0))  # Clamp to [0, 1]

        # Apply fractional Kelly (more conservative)
        suggested_frac = kelly_frac * self.kelly_fraction

        opportunity = BettingOpportunity(
            timestamp=datetime.utcnow(),
            event_id=event_id,
            opportunity_type=OpportunityType.VALUE_BET,
            gross_profit_pct=ev_pct,
            net_profit_pct=ev_pct - self.estimated_fees_pct,
            probability_win=fair_probability,
            legs=[
                {
                    "action": "back",
                    "outcome": outcome_name,
                    "odds": market_odds,
                    "bookmaker": bookmaker,
                    "suggested_kelly_fraction": suggested_frac,
                }
            ],
            required_liquidity=float("inf"),  # Depends on stake
            max_loss=100.0,  # Full stake loss possible
            execution_urgency=0.3,  # Value bets less time-sensitive
            confidence=0.5 + fair_probability * 0.5,  # Higher if more confident in estimate
            notes=f"Value: fair_prob={fair_probability:.2%} vs market={market_probability:.2%}, EV={ev_pct:.2f}%",
        )

        self.opportunities.append(opportunity)
        return opportunity

    def detect_middle(
        self,
        event_id: str,
        low_odds: float,
        high_odds: float,
        low_volume: float,
        high_volume: float,
    ) -> Optional[BettingOpportunity]:
        """
        Detect "middle" opportunity: exploit differing quotes.

        Example:
        - Bookmaker A: Back Under 2.5 @ 1.95
        - Bookmaker B: Back Over 2.5 @ 2.00
        - If you lay both, you can "middle" if the actual result is 2.5-3.0 goals.

        This is more complex; simplified version here.
        """
        if low_odds >= high_odds:
            return None

        implied_low = 1.0 / low_odds
        implied_high = 1.0 / high_odds

        # Crude check: if combined implied prob < 100%, there's potential value
        combined_prob = implied_low + implied_high
        if combined_prob > 1.05:  # Buffer for fees
            return None

        middle_profit_pct = (1.0 / combined_prob - 1.0) * 100 - self.estimated_fees_pct

        if middle_profit_pct < self.min_profit_pct:
            return None

        # Not fully implemented (complex); return placeholder
        opportunity = BettingOpportunity(
            timestamp=datetime.utcnow(),
            event_id=event_id,
            opportunity_type=OpportunityType.MIDDLE,
            gross_profit_pct=middle_profit_pct,
            net_profit_pct=middle_profit_pct,
            probability_win=0.5,
            legs=[],
            required_liquidity=float("inf"),
            max_loss=100.0,
            execution_urgency=0.7,
            confidence=0.4,
            notes="Middle opportunity (simplified); needs manual validation",
        )

        self.opportunities.append(opportunity)
        return opportunity

    def evaluate_anomaly(
        self,
        anomaly_type: str,
        fair_odds: float,
        current_odds: float,
        current_volume: float,
        deviation_pct: float,
    ) -> Optional[BettingOpportunity]:
        """
        Convert an AnomalySignal into a betting opportunity.

        Strategy:
        - If odds reversed (too high for outcome), look for value lay.
        - If odds reversed (too low), look for value back.
        
        Args:
            anomaly_type: "reversal", "spread_inflation", "liquidity_shock"
            fair_odds: Baseline/estimated fair value
            current_odds: Current market odds
            current_volume: Available liquidity
            deviation_pct: How much odds deviated
        """
        if fair_odds <= 1.0 or current_odds <= 1.0:
            return None

        fair_prob = 1.0 / fair_odds
        current_prob = 1.0 / current_odds

        # Decide action
        if current_prob < fair_prob:
            # Market is too pessimistic (odds too high) -> value backing
            action = "back"
            value_edge = current_prob - fair_prob
        else:
            # Market is too optimistic (odds too low) -> value laying
            action = "lay"
            value_edge = fair_prob - current_prob

        if value_edge < 0.02:  # Less than 2% EV
            return None

        if current_volume < self.min_liquidity_stake:
            return None

        ev_pct = value_edge * 100

        opportunity = BettingOpportunity(
            timestamp=datetime.utcnow(),
            event_id="",
            opportunity_type=OpportunityType.VALUE_BET,
            gross_profit_pct=ev_pct,
            net_profit_pct=ev_pct - self.estimated_fees_pct,
            probability_win=fair_prob,
            legs=[
                {
                    "action": action,
                    "odds": current_odds,
                    "value_edge_pct": value_edge * 100,
                }
            ],
            required_liquidity=current_volume,
            max_loss=100.0,
            execution_urgency=0.8 if deviation_pct > 10 else 0.5,
            confidence=0.6 if anomaly_type == "reversal" else 0.4,
            anomaly_source=anomaly_type,
            notes=f"Value from {anomaly_type}: deviation={deviation_pct:.1f}%, fair={fair_odds:.2f}, current={current_odds:.2f}",
        )

        self.opportunities.append(opportunity)
        return opportunity

    def get_ranked_opportunities(
        self, max_age_seconds: int = 60
    ) -> List[BettingOpportunity]:
        """
        Get opportunities ranked by:
        1. Net profit % (highest first)
        2. Execution urgency (more urgent first)
        3. Confidence
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        recent = [o for o in self.opportunities if o.timestamp > cutoff]

        # Score each opportunity
        scored = []
        for opp in recent:
            score = (
                opp.net_profit_pct * 0.5
                + opp.execution_urgency * opp.confidence * 50  # Prioritize urgent, high-confidence
            )
            scored.append((score, opp))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [opp for _, opp in scored]

    def export_opportunities(self) -> List[Dict]:
        """Export all recorded opportunities as dicts."""
        return [asdict(opp) for opp in self.opportunities[-100:]]
