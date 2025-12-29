"""
Odds Snapshot - Core Data Structures for LOAD System

Defines dataclasses for odds snapshots, anomaly signals, and related enums.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class Sport(Enum):
    """Supported sports for market scanning."""

    FOOTBALL = "football"
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    ESPORTS = "esports"
    HORSE_RACING = "horse_racing"


class MarketType(Enum):
    """Common Betfair market types."""

    MATCH_ODDS = "MATCH_ODDS"
    OVER_UNDER_25 = "OVER_UNDER_25"
    OVER_UNDER_35 = "OVER_UNDER_35"
    CORRECT_SCORE = "CORRECT_SCORE"
    BOTH_TO_SCORE = "BOTH_TEAMS_TO_SCORE"
    SET_WINNER = "SET_WINNER"
    MONEYLINE = "MONEYLINE"


# Betfair Event Type IDs
EVENT_TYPE_IDS = {
    Sport.FOOTBALL: "1",
    Sport.TENNIS: "2",
    Sport.BASKETBALL: "7522",
    Sport.HORSE_RACING: "7",
    Sport.ESPORTS: "6423",
}


@dataclass
class OddsSnapshot:
    """
    Single observation of Betfair live odds.

    Captures all relevant data for anomaly detection including
    prices, volumes, and derived metrics.
    """

    timestamp: datetime
    market_id: str
    selection_id: int
    sport: Sport
    competition: str
    event_name: str
    runner_name: str

    # Core prices
    back_price: float
    lay_price: float

    # Volumes
    back_volume: float
    lay_volume: float
    last_traded_price: float
    total_matched: float

    # Optional metadata
    in_play: bool = False
    market_status: str = "OPEN"

    # Derived fields (computed in __post_init__)
    implied_prob_back: float = field(init=False)
    implied_prob_lay: float = field(init=False)
    spread_pct: float = field(init=False)
    mid_price: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields."""
        # Implied probabilities
        self.implied_prob_back = 1 / self.back_price if self.back_price > 1 else 0
        self.implied_prob_lay = 1 / self.lay_price if self.lay_price > 1 else 0

        # Bid-ask spread as percentage
        if self.back_price > 0:
            self.spread_pct = (self.lay_price - self.back_price) / self.back_price * 100
        else:
            self.spread_pct = 0

        # Mid price
        if self.back_price > 0 and self.lay_price > 0:
            self.mid_price = (self.back_price + self.lay_price) / 2
        else:
            self.mid_price = self.back_price or self.lay_price

    @property
    def is_liquid(self) -> bool:
        """Check if market has reasonable liquidity."""
        return self.total_matched > 1000 and self.back_volume > 50

    @property
    def fair_probability(self) -> float:
        """Estimate fair probability (average of back/lay implied probs)."""
        return (self.implied_prob_back + self.implied_prob_lay) / 2


@dataclass
class AnomalySignal:
    """
    Detected anomaly signal for potential trading opportunity.

    Contains all information needed to evaluate and execute a trade.
    """

    timestamp: datetime
    market_id: str
    selection_id: int
    runner_name: str

    # Signal classification
    signal_type: str  # reversal, spread_inflation, liquidity_shock, arbitrage
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL

    # Quantitative metrics
    deviation_pct: float
    confidence: float  # 0-1 probability signal is valid
    persistence_prob: float  # 0-1 probability anomaly persists

    # Trade recommendation
    suggested_side: str  # BACK or LAY
    suggested_price: float
    suggested_stake: float
    expected_value: float  # Expected edge

    # Context
    sport: Sport = Sport.FOOTBALL
    competition: str = ""
    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.signal_type.upper()}: "
            f"{self.runner_name} @ {self.suggested_price:.2f} "
            f"(EV: {self.expected_value:.1%}, Conf: {self.confidence:.0%})"
        )

    @property
    def is_tradeable(self) -> bool:
        """Check if signal meets minimum criteria for trading."""
        return (
            self.confidence >= 0.5
            and self.expected_value > 0.01
            and self.suggested_side in ("BACK", "LAY")
        )


@dataclass
class BaselineSnapshot:
    """
    Pre-match baseline for comparison with live odds.

    Used to detect reversals and significant deviations.
    """

    market_id: str
    selection_id: int
    runner_name: str
    sport: Sport

    # Pre-match prices
    back_price: float
    lay_price: float
    total_matched: float

    # Metadata
    recorded_at: datetime = field(default_factory=datetime.now)

    @property
    def implied_prob(self) -> float:
        """Fair implied probability from pre-match odds."""
        return (
            (1 / self.back_price + 1 / self.lay_price) / 2 if self.back_price > 1 else 0
        )
