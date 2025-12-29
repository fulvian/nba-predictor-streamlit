"""
Value Betting Engine - EV Calculation and Position Sizing

Calculates expected value for trading opportunities and determines
optimal stake sizing using fractional Kelly Criterion.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .odds_snapshot import AnomalySignal, Sport

logger = logging.getLogger(__name__)


@dataclass
class PaperTradeConfig:
    """Configuration for paper trading mode."""

    initial_bankroll: float = 10_000.0
    max_stake_pct: float = 1.0  # Max % of bankroll per trade
    min_stake: float = 2.0  # Betfair minimum
    kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    max_exposure_pct: float = 5.0  # Max total exposure
    daily_loss_limit_pct: float = 10.0  # Daily stop-loss
    cooldown_seconds: int = 30
    commission_rate: float = 0.05  # 5% Betfair commission
    slippage_pct: float = 0.5  # Simulated slippage
    min_ev_threshold: float = 0.02  # Min 2% EV to trade


@dataclass
class TradeRecommendation:
    """Trading recommendation from the value engine."""

    signal: AnomalySignal
    action: str  # BACK, LAY, SKIP
    stake: float
    expected_value: float
    kelly_fraction: float
    risk_reward_ratio: float
    notes: str

    # Execution details (filled after trade)
    executed: bool = False
    executed_price: float = 0
    executed_stake: float = 0
    pnl: float = 0

    def __str__(self) -> str:
        return (
            f"{self.action} {self.signal.runner_name} "
            f"@ {self.signal.suggested_price:.2f} "
            f"stake €{self.stake:.2f} (EV: {self.expected_value:.1%})"
        )


@dataclass
class TradingSession:
    """Tracks trading session statistics."""

    start_time: datetime = field(default_factory=datetime.now)
    trades: List[TradeRecommendation] = field(default_factory=list)
    starting_bankroll: float = 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.executed)

    @property
    def win_rate(self) -> float:
        executed = [t for t in self.trades if t.executed]
        if not executed:
            return 0
        wins = sum(1 for t in executed if t.pnl > 0)
        return wins / len(executed)

    @property
    def trade_count(self) -> int:
        return sum(1 for t in self.trades if t.executed)


class ValueBettingEngine:
    """
    Calculates expected value and optimal position sizing.

    Implements:
    - Expected Value (EV) calculation
    - Fractional Kelly Criterion sizing
    - Risk management constraints
    - Paper trading simulation

    Usage:
        engine = ValueBettingEngine(bankroll=10000)

        for signal in signals:
            rec = engine.evaluate_signal(signal)
            if rec.action != "SKIP":
                result = engine.execute_paper_trade(rec)
    """

    def __init__(
        self,
        config: Optional[PaperTradeConfig] = None,
    ):
        """
        Initialize the value betting engine.

        Args:
            config: Paper trading configuration
        """
        self.config = config or PaperTradeConfig()
        self.bankroll = self.config.initial_bankroll

        # Tracking
        self.session = TradingSession(starting_bankroll=self.bankroll)
        self.daily_pnl = 0.0
        self.current_exposure = 0.0

        logger.info(
            f"ValueBettingEngine initialized: "
            f"bankroll=€{self.bankroll:,.2f}, "
            f"kelly={self.config.kelly_fraction:.0%}"
        )

    def evaluate_signal(self, signal: AnomalySignal) -> TradeRecommendation:
        """
        Evaluate an anomaly signal and generate a trade recommendation.

        Args:
            signal: The anomaly signal to evaluate

        Returns:
            TradeRecommendation with action, stake, and expected value
        """
        # Skip non-tradeable signals
        if signal.signal_type == "spread_inflation":
            return self._skip_recommendation(
                signal, "Spread inflation is informational only"
            )

        if not signal.suggested_side:
            return self._skip_recommendation(signal, "No suggested side")

        # Check risk limits
        limit_check = self._check_risk_limits()
        if limit_check:
            return self._skip_recommendation(signal, limit_check)

        # Calculate EV
        ev, p_win = self._calculate_ev(signal)

        if ev < self.config.min_ev_threshold:
            return self._skip_recommendation(
                signal,
                f"EV {ev:.1%} below threshold {self.config.min_ev_threshold:.1%}",
            )

        # Calculate Kelly stake
        kelly_stake = self._kelly_criterion(signal.suggested_price, p_win)

        # Apply stake limits
        max_stake = self.bankroll * (self.config.max_stake_pct / 100)
        final_stake = min(kelly_stake, max_stake)
        final_stake = max(self.config.min_stake, final_stake)

        # Risk/reward ratio
        potential_profit = final_stake * (signal.suggested_price - 1)
        risk_reward = potential_profit / final_stake if final_stake > 0 else 0

        return TradeRecommendation(
            signal=signal,
            action=signal.suggested_side,
            stake=round(final_stake, 2),
            expected_value=ev,
            kelly_fraction=kelly_stake / self.bankroll if self.bankroll > 0 else 0,
            risk_reward_ratio=risk_reward,
            notes=f"Confidence: {signal.confidence:.0%}, Persistence: {signal.persistence_prob:.0%}",
        )

    def _calculate_ev(self, signal: AnomalySignal) -> Tuple[float, float]:
        """
        Calculate Expected Value based on signal characteristics.

        Returns:
            Tuple of (expected_value, estimated_win_probability)
        """
        if signal.signal_type == "reversal":
            # EV based on deviation from baseline
            deviation = signal.deviation_pct / 100

            # Estimate win probability
            # Higher confidence + larger deviation = higher win prob
            p_win = 0.5 + (signal.confidence * deviation * 0.3)
            p_win = min(0.70, max(0.35, p_win))  # Cap at 35-70%

            # EV = p_win * profit - (1-p_win) * loss
            odds = signal.suggested_price
            profit_if_win = (odds - 1) * (1 - self.config.commission_rate)
            loss_if_lose = 1

            ev = p_win * profit_if_win - (1 - p_win) * loss_if_lose

            return max(0, ev), p_win

        elif signal.signal_type == "liquidity_shock":
            # More uncertain - use conservative estimates
            p_win = 0.52
            ev = 0.02  # 2% EV estimate
            return ev, p_win

        return 0, 0.5

    def _kelly_criterion(self, odds: float, p_win: float) -> float:
        """
        Calculate stake using fractional Kelly Criterion.

        Formula: f* = (bp - q) / b
        Where: b = odds - 1, p = win probability, q = 1 - p

        Args:
            odds: Decimal odds for the bet
            p_win: Estimated probability of winning

        Returns:
            Recommended stake amount
        """
        if odds <= 1 or p_win <= 0 or p_win >= 1:
            return 0

        b = odds - 1
        p = p_win
        q = 1 - p

        # Full Kelly
        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0

        # Apply fractional Kelly
        fractional = kelly * self.config.kelly_fraction

        # Calculate stake
        stake = self.bankroll * fractional

        return max(self.config.min_stake, stake)

    def _check_risk_limits(self) -> Optional[str]:
        """
        Check if current risk limits allow new trades.

        Returns:
            Error message if limit breached, None if OK
        """
        # Daily loss limit
        daily_limit = self.config.initial_bankroll * (
            self.config.daily_loss_limit_pct / 100
        )
        if self.daily_pnl < -daily_limit:
            return f"Daily loss limit reached (€{-self.daily_pnl:.2f})"

        # Max exposure
        max_exposure = self.bankroll * (self.config.max_exposure_pct / 100)
        if self.current_exposure >= max_exposure:
            return f"Max exposure reached (€{self.current_exposure:.2f})"

        # Bankroll depletion
        if self.bankroll < self.config.min_stake * 10:
            return "Insufficient bankroll"

        return None

    def _skip_recommendation(
        self, signal: AnomalySignal, reason: str
    ) -> TradeRecommendation:
        """Create a SKIP recommendation."""
        return TradeRecommendation(
            signal=signal,
            action="SKIP",
            stake=0,
            expected_value=0,
            kelly_fraction=0,
            risk_reward_ratio=0,
            notes=reason,
        )

    def execute_paper_trade(
        self,
        rec: TradeRecommendation,
        outcome: Optional[bool] = None,
    ) -> TradeRecommendation:
        """
        Execute a paper trade (simulation).

        Args:
            rec: Trade recommendation to execute
            outcome: If known, the actual outcome (won/lost)
                    If None, simulates based on probability

        Returns:
            Updated recommendation with execution details
        """
        if rec.action == "SKIP":
            return rec

        # Simulate slippage
        slippage = rec.signal.suggested_price * (self.config.slippage_pct / 100)
        if rec.action == "BACK":
            executed_price = rec.signal.suggested_price - slippage
        else:
            executed_price = rec.signal.suggested_price + slippage

        rec.executed = True
        rec.executed_price = executed_price
        rec.executed_stake = rec.stake

        # Determine outcome
        if outcome is None:
            # Simulate based on expected probability
            import random

            _, p_win = self._calculate_ev(rec.signal)
            outcome = random.random() < p_win

        # Calculate PnL
        if outcome:
            gross_profit = rec.stake * (executed_price - 1)
            commission = gross_profit * self.config.commission_rate
            rec.pnl = gross_profit - commission
        else:
            rec.pnl = -rec.stake

        # Update tracking
        self.bankroll += rec.pnl
        self.daily_pnl += rec.pnl
        self.session.trades.append(rec)

        logger.info(
            f"📝 Paper trade: {rec.action} {rec.signal.runner_name} "
            f"@ {executed_price:.2f} stake €{rec.stake:.2f} "
            f"→ PnL: €{rec.pnl:+.2f} (Bankroll: €{self.bankroll:,.2f})"
        )

        return rec

    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            "bankroll": self.bankroll,
            "starting_bankroll": self.session.starting_bankroll,
            "total_pnl": self.session.total_pnl,
            "pnl_pct": (
                self.session.total_pnl / self.session.starting_bankroll * 100
                if self.session.starting_bankroll > 0
                else 0
            ),
            "trade_count": self.session.trade_count,
            "win_rate": self.session.win_rate,
            "daily_pnl": self.daily_pnl,
        }

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L tracking (call at start of new day)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")

    def reset_session(self) -> None:
        """Reset entire session and restore initial bankroll."""
        self.bankroll = self.config.initial_bankroll
        self.daily_pnl = 0.0
        self.current_exposure = 0.0
        self.session = TradingSession(starting_bankroll=self.bankroll)
        logger.info(f"Session reset - bankroll restored to €{self.bankroll:,.2f}")
