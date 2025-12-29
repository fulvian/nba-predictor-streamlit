"""
Position Manager for Betfair Trading.
Manages open positions and exit strategy.

Best Practices 2025 (from Perplexity research):
- Take Profit: 2-5% ROI or 1-2 ticks for scalping
- Stop Loss: 2% hard limit or fixed ticks
- Trailing stop on profitable trades
- Time-based exit (avoid capital lock)
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.nba_predictor.betfair.trade_executor import TradeExecutor, TradeResult
from src.nba_predictor.betfair.panic_detector import PanicAlert

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Represents an open trading position."""

    bet_id: str
    market_id: str
    selection_id: int
    side: str  # BACK or LAY
    entry_price: float
    stake: float
    entry_time: float
    status: str = "OPEN"  # OPEN, CLOSED
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    is_paper: bool = False


class PositionManager:
    """
    Manages open positions and implements exit strategy.

    Exit Conditions (2025 Best Practices):
    1. Take Profit: +5 ticks from entry
    2. Stop Loss: -10 ticks from entry
    3. Timeout: 5 minutes max hold time
    """

    def __init__(
        self,
        executor: TradeExecutor,
        take_profit_ticks: int = 5,
        stop_loss_ticks: int = 10,
        timeout_seconds: int = 300,  # 5 minutes
        max_positions: int = 3,
    ):
        self.executor = executor
        self.take_profit_ticks = take_profit_ticks
        self.stop_loss_ticks = stop_loss_ticks
        self.timeout_seconds = timeout_seconds
        self.max_positions = max_positions

        self.positions: Dict[str, OpenPosition] = {}
        self.closed_positions: List[OpenPosition] = []

        # Daily stats
        self.daily_pnl = 0.0
        self.daily_trades = 0

        logger.info(
            f"PositionManager initialized: TP={take_profit_ticks} ticks, "
            f"SL={stop_loss_ticks} ticks, Timeout={timeout_seconds}s"
        )

    def open_position(
        self,
        alert: PanicAlert,
        stake: float,
        current_price: float,
    ) -> Optional[str]:
        """
        Opens a position based on a panic alert.

        Strategy: LAY the drifting runner (bet it loses).
        """
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached. Skipping.")
            return None

        # Place LAY order (bet against the drifting favorite)
        result = self.executor.place_lay_order(
            market_id=alert.market_id,
            selection_id=alert.runner_id,
            stake=stake,
            price=current_price,
        )

        if not result.success:
            logger.error(f"Failed to open position: {result.error}")
            return None

        position = OpenPosition(
            bet_id=result.bet_id,
            market_id=alert.market_id,
            selection_id=alert.runner_id,
            side="LAY",
            entry_price=result.average_price or current_price,
            stake=stake,
            entry_time=time.time(),
            is_paper=result.is_paper,
        )

        self.positions[result.bet_id] = position
        self.daily_trades += 1

        logger.info(f"📈 Position OPENED: {result.bet_id} LAY @ {position.entry_price}")

        return result.bet_id

    def check_exit_conditions(self, market_book: Any) -> List[str]:
        """
        Checks all open positions for exit conditions.
        Returns list of closed bet_ids.
        """
        closed = []
        now = time.time()

        for bet_id, pos in list(self.positions.items()):
            if pos.market_id != market_book.market_id:
                continue

            # Find current price for this selection
            current_price = None
            for runner in market_book.runners:
                if runner.selection_id == pos.selection_id:
                    if runner.ex.available_to_back:
                        current_price = runner.ex.available_to_back[0].price
                    break

            if current_price is None:
                continue

            # Calculate tick difference
            tick_diff = self._calculate_tick_diff(pos.entry_price, current_price)

            exit_reason = None

            # 1. Take Profit (LAY wins when price goes UP)
            if pos.side == "LAY" and tick_diff >= self.take_profit_ticks:
                exit_reason = f"TAKE_PROFIT (+{tick_diff} ticks)"

            # 2. Stop Loss (LAY loses when price goes DOWN)
            elif pos.side == "LAY" and tick_diff <= -self.stop_loss_ticks:
                exit_reason = f"STOP_LOSS ({tick_diff} ticks)"

            # 3. Timeout
            elif (now - pos.entry_time) > self.timeout_seconds:
                exit_reason = "TIMEOUT"

            if exit_reason:
                self._close_position(pos, current_price, exit_reason)
                closed.append(bet_id)

        return closed

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        reason: str,
    ):
        """Closes a position by placing opposite order."""

        # For LAY, we BACK to close
        opposite_side = "BACK" if pos.side == "LAY" else "LAY"

        result = self.executor.place_back_order(
            market_id=pos.market_id,
            selection_id=pos.selection_id,
            stake=pos.stake,
            price=exit_price,
        )

        if result.success:
            # Calculate P&L for LAY trade
            # LAY wins when price goes UP (we keep stake if selection loses)
            # Simplified: P&L = (exit_price - entry_price) * stake / entry_price
            if pos.side == "LAY":
                pnl = (exit_price - pos.entry_price) * pos.stake / pos.entry_price
            else:
                pnl = (pos.entry_price - exit_price) * pos.stake / exit_price

            pos.status = "CLOSED"
            pos.exit_price = exit_price
            pos.exit_reason = reason
            pos.pnl = pnl

            self.daily_pnl += pnl

            # Move to closed
            self.closed_positions.append(pos)
            del self.positions[pos.bet_id]

            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                f"{emoji} Position CLOSED: {pos.bet_id} Reason={reason}, P&L=€{pnl:.2f}"
            )
        else:
            logger.error(f"Failed to close position {pos.bet_id}: {result.error}")

    def _calculate_tick_diff(self, price1: float, price2: float) -> int:
        """Calculate tick difference between two prices."""
        # Simplified tick calculation
        if price1 == price2:
            return 0

        low, high = sorted([price1, price2])
        ticks = 0
        curr = low

        while curr < high:
            if curr < 2.0:
                inc = 0.01
            elif curr < 3.0:
                inc = 0.02
            elif curr < 4.0:
                inc = 0.05
            elif curr < 6.0:
                inc = 0.1
            elif curr < 10.0:
                inc = 0.2
            elif curr < 20.0:
                inc = 0.5
            elif curr < 30.0:
                inc = 1.0
            elif curr < 50.0:
                inc = 2.0
            elif curr < 100.0:
                inc = 5.0
            else:
                inc = 10.0

            curr += inc
            ticks += 1

        return ticks if price2 > price1 else -ticks

    def get_stats(self) -> Dict[str, Any]:
        """Returns current trading stats."""
        return {
            "open_positions": len(self.positions),
            "closed_today": len(self.closed_positions),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "paper_mode": self.executor.paper_mode,
        }
