"""
Trade Executor for Betfair Exchange.
Handles order placement with paper trading mode for development.
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from betfairlightweight import filters

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a trade attempt."""

    success: bool
    bet_id: Optional[str] = None
    average_price: Optional[float] = None
    size_matched: float = 0.0
    error: Optional[str] = None
    is_paper: bool = False


class TradeExecutor:
    """
    Executes trades on Betfair Exchange.

    Best Practices 2025 (from Perplexity research):
    - Paper trading mode for development (no Betfair sandbox exists)
    - Stake sizing: 1-2% of bankroll max
    - Fast execution for scalping
    """

    def __init__(
        self,
        client,  # BetfairClient instance
        paper_mode: bool = True,  # Default to paper trading
        max_stake: float = 20.0,  # Hard limit per bet
    ):
        self.client = client
        self.paper_mode = paper_mode
        self.max_stake = max_stake

        # Paper trading state
        self._paper_trades: Dict[str, TradeResult] = {}
        self._paper_bet_counter = 0

        logger.info(f"TradeExecutor initialized. Paper Mode: {paper_mode}")

    def place_back_order(
        self,
        market_id: str,
        selection_id: int,
        stake: float,
        price: float,
    ) -> TradeResult:
        """
        Place a BACK order (bet that selection wins).
        Returns TradeResult with bet_id if successful.
        """
        return self._place_order(market_id, selection_id, stake, price, side="BACK")

    def place_lay_order(
        self,
        market_id: str,
        selection_id: int,
        stake: float,
        price: float,
    ) -> TradeResult:
        """
        Place a LAY order (bet that selection loses).
        Returns TradeResult with bet_id if successful.
        """
        return self._place_order(market_id, selection_id, stake, price, side="LAY")

    def _place_order(
        self,
        market_id: str,
        selection_id: int,
        stake: float,
        price: float,
        side: str,
    ) -> TradeResult:
        """Internal order placement logic."""

        # Validate stake
        if stake > self.max_stake:
            logger.warning(f"Stake {stake} exceeds max {self.max_stake}. Capping.")
            stake = self.max_stake

        if stake < 2.0:
            return TradeResult(success=False, error="Stake below Betfair minimum (€2)")

        # Paper Trading Mode
        if self.paper_mode:
            return self._paper_trade(market_id, selection_id, stake, price, side)

        # Real Trading
        try:
            limit_order = filters.limit_order(
                size=stake,
                price=price,
                persistence_type="LAPSE",  # Cancel unmatched at market end
            )

            instruction = filters.place_instruction(
                order_type="LIMIT",
                selection_id=selection_id,
                side=side,
                limit_order=limit_order,
            )

            result = self.client.client.betting.place_orders(
                market_id=market_id, instructions=[instruction]
            )

            if result.status == "SUCCESS":
                report = result.place_instruction_reports[0]
                logger.info(
                    f"✅ {side} order placed: BetId={report.bet_id}, "
                    f"Price={price}, Stake={stake}"
                )
                return TradeResult(
                    success=True,
                    bet_id=report.bet_id,
                    average_price=report.average_price_matched,
                    size_matched=report.size_matched,
                    is_paper=False,
                )
            else:
                error_msg = str(result.error_code) if result.error_code else "Unknown"
                logger.error(f"❌ Order failed: {error_msg}")
                return TradeResult(success=False, error=error_msg)

        except Exception as e:
            logger.error(f"❌ Order exception: {e}")
            return TradeResult(success=False, error=str(e))

    def _paper_trade(
        self,
        market_id: str,
        selection_id: int,
        stake: float,
        price: float,
        side: str,
    ) -> TradeResult:
        """Simulate a trade without real money."""
        self._paper_bet_counter += 1
        paper_bet_id = f"PAPER_{self._paper_bet_counter}_{int(time.time())}"

        result = TradeResult(
            success=True,
            bet_id=paper_bet_id,
            average_price=price,  # Assume instant fill at requested price
            size_matched=stake,
            is_paper=True,
        )

        self._paper_trades[paper_bet_id] = result

        logger.info(
            f"📝 PAPER {side}: BetId={paper_bet_id}, "
            f"Market={market_id}, Selection={selection_id}, "
            f"Price={price}, Stake={stake}"
        )

        return result

    def cancel_order(self, market_id: str, bet_id: str) -> bool:
        """Cancel an unmatched order."""
        if self.paper_mode or bet_id.startswith("PAPER_"):
            logger.info(f"📝 PAPER CANCEL: {bet_id}")
            return True

        try:
            instruction = filters.cancel_instruction(bet_id=bet_id)
            result = self.client.client.betting.cancel_orders(
                market_id=market_id, instructions=[instruction]
            )
            return result.status == "SUCCESS"
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
