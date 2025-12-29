"""
Replay Engine - Backtesting Framework for LOAD System

Simulates live trading using historical odds data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import time

from ..odds_snapshot import OddsSnapshot, Sport, AnomalySignal
from ..anomaly_detector import AnomalyDetector
from ..value_betting_engine import (
    ValueBettingEngine,
    TradeRecommendation,
    PaperTradeConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_pnl: float
    roi: float
    win_rate: float
    trade_count: int
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame

    def __str__(self):
        return (
            f"Backtest Result:\n"
            f"  PnL: €{self.total_pnl:+.2f} (ROI: {self.roi:.1%})\n"
            f"  Trades: {self.trade_count} (Win Rate: {self.win_rate:.1%})\n"
            f"  Max Drawdown: {self.max_drawdown:.1%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}"
        )


class ReplayEngine:
    """
    Simulates market replay for strategy backtesting.
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        config: Optional[PaperTradeConfig] = None,
    ):
        self.config = config or PaperTradeConfig(initial_bankroll=initial_bankroll)
        self.detector = AnomalyDetector()
        self.engine = ValueBettingEngine(config=self.config)
        self.trades: List[TradeRecommendation] = []
        self.equity_history = []

    def run_backtest(
        self,
        odds_data: pd.DataFrame,
        market_outcomes: Optional[Dict[str, Dict[int, bool]]] = None,
        simulated_delay_ms: int = 0,
    ) -> BacktestResult:
        """
        Run backtest on historical odds DataFrame.

        Args:
            odds_data: DataFrame with columns matching OddsSnapshot fields.
                      Must be sorted by timestamp.
            market_outcomes: Dict mapping market_id -> {selection_id: is_winner}
                           Used for accurate PnL calculation.
            simulated_delay_ms: Artificial delay per tick for realism (0 = max speed)

        Returns:
            BacktestResult object
        """
        logger.info(f"Starting backtest on {len(odds_data)} tick updates...")
        start_time = time.time()

        # Reset state
        self.detector.clear_history()
        self.engine.reset_session()
        self.trades = []
        self.equity_history = [
            {
                "timestamp": odds_data.iloc[0]["timestamp"],
                "equity": self.engine.bankroll,
            }
        ]

        # Process ticks
        for _, row in odds_data.iterrows():
            snapshot = self._row_to_snapshot(row)

            # 1. Detect
            signals = self.detector.process_update(snapshot)

            # 2. Trade
            for signal in signals:
                rec = self.engine.evaluate_signal(signal)

                if rec.action != "SKIP":
                    # Determine outcome if known
                    outcome = None
                    if market_outcomes:
                        winners = market_outcomes.get(snapshot.market_id, {})
                        outcome = winners.get(
                            snapshot.selection_id, False
                        )  # Default assumption

                        # Invert for LAY
                        if rec.action == "LAY":
                            outcome = not outcome

                    # Execute
                    executed_rec = self.engine.execute_paper_trade(rec, outcome=outcome)
                    self.trades.append(executed_rec)

                    # Record Equity
                    self.equity_history.append(
                        {
                            "timestamp": snapshot.timestamp,
                            "equity": self.engine.bankroll,
                        }
                    )

            if simulated_delay_ms > 0:
                time.sleep(simulated_delay_ms / 1000)

        duration = time.time() - start_time
        logger.info(f"Backtest completed in {duration:.2f}s")

        return self._calculate_metrics()

    def _row_to_snapshot(self, row: pd.Series) -> OddsSnapshot:
        """Convert DataFrame row to OddsSnapshot."""
        # Clean sport string
        sport_str = row.get("sport", "football").lower()
        if "tennis" in sport_str:
            sport = Sport.TENNIS
        elif "basket" in sport_str:
            sport = Sport.BASKETBALL
        elif "horse" in sport_str:
            sport = Sport.HORSE_RACING
        else:
            sport = Sport.FOOTBALL

        return OddsSnapshot(
            timestamp=row["timestamp"],
            market_id=str(row["market_id"]),
            selection_id=int(row["selection_id"]),
            sport=sport,
            competition=row.get("competition", "Unknown"),
            event_name=row.get("event_name", "Unknown Event"),
            runner_name=row.get("runner_name", "Runner"),
            back_price=float(row.get("back_price", 0)),
            lay_price=float(row.get("lay_price", 0)),
            back_volume=float(row.get("back_volume", 0)),
            lay_volume=float(row.get("lay_volume", 0)),
            last_traded_price=float(row.get("ltp", 0)),
            total_matched=float(row.get("total_matched", 0)),
            in_play=bool(row.get("in_play", True)),
        )

    def _calculate_metrics(self) -> BacktestResult:
        """Compute performance metrics."""
        equity_df = pd.DataFrame(self.equity_history)
        trades_df = (
            pd.DataFrame([vars(t) for t in self.trades])
            if self.trades
            else pd.DataFrame()
        )

        total_pnl = self.engine.session.total_pnl
        roi = total_pnl / self.config.initial_bankroll

        win_rate = 0.0
        if not trades_df.empty:
            win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)

        # Drawdown
        if not equity_df.empty:
            equity_df["peak"] = equity_df["equity"].cummax()
            equity_df["drawdown"] = (
                equity_df["equity"] - equity_df["peak"]
            ) / equity_df["peak"]
            max_drawdown = abs(equity_df["drawdown"].min())

            # Sharpe (Daily approximation)
            # Resample strictly if timestamps provided
            # For now, approximate based on trade returns
            returns = equity_df["equity"].pct_change().fillna(0)
            sharpe = (
                (returns.mean() / returns.std() * np.sqrt(252))
                if returns.std() != 0
                else 0
            )
        else:
            max_drawdown = 0.0
            sharpe = 0.0
            equity_df = pd.DataFrame(columns=["timestamp", "equity"])

        return BacktestResult(
            total_pnl=total_pnl,
            roi=roi,
            win_rate=win_rate,
            trade_count=len(self.trades),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            equity_curve=equity_df,
            trades=trades_df,
        )
