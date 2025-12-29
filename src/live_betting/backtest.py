"""
Live Betting Strategy Backtest Framework

Validates LOAD strategy on historical odds data:
1. Replay recorded odds snapshots
2. Simulate anomaly detection and opportunity generation
3. Calculate realized PnL vs. simulated trades
4. Analyze by sport, market type, time-of-day, liquidity tier

Usage:
    bt = BacktestEngine()
    bt.load_odds_file("odds_history.csv")
    bt.run(start_time=..., end_time=...)
    report = bt.generate_report()
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

import pandas as pd
import numpy as np

from .anomaly_detector import AnomalyDetector, OddsSnapshot, AnomalySignal
from .arbitrage_engine import ArbitrageEngine, BettingOpportunity, OpportunityType

logger = logging.getLogger(__name__)


class SimulatedTradeStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    CLOSED = "closed"


@dataclass
class SimulatedTrade:
    """A hypothetical trade for backtesting."""
    trade_id: str
    timestamp: datetime
    opportunity: BettingOpportunity
    
    # Entry
    execution_price: float
    execution_volume: float
    stake: float
    
    # Exit / Resolution
    resolved_at: Optional[datetime] = None
    outcome_result: Optional[str] = None  # "won", "lost", "cancelled"
    realized_pnl: float = 0.0
    status: SimulatedTradeStatus = SimulatedTradeStatus.PENDING
    
    # Metrics
    slippage_pct: float = 0.0
    fees_pct: float = 5.0


@dataclass
class BacktestMetrics:
    """Summary statistics from a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    cancelled_trades: int
    
    win_rate: float  # %
    avg_win: float  # % per trade
    avg_loss: float  # %
    profit_factor: float  # Total wins / Total losses
    total_pnl: float  # Net P&L
    total_pnl_pct: float  # % return
    
    max_drawdown: float  # %
    sharpe_ratio: float
    
    anomalies_detected: int
    opportunities_generated: int
    trades_per_anomaly: float


class BacktestEngine:
    """
    Replay historical odds and simulate strategy performance.
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        stake_pct_per_trade: float = 1.0,
        slippage_assumption_pct: float = 0.5,
    ):
        """
        Args:
            initial_bankroll: Starting capital for backtest
            stake_pct_per_trade: What % of bankroll to stake (Kelly-adjusted)
            slippage_assumption_pct: Assumed execution slippage from quoted price
        """
        self.initial_bankroll = initial_bankroll
        self.stake_pct_per_trade = stake_pct_per_trade
        self.slippage_assumption_pct = slippage_assumption_pct

        # Data
        self.odds_snapshots: List[OddsSnapshot] = []
        self.simulated_trades: List[SimulatedTrade] = []
        
        # Engines
        self.detector = AnomalyDetector()
        self.arbitrage = ArbitrageEngine()

    def load_odds_file(self, filepath: str) -> bool:
        """
        Load odds history from CSV.

        Expected columns:
            timestamp, sport, competition, event_id, bookmaker, market_type,
            outcome, odds, backing_odds, laying_odds, back_volume, lay_volume
        """
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            for _, row in df.iterrows():
                snapshot = OddsSnapshot(
                    timestamp=row['timestamp'],
                    sport=row.get('sport', 'unknown'),
                    competition=row.get('competition', 'unknown'),
                    event_id=row['event_id'],
                    bookmaker=row.get('bookmaker', 'unknown'),
                    market_type=row.get('market_type', 'unknown'),
                    outcome=row['outcome'],
                    odds=row['odds'],
                    backing_odds=row.get('backing_odds', row['odds']),
                    laying_odds=row.get('laying_odds', row['odds'] * 1.02),
                    back_volume=row.get('back_volume', 0),
                    lay_volume=row.get('lay_volume', 0),
                    implied_prob=1.0 / row['odds'],
                )
                self.odds_snapshots.append(snapshot)

            logger.info(f"Loaded {len(self.odds_snapshots)} odds snapshots from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load odds file: {e}")
            return False

    def load_odds_list(self, snapshots: List[OddsSnapshot]):
        """Load odds from in-memory list."""
        self.odds_snapshots = snapshots
        logger.info(f"Loaded {len(snapshots)} odds snapshots")

    def run(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        """
        Run backtest over loaded odds data.
        """
        if not self.odds_snapshots:
            logger.warning("No odds snapshots loaded; cannot run backtest")
            return

        # Filter by time
        data = self.odds_snapshots
        if start_time:
            data = [s for s in data if s.timestamp >= start_time]
        if end_time:
            data = [s for s in data if s.timestamp <= end_time]

        logger.info(f"Running backtest on {len(data)} snapshots")

        # Track recorded baselines
        baseline_recorded = set()

        for snapshot in data:
            # Record baselines for new events (simulate pre-match phase)
            if snapshot.event_id not in baseline_recorded:
                self.detector.record_baseline(
                    snapshot.event_id,
                    snapshot.market_type,
                    snapshot.outcome,
                    snapshot.odds,
                    snapshot.sport,
                )
                baseline_recorded.add(snapshot.event_id)

            # Add snapshot to detector
            self.detector.add_snapshot(snapshot)

            # Check for opportunities
            anomalies = self.detector.get_recent_anomalies(seconds=5)
            for anomaly in anomalies:
                # Attempt to create opportunity from anomaly
                opp = self.arbitrage.evaluate_anomaly(
                    anomaly_type=anomaly.anomaly_type,
                    fair_odds=anomaly.baseline_odds,
                    current_odds=anomaly.current_odds,
                    current_volume=snapshot.back_volume,  # Proxy
                    deviation_pct=anomaly.deviation_pct,
                )

                if opp:
                    # Simulate trade
                    trade = self._simulate_trade(
                        opp, snapshot, baseline_recorded
                    )
                    if trade:
                        self.simulated_trades.append(trade)

        logger.info(
            f"Backtest complete: {len(self.simulated_trades)} trades generated from {len(self.detector.detected_anomalies)} anomalies"
        )

    def _simulate_trade(
        self, opportunity: BettingOpportunity, snapshot: OddsSnapshot, event_ids: set
    ) -> Optional[SimulatedTrade]:
        """
        Create a simulated trade from an opportunity.
        """
        # Simplification: for value bets, assume we execute at odds quoted
        # and the outcome resolves based on fair probability
        
        trade_id = f"trade_{len(self.simulated_trades):06d}"
        stake = self.initial_bankroll * self.stake_pct_per_trade / 100.0
        
        # Slippage on execution
        execution_price = snapshot.odds * (1 - self.slippage_assumption_pct / 100.0)
        
        trade = SimulatedTrade(
            trade_id=trade_id,
            timestamp=snapshot.timestamp,
            opportunity=opportunity,
            execution_price=execution_price,
            execution_volume=snapshot.back_volume,
            stake=stake,
        )
        
        # Simulate outcome (simplified: assume fair probability)
        # In real backtest, would match actual event result
        win_prob = opportunity.probability_win
        
        # Resolve immediately (in production, would wait for market closure)
        trade.resolved_at = snapshot.timestamp + timedelta(seconds=30)
        
        if np.random.random() < win_prob:
            trade.outcome_result = "won"
            # Profit = (odds - 1) * stake
            gross_pnl = (execution_price - 1) * stake
        else:
            trade.outcome_result = "lost"
            gross_pnl = -stake
        
        # Apply fees and slippage
        fees = stake * (opportunity.net_profit_pct / 100.0)  # Crude
        trade.realized_pnl = gross_pnl - fees
        trade.status = SimulatedTradeStatus.CLOSED
        
        return trade

    def generate_report(self) -> Dict:
        """
        Generate backtest performance report.
        """
        if not self.simulated_trades:
            return {
                "status": "No trades",
                "message": "No simulated trades to analyze",
            }

        # Calculate metrics
        trades_by_status = {}
        for status in SimulatedTradeStatus:
            trades_by_status[status.value] = [t for t in self.simulated_trades if t.status == status]

        executed = [
            t for t in self.simulated_trades if t.status == SimulatedTradeStatus.CLOSED
        ]
        
        if not executed:
            return {"status": "No closed trades", "trades": len(self.simulated_trades)}

        winning = [t for t in executed if t.realized_pnl > 0]
        losing = [t for t in executed if t.realized_pnl < 0]

        win_rate = len(winning) / len(executed) * 100 if executed else 0
        avg_win = np.mean([t.realized_pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.realized_pnl for t in losing]) if losing else 0
        
        profit_factor = abs(sum(t.realized_pnl for t in winning) / sum(t.realized_pnl for t in losing)) if losing else float('inf')
        
        total_pnl = sum(t.realized_pnl for t in executed)
        total_pnl_pct = (total_pnl / self.initial_bankroll) * 100

        # Drawdown
        cumulative_pnl = np.cumsum([t.realized_pnl for t in executed])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) / self.initial_bankroll * 100 if len(drawdown) > 0 else 0

        # Sharpe (simplified)
        returns = np.array([t.realized_pnl / self.initial_bankroll for t in executed])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        metrics = BacktestMetrics(
            total_trades=len(executed),
            winning_trades=len(winning),
            losing_trades=len(losing),
            cancelled_trades=len(trades_by_status[SimulatedTradeStatus.CANCELLED.value]),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            anomalies_detected=len(self.detector.detected_anomalies),
            opportunities_generated=len(self.arbitrage.opportunities),
            trades_per_anomaly=len(executed) / len(self.detector.detected_anomalies) if self.detector.detected_anomalies else 0,
        )

        return {
            "status": "success",
            "metrics": asdict(metrics),
            "sample_trades": [asdict(t) for t in executed[-10:]],  # Last 10 trades
            "anomaly_summary": self.detector.export_analysis(),
        }

    def export_trades_csv(self, filepath: str):
        """Export all trades to CSV for further analysis."""
        trades_data = []
        for trade in self.simulated_trades:
            record = asdict(trade)
            record['opportunity_type'] = record['opportunity'].get('opportunity_type', 'unknown')
            record['net_profit_pct'] = record['opportunity'].get('net_profit_pct', 0)
            del record['opportunity']  # Remove complex nested object
            trades_data.append(record)

        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(trades_data)} trades to {filepath}")

    def plot_equity_curve(self, output_file: Optional[str] = None):
        """
        Generate equity curve visualization.
        (Returns data suitable for Plotly/Streamlit)
        """
        if not self.simulated_trades:
            return None

        executed = [t for t in self.simulated_trades if t.status == SimulatedTradeStatus.CLOSED]
        executed.sort(key=lambda t: t.timestamp)

        cumulative_pnl = [self.initial_bankroll]
        for trade in executed:
            cumulative_pnl.append(cumulative_pnl[-1] + trade.realized_pnl)

        timestamps = [executed[0].timestamp if executed else datetime.utcnow()]
        timestamps += [t.timestamp for t in executed]

        return {
            "timestamps": timestamps,
            "equity": cumulative_pnl,
            "initial_bankroll": self.initial_bankroll,
        }
