"""
Live Odds Anomaly Detector (LOAD)

Identifies and exploits inefficiencies in live betting markets,
particularly for low-liquidity sporting events where quote distortions persist longer.

Architecture:
- odds_collector: Real-time odds ingestion (WebSocket/API)
- anomaly_detector: Deviation from fair value identification
- arbitrage_engine: Hedged betting opportunity finder
- execution_manager: Order placement and risk management
- backtest: Historical validation and performance analysis

Usage:
    from live_betting import (
        BetfairOddsCollector,
        AnomalyDetector,
        ArbitrageEngine,
        ExecutionManager,
    )
    from live_betting.backtest import BacktestEngine

Example Backtest:
    python examples/live_betting_backtest_example.py

References:
    - Angelini et al. (2022): Liquidity and Information in Betting Markets
    - Vlastakis et al. (2009): Information Asymmetries in Betting
    - Van der Sluijs (2013): Market Efficiency in Betting Odds

Quick Start:
    See docs/LIVE_BETTING_STRATEGY.md or LOAD_QUICKSTART.md
"""

__version__ = "0.1.0"
__author__ = "Fulvian Dev"

from .odds_collector import BetfairOddsCollector, BetfairMarketType
from .anomaly_detector import AnomalyDetector, OddsSnapshot, AnomalySignal
from .arbitrage_engine import ArbitrageEngine, BettingOpportunity, OpportunityType
from .execution_manager import ExecutionManager, ExecutionOrder, OrderStatus, OrderSide

__all__ = [
    # Collectors
    "BetfairOddsCollector",
    "BetfairMarketType",
    # Anomaly Detection
    "AnomalyDetector",
    "OddsSnapshot",
    "AnomalySignal",
    # Arbitrage & Value
    "ArbitrageEngine",
    "BettingOpportunity",
    "OpportunityType",
    # Execution
    "ExecutionManager",
    "ExecutionOrder",
    "OrderStatus",
    "OrderSide",
]
