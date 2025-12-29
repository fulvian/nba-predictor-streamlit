"""
Live Betting Module - LOAD System
Live Odds Anomaly Detector for Betfair Exchange

Detects and exploits market inefficiencies in live betting markets,
focusing on minor leagues and illiquid markets.
"""

from .odds_snapshot import OddsSnapshot, AnomalySignal, Sport, MarketType
from .anomaly_detector import AnomalyDetector
from .market_scanner import MarketScanner, MarketCandidate
from .value_betting_engine import ValueBettingEngine, TradeRecommendation
from .backtest.replay_engine import ReplayEngine, BacktestResult

__all__ = [
    "OddsSnapshot",
    "AnomalySignal",
    "Sport",
    "MarketType",
    "AnomalyDetector",
    "MarketScanner",
    "MarketCandidate",
    "ValueBettingEngine",
    "TradeRecommendation",
    "ReplayEngine",
    "BacktestResult",
]

__version__ = "0.1.0"
