"""
Live Odds Anomaly Detector (LOAD)

Identifies and exploits inefficiencies in live betting markets,
particularly for low-liquidity sporting events where quote distortions persist longer.

Architecture:
- odds_collector: Real-time odds ingestion (WebSocket/API)
- anomaly_detector: Deviation from fair value identification
- arbitrage_engine: Hedged betting opportunity finder
- execution_manager: Order placement and risk management
- analytics: Performance tracking and pattern learning
"""

__version__ = "0.1.0"
__author__ = "Fulvian Dev"

from .odds_collector import OddsCollector
from .anomaly_detector import AnomalyDetector
from .arbitrage_engine import ArbitrageEngine
from .execution_manager import ExecutionManager

__all__ = [
    "OddsCollector",
    "AnomalyDetector",
    "ArbitrageEngine",
    "ExecutionManager",
]
