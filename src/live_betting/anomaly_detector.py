"""
Anomaly Detector - Core Detection Engine for LOAD System

Detects 4 types of market anomalies:
1. Reversal - Live odds deviate significantly from pre-match baseline
2. Spread Inflation - Bid-ask spread exceeds normal thresholds
3. Liquidity Shock - Sudden drop in available volume
4. Cross-Market Arbitrage - Price discrepancies across correlated markets
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .odds_snapshot import OddsSnapshot, AnomalySignal, BaselineSnapshot, Sport

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for anomaly detection thresholds."""

    # Reversal detection
    reversal_pct_min: float = 8.0  # Minimum % deviation to trigger
    reversal_pct_critical: float = 15.0  # Critical threshold

    # Spread inflation
    spread_pct_warning: float = 4.0  # Warning threshold
    spread_pct_critical: float = 8.0  # Critical threshold

    # Liquidity shock
    volume_drop_pct: float = 50.0  # % volume drop = shock
    min_volume_for_shock: float = 100.0  # Min volume to consider

    # General
    min_confidence: float = 0.5
    cooldown_seconds: int = 30
    history_max_size: int = 100


# Sport-specific configurations
SPORT_CONFIGS: Dict[Sport, DetectorConfig] = {
    Sport.FOOTBALL: DetectorConfig(
        reversal_pct_min=10.0,
        spread_pct_warning=5.0,
        volume_drop_pct=60.0,
    ),
    Sport.TENNIS: DetectorConfig(
        reversal_pct_min=8.0,
        spread_pct_warning=4.0,
        volume_drop_pct=50.0,
    ),
    Sport.BASKETBALL: DetectorConfig(
        reversal_pct_min=12.0,
        spread_pct_warning=6.0,
        volume_drop_pct=50.0,
    ),
    Sport.HORSE_RACING: DetectorConfig(
        reversal_pct_min=15.0,
        spread_pct_warning=3.0,
        volume_drop_pct=40.0,
    ),
    Sport.ESPORTS: DetectorConfig(
        reversal_pct_min=10.0,
        spread_pct_warning=5.0,
        volume_drop_pct=50.0,
    ),
}


class AnomalyDetector:
    """
    Core anomaly detection engine for live betting markets.

    Maintains baselines, tracks history, and generates trading signals
    when market anomalies are detected.

    Usage:
        detector = AnomalyDetector()

        # Set pre-match baseline
        detector.set_baseline(pre_match_snapshot)

        # Process live updates
        for update in live_stream:
            signals = detector.process_update(update)
            for signal in signals:
                if signal.is_tradeable:
                    execute_trade(signal)
    """

    def __init__(self, default_config: Optional[DetectorConfig] = None):
        """
        Initialize detector with optional custom configuration.

        Args:
            default_config: Default config for sports without specific settings
        """
        self.default_config = default_config or DetectorConfig()

        # Storage
        self.baselines: Dict[str, Dict[int, BaselineSnapshot]] = {}
        self.live_history: Dict[str, Dict[int, deque]] = {}
        self.active_signals: List[AnomalySignal] = []

        # Cooldown tracking: {f"{market_id}_{selection_id}": last_alert_time}
        self.cooldowns: Dict[str, datetime] = {}

        # Statistics
        self.stats = {
            "updates_processed": 0,
            "signals_generated": 0,
            "reversals": 0,
            "spread_alerts": 0,
            "liquidity_shocks": 0,
        }

    def set_baseline(self, snapshot: OddsSnapshot) -> None:
        """
        Record pre-match odds as baseline for comparison.

        Args:
            snapshot: Pre-match odds snapshot to use as baseline
        """
        if snapshot.market_id not in self.baselines:
            self.baselines[snapshot.market_id] = {}

        baseline = BaselineSnapshot(
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            runner_name=snapshot.runner_name,
            sport=snapshot.sport,
            back_price=snapshot.back_price,
            lay_price=snapshot.lay_price,
            total_matched=snapshot.total_matched,
            recorded_at=snapshot.timestamp,
        )

        self.baselines[snapshot.market_id][snapshot.selection_id] = baseline
        logger.debug(
            f"Baseline set: {snapshot.runner_name} @ {snapshot.back_price:.2f} "
            f"(market: {snapshot.market_id})"
        )

    def process_update(self, snapshot: OddsSnapshot) -> List[AnomalySignal]:
        """
        Process a live odds update and detect any anomalies.

        Args:
            snapshot: Live odds snapshot to analyze

        Returns:
            List of detected anomaly signals (may be empty)
        """
        self.stats["updates_processed"] += 1
        signals: List[AnomalySignal] = []

        # Update history
        self._update_history(snapshot)

        # Get sport-specific config
        config = SPORT_CONFIGS.get(snapshot.sport, self.default_config)

        # Check cooldown
        key = f"{snapshot.market_id}_{snapshot.selection_id}"
        if self._in_cooldown(key, config.cooldown_seconds):
            return signals

        # Run all detectors
        reversal = self._detect_reversal(snapshot, config)
        if reversal:
            signals.append(reversal)
            self.stats["reversals"] += 1

        spread = self._detect_spread_inflation(snapshot, config)
        if spread:
            signals.append(spread)
            self.stats["spread_alerts"] += 1

        liquidity = self._detect_liquidity_shock(snapshot, config)
        if liquidity:
            signals.append(liquidity)
            self.stats["liquidity_shocks"] += 1

        # Trigger cooldown if signals generated
        if signals:
            self._trigger_cooldown(key)
            self.active_signals.extend(signals)
            self.stats["signals_generated"] += len(signals)

            for sig in signals:
                logger.info(f"🚨 Signal: {sig}")

        return signals

    def _detect_reversal(
        self, snapshot: OddsSnapshot, config: DetectorConfig
    ) -> Optional[AnomalySignal]:
        """
        Detect reversal: live odds deviate significantly from pre-match.

        A reversal indicates the market view has shifted, potentially
        creating value on the opposite side.
        """
        baseline = self.baselines.get(snapshot.market_id, {}).get(snapshot.selection_id)
        if not baseline:
            return None

        # Calculate price deviation
        price_change = snapshot.back_price - baseline.back_price
        deviation_pct = abs(price_change / baseline.back_price * 100)

        if deviation_pct < config.reversal_pct_min:
            return None

        # Determine severity
        severity = "HIGH" if deviation_pct >= config.reversal_pct_critical else "MEDIUM"

        # Direction: price up = worse for BACK, better for LAY
        # If price dropped, might be value on BACK
        is_drifting = price_change > 0  # Odds getting worse
        suggested_side = "LAY" if is_drifting else "BACK"

        # Confidence based on deviation magnitude and liquidity
        base_confidence = min(0.9, 0.4 + deviation_pct / 50)
        liquidity_factor = min(1.0, snapshot.total_matched / 10000)
        confidence = base_confidence * (0.5 + 0.5 * liquidity_factor)

        # Persistence probability
        persistence = self._estimate_persistence(snapshot)

        # Expected value estimation
        ev = deviation_pct / 100 * confidence * 0.5  # Conservative

        return AnomalySignal(
            timestamp=snapshot.timestamp,
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            runner_name=snapshot.runner_name,
            signal_type="reversal",
            severity=severity,
            deviation_pct=deviation_pct,
            confidence=confidence,
            persistence_prob=persistence,
            suggested_side=suggested_side,
            suggested_price=snapshot.lay_price if is_drifting else snapshot.back_price,
            suggested_stake=0,  # Set by ValueBettingEngine
            expected_value=ev,
            sport=snapshot.sport,
            competition=snapshot.competition,
            details={
                "baseline_price": baseline.back_price,
                "current_price": snapshot.back_price,
                "price_change": price_change,
                "direction": "drift" if is_drifting else "steam",
            },
        )

    def _detect_spread_inflation(
        self, snapshot: OddsSnapshot, config: DetectorConfig
    ) -> Optional[AnomalySignal]:
        """
        Detect spread inflation: excessive bid-ask spread indicating illiquidity.

        Wide spreads can signal uncertainty or opportunity for liquidity provision.
        """
        if snapshot.spread_pct < config.spread_pct_warning:
            return None

        severity = (
            "HIGH" if snapshot.spread_pct >= config.spread_pct_critical else "LOW"
        )

        return AnomalySignal(
            timestamp=snapshot.timestamp,
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            runner_name=snapshot.runner_name,
            signal_type="spread_inflation",
            severity=severity,
            deviation_pct=snapshot.spread_pct,
            confidence=0.7,
            persistence_prob=0.8,  # Wide spreads tend to persist
            suggested_side="",  # Informational only
            suggested_price=0,
            suggested_stake=0,
            expected_value=0,
            sport=snapshot.sport,
            competition=snapshot.competition,
            details={
                "back_price": snapshot.back_price,
                "lay_price": snapshot.lay_price,
                "spread_pct": snapshot.spread_pct,
            },
        )

    def _detect_liquidity_shock(
        self, snapshot: OddsSnapshot, config: DetectorConfig
    ) -> Optional[AnomalySignal]:
        """
        Detect liquidity shock: sudden volume drop indicating market stress.

        Liquidity shocks can precede price movements or indicate insider activity.
        """
        history = self.live_history.get(snapshot.market_id, {}).get(
            snapshot.selection_id
        )
        if not history or len(history) < 5:
            return None

        # Get recent volume average
        recent = list(history)[-10:]
        volumes = [s.total_matched for s in recent]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        if avg_volume < config.min_volume_for_shock:
            return None

        # Check for significant drop
        current_volume = snapshot.total_matched
        if avg_volume == 0:
            return None

        volume_ratio = current_volume / avg_volume
        drop_pct = (1 - volume_ratio) * 100

        if drop_pct < config.volume_drop_pct:
            return None

        return AnomalySignal(
            timestamp=snapshot.timestamp,
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            runner_name=snapshot.runner_name,
            signal_type="liquidity_shock",
            severity="HIGH",
            deviation_pct=drop_pct,
            confidence=0.6,
            persistence_prob=0.3,  # Shocks are often temporary
            suggested_side="",  # Requires deeper analysis
            suggested_price=0,
            suggested_stake=0,
            expected_value=0,
            sport=snapshot.sport,
            competition=snapshot.competition,
            details={
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "drop_pct": drop_pct,
            },
        )

    def _update_history(self, snapshot: OddsSnapshot) -> None:
        """Add snapshot to market history."""
        if snapshot.market_id not in self.live_history:
            self.live_history[snapshot.market_id] = {}

        if snapshot.selection_id not in self.live_history[snapshot.market_id]:
            self.live_history[snapshot.market_id][snapshot.selection_id] = deque(
                maxlen=self.default_config.history_max_size
            )

        self.live_history[snapshot.market_id][snapshot.selection_id].append(snapshot)

    def _estimate_persistence(self, snapshot: OddsSnapshot) -> float:
        """
        Estimate probability that the anomaly will persist.

        Based on market liquidity: illiquid markets = longer persistence.
        """
        if snapshot.total_matched < 1000:
            return 0.8  # Very illiquid - high persistence
        elif snapshot.total_matched < 10000:
            return 0.5  # Moderate liquidity
        else:
            return 0.2  # Liquid market - quick correction

    def _in_cooldown(self, key: str, cooldown_seconds: int) -> bool:
        """Check if key is in cooldown period."""
        last_alert = self.cooldowns.get(key)
        if not last_alert:
            return False

        elapsed = (datetime.now() - last_alert).total_seconds()
        return elapsed < cooldown_seconds

    def _trigger_cooldown(self, key: str) -> None:
        """Start cooldown period for key."""
        self.cooldowns[key] = datetime.now()

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return self.stats.copy()

    def clear_history(self) -> None:
        """Clear all stored history (useful for backtesting)."""
        self.live_history.clear()
        self.active_signals.clear()
        self.cooldowns.clear()

    def clear_baselines(self) -> None:
        """Clear all baselines."""
        self.baselines.clear()
