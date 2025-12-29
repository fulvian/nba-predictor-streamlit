"""
Live Odds Anomaly Detector

Identifies deviations from fair value odds using:
1. Historical pre-match odds (baseline)
2. Real-time odds across multiple bookmakers
3. Implied probability mismatches (reverse favorite-longshot bias)
4. Bid-ask spread inflation (liquidity proxy)

References:
- Angelini et al. (2022): Significant mispricing persists ~20 seconds after unexpected events
- Van der Sluijs (2013): Arbitrage opportunities last median 15 minutes
- Vlastakis et al. (2009): Higher returns in low-liquidity matches
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class OddsSnapshot:
    """Single point-in-time odds observation."""
    timestamp: datetime
    sport: str
    competition: str
    event_id: str
    bookmaker: str
    market_type: str  # "match_odds", "over_under", "handicap", etc.
    outcome: str
    odds: float
    backing_odds: float  # Best bid
    laying_odds: float   # Best ask
    back_volume: float   # Liquidity at back price
    lay_volume: float    # Liquidity at lay price
    implied_prob: float  # 1 / odds (for European odds)
    confidence_score: float = 0.9


@dataclass
class AnomalySignal:
    """Detected market anomaly with severity score."""
    timestamp: datetime
    event_id: str
    anomaly_type: str  # "mispricing", "spread_inflation", "reversal", "liquidity_shock"
    severity_score: float  # 0-1, where 1 = strongest anomaly
    baseline_odds: float
    current_odds: float
    deviation_pct: float
    persistence_prob: float  # Estimated probability anomaly persists >20 sec
    description: str
    supporting_data: Dict


class AnomalyDetector:
    """Detects live odds anomalies in low-liquidity markets."""

    def __init__(
        self,
        baseline_odds_retention_hours: int = 48,
        reversal_threshold_pct: float = 5.0,
        spread_inflation_threshold_pct: float = 2.0,
        min_recency_seconds: int = 20,
    ):
        """
        Args:
            baseline_odds_retention_hours: How long to keep pre-match baselines
            reversal_threshold_pct: Trigger anomaly if odds deviate >X% from pre-match
            spread_inflation_threshold_pct: Trigger anomaly if bid-ask spread >X%
            min_recency_seconds: Only consider data fresher than X seconds
        """
        self.baseline_odds_retention_hours = baseline_odds_retention_hours
        self.reversal_threshold_pct = reversal_threshold_pct
        self.spread_inflation_threshold_pct = spread_inflation_threshold_pct
        self.min_recency_seconds = min_recency_seconds

        # Storage
        self.baseline_odds: Dict[str, Dict] = {}  # event_id -> {outcome: odds, timestamp}
        self.live_odds_history: Dict[str, deque] = {}  # event_id -> deque of snapshots
        self.detected_anomalies: List[AnomalySignal] = []

    def record_baseline(self, event_id: str, market_type: str, outcome: str, odds: float, sport: str):
        """Record pre-match odds as baseline."""
        if event_id not in self.baseline_odds:
            self.baseline_odds[event_id] = {
                "sport": sport,
                "market_type": market_type,
                "outcomes": {},
                "recorded_at": datetime.now(),
            }

        self.baseline_odds[event_id]["outcomes"][outcome] = {
            "odds": odds,
            "implied_prob": 1.0 / odds,
            "recorded_at": datetime.now(),
        }

        logger.debug(f"Recorded baseline for {event_id}/{outcome}: {odds}")

    def add_snapshot(self, snapshot: OddsSnapshot):
        """Add new live odds observation."""
        if snapshot.event_id not in self.live_odds_history:
            self.live_odds_history[snapshot.event_id] = deque(maxlen=1000)

        self.live_odds_history[snapshot.event_id].append(snapshot)

        # Immediately analyze for anomalies
        anomaly = self._detect_anomaly(snapshot)
        if anomaly:
            self.detected_anomalies.append(anomaly)

    def _detect_anomaly(self, snapshot: OddsSnapshot) -> Optional[AnomalySignal]:
        """Analyze single snapshot for anomalies."""
        event_id = snapshot.event_id

        # Check data freshness
        age_seconds = (datetime.now() - snapshot.timestamp).total_seconds()
        if age_seconds > self.min_recency_seconds:
            return None

        # 1. REVERSAL DETECTION (vs. pre-match baseline)
        reversal = self._detect_reversal(event_id, snapshot)
        if reversal:
            return reversal

        # 2. SPREAD INFLATION DETECTION
        spread_anomaly = self._detect_spread_inflation(snapshot)
        if spread_anomaly:
            return spread_anomaly

        # 3. LIQUIDITY SHOCK DETECTION
        liquidity_anomaly = self._detect_liquidity_shock(event_id, snapshot)
        if liquidity_anomaly:
            return liquidity_anomaly

        return None

    def _detect_reversal(self, event_id: str, snapshot: OddsSnapshot) -> Optional[AnomalySignal]:
        """
        Detect reverse favorite-longshot bias.
        
        Case study (Angelini et al. 2022): When underdog scores late goal,
        market undervalues their probability to win. We detect this via:
        - Odds moved >X% from baseline in unexpected direction
        - Implied probability drop indicates market overreaction
        """
        if event_id not in self.baseline_odds:
            return None

        baseline_data = self.baseline_odds[event_id]
        if snapshot.outcome not in baseline_data["outcomes"]:
            return None

        baseline_odds = baseline_data["outcomes"][snapshot.outcome]["odds"]
        baseline_prob = 1.0 / baseline_odds

        # Calculate deviation
        deviation_pct = abs(snapshot.odds - baseline_odds) / baseline_odds * 100
        prob_deviation = abs(snapshot.implied_prob - baseline_prob) / baseline_prob * 100

        if deviation_pct > self.reversal_threshold_pct:
            # Estimate persistence probability (from Angelini et al.)
            # Mispricing lasts ~20 sec max (coefficient from paper)
            time_since_change = self._estimate_change_recency(event_id, snapshot)
            persistence_prob = max(0, 1.0 - (time_since_change / 20.0))

            return AnomalySignal(
                timestamp=snapshot.timestamp,
                event_id=event_id,
                anomaly_type="reversal",
                severity_score=min(1.0, deviation_pct / self.reversal_threshold_pct),
                baseline_odds=baseline_odds,
                current_odds=snapshot.odds,
                deviation_pct=deviation_pct,
                persistence_prob=persistence_prob,
                description=f"Odds reversed {deviation_pct:.2f}% from baseline ({baseline_odds:.2f}→{snapshot.odds:.2f})",
                supporting_data={
                    "baseline_prob": baseline_prob,
                    "current_prob": snapshot.implied_prob,
                    "prob_deviation_pct": prob_deviation,
                    "baseline_timestamp": baseline_data["recorded_at"].isoformat(),
                },
            )

        return None

    def _detect_spread_inflation(self, snapshot: OddsSnapshot) -> Optional[AnomalySignal]:
        """
        Detect abnormal bid-ask spreads.
        
        Low-liquidity markets show spreads of 4-5% (vs. 2-3% for major markets).
        Sharp inflation indicates liquidity withdrawal = coming correction.
        """
        if snapshot.backing_odds == 0 or snapshot.laying_odds == 0:
            return None

        spread_pct = (snapshot.laying_odds - snapshot.backing_odds) / snapshot.backing_odds * 100

        if spread_pct > self.spread_inflation_threshold_pct:
            return AnomalySignal(
                timestamp=snapshot.timestamp,
                event_id=snapshot.event_id,
                anomaly_type="spread_inflation",
                severity_score=min(1.0, spread_pct / (self.spread_inflation_threshold_pct * 2)),
                baseline_odds=snapshot.backing_odds,
                current_odds=snapshot.laying_odds,
                deviation_pct=spread_pct,
                persistence_prob=0.4,  # Spreads tighten quickly with volume
                description=f"Bid-ask spread inflated to {spread_pct:.2f}% (backing: {snapshot.backing_odds:.2f}, laying: {snapshot.laying_odds:.2f})",
                supporting_data={
                    "backing_volume": snapshot.back_volume,
                    "laying_volume": snapshot.lay_volume,
                },
            )

        return None

    def _detect_liquidity_shock(self, event_id: str, snapshot: OddsSnapshot) -> Optional[AnomalySignal]:
        """Detect sudden volume withdrawal = market drying up."""
        if event_id not in self.live_odds_history:
            return None

        history = list(self.live_odds_history[event_id])
        if len(history) < 3:
            return None

        # Compare current volume to recent average
        recent_volumes = [
            max(h.back_volume, h.lay_volume) for h in history[-10:] if h.outcome == snapshot.outcome
        ]

        if not recent_volumes:
            return None

        avg_volume = np.mean(recent_volumes)
        current_volume = max(snapshot.back_volume, snapshot.lay_volume)

        if avg_volume > 0 and current_volume < avg_volume * 0.3:  # 70% volume drop
            return AnomalySignal(
                timestamp=snapshot.timestamp,
                event_id=event_id,
                anomaly_type="liquidity_shock",
                severity_score=min(1.0, 1.0 - (current_volume / avg_volume)),
                baseline_odds=snapshot.odds,
                current_odds=snapshot.odds,
                deviation_pct=0,
                persistence_prob=0.2,  # Liquidity often returns quickly
                description=f"Volume dropped {(1 - current_volume/avg_volume)*100:.1f}% (was {avg_volume:.0f}, now {current_volume:.0f})",
                supporting_data={"avg_recent_volume": avg_volume, "current_volume": current_volume},
            )

        return None

    def _estimate_change_recency(self, event_id: str, snapshot: OddsSnapshot) -> float:
        """Estimate seconds since odds last changed."""
        if event_id not in self.live_odds_history:
            return 0

        history = list(self.live_odds_history[event_id])
        if len(history) < 2:
            return 0

        for i in range(len(history) - 1, 0, -1):
            if history[i].odds != history[i - 1].odds:
                return (history[i].timestamp - history[i - 1].timestamp).total_seconds()

        return 0

    def get_recent_anomalies(self, seconds: int = 300) -> List[AnomalySignal]:
        """Get anomalies from last N seconds."""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return [a for a in self.detected_anomalies if a.timestamp > cutoff]

    def export_analysis(self, output_file: Optional[str] = None) -> Dict:
        """Export analysis for audit/research."""
        export_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "baseline_odds_count": len(self.baseline_odds),
            "detected_anomalies": [asdict(a) for a in self.detected_anomalies[-100:]],
            "anomaly_types": {
                "reversal": len([a for a in self.detected_anomalies if a.anomaly_type == "reversal"]),
                "spread_inflation": len([a for a in self.detected_anomalies if a.anomaly_type == "spread_inflation"]),
                "liquidity_shock": len([a for a in self.detected_anomalies if a.anomaly_type == "liquidity_shock"]),
            },
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        return export_data
