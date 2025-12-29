import logging
import time
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PanicAlert:
    timestamp: float
    market_id: str
    runner_id: int
    alert_type: str  # "DRIFT", "VOLUME", "PANIC_REVERSAL"
    details: str
    severity: str  # "INFO", "WARNING", "CRITICAL"


class PanicDetector:
    """
    Analyzes a stream of Betfair MarketBooks to detect Microstructure Anomalies.

    Strategies:
    1. Rapid Drift: Price moves > X ticks in Y seconds.
    2. Volume Spike: Traded Volume > $Z in Y seconds.
    3. Panic Reversal: Combination of Drift + Spike on a Favorite that isn't losing yet.
    """

    def __init__(
        self,
        drift_ticks: int = 10,
        drift_seconds: int = 3,
        volume_threshold: float = 1000.0,
        volume_seconds: int = 5,
    ):
        self.DRIFT_TICKS = drift_ticks
        self.DRIFT_SECONDS = drift_seconds
        self.VOLUME_THRESHOLD = volume_threshold
        self.VOLUME_SECONDS = volume_seconds

        # History: { market_id: { runner_id: deque([(ts, price, volume), ...]) } }
        self.history: Dict[str, Dict[int, deque]] = {}

        # Last Alert: { market_id: { runner_id: last_alert_ts } } to avoid spam
        self.last_alerts: Dict[str, Dict[int, float]] = {}
        self.COOLDOWN_SECONDS = 30

    def process_update(self, market_book: Any) -> List[PanicAlert]:
        """
        Ingests a MarketBook update and returns triggered alerts.
        """
        alerts = []
        ts = time.time()  # Use local receipt time for stability
        market_id = market_book.market_id

        if market_id not in self.history:
            self.history[market_id] = {}
            self.last_alerts[market_id] = {}

        for runner in market_book.runners:
            if runner.status != "ACTIVE":
                continue

            rid = runner.selection_id

            # Extract metrics
            current_price = (
                runner.ex.available_to_back[0].price
                if runner.ex.available_to_back
                else None
            )
            total_matched = runner.total_matched if runner.total_matched else 0.0

            if current_price is None:
                continue

            # distinct runner history
            if rid not in self.history[market_id]:
                self.history[market_id][rid] = deque()

            history = self.history[market_id][rid]

            # 1. Update History
            history.append((ts, current_price, total_matched))

            # 2. Prune Old Data (Max window needed is max(drift_sec, vol_sec))
            max_window = max(self.DRIFT_SECONDS, self.VOLUME_SECONDS)
            while history and (ts - history[0][0] > max_window):
                history.popleft()

            if len(history) < 2:
                continue

            # 3. Analyze Patterns

            # A. Traded Volume Spike (Last 5s)
            # Find closest snapshot to VOLUME_SECONDS ago
            vol_start_snap = self._get_snapshot_at_age(history, ts, self.VOLUME_SECONDS)
            volume_delta = total_matched - vol_start_snap[2]

            is_volume_spike = volume_delta >= self.VOLUME_THRESHOLD

            # B. Price Drift (Last 3s)
            drift_start_snap = self._get_snapshot_at_age(
                history, ts, self.DRIFT_SECONDS
            )
            start_price = drift_start_snap[1]
            price_delta_ticks = self._calculate_ticks(start_price, current_price)

            # "Worsened" means Price went UP (Back odds increased)
            is_rapid_drift = price_delta_ticks >= self.DRIFT_TICKS

            # C. Check Alerts
            if self._in_cooldown(market_id, rid, ts):
                continue

            if is_rapid_drift and is_volume_spike:
                msg = f"PANIC REVERSAL! Odds drifted {price_delta_ticks} ticks ({start_price}->{current_price}) with ${volume_delta:.0f} volume in <{self.VOLUME_SECONDS}s."
                alerts.append(
                    PanicAlert(ts, market_id, rid, "PANIC_REVERSAL", msg, "CRITICAL")
                )
                self._trigger_cooldown(market_id, rid, ts)

            elif is_rapid_drift:
                msg = f"Rapid Drift: +{price_delta_ticks} ticks ({start_price}->{current_price}) in {self.DRIFT_SECONDS}s."
                alerts.append(PanicAlert(ts, market_id, rid, "DRIFT", msg, "WARNING"))
                self._trigger_cooldown(market_id, rid, ts)

            elif is_volume_spike:
                msg = f"Volume Spike: +${volume_delta:.0f} matched in {self.VOLUME_SECONDS}s."
                alerts.append(PanicAlert(ts, market_id, rid, "VOLUME", msg, "INFO"))
                self._trigger_cooldown(market_id, rid, ts)

        return alerts

    def _get_snapshot_at_age(self, history, current_ts, target_age):
        """Finds the snapshot closest to (current_ts - target_age)."""
        target_ts = current_ts - target_age
        # history is implicitly sorted by time
        # Linear scan is fine for small deque size (~5-10 items)
        best_snap = history[0]
        for item in history:
            if item[0] >= target_ts:
                best_snap = item
                break
        return best_snap

    def _calculate_ticks(self, start_price, end_price):
        """
        Approximation of Betfair ticks.
        Real implementation needs the ladder lookup, but for approximation:
        Price < 2: 0.01
        Price < 3: 0.02
        Price < 4: 0.05
        Price < 6: 0.1
        Price < 10: 0.2
        ...
        Simple diff for now, treating rough delta.
        """
        # Just return raw diff * 100 for now to be safe/simple,
        # or just raw diff. The user asked for "10 ticks", let's approximate
        # generic tick size average ~0.02 for favorites?
        # Actually implementation needs to be better.
        # Let's use a simplified tick counter logic for common ranges.
        return self._get_tick_diff(start_price, end_price)

    def _get_tick_diff(self, p1, p2):
        # Very rough approximation logic for speed
        # Counts how many "minimum increments" exist between p1 and p2 based on p1 bracket
        if p1 == p2:
            return 0

        # Direction
        low, high = sorted([p1, p2])

        # Simplified brackets
        ticks = 0
        curr = low
        while curr < high:
            inc = 0.01
            if curr >= 100:
                inc = 1000  # irrelevant
            elif curr >= 50:
                inc = 1.0  # 50-100
            elif curr >= 30:
                inc = 0.5  # 30-50
            elif curr >= 20:
                inc = 0.2  # 20-30
            elif curr >= 10:
                inc = 0.1  # 10-20
            elif curr >= 6:
                inc = 0.1  # 6-10 (Wait, 6-10 is 0.1? No 0.2? Checking docs...)
            # Standard: 1.01-2 (0.01), 2-3 (0.02), 3-4 (0.05), 4-6 (0.1), 6-10 (0.2), 10-20 (0.5), 20-30 (1), 30-50 (2), 50-100 (5), 100+ (10)

            # Corrected Standard
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

        return ticks if p2 > p1 else -ticks

    def _in_cooldown(self, mid, rid, ts):
        last = self.last_alerts.get(mid, {}).get(rid, 0)
        return (ts - last) < self.COOLDOWN_SECONDS

    def _trigger_cooldown(self, mid, rid, ts):
        self.last_alerts[mid][rid] = ts
