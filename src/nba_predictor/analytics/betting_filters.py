#!/usr/bin/env python3
"""
🛡️ Betting Filters Module - Protective Filters for NBA Betting

Implements empirical-based filters to protect against unfavorable betting conditions:
- Line range filters (avoid DANGER zone 220-230)
- Minimum edge requirements (12% based on analysis)
- Zone-based edge adjustments
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying betting filters."""

    should_bet: bool
    adjusted_edge: float
    filter_reason: str
    zone: str
    confidence_modifier: float


class BettingFilters:
    """
    Protective filters for NBA betting based on empirical analysis.

    Analysis of 97 bets revealed:
    - Lines 220-230: WR 41.7%, P&L -€37.07 (DANGER ZONE)
    - Lines 230-240: WR 57.5%, P&L +€10.76 (OPTIMAL ZONE)
    - Overall breakeven WR: 55.9%

    These filters implement protective measures to avoid unfavorable conditions.
    """

    # Minimum edge requirements by zone
    MIN_EDGE_DEFAULT = 0.08  # 8% edge minimum (was 2.5%)
    MIN_EDGE_DANGER_ZONE = 0.15  # 15% edge required in danger zone
    MIN_EDGE_OPTIMAL_ZONE = 0.06  # 6% edge acceptable in optimal zone

    # Line zone thresholds
    DANGER_ZONE = (220, 230)
    OPTIMAL_ZONE = (230, 240)

    # Filter modes
    STRICT_MODE = True  # If True, completely blocks DANGER zone bets

    def __init__(self, strict_mode: bool = True):
        """
        Initialize betting filters.

        Args:
            strict_mode: If True, completely blocks bets in DANGER zone.
                         If False, allows with higher edge requirement.
        """
        self.strict_mode = strict_mode
        self._filter_history = []
        logger.info(f"🛡️ BettingFilters initialized (strict_mode={strict_mode})")

    def get_line_zone(self, market_line: float) -> str:
        """Categorize the market line into a zone."""
        if market_line < self.DANGER_ZONE[0]:
            return "LOW"
        elif self.DANGER_ZONE[0] <= market_line < self.DANGER_ZONE[1]:
            return "DANGER"
        elif self.OPTIMAL_ZONE[0] <= market_line < self.OPTIMAL_ZONE[1]:
            return "OPTIMAL"
        else:
            return "HIGH"

    def get_min_edge_for_zone(self, zone: str) -> float:
        """Get minimum edge requirement based on zone."""
        if zone == "DANGER":
            return self.MIN_EDGE_DANGER_ZONE
        elif zone == "OPTIMAL":
            return self.MIN_EDGE_OPTIMAL_ZONE
        else:
            return self.MIN_EDGE_DEFAULT

    def apply_filters(
        self,
        market_line: float,
        edge: float,
        bet_type: str = "UNDER",
    ) -> FilterResult:
        """
        Apply all protective filters to a potential bet.

        Args:
            market_line: The betting line from the market
            edge: Calculated edge for this bet (as decimal, e.g., 0.08 for 8%)
            bet_type: "OVER" or "UNDER"

        Returns:
            FilterResult with decision and metadata
        """
        zone = self.get_line_zone(market_line)
        min_edge_required = self.get_min_edge_for_zone(zone)

        # Default values
        should_bet = True
        adjusted_edge = edge
        filter_reason = ""
        confidence_modifier = 0.0

        # 1. DANGER Zone Check
        if zone == "DANGER":
            if self.strict_mode:
                # Complete block
                should_bet = False
                filter_reason = f"BLOCKED: Line {market_line} in DANGER zone (220-230). WR=41.7% historical."
                confidence_modifier = -0.20
            else:
                # Allow with higher edge requirement
                if edge < min_edge_required:
                    should_bet = False
                    filter_reason = f"Edge {edge:.1%} below DANGER zone minimum {min_edge_required:.1%}"
                else:
                    filter_reason = f"DANGER zone bet allowed (edge {edge:.1%} >= {min_edge_required:.1%})"
                confidence_modifier = -0.15

        # 2. Minimum Edge Check for other zones
        elif edge < min_edge_required:
            should_bet = False
            filter_reason = (
                f"Edge {edge:.1%} below zone minimum {min_edge_required:.1%}"
            )

        # 3. OPTIMAL Zone boost
        elif zone == "OPTIMAL":
            confidence_modifier = +0.05
            filter_reason = f"OPTIMAL zone bet (edge {edge:.1%})"

        # 4. Standard zones
        else:
            filter_reason = f"Standard zone bet (edge {edge:.1%})"

        result = FilterResult(
            should_bet=should_bet,
            adjusted_edge=adjusted_edge,
            filter_reason=filter_reason,
            zone=zone,
            confidence_modifier=confidence_modifier,
        )

        self._filter_history.append(result)

        log_level = logging.INFO if should_bet else logging.WARNING
        logger.log(
            log_level,
            f"🛡️ Filter: line={market_line}, edge={edge:.1%}, zone={zone} → {'PASS' if should_bet else 'BLOCK'} ({filter_reason})",
        )

        return result

    def get_filter_stats(self) -> dict:
        """Get statistics about filter applications."""
        if not self._filter_history:
            return {"total_checks": 0}

        passed = len([f for f in self._filter_history if f.should_bet])
        blocked = len([f for f in self._filter_history if not f.should_bet])

        return {
            "total_checks": len(self._filter_history),
            "passed": passed,
            "blocked": blocked,
            "block_rate": blocked / len(self._filter_history)
            if self._filter_history
            else 0,
            "zone_distribution": {
                zone: len([f for f in self._filter_history if f.zone == zone])
                for zone in ["LOW", "DANGER", "OPTIMAL", "HIGH"]
            },
        }


# Singleton instance
_default_filters: Optional[BettingFilters] = None


def get_betting_filters(strict_mode: bool = True) -> BettingFilters:
    """Get or create the default betting filters instance."""
    global _default_filters
    if _default_filters is None:
        _default_filters = BettingFilters(strict_mode=strict_mode)
    return _default_filters
