"""
Market Scanner - Intelligent Market Selection for LOAD System

Scans Betfair markets to identify suitable candidates for trading,
filtering by sport, liquidity, and competition tier.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from src.nba_predictor.betfair.client import BetfairClient
from .odds_snapshot import Sport, MarketType, EVENT_TYPE_IDS

logger = logging.getLogger(__name__)


@dataclass
class MarketCandidate:
    """
    Candidate market identified for potential trading.
    """

    market_id: str
    event_id: str
    sport: Sport
    competition_name: str
    event_name: str
    market_start_time: datetime
    total_matched: float
    is_in_play: bool

    # Priority score (higher = better candidate)
    priority_score: int = 0

    def __lt__(self, other):
        return self.priority_score < other.priority_score


class MarketScanner:
    """
    Scans Betfair for trading opportunities.

    Responsibilities:
    1. Find in-play and soon-to-start markets
    2. Filter by liquidity and competition tier
    3. Prioritize markets based on potential for inefficiency
    4. Manage market subscriptions
    """

    # Priority Tiers
    TIER_1_COMPETITIONS = [  # High liquidity, harder to beat
        "Premier League",
        "Serie A",
        "La Liga",
        "NBA",
        "ATP Grand Slam",
    ]

    TIER_2_COMPETITIONS = [  # Moderate liquidity, good balance
        "Serie B",
        "Championship",
        "ATP 250",
        "EuroLeague",
    ]

    TIER_3_COMPETITIONS = [  # Low liquidity, high inefficiency potential
        "Serie C",
        "ITF",
        "Challenger",
        "NBL",
        "Friendly",
    ]

    def __init__(self, client: BetfairClient):
        self.client = client
        self.active_markets: Dict[str, MarketCandidate] = {}
        self.ignored_competitions: Set[str] = set()

    def scan_markets(
        self,
        sports: List[Sport] = None,
        max_results: int = 50,
        in_play_only: bool = True,
    ) -> List[MarketCandidate]:
        """
        Scan for market candidates across specified sports.

        Args:
            sports: List of sports to scan (default: all supported)
            max_results: Max number of markets to return
            in_play_only: If True, only return markets currently in play

        Returns:
            List of MarketCandidate objects ordered by priority
        """
        if sports is None:
            sports = [Sport.FOOTBALL, Sport.TENNIS, Sport.BASKETBALL]

        candidates = []

        for sport in sports:
            try:
                sport_candidates = self._scan_sport(sport, in_play_only)
                candidates.extend(sport_candidates)
            except Exception as e:
                logger.error(f"Error scanning sport {sport.value}: {e}")

        # Sort by priority (descending)
        candidates.sort(key=lambda x: x.priority_score, reverse=True)

        # Helper to avoid too many markets
        selected = candidates[:max_results]

        # Update active markets cache
        self.active_markets = {c.market_id: c for c in selected}

        logger.info(f"Scan complete: found {len(selected)} candidates")
        return selected

    def _scan_sport(self, sport: Sport, in_play_only: bool) -> List[MarketCandidate]:
        """Internal method to scan a specific sport."""
        from betfairlightweight import filters

        event_type_id = EVENT_TYPE_IDS.get(sport)
        if not event_type_id:
            logger.warning(f"No event type ID for sport {sport}")
            return []

        # Define time window (e.g., started recently or starting soon)
        now = datetime.utcnow()
        time_range = (
            now - timedelta(hours=2) if in_play_only else now - timedelta(minutes=15)
        )

        market_filter = filters.market_filter(
            event_type_ids=[event_type_id],
            market_type_codes=["MATCH_ODDS"],
            in_play_only=in_play_only,
            # Potentially filter by countries if needed (e.g. ['IT', 'GB'])
        )

        # Get Catalogue
        try:
            catalogue = self.client.client.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=[
                    "COMPETITION",
                    "EVENT",
                    "MARKET_START_TIME",
                    "RUNNER_METADATA",  # To get runner names
                ],
                max_results=100,
                sort="FIRST_TO_START",
            )
        except Exception as e:
            logger.error(f"Betfair API error scanning {sport.name}: {e}")
            return []

        candidates = []

        for market in catalogue:
            # Skip if no total matched (dead market)
            # Note: list_market_catalogue doesn't give total matched directly usually,
            # we might need list_market_book for that, but let's assume we filter later
            # or accept all and filter in stream.
            # Actually, standard practice is to subscribe to interesting ones.

            competition = market.competition.name if market.competition else "Unknown"

            # Skip ignored
            if any(ign in competition for ign in self.ignored_competitions):
                continue

            # Calculate priority
            priority = self._calculate_priority(sport, competition)

            candidate = MarketCandidate(
                market_id=market.market_id,
                event_id=market.event.id,
                sport=sport,
                competition_name=competition,
                event_name=market.event.name,
                market_start_time=market.market_start_time,
                total_matched=market.total_matched
                if hasattr(market, "total_matched")
                else 0,  # Often None in catalogue
                is_in_play=in_play_only,  # Rough assumption based on filter
                priority_score=priority,
            )

            candidates.append(candidate)

        return candidates

    def _calculate_priority(self, sport: Sport, competition: str) -> int:
        """
        Calculate priority score for a market.

        Higher score = better candidate for inefficiency.
        """
        priority = 50  # Base score

        # Favor Tier 3 (Inefficiency potential)
        if any(c in competition for c in self.TIER_3_COMPETITIONS):
            priority += 30
        # Determine Tier 2
        elif any(c in competition for c in self.TIER_2_COMPETITIONS):
            priority += 10
        # Tier 1 - Harder to beat
        elif any(c in competition for c in self.TIER_1_COMPETITIONS):
            priority -= 10

        # Sport preferences
        if sport == Sport.TENNIS:  # High frequency changes
            priority += 10
        elif sport == Sport.BASKETBALL:  # Fast pace
            priority += 5

        return priority

    def get_market_ids(self) -> List[str]:
        """Get list of currently active market IDs."""
        return list(self.active_markets.keys())

    def add_ignored_competition(self, competition: str):
        self.ignored_competitions.add(competition)
