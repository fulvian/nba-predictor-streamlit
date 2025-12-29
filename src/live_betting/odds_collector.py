"""
Betfair Live Odds Collector

Ingests live odds from Betfair API via:
1. WebSocket streaming (market stream, EX_BEST_OFFERS)
2. REST polling fallback (listMarketBook)

Handles:
- Connection management and reconnection
- Quote normalization to OddsSnapshot format
- Volume inference from matched bets
- Market state tracking

References:
- Betfair API docs: https://docs.betfair.com/
- EX_BEST_OFFERS: best bid/ask with depth
"""

import json
import logging
import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import hashlib

import httpx
import websockets

from .anomaly_detector import OddsSnapshot

logger = logging.getLogger(__name__)


class BetfairMarketType(Enum):
    """Common Betfair market types for sports."""
    MATCH_ODDS = "WIN"  # 1X2 / Match Winner
    OVER_UNDER = "OVER_UNDER_25"  # O/U 2.5 goals
    HANDICAP = "HANDICAP"
    CORRECT_SCORE = "CORRECT_SCORE"
    BOTH_TEAMS_SCORE = "BTTS"  # Both Teams to Score


class BetfairOddsCollector:
    """
    Collects live odds from Betfair in real-time.

    Usage:
        collector = BetfairOddsCollector(
            app_key="your-app-key",
            username="your-username",
            password="your-password"
        )
        collector.on_snapshot = my_callback  # receives OddsSnapshot
        await collector.connect()
        await collector.stream_market(market_id="1.123456")
    """

    def __init__(
        self,
        app_key: str,
        username: str,
        password: str,
        polling_interval_seconds: int = 5,
    ):
        """
        Args:
            app_key: Betfair application key (free tier available)
            username: Betfair username
            password: Betfair password
            polling_interval_seconds: Fallback polling rate if WebSocket unavailable
        """
        self.app_key = app_key
        self.username = username
        self.password = password
        self.polling_interval = polling_interval_seconds

        # Session state
        self.session_token: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws_connection = None

        # Streaming configuration
        self.on_snapshot: Optional[Callable[[OddsSnapshot], None]] = None
        self.market_state: Dict[str, Dict] = {}  # market_id -> last known state

    async def connect(self) -> bool:
        """Authenticate and establish session with Betfair."""
        self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Login to get session token
            login_response = await self.http_client.post(
                "https://api.betfair.com/exchange/betting/json-rpc/v1",
                json={
                    "jsonrpc": "2.0",
                    "method": "AuthenticationService/login",
                    "params": {
                        "appKey": self.app_key,
                        "username": self.username,
                        "password": self.password,
                    },
                    "id": 1,
                },
            )

            result = login_response.json()
            if result.get("result", {}).get("sessionToken"):
                self.session_token = result["result"]["sessionToken"]
                logger.info(f"Betfair authenticated: {self.username}")
                return True
            else:
                error = result.get("error", {}).get("message", "Unknown error")
                logger.error(f"Betfair login failed: {error}")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Close session."""
        if self.http_client:
            await self.http_client.aclose()
        if self.ws_connection:
            await self.ws_connection.close()

    async def stream_market(
        self,
        market_id: str,
        sport: str = "unknown",
        competition: str = "unknown",
        use_websocket: bool = True,
    ):
        """
        Stream live odds for a Betfair market.

        Args:
            market_id: Betfair market ID (e.g., "1.123456")
            sport: Sport name (e.g., "football", "tennis")
            competition: League/tournament (e.g., "Serie C")
            use_websocket: Prefer WebSocket if available; fallback to REST polling
        """
        if use_websocket:
            await self._stream_websocket(market_id, sport, competition)
        else:
            await self._stream_polling(market_id, sport, competition)

    async def _stream_websocket(self, market_id: str, sport: str, competition: str):
        """Stream via WebSocket (preferred method)."""
        try:
            uri = "wss://stream-api.betfair.com/exchange"
            headers = {"X-Application": self.app_key}

            async with websockets.connect(uri, subprotocols=["betfair"], extra_headers=headers) as ws:
                self.ws_connection = ws

                # Subscribe to market
                await ws.send(
                    json.dumps({
                        "op": "connection",
                        "connectionId": "LOAD-collector-" + hashlib.md5(market_id.encode()).hexdigest()[:8],
                    })
                )

                subscription = {
                    "op": "subscribe",
                    "subscriptionData": {
                        "requestId": 1,
                        "clk": None,
                        "pt": 0,
                        "heartbeatMs": 5000,
                        "initialClk": None,
                        "mc": [
                            {
                                "marketFilter": {
                                    "marketIds": [market_id],
                                    "bspMarket": False,
                                },
                                "marketDataFilter": {
                                    "fields": ["EX_BEST_OFFERS", "EX_TRADED", "TV"],
                                    "ladderLevels": 3,
                                },
                            }
                        ],
                    },
                }

                await ws.send(json.dumps(subscription))
                logger.info(f"Subscribed to {market_id} via WebSocket")

                # Receive stream
                async for message in ws:
                    await self._process_stream_message(
                        json.loads(message), market_id, sport, competition
                    )

        except Exception as e:
            logger.warning(f"WebSocket disconnected ({market_id}): {e}. Falling back to polling.")
            await self._stream_polling(market_id, sport, competition)

    async def _stream_polling(self, market_id: str, sport: str, competition: str):
        """Fallback: Poll REST API periodically."""
        while True:
            try:
                market_book = await self._get_market_book(market_id)

                if market_book:
                    self._process_market_book(market_book, market_id, sport, competition)

                await asyncio.sleep(self.polling_interval)

            except Exception as e:
                logger.error(f"Polling error for {market_id}: {e}")
                await asyncio.sleep(self.polling_interval)

    async def _get_market_book(self, market_id: str) -> Optional[Dict]:
        """Fetch current market book via REST API."""
        if not self.session_token or not self.http_client:
            return None

        try:
            response = await self.http_client.post(
                "https://api.betfair.com/exchange/betting/json-rpc/v1",
                json={
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/listMarketBook",
                    "params": {
                        "marketIds": [market_id],
                        "priceData": ["EX_BEST_OFFERS"],
                        "orderProjection": "ALL",
                    },
                    "id": 1,
                },
                headers={"X-Authentication": self.session_token, "X-Application": self.app_key},
            )

            result = response.json()
            if result.get("result"):
                return result["result"][0]
            else:
                logger.debug(f"Market book fetch failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"REST call failed: {e}")
            return None

    async def _process_stream_message(self, message: Dict, market_id: str, sport: str, competition: str):
        """Process incoming WebSocket market stream message."""
        if "mc" not in message:
            return

        for market_change in message.get("mc", []):
            market_id_from_msg = market_change.get("id")
            if market_id_from_msg != market_id:
                continue

            for runner in market_change.get("rc", []):
                await self._extract_odds_from_runner(
                    runner, market_id, sport, competition, market_change.get("tv", 0)
                )

    def _process_market_book(self, market_book: Dict, market_id: str, sport: str, competition: str):
        """Process REST API market book response."""
        event = market_book.get("description", {})
        market_type = event.get("marketType", "UNKNOWN")
        total_matched = market_book.get("totalMatched", 0)

        for runner in market_book.get("runners", []):
            selection_id = runner.get("selectionId")
            selection_name = runner.get("status", "ACTIVE")

            ex = runner.get("ex", {})
            back_offers = ex.get("availableToBack", [])
            lay_offers = ex.get("availableToLay", [])

            if back_offers and lay_offers:
                best_back = back_offers[0]
                best_lay = lay_offers[0]

                snapshot = OddsSnapshot(
                    timestamp=datetime.utcnow(),
                    sport=sport,
                    competition=competition,
                    event_id=market_id,
                    bookmaker="betfair",
                    market_type=market_type,
                    outcome=str(selection_id),
                    odds=best_back["price"],
                    backing_odds=best_back["price"],
                    laying_odds=best_lay["price"],
                    back_volume=best_back.get("size", 0),
                    lay_volume=best_lay.get("size", 0),
                    implied_prob=1.0 / best_back["price"] if best_back["price"] > 0 else 0,
                )

                if self.on_snapshot:
                    self.on_snapshot(snapshot)

    async def _extract_odds_from_runner(
        self, runner: Dict, market_id: str, sport: str, competition: str, total_matched: float
    ):
        """Extract OddsSnapshot from WebSocket runner message."""
        selection_id = runner.get("id")

        ex = runner.get("ex", {})
        back_offers = ex.get("b", [])
        lay_offers = ex.get("l", [])

        if back_offers and lay_offers:
            best_back = back_offers[0]
            best_lay = lay_offers[0]

            snapshot = OddsSnapshot(
                timestamp=datetime.utcnow(),
                sport=sport,
                competition=competition,
                event_id=market_id,
                bookmaker="betfair",
                market_type="match_odds",
                outcome=str(selection_id),
                odds=best_back[0],
                backing_odds=best_back[0],
                laying_odds=best_lay[0],
                back_volume=best_back[1],
                lay_volume=best_lay[1],
                implied_prob=1.0 / best_back[0] if best_back[0] > 0 else 0,
            )

            if self.on_snapshot:
                self.on_snapshot(snapshot)

    async def get_market_catalogue(
        self,
        event_type_ids: List[str] = None,
        market_filter: Dict = None,
    ) -> List[Dict]:
        """
        Fetch available markets from Betfair.

        Args:
            event_type_ids: E.g., ["1"] (soccer), ["2"] (tennis), ["7"] (horse racing)
            market_filter: Custom filter (e.g., country codes)

        Returns:
            List of market objects
        """
        if not self.session_token or not self.http_client:
            return []

        try:
            response = await self.http_client.post(
                "https://api.betfair.com/exchange/betting/json-rpc/v1",
                json={
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/listMarketCatalogue",
                    "params": {
                        "filter": {
                            "eventTypeIds": event_type_ids or [],
                            "marketFilter": market_filter or {},
                        },
                        "maxResults": 200,
                    },
                    "id": 1,
                },
                headers={"X-Authentication": self.session_token, "X-Application": self.app_key},
            )

            result = response.json()
            return result.get("result", [])

        except Exception as e:
            logger.error(f"Failed to fetch market catalogue: {e}")
            return []
