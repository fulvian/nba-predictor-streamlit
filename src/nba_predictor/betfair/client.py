import logging
import os
from typing import List, Optional, Dict, Any
import betfairlightweight
from betfairlightweight import filters

logger = logging.getLogger(__name__)


class BetfairClient:
    """
    Wrapper around betfairlightweight APIClient.
    Uses 'Delay' App Key by default for testing.
    Authentication is handled via Session Token (SSOID) provided manually or via env.
    """

    def __init__(
        self,
        app_key: str,
        ssoid: str = None,
        username: str = None,
        password: str = None,
        certs_path: str = None,
        locale: str = None,
    ):
        """
        Initializes the Betfair Client.

        Args:
            app_key: Betfair Application Key
            ssoid: Session Token (Optional, for manual interactive login)
            username: Betfair Username (Optional, for cert login)
            password: Betfair Password (Optional, for cert login)
            certs_path: Path to directory containing client-2048.crt and client-2048.key (Optional, for cert login)
            locale: Betfair locale (e.g. 'italy'), defaults to None (Global .com)
        """
        self.app_key = app_key
        self.ssoid = ssoid
        self.username = username
        self.password = password
        self.certs_path = certs_path
        self.locale = locale

        # Configure client
        self.client = betfairlightweight.APIClient(
            username=self.username if self.username else "",
            password=self.password if self.password else "",
            app_key=self.app_key,
            certs=self.certs_path,
            locale=self.locale,
        )

    def login(self):
        """
        Performs login.
        If SSOID is provided, uses it (Interactive/Manual).
        If Certs/User/Pass are provided, uses Non-Interactive (Automated).
        """
        if self.ssoid:
            logger.info("🔐 Logging in via SSOID (Manual Manual Session)...")
            self.client.set_session_token(self.ssoid)
        elif self.username and self.password and self.certs_path:
            logger.info(
                f"🔐 Logging in via Certificate (Automated) for user {self.username}..."
            )
            self.client.login()  # This uses the non-interactive cert flow
        else:
            logger.warning(
                "❌ No valid credentials provided (SSOID or Certs). Login may fail."
            )

        logger.info(f"Betfair Client initialized with AppKey: {self.app_key[:4]}...")

    def list_nba_events(self) -> List[Any]:
        """
        Uses Navigation API to find NBA events.
        Parses the list-of-dicts response structure.
        """
        try:
            # list_navigation() returns a LIST of dicts (Root Categories)
            nav = self.client.navigation.list_navigation()
        except Exception as e:
            logger.error(f"Navigation API failed: {e}")
            return []

        nba_events = []

        # list_navigation() returns a ROOT dict with keys: type, name, id, children
        # 'children' is a list of sport categories (like 'Soccer', 'Basketball')
        if not isinstance(nav, dict):
            logger.warning(f"Navigation API returned unexpected type: {type(nav)}")
            return []

        root_children = nav.get("children", [])
        if not isinstance(root_children, list):
            logger.warning(
                f"Navigation 'children' is not a list: {type(root_children)}"
            )
            return []

        # 1. Find Basketball Root Node (Note: In Italian locale, it's "Basket")
        basket_node = None
        for category in root_children:
            if not isinstance(category, dict):
                continue
            cat_name = category.get("name", "").lower()
            if cat_name in ("basket", "basketball"):
                basket_node = category
                break

        if not basket_node:
            logger.warning("Basketball/Basket category not found in Navigation.")
            return []

        # 2. Collect ALL basketball events (NBA not available on Betfair.it)
        # Traverse the entire Basketball subtree

        def collect_all_events(node):
            """Recursively collect ALL events in a subtree."""
            if not isinstance(node, dict):
                return

            if node.get("type") == "EVENT":
                nba_events.append(node)
                return

            for child in node.get("children", []):
                collect_all_events(child)

        collect_all_events(basket_node)

        logger.info(f"Found {len(nba_events)} basketball events")

        # 3. Adapter: Convert dicts to Objects compatible with UI
        # UI expects: item.event.name, item.event.id
        class EventAdapter:
            def __init__(self, raw_node):
                self.event = type(
                    "obj",
                    (object,),
                    {"name": raw_node.get("name"), "id": raw_node.get("id")},
                )()

        return [EventAdapter(e) for e in nba_events]

    def get_market_catalogue(self, event_ids: List[str], max_results=10) -> List[Any]:
        """
        Get market catalogue (Match Odds, Moneyline) for given events.
        """
        market_filter = filters.market_filter(
            event_ids=event_ids,
            market_type_codes=["MATCH_ODDS"],
        )

        # Standard projection: RUNNER_METADATA, MARKET_START_TIME
        item_projection = ["RUNNER_METADATA", "MARKET_START_TIME", "EVENT"]

        catalogue = self.client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=item_projection,
            max_results=max_results,
        )
        return catalogue
