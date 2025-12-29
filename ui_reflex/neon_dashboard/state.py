import reflex as rx
import sys
import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Add project root to sys.path to import src
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Singleton Service
from src.nba_predictor.live.betfair_service import get_betfair_service
from src.nba_predictor.betfair.client import BetfairClient


# Helper for singleton access
def get_service():
    return get_betfair_service()


class State(rx.State):
    """The App State."""

    # UI State Variables
    connected: bool = False

    # Data Buckets
    market_ids: List[str] = []
    market_names: Dict[str, str] = {}

    # Dashboard Data
    dashboard_grid: List[Dict[str, Any]] = []
    global_stats: Dict[str, Any] = {"open_positions": 0, "daily_pnl": 0.0}
    alerts: List[str] = []

    # Focused Market
    selected_market_id: str = ""
    focused_odds: List[Dict[str, Any]] = []

    # Control Flags
    auto_trade_enabled: bool = False

    # LOAD System State
    load_system_enabled: bool = False
    load_stats: Dict[str, Any] = {}
    active_anomalies: List[Dict[str, Any]] = []
    recent_trades: List[Dict[str, Any]] = []

    def toggle_load_system(self):
        """Toggle the LOAD system on/off."""
        service = get_service()
        if not self.load_system_enabled:
            # Enable
            service.enable_load_system()
            self.load_system_enabled = True
            return rx.toast.success("LOAD System ACTIVATED")
        else:
            # Disable (Just flag for now, need explicit method in service if full tear down needed)
            service.load_enabled = False
            self.load_system_enabled = False
            return rx.toast.info("LOAD System PAUSED")

    def start_monitoring(self):
        """Start the Betfair Service from the UI."""
        service = get_service()
        if not service.is_running:
            # For demo, we do a scan like in streamlit
            try:
                # We need a client to scan
                client = BetfairClient(
                    app_key=os.getenv("BETFAIR_APP_KEY", "QkYxxm82m7tiUQrI"),
                    username=os.getenv("BETFAIR_USERNAME", "fulviold@gmail.com"),
                    password=os.getenv("BETFAIR_PASSWORD", "9#!Vq-!45ukvu&6"),
                    certs_path=os.getenv(
                        "BETFAIR_CERTS_PATH",
                        "/Users/fulvioventura/nba-predictor-streamlit/certs",
                    ),
                    locale="italy",
                )
                client.login()
                events = client.list_nba_events()

                if events:
                    # Take top 5
                    event_ids = [e.event.id for e in events][:5]
                    cats = client.get_market_catalogue(
                        event_ids, max_results=len(event_ids)
                    )

                    m_ids = [c.market_id for c in cats]
                    m_names = {c.market_id: c.event.name for c in cats}

                    # Extract runner names for the detail view
                    r_names = {}
                    for cat in cats:
                        r_names[cat.market_id] = {
                            r.selection_id: r.runner_name for r in cat.runners
                        }

                    service.start_monitoring(m_ids, m_names, r_names)
                    self.connected = True
                    return rx.toast.info(f"Started Monitoring {len(m_ids)} Markets")
                else:
                    return rx.toast.warning("No Live Events Found")

            except Exception as e:
                return rx.toast.error(f"Error Starting: {e}")
        else:
            return rx.toast.info("Service already running")

    def stop_monitoring(self):
        service = get_service()
        service.stop()
        self.connected = False

    # Trend Tracking
    pnl_trend: float = 0.0
    _last_pnl: float = 0.0

    @rx.background
    async def update_hud_tick(self):
        """High-frequency HUD updates (500ms)."""
        while True:
            await asyncio.sleep(0.5)
            if not self.connected:
                continue

            async with self:
                service = get_service()
                if service.is_running:
                    # Update global stats (PnL, positions)
                    stats = service.get_trading_stats()

                    # Calculate trend
                    current_pnl = stats.get("daily_pnl", 0.0)
                    if current_pnl != self._last_pnl:
                        self.pnl_trend = current_pnl - self._last_pnl
                        self._last_pnl = current_pnl

                    self.global_stats = stats

                    # Update focused market odds (critical for trader response)
                    if self.selected_market_id:
                        self.focused_odds = service.get_live_odds(
                            self.selected_market_id
                        )

    @rx.background
    async def update_scanner_tick(self):
        """Lower-frequency scanner and metadata updates (2s)."""
        while True:
            await asyncio.sleep(2.0)
            if not self.connected:
                continue

            async with self:
                service = get_service()
                if service.is_running:
                    # 1. Update Grid
                    self.dashboard_grid = service.get_live_dashboard_data()

                    # 2. Update Alerts
                    new = service.get_new_alerts()
                    if new:
                        msgs = [f"[{a.severity}] {a.details}" for a in new]
                        self.alerts = (self.alerts + msgs)[-20:]

                    # 3. Metadata
                    self.market_ids = service.monitored_market_ids
                    self.market_names = getattr(service, "market_names", {})

                    # 4. LOAD System Data
                    if self.load_system_enabled or service.load_enabled:
                        self.load_system_enabled = service.load_enabled
                        self.load_stats = service.get_load_stats()
                        self.active_anomalies = service.get_active_anomalies()
                        self.recent_trades = service.get_recent_trades()

    def on_load(self):
        """Called when page loads."""
        service = get_service()
        if service.is_running:
            self.connected = True

        # Start split background tickers
        return [State.update_hud_tick, State.update_scanner_tick]
