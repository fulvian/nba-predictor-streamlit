import threading
import queue
import logging
import os
from typing import List, Optional, Dict, Any

from src.nba_predictor.betfair.client import BetfairClient
from src.nba_predictor.betfair.streamer import MarketStreamer
from src.nba_predictor.betfair.panic_detector import PanicDetector, PanicAlert
from src.nba_predictor.betfair.trade_executor import TradeExecutor
from src.nba_predictor.betfair.position_manager import PositionManager

logger = logging.getLogger(__name__)


# Singleton instance and lock
_service_instance = None
_service_lock = threading.Lock()


def get_betfair_service() -> "BetfairService":
    """Thread-safe singleton accessor."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = BetfairService()
    return _service_instance


class BetfairService:
    """
    Singleton service to manage Betfair Connection, Streaming,
    and Auto-Trading in a background thread safe for Streamlit.
    Supports MULTI-MARKET monitoring.
    """

    def __init__(self):
        # Fallback Credentials (for local testing without env vars)
        self.app_key = os.getenv("BETFAIR_APP_KEY", "QkYxxm82m7tiUQrI")
        self.username = os.getenv("BETFAIR_USERNAME", "fulviold@gmail.com")
        self.password = os.getenv("BETFAIR_PASSWORD", "9#!Vq-!45ukvu&6")
        self.certs_path = os.getenv(
            "BETFAIR_CERTS_PATH", "/Users/fulvioventura/nba-predictor-streamlit/certs"
        )
        self.locale = "italy"  # Force Italy as per fix

        self.streamer: Optional[MarketStreamer] = None
        self.client: Optional[BetfairClient] = None

        # Multi-Market: One Detector per market_id
        self.detectors: Dict[str, PanicDetector] = {}

        # Trading Components
        self.executor: Optional[TradeExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.auto_trade_enabled = False
        self.default_stake = 5.0  # €5 default
        self.paper_mode = True  # Default to paper trading

        self.alert_queue = queue.Queue()
        self.is_running = False
        self.monitored_market_ids: List[str] = []

        # Live market data buffer (for UI display)
        # Key: market_id, Value: MarketBook object
        self._last_market_books: Dict[str, Any] = {}
        self._market_lock = threading.Lock()

        # internal
        self._monitor_thread = None

        # Additional metadata for UI (market names)
        self.market_names: Dict[str, str] = {}

    def start_monitoring(
        self,
        market_ids: List[str],
        market_names: Dict[str, str] = None,
        runner_names: Dict[str, Dict[int, str]] = None,
    ):
        """Starts the streamer for a list of market IDs."""
        if self.is_running:
            logger.warning("Streamer already running. Stop first.")
            return

        if not (self.username and self.password):
            raise ValueError("Credentials missing in Env Vars.")

        if not market_ids:
            logger.warning("No markets provided to monitor.")
            return

        logger.info(f"Starting Betfair Service for {len(market_ids)} Markets")
        self.monitored_market_ids = market_ids
        if market_names:
            self.market_names = market_names
        self.runner_names = runner_names or {}

        # 1. Init Client for Trading
        self.client = BetfairClient(
            app_key=self.app_key,
            username=self.username,
            password=self.password,
            certs_path=self.certs_path,
            locale=self.locale,
        )
        self.client.login()

        # 2. Init Panic Detectors (One per market)
        self.detectors = {}
        for mid in market_ids:
            self.detectors[mid] = PanicDetector(
                drift_ticks=8,
                drift_seconds=4,
                volume_threshold=1000.0,
                volume_seconds=5,
            )

        # 3. Init Trading Components
        self.executor = TradeExecutor(
            client=self.client,
            paper_mode=self.paper_mode,
            max_stake=20.0,
        )
        self.position_manager = PositionManager(
            executor=self.executor,
            take_profit_ticks=5,
            stop_loss_ticks=10,
            timeout_seconds=300,
        )

        # 4. Init Streamer (Multi-Market)
        self.streamer = MarketStreamer(
            app_key=self.app_key,
            market_ids=market_ids,
            username=self.username,
            password=self.password,
            certs_path=self.certs_path,
            locale=self.locale,
        )

        # 5. Start Background Loop
        try:
            self.is_running = True
            self.streamer.start()

            self._monitor_thread = threading.Thread(
                target=self._update_loop, daemon=True
            )
            self._monitor_thread.start()
            logger.info("Streamer and Background Loop Started")

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.stop()
            raise e

    def enable_auto_trading(self, stake: float = 5.0, paper_mode: bool = True):
        """Enable or update auto-trading settings."""
        self.auto_trade_enabled = True
        self.default_stake = stake
        self.paper_mode = paper_mode

        if self.executor:
            self.executor.paper_mode = paper_mode

        mode_str = "📝 PAPER" if paper_mode else "💰 LIVE"
        logger.info(f"Auto-trading ENABLED: {mode_str}, Stake=€{stake}")

    def disable_auto_trading(self):
        """Disable auto-trading."""
        self.auto_trade_enabled = False
        logger.info("Auto-trading DISABLED")

    def _update_loop(self):
        """Background loop consuming streamer updates and feeding detectors."""
        logger.info("Service Loop Started.")
        try:
            for market_book in self.streamer.get_updates():
                if not self.is_running:
                    break

                mid = market_book.market_id

                # Store latest market book for UI (thread-safe)
                with self._market_lock:
                    self._last_market_books[mid] = market_book

                # Feed Correct Detector
                detector = self.detectors.get(mid)
                if detector:
                    alerts = detector.process_update(market_book)

                    # Enqueue Alerts
                    for alert in alerts:
                        self.alert_queue.put(alert)

                        # AUTO-TRADE: Execute on CRITICAL alerts
                        if self.auto_trade_enabled and alert.severity == "CRITICAL":
                            self._execute_trade(alert, market_book)

                # Check exit conditions for open positions (global check)
                if self.position_manager:
                    self.position_manager.check_exit_conditions(market_book)

        except Exception as e:
            logger.error(f"Service Loop Error: {e}")
            self.stop()

    def _execute_trade(self, alert: PanicAlert, market_book):
        """Execute a trade based on alert."""
        # Find current price for the runner
        current_price = None
        for runner in market_book.runners:
            if runner.selection_id == alert.runner_id:
                if runner.ex.available_to_back:
                    current_price = runner.ex.available_to_back[0].price
                break

        if current_price is None:
            logger.warning(
                f"Cannot execute trade: no price for runner {alert.runner_id}"
            )
            return

        bet_id = self.position_manager.open_position(
            alert=alert,
            stake=self.default_stake,
            current_price=current_price,
        )

        if bet_id:
            logger.info(f"🎯 Trade executed: {bet_id}")

    def stop(self):
        self.is_running = False
        if self.streamer:
            self.streamer.stop()
        logger.info("Service Stopped.")

    def get_new_alerts(self) -> List[PanicAlert]:
        """Dequeues all pending alerts (non-blocking)."""
        alerts = []
        try:
            while True:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
        except queue.Empty:
            pass
        return alerts

    def get_trading_stats(self) -> Dict[str, Any]:
        """Returns current trading statistics."""
        if self.position_manager:
            return self.position_manager.get_stats()
        return {
            "open_positions": 0,
            "closed_today": 0,
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "paper_mode": self.paper_mode,
        }

    def get_live_dashboard_data(self) -> List[Dict[str, Any]]:
        """
        Returns aggregate status for ALL monitored markets for the main dashboard grid.
        Rows: Market Name, Status, Volume, Drift Indicator
        """
        data = []
        with self._market_lock:
            for mid, market_book in self._last_market_books.items():
                # Market Name
                market_name = self.market_names.get(mid, f"Market {mid}")

                # Calculate Total Volume
                total_volume = market_book.total_matched

                # Status: "Active", "Suspended", "Closed"
                status = market_book.status

                # Panic State (Check recent alerts from history if posisble, or simple logic)
                # For now, just show status icon
                status_icon = "🟢" if status == "OPEN" else "🔴"
                if market_book.inplay:
                    status_icon = "⚡"  # Live In-Play

                data.append(
                    {
                        "market_id": mid,
                        "market_name": market_name,
                        "status": status_icon,
                        "volume": total_volume,
                        "in_play": market_book.inplay,
                    }
                )
        return data

    def get_live_odds(self, market_id: str) -> List[Dict[str, Any]]:
        """
        Returns current live odds for specific market runners.
        """
        with self._market_lock:
            market_book = self._last_market_books.get(market_id)
            if market_book is None:
                # logger.debug(f"No market book found for {market_id}")
                return []

            runners_data = []
            for runner in market_book.runners:
                # Some markets might not have 'ACTIVE' status yet but still have prices
                # or we want to see them anyway.

                # Extract best back/lay prices
                best_back = (
                    runner.ex.available_to_back[0]
                    if runner.ex.available_to_back
                    else None
                )
                best_lay = (
                    runner.ex.available_to_lay[0]
                    if runner.ex.available_to_lay
                    else None
                )

                runner_info = {
                    "selection_id": runner.selection_id,
                    "runner_name": self.runner_names.get(market_id, {}).get(
                        runner.selection_id, f"ID: {runner.selection_id}"
                    ),
                    "status": runner.status,
                    "back_price": best_back.price if best_back else 0.0,
                    "back_size": best_back.size if best_back else 0.0,
                    "lay_price": best_lay.price if best_lay else 0.0,
                    "lay_size": best_lay.size if best_lay else 0.0,
                    "total_matched": runner.total_matched or 0,
                    "last_price_traded": runner.last_price_traded
                    if hasattr(runner, "last_price_traded")
                    else None,
                }

                # Calculate spread
                if runner_info["back_price"] and runner_info["lay_price"]:
                    runner_info["spread"] = round(
                        runner_info["lay_price"] - runner_info["back_price"], 2
                    )
                else:
                    runner_info["spread"] = None

                runners_data.append(runner_info)

            return runners_data
