import logging
import queue
import threading
from typing import List, Optional, Callable
import betfairlightweight
from betfairlightweight import filters

logger = logging.getLogger(__name__)


class MarketStreamer:
    """
    Manages a Betfair WebSocket stream for real-time market data.
    """

    def __init__(
        self,
        app_key: str,
        market_ids: List[str],
        ssoid: str = None,
        username: str = None,
        password: str = None,
        certs_path: str = None,
        locale: str = None,
    ):
        self.app_key = app_key
        self.market_ids = market_ids
        self.ssoid = ssoid
        self.username = username
        self.password = password
        self.certs_path = certs_path
        self.locale = locale

        self.client = betfairlightweight.APIClient(
            username=self.username if self.username else "",
            password=self.password if self.password else "",
            app_key=self.app_key,
            certs=self.certs_path,
            locale=self.locale,
        )

        # Login Logic for Streamer
        if self.ssoid:
            self.client.set_session_token(self.ssoid)
        elif self.username and self.password and self.certs_path:
            # Streaming client needs a valid session.
            # usually client.login() sets it in the client object
            self.client.login()

        self.stream = None
        self.queue = queue.Queue()
        self.stream_thread = None
        self.running = False

    def start(self, update_callback: Optional[Callable] = None):
        """
        Starts the stream in a separate thread.
        update_callback(market_book) will be called on every update.
        """
        logger.info(f"Starting Stream for {len(self.market_ids)} markets...")

        # 1. Create Listener
        # We use a Queue to inspect updates, or a direct callback
        self.listener = betfairlightweight.StreamListener(
            output_queue=self.queue if not update_callback else None
        )

        # 2. Create Stream
        self.stream = self.client.streaming.create_stream(listener=self.listener)

        # 3. Define Subscription
        # Subscribe to MARKET_DEFINITIONS and EX_BEST_OFFERS (Order Book)
        market_filter = filters.streaming_market_filter(market_ids=self.market_ids)
        market_data_filter = filters.streaming_market_data_filter(
            fields=["EX_BEST_OFFERS", "EX_MARKET_DEF"],
            ladder_levels=3,  # Top 3 prices
        )

        # 4. Subscribe
        self.stream.subscribe_to_markets(
            market_filter=market_filter,
            market_data_filter=market_data_filter,
            conflate_ms=500,  # Throttle updates to 500ms
            initial_clk=None,
            clk=None,
        )

        # 5. Start (Blocking if in main thread, so we start in thread usually)
        # But betfairlightweight stream.start() blocks.
        # So we run it in a thread if we want non-blocking.
        self.stream_thread = threading.Thread(target=self.stream.start, daemon=True)
        self.stream_thread.start()
        logger.info("Stream thread started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            logger.info("Stream stopped.")

    def get_updates(self):
        """Generator that yields updates from the queue."""
        while True:
            try:
                # snap_list is a list of MarketBooks
                snap_list = self.queue.get(timeout=1)
                for snap in snap_list:
                    yield snap
            except queue.Empty:
                continue
