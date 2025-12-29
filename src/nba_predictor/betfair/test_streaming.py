import logging
import sys
import os
import time

sys.path.append(os.getcwd())

from src.nba_predictor.betfair.client import BetfairClient
from src.nba_predictor.betfair.streamer import MarketStreamer
from src.nba_predictor.betfair.panic_detector import PanicDetector

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Credentials
# Note: "Delay" Application Keys work for both Interactive and Non-Interactive
APP_KEY_DELAY = os.getenv("BETFAIR_APP_KEY", "QkYxxm82m7tiUQrI")
USERNAME = os.getenv("BETFAIR_USERNAME")
PASSWORD = os.getenv("BETFAIR_PASSWORD")
CERTS_PATH = os.getenv(
    "BETFAIR_CERTS_PATH", "/Users/fulvioventura/nba-predictor-streamlit/certs"
)


def test_stream():
    print("🏀 Finding a Live Market for Streaming Test (Cert Auth)...")

    if not USERNAME or not PASSWORD:
        print(
            "❌ Missing Credentials. Please export BETFAIR_USERNAME and BETFAIR_PASSWORD."
        )
        return

    # 1. Get a Market ID (Auto Login via Certs)
    client = BetfairClient(
        app_key=APP_KEY_DELAY,
        username=USERNAME,
        password=PASSWORD,
        certs_path=CERTS_PATH,
        locale="italy",
    )
    client.login()

    events = client.list_nba_events()
    if not events:
        print("❌ No events found to stream.")
        return

    # Check for "In Play" or just take the first one
    target_event = events[0]
    print(f"🎯 Target Event: {target_event.event.name}")

    # Get Market Catalogue
    catalogue = client.get_market_catalogue([target_event.event.id], max_results=1)
    if not catalogue:
        print("❌ No markets found for this event.")
        return

    market_id = catalogue[0].market_id
    market_name = catalogue[0].market_name
    print(f"📡 Streaming Market: {market_name} (ID: {market_id})...")

    # Initialize Detector
    detector = PanicDetector(
        drift_ticks=2,  # Low threshold for testing (usually 10)
        drift_seconds=10,  # Longer window for testing
        volume_threshold=100,  # Low threshold for testing matched volume
        volume_seconds=10,
    )
    print("🕵️ Panic Detector Initialized (Sensitive Mode for testing)")

    # 2. Start Stream (Auto Login via Certs)
    streamer = MarketStreamer(
        app_key=APP_KEY_DELAY,
        market_ids=[market_id],
        username=USERNAME,
        password=PASSWORD,
        certs_path=CERTS_PATH,
        locale="italy",
    )
    streamer.start()

    print("⏳ Listening for 30 seconds... (Watch for PANIC alerts)")

    start_time = time.time()
    try:
        for market_book in streamer.get_updates():
            # Process Update via Detector
            alerts = detector.process_update(market_book)

            # Print Alert if any
            if alerts:
                for alert in alerts:
                    icon = "🚨" if alert.severity == "CRITICAL" else "⚠️"
                    print(f"\n{icon} {alert.alert_type}: {alert.details}")

            # Periodic Heartbeat
            # print(".", end="", flush=True)

            if time.time() - start_time > 30:
                print("\n🛑 Test Duration Exceeded.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        print("See you later!")


if __name__ == "__main__":
    test_stream()
