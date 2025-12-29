import logging
import sys
import os

sys.path.append(os.getcwd())

from src.nba_predictor.betfair.client import BetfairClient

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Hardcoded Creds for Test (User provided these)
APP_KEY_DELAY = "QkYxxm82m7tiUQrI"
SSOID = "ViZ3wYJTjQdVBk3X8plwQbxSRPGwUWuWI3vLTs4NW+0="


def test_connection():
    print("🏀 Testing Betfair Connection...")

    client = BetfairClient(app_key=APP_KEY_DELAY, ssoid=SSOID)
    client.login()

    print("📡 Fetching NBA Events...")
    events = client.list_nba_events()

    if not events:
        print("⚠️ No NBA events found. (Might be off-season or filter issue)")
    else:
        print(f"✅ Found {len(events)} NBA Events:")

        # Get Catalogue for first 5 events
        event_ids = [e.event.id for e in events[:5]]

        print("\n🔍 Fetching Market Catalogue (Match Odds) for first 5 events...")
        catalogues = client.get_market_catalogue(event_ids, max_results=5)

        for cat in catalogues:
            print(
                f"   - {cat.event.name} | Market: {cat.market_name} | ID: {cat.market_id}"
            )
            for runner in cat.runners:
                print(f"      > {runner.runner_name} (ID: {runner.selection_id})")
            print("")


if __name__ == "__main__":
    test_connection()
