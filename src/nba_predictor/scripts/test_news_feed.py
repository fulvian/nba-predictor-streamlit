import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.nba_predictor.intelligence.news_aggregator import CompositeNewsAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_news_feed():
    print("🚀 Starting News Feed Verification...")

    aggregator = CompositeNewsAggregator()

    print("\n1. Testing Injury Scraping (Rotowire)...")
    try:
        # Force refresh
        aggregator._refresh_data_if_needed()

        # Fetch some injuries
        injuries = aggregator.get_latest_news()
        injury_count = len([i for i in injuries if i["type"] == "injury"])

        print(f"✅ Found {injury_count} total injury reports.")

        if injury_count > 0:
            print("\n   Sample Injury:")
            print(f"   - {injuries[0]['text']}")
        else:
            print("⚠️ No injuries found. This might be off-season or scraper issue.")

    except Exception as e:
        print(f"❌ Injury Test Failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Testing RSS News...")
    try:
        news_items = [i for i in aggregator.get_latest_news() if i["type"] == "news"]
        print(f"✅ Found {len(news_items)} news items.")

        if news_items:
            print("\n   Sample News:")
            print(f"   - {news_items[0]['text']}")
            print(f"     (Source: {news_items[0]['source']})")

    except Exception as e:
        print(f"❌ RSS Test Failed: {e}")


if __name__ == "__main__":
    test_news_feed()
