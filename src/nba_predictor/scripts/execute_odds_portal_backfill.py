#!/usr/bin/env python3
"""
🏀 Odds Portal Backfill Orchestrator
Orchestrates massive data scraping for NBA Closing Line Value (CLV).

Enhanced with Tor integration for IP rotation and anti-ban protection.
"""

import sys
import logging
from pathlib import Path
import json
import time
from typing import Optional
import datetime
import argparse
from selenium.common.exceptions import NoSuchWindowException, WebDriverException

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nba_predictor.intelligence.odds_portal_scraper import (
    OddsPortalScraper,
    TorRotator,
    TOR_AVAILABLE,
    UNDETECTED_AVAILABLE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "data/raw/odds_portal/backfill_checkpoint.json"


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"seasons": {}, "last_scraped_game": None}


def save_checkpoint(checkpoint):
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)


def execute_backfill(
    seasons: list[tuple[int, int]], pilot: bool = False, use_tor: bool = True
):
    """Execute backfill with optional Tor integration for IP rotation."""

    # Initialize Tor rotator if available and requested
    tor_rotator = None
    proxy = None
    if use_tor and TOR_AVAILABLE:
        try:
            tor_rotator = TorRotator()
            proxy = TorRotator.get_socks_proxy()
            logger.info(f"🔒 Tor integration enabled (proxy: {proxy})")
        except Exception as e:
            logger.warning(f"Failed to initialize Tor: {e}, proceeding without")

    logger.info(
        f"📊 Starting with: Undetected={UNDETECTED_AVAILABLE}, Tor={TOR_AVAILABLE}"
    )

    scraper = OddsPortalScraper(headless=False, proxy=proxy)
    checkpoint = load_checkpoint()

    try:
        for start_year, end_year in seasons:
            season_str = f"{start_year}-{end_year}"
            if (
                checkpoint["seasons"].get(season_str, {}).get("completed", False)
                and not pilot
            ):
                logger.info(f"Skipping already completed season: {season_str}")
                continue

            logger.info(f"🚀 Starting backfill for season: {season_str}")

            # 1. Get Game URLs for this season
            season_urls = scraper.get_season_results_urls(start_year, end_year)
            game_urls = []
            for s_url in season_urls:
                # Retry logic for fetching season URLs
                for attempt in range(3):
                    try:
                        season_games = scraper.fetch_game_urls_from_season(s_url)
                        game_urls.extend(season_games)
                        break
                    except (NoSuchWindowException, WebDriverException) as e:
                        logger.warning(
                            f"Driver crash fetching season {s_url}: {e}. Restarting scraper..."
                        )
                        try:
                            scraper.close()
                        except:
                            pass
                        time.sleep(10)
                        scraper = OddsPortalScraper(headless=False, proxy=proxy)
                else:
                    logger.error(f"Failed to fetch games for {s_url} after retries")

            logger.info(f"Found {len(game_urls)} games for {season_str}")

            # 2. Scrape games
            season_data = []
            count = 0
            for url in game_urls:
                # Basic checkpoint check: skip if already in season_data?
                # Better to save incrementally

                logger.info(f"Scraping [{count + 1}/{len(game_urls)}]: {url}")

                game_data = None
                # Retry logic for individual games
                for attempt in range(2):
                    try:
                        game_data = scraper.scrape_game_data(url)
                        # Check if driver is still alive/valid by checking if data is None/empty due to crash
                        if (
                            not game_data
                        ):  # If scrape_game_data swallowed exception but failed hard
                            # This assumes scrape_game_data returns valid dict even on error, so we rely on exceptions primarily
                            pass
                        break
                    except (NoSuchWindowException, WebDriverException) as e:
                        logger.warning(
                            f"Driver crash scraping game {url}: {e}. Restarting scraper..."
                        )
                        try:
                            scraper.close()
                        except:
                            pass
                        time.sleep(5)
                        scraper = OddsPortalScraper(headless=False, proxy=proxy)

                if game_data and game_data.get(
                    "closing_lines"
                ):  # Changed from over_under_data to closing_lines
                    game_data["url"] = url
                    game_data["season"] = season_str
                    season_data.append(game_data)

                count += 1

                # Rotate Tor IP every 15 requests for better anti-ban
                if tor_rotator and count % 15 == 0:
                    logger.info("🔄 Rotating Tor IP...")
                    if tor_rotator.rotate_ip():
                        time.sleep(8)  # Wait for new circuit to establish
                    else:
                        logger.warning(
                            "Tor rotation failed, continuing with current IP"
                        )

                # Adaptive sleep based on request count
                sleep_time = 2 + (count % 10) * 0.3  # 2-5 seconds
                time.sleep(sleep_time)

                # Checkpointing every 10 games
                if count % 10 == 0:
                    scraper.save_to_parquet(
                        season_data, f"nba_clv_{season_str}_batch_{count // 10}.parquet"
                    )
                    season_data = []  # Clear batch

                if pilot and count >= 5:
                    logger.info("Pilot mode: stopping after 5 games")
                    break

            # Save any remaining data
            if season_data:
                scraper.save_to_parquet(
                    season_data, f"nba_clv_{season_str}_final.parquet"
                )

            if not pilot:
                if season_str not in checkpoint["seasons"]:
                    checkpoint["seasons"][season_str] = {}
                checkpoint["seasons"][season_str]["completed"] = True
                save_checkpoint(checkpoint)

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        scraper.close()
        logger.info("Backfill session closed.")


if __name__ == "__main__":
    # Seasons: 2021-2026 (Excluding 2020-2021 due to lazy loading issues)
    # Dec 2025: 2024-2025 completed, 2025-2026 ongoing
    # Seasons: 2021-2026 (Excluding 2020-2021 due to lazy loading issues)
    # Dec 2025: 2025-2026 ongoing
    # Split into individual tasks for granular checkpointing
    TARGET_SEASONS = [(y, y) for y in range(2021, 2026)]  # 2021..2025 -> 21-22 to 25-26

    import argparse

    parser = argparse.ArgumentParser(description="Odds Portal NBA Backfill with Tor")
    parser.add_argument(
        "--pilot", action="store_true", help="Run in pilot mode (5 games)"
    )
    parser.add_argument(
        "--no-tor", action="store_true", help="Disable Tor proxy rotation"
    )
    args = parser.parse_args()

    logger.info("🏀 Odds Portal NBA Backfill Starting...")
    logger.info(f"   Tor: {'Disabled' if args.no_tor else 'Enabled'}")
    logger.info(f"   Mode: {'Pilot (5 games)' if args.pilot else 'Full backfill'}")

    execute_backfill(TARGET_SEASONS, pilot=args.pilot, use_tor=not args.no_tor)
