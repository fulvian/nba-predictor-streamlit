"""
Odds Portal Scraper - Closing Line Value (CLV) Collector
--------------------------------------------------------
Specialized scraper for fetching historical NBA closing odds and totals
specifically from Bet365 and Pinnacle to optimize models for CLV.

Enhanced with undetected-chromedriver for anti-detection.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path

from typing import Any, Optional

import pandas as pd

# Use undetected-chromedriver for stealth
try:
    import undetected_chromedriver as uc

    UNDETECTED_AVAILABLE = True
except ImportError:
    from selenium import webdriver

    UNDETECTED_AVAILABLE = False

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    WebDriverException,
    TimeoutException,
)

from src.nba_predictor.intelligence.scrapers import BaseScraper

logger = logging.getLogger(__name__)

# Tor integration for free IP rotation
try:
    from stem import Signal
    from stem.control import Controller

    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False


class TorRotator:
    """
    Rotate Tor circuits for new IP addresses.

    Requires Tor to be running with ControlPort enabled:
    - SOCKSPort 9050
    - ControlPort 9051

    To install Tor on macOS: brew install tor
    To start: tor &
    """

    def __init__(self, control_port: int = 9051, password: str | None = None):
        self.control_port = control_port
        self.password = password
        self.rotation_count = 0

    def rotate_ip(self) -> bool:
        """Rotate Tor circuit to get a new IP."""
        if not TOR_AVAILABLE:
            logger.warning("stem library not available, cannot rotate Tor IP")
            return False

        try:
            with Controller.from_port(port=self.control_port) as controller:
                if self.password:
                    controller.authenticate(password=self.password)
                else:
                    controller.authenticate()
                controller.signal(Signal.NEWNYM)
                self.rotation_count += 1
                logger.info(f"Tor circuit rotated (count: {self.rotation_count})")
                return True
        except Exception as e:
            logger.warning(f"Failed to rotate Tor IP: {e}")
            return False

    @staticmethod
    def get_socks_proxy() -> str:
        """Get the Tor SOCKS5 proxy address for Selenium."""
        return "socks5://127.0.0.1:9050"


class OddsPortalScraper(BaseScraper):
    """
    Scraper for Odds Portal (and regional redirects like CentroQuote)
    to collect historical NBA closing line data.
    """

    BASE_URL = (
        "https://www.centroquote.it"  # Italian domain (redirected from oddsportal.com)
    )
    NBA_RESULTS_PATH = "/basketball/usa/nba/results/"

    # Selectors based on research
    SELECTORS = {
        "game_links": 'a[href*="/basketball/usa/nba"]',  # More general to match /nba-2023-2024/
        "pagination_next": "//a[contains(@class, 'pagination-link') and (text()='Avanti' or text()='»' or contains(text(), 'Next'))]",
        "over_under_row": "div.flex.justify-center",  # Updated based on 2025-12 inspection
        "over_under_line_text": "p",  # All p tags within row
    }

    def __init__(self, headless: bool = False, proxy: Optional[str] = None):
        """Initialize the scraper with enhanced stealth settings."""
        self.headless = headless
        self.proxy = proxy
        self.driver = None
        self.default_timeout = 15
        self.request_count = 0
        self.session_start = None
        self.max_requests_per_session = 25  # Rotate session after this many requests

        # Ensure raw data directory exists
        self.output_dir = Path("data/raw/odds_portal")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"OddsPortalScraper initialized (undetected: {UNDETECTED_AVAILABLE})"
        )

    def _init_driver(self):
        """Initialize the WebDriver with anti-detection measures."""
        if self.driver:
            return

        if UNDETECTED_AVAILABLE:
            # Use undetected-chromedriver for better stealth
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")

            if self.proxy:
                options.add_argument(f"--proxy-server={self.proxy}")

            try:
                self.driver = uc.Chrome(
                    options=options, headless=self.headless, use_subprocess=True
                )
                self.driver.set_page_load_timeout(30)
                self.session_start = datetime.now()
                self.request_count = 0
                logger.info("Undetected ChromeDriver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize undetected driver: {e}")
                raise
        else:
            # Fallback to standard Selenium
            from selenium import webdriver

            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument(f"user-agent={random.choice(self.USER_AGENTS)}")

            if self.proxy:
                chrome_options.add_argument(f"--proxy-server={self.proxy}")

            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                self.driver.set_page_load_timeout(30)
                self.session_start = datetime.now()
                self.request_count = 0
                logger.info("WebDriver initialized (fallback mode)")
            except WebDriverException as e:
                logger.error(f"Failed to initialize WebDriver: {e}")
                raise

    def _maybe_rotate_session(self):
        """Rotate session after max requests to avoid detection."""
        self.request_count += 1
        if self.request_count >= self.max_requests_per_session:
            logger.info(f"Rotating session after {self.request_count} requests")
            self.close()
            time.sleep(random.uniform(5, 10))  # Pause before new session
            self._init_driver()

    def _adaptive_sleep(self, base_min: float = 3, base_max: float = 6):
        """Sleep with exponential backoff based on request count."""
        # Increase delay as we make more requests in a session
        multiplier = 1 + (self.request_count / self.max_requests_per_session) * 0.5
        delay = random.uniform(base_min * multiplier, base_max * multiplier)
        time.sleep(delay)

    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("WebDriver closed")

    def get_season_results_urls(
        self, start_year: int = 2018, end_year: int = 2024
    ) -> list[str]:
        """
        Generate URLs for historical NBA seasons.

        For the current season, Odds Portal uses '/nba/results/' instead of '/nba-YYYY-YYYY/'.
        """
        from datetime import datetime

        current_year = datetime.now().year

        urls = []
        for year in range(start_year, end_year + 1):  # Include end_year
            if year >= current_year - 1:
                # Current season uses different URL pattern
                urls.append(f"{self.BASE_URL}/basketball/usa/nba/results/")
                break  # Only add current season once
            else:
                season_str = f"nba-{year}-{year + 1}"
                urls.append(f"{self.BASE_URL}/basketball/usa/{season_str}/results/")
        return urls

    def fetch_game_urls_from_season(self, season_url: str) -> list[str]:
        """Extract all individual game URLs from a season results page."""
        self._init_driver()
        logger.info(f"Fetching game URLs from {season_url}")

        game_urls = set()
        current_url = season_url

        # Extract season pattern from URL (e.g., "nba-2020-2021" or just "nba" for current)
        # URL formats:
        # - Historical: https://www.centroquote.it/basketball/usa/nba-2020-2021/results/
        # - Current: https://www.centroquote.it/basketball/usa/nba/results/
        season_match = season_url.split("/basketball/usa/")[-1].split("/")[0]
        # season_match will be "nba-2020-2021" or "nba"

        logger.info(f"Season pattern: {season_match}")

        while True:
            self.driver.get(current_url) if current_url else None
            self._sleep_random(3, 5)

            # Wait for content
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, self.SELECTORS["game_links"])
                    )
                )
            except:
                logger.warning(
                    f"Timeout waiting for games on {self.driver.current_url}"
                )

            # Scroll and extract - increased for heavy lazy loading (especially 2020-2021)
            logger.info("Scrolling for lazy loading...")
            for _ in range(10):  # Increased from 4 to 10 iterations
                self.driver.execute_script(
                    "window.scrollBy(0, 1200);"
                )  # Increased from 800 to 1200px
                self._sleep_random(1, 1.5)

            # Use season-specific selector pattern
            season_specific_selector = f'a[href*="/basketball/usa/{season_match}/"]'
            links = self.driver.find_elements(By.CSS_SELECTOR, season_specific_selector)

            page_count = 0
            for link in links:
                href = link.get_attribute("href")
                if href:
                    # Game URLs have format: /basketball/usa/{season}/{home}-{away}-{id}/
                    # Filter out non-game URLs (e.g., /results/, /standings/)
                    path_parts = href.rstrip("/").split("/")
                    if (
                        len(path_parts) >= 5
                    ):  # Must have at least: '', 'basketball', 'usa', season, game
                        last_part = path_parts[-1]
                        # Game IDs contain at least 2 dashes (teams) + 1 hash = 3 total
                        # E.g: "denver-nuggets-phoenix-suns-fu1WoxiF"
                        if last_part.count("-") >= 3:
                            game_urls.add(href)
                            page_count += 1

            logger.info(f"Extracted {page_count} games (Total: {len(game_urls)})")

            # Try to navigate to next page via click
            try:
                next_btn = self.driver.find_element(
                    By.XPATH, self.SELECTORS["pagination_next"]
                )
                # Compare current URL or state to avoid infinite loop
                prev_url = self.driver.current_url

                # JS click is safer against overlays (banners)
                logger.info("Clicking 'Avanti' via JavaScript...")
                self.driver.execute_script("arguments[0].click();", next_btn)

                # Wait for URL to change OR some time for JS to work
                time.sleep(4)
                if (
                    self.driver.current_url == prev_url
                    and not "#/page/" in self.driver.current_url
                ):
                    # If URL didn't change and no hash, we might be stuck
                    break
                current_url = None  # Don't reload, we just clicked
            except NoSuchElementException:
                logger.info("No more pages found.")
                break
            except Exception as e:
                logger.warning(f"Pagination error: {e}")
                break

        logger.info(f"Found {len(game_urls)} games in season")
        return list(game_urls)

    def scrape_game_data(self, game_url: str) -> dict[str, Any]:
        """
        Scrape Over/Under closing line and odds for a specific game.

        Extracts:
        - Closing line value (e.g., 211.0, 215.5)
        - Over odds (e.g., 1.91)
        - Under odds (e.g., 1.91)
        - Bookmaker (bet365, Pinnacle)
        - Metadata: Teams, Date, Score (best effort)
        """
        self._init_driver()
        logger.info(f"Scraping game: {game_url}")
        import re

        # Metadata from URL (Fallback)
        # URL format: .../nba/home-team-away-team-HASH/
        try:
            slug = game_url.rstrip("/").split("/")[-1]
            # Remove hash (last part after last dash)
            if "-" in slug:
                parts = slug.split("-")
                # Heuristic: Hash is usually mixed case/alphanum, teams are lowercase words
                # But safer to just assume last part is hash if it looks like one (e.g. 8 chars)
                # Actually, URL structure is reliable.
                # e.g. atlanta-hawks-chicago-bulls-AVibthnK
                # The hash is always at end.
                # But how to split home/away? "new-york-knicks" vs "utah-jazz". ambiguous.
                # We will rely on extracting FROM PAGE if possible, URL as fallback.
                pass
        except:
            pass

        data = {
            "url": game_url,
            "timestamp": datetime.now().isoformat(),
            "game_id": game_url.rstrip("/").split("-")[-1],
            "closing_lines": [],  # List of {line, over_odds, under_odds, bookmaker}
            "home_team": None,
            "away_team": None,
            "score_home": None,
            "score_away": None,
            "game_date": None,
        }

        try:
            # Navigate directly to O/U tab via URL hash
            ou_url = game_url.rstrip("/") + "/#over-under;1"
            self.driver.get(ou_url)
            self._sleep_random(3, 5)
            logger.info("Navigated to Over/Under tab")

            # Extract Metadata from Header
            try:
                # Teams from H1
                h1 = self.driver.find_element(By.TAG_NAME, "h1")
                h1_text = h1.text.strip()
                # Format: "Atlanta Hawks - Chicago Bulls" or "Atlanta Hawks vs Chicago Bulls"
                if "-" in h1_text:
                    teams = h1_text.split("-")
                elif "vs" in h1_text:
                    teams = h1_text.split("vs")
                elif "–" in h1_text:  # En dash
                    teams = h1_text.split("–")
                else:
                    teams = []

                if len(teams) >= 2:
                    data["home_team"] = teams[0].strip()
                    data["away_team"] = (
                        teams[1].split("Quote")[0].strip()
                    )  # Remove "Quote..." suffix if present

                # Date and Score
                # Usually in a div under H1 or parallel
                # Look for score pattern regex in the header region
                header_text = self.driver.find_element(
                    By.CSS_SELECTOR, "div.flex.w-full"
                ).text  # Generic container

                # Regex for score: "100:98" or "100 - 98"
                score_match = re.search(r"(\d{2,3})[:\-](\d{2,3})", header_text)
                if score_match:
                    data["score_home"] = int(score_match.group(1))
                    data["score_away"] = int(score_match.group(2))

                # Regex for date: "22 Dec 2023" or "22.12.2023"
                # OddsPortal often uses "22.12.2023, 01:30"
                date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", header_text)
                if date_match:
                    data["game_date"] = date_match.group(1)  # Keep as string for now

            except Exception as meta_e:
                logger.warning(f"Metadata extraction failed: {meta_e}")

            # Scroll to load all lines
            for _ in range(3):
                self.driver.execute_script("window.scrollBy(0, 300);")
                time.sleep(0.5)
            self._sleep_random(1, 2)

            # Find all p tags in the main odds container
            # Updated 2025-12 logic: Use sequential scanning of all p tags
            # The structure is flattened inside large containers.
            all_p_tags = self.driver.find_elements(
                By.CSS_SELECTOR, "div.flex.justify-center p"
            )

            logger.info(f"Scanning {len(all_p_tags)} text elements for O/U data")

            p_texts = [p.text.strip() for p in all_p_tags]
            extracted_count = 0

            i = 0
            while i < len(p_texts):
                text = p_texts[i]

                # Check for "Over/Under +216.5" or "O/U"
                if "Over/Under" in text or "O/U" in text:
                    # Found a potential line header
                    # Expected sequence from probe:
                    # i+0: "Over/Under +216.5"
                    # ...
                    # i+3: Over Odds (e.g. "1.32")
                    # i+4: Under Odds (e.g. "3.19")

                    try:
                        # Extract line value
                        line_match = re.search(r"[+-]?(\d{2,3}(?:\.\d)?)", text)
                        if not line_match:
                            i += 1
                            continue

                        line_value = float(line_match.group(1))

                        # Check bounds for odds
                        if i + 4 >= len(p_texts):
                            break

                        over_text = p_texts[i + 3]
                        under_text = p_texts[i + 4]

                        # Skip empty or dashed odds
                        if (
                            not over_text
                            or over_text == "-"
                            or not under_text
                            or under_text == "-"
                        ):
                            i += 1
                            continue

                        over_odds = float(over_text)
                        under_odds = float(under_text)

                        # Validate reasonable odds
                        if not (
                            1.01 <= over_odds <= 50.0 and 1.01 <= under_odds <= 50.0
                        ):
                            i += 1
                            continue

                        # Success
                        line_data = {
                            "line": line_value,
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "bookmakers": [],
                        }

                        # Avoid duplicates (if same line appears multiple times, take first or overwrite?)
                        # We append all unique lines found
                        if not any(
                            d["line"] == line_value for d in data["closing_lines"]
                        ):
                            data["closing_lines"].append(line_data)
                            extracted_count += 1
                            logger.info(
                                f"Extracted line {line_value}: Over={over_odds}, Under={under_odds}"
                            )

                        # Advance index to skip this block?
                        # The block seems to repeat every ~6-10 elements.
                        # Safe to just increment i normally, but jumping ahead slightly is faster
                        # However, probe showed next "Over/Under" at index 19 (prev was 13). +6 difference.
                        # So we can safely skip 5 elements.
                        i += 5
                        continue

                    except (ValueError, IndexError):
                        pass

                i += 1

            if not data["closing_lines"]:
                logger.warning("No O/U lines extracted from page")

        except Exception as e:
            logger.error(f"Error scraping {game_url}: {e}")

        return data

    def _extract_bookmaker_odds(self) -> dict[str, dict]:
        """
        Extract odds from expanded bookmaker table.
        Targets: bet365, Pinnacle
        """
        import re

        bookmaker_data = {}
        targets = ["bet365", "pinnacle"]

        try:
            # Based on browser analysis: bookmaker names in a.max-mm:hidden p
            # Fallback: find all visible bookmaker rows
            bm_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "a[href*='bookmaker'] p, a.max-mm\\:hidden p"
            )

            if not bm_elements:
                # Alternative: find rows with bookmaker logos/names
                bm_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "div.flex.items-center p"
                )

            for elem in bm_elements:
                name = elem.text.strip().lower()
                for target in targets:
                    if target in name:
                        # Find sibling odds elements
                        parent = elem.find_element(
                            By.XPATH, "./ancestor::div[contains(@class, 'flex')]"
                        )
                        odds_p = parent.find_elements(By.CSS_SELECTOR, "p")

                        odds_vals = []
                        for p in odds_p:
                            txt = p.text.strip()
                            if re.match(r"^\d+\.\d{2}$", txt):
                                odds_vals.append(float(txt))

                        if len(odds_vals) >= 2:
                            bookmaker_data[target] = {
                                "over": odds_vals[0],
                                "under": odds_vals[1],
                            }
                            logger.debug(
                                f"Found {target}: Over={odds_vals[0]}, Under={odds_vals[1]}"
                            )
                        break

        except Exception as e:
            logger.debug(f"Error extracting bookmaker odds: {e}")

        return bookmaker_data

    def save_to_parquet(self, data: list[dict[str, Any]], filename: str):
        """Save collected data to Parquet."""
        if not data:
            logger.warning("No data to save")
            return

        import json

        # Serialize complex nested structures to JSON strings to avoid PyArrow schema errors
        # especially with Empty Structs or variable schemas
        clean_data = []
        for item in data:
            item_copy = item.copy()
            for key, val in item_copy.items():
                if isinstance(val, (list, dict)):
                    item_copy[key] = json.dumps(val)
            clean_data.append(item_copy)

        df = pd.DataFrame(clean_data)
        file_path = self.output_dir / filename
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(data)} records to {file_path}")


if __name__ == "__main__":
    # Quick test if run directly
    logging.basicConfig(level=logging.INFO)
    scraper = OddsPortalScraper(headless=False)
    try:
        urls = scraper.get_season_results_urls(2023, 2024)
        print(f"Season URL: {urls[0]}")
    finally:
        scraper.close()
