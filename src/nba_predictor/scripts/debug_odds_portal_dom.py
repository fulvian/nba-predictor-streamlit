#!/usr/bin/env python3
"""Debug V5 - scroll the page and wait for lazy-loaded content."""

import sys
from pathlib import Path
import time

project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def debug_scroll():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)

    try:
        url = "https://www.centroquote.it/basketball/usa/nba-2023-2024/results/"
        print(f"Navigating to: {url}")
        driver.get(url)

        # Initial wait
        time.sleep(5)
        print(f"URL: {driver.current_url}")

        # Scroll down multiple times to trigger lazy loading
        print("\n--- Scrolling to load content ---")
        for i in range(5):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1)
            print(
                f"  Scroll {i + 1}: height={driver.execute_script('return document.body.scrollHeight')}"
            )

        time.sleep(3)

        # Now count elements
        event_divs = driver.find_elements(By.XPATH, "//*[contains(@class, 'event')]")
        print(f"\nAfter scrolling: {len(event_divs)} elements with 'event' in class")

        # Print sample of page source around "event"
        page_source = driver.page_source
        print(f"Page source length: {len(page_source)}")

        # Find "event" occurrences
        import re

        event_matches = re.findall(r'class="[^"]*event[^"]*"', page_source)
        print(f"'event' class occurrences in source: {len(event_matches)}")
        for m in event_matches[:10]:
            print(f"  {m}")

        # Look for game score patterns (e.g., "108 - 97")
        score_patterns = re.findall(r"\d{2,3}\s*[-–]\s*\d{2,3}", page_source)
        print(f"\nScore patterns found: {len(score_patterns)}")
        for s in score_patterns[:10]:
            print(f"  {s}")

        # Try finding links directly
        print("\n--- All links with /nba-2023-2024/ or team names ---")
        all_links = driver.find_elements(By.TAG_NAME, "a")
        game_links = []
        for link in all_links:
            href = link.get_attribute("href") or ""
            if (
                "celtics" in href.lower()
                or "lakers" in href.lower()
                or "maverick" in href.lower()
            ):
                game_links.append(href)
                print(f"  {href}")

        print(f"\n{len(game_links)} links with team names found")

    finally:
        driver.quit()


if __name__ == "__main__":
    debug_scroll()
