#!/usr/bin/env python3
"""Debug game page V2 - use JavaScript click for Over/Under tab."""

import sys
from pathlib import Path
import time

project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def debug_game_page_v2():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )

    driver = webdriver.Chrome(options=options)

    try:
        url = "https://www.centroquote.it/basketball/usa/nba-2023-2024/boston-celtics-dallas-mavericks-dUhxW4j2/"
        print(f"Navigating to: {url}")
        driver.get(url)
        time.sleep(5)

        print(f"Page title: {driver.title}")

        # 1. Find and click Over/Under tab using JavaScript
        print("\n=== CLICKING OVER/UNDER TAB ===")
        try:
            # Find the correct Over/Under tab element
            ou_tabs = driver.find_elements(
                By.XPATH, "//*[contains(text(), 'Over/Under')]"
            )
            print(f"Found {len(ou_tabs)} O/U elements")

            for tab in ou_tabs:
                try:
                    # Use JavaScript to click
                    driver.execute_script("arguments[0].click();", tab)
                    print(f"Clicked via JS: {tab.tag_name}, text='{tab.text}'")
                    time.sleep(3)
                    break
                except Exception as e:
                    print(f"  Click failed: {e}")
                    continue

        except Exception as e:
            print(f"Error finding O/U tab: {e}")

        print(f"\nCurrent URL after click: {driver.current_url}")

        # 2. Look for score in the header/title area
        print("\n=== LOOKING FOR FINAL SCORE ===")

        # The score is often in the page title or header
        title = driver.title
        print(f"Page title: {title}")

        # Try to find score elements near team names
        team_elements = driver.find_elements(
            By.CSS_SELECTOR, "a[href*='/basketball/usa/nba']"
        )
        for el in team_elements[:5]:
            parent = el.find_element(By.XPATH, "..")
            print(
                f"Team link: {el.text}, Parent: {parent.text[:50] if parent.text else '(empty)'}"
            )

        # Look for numeric elements that could be scores
        numbers = driver.find_elements(
            By.CSS_SELECTOR, "p.leading-none, span.leading-none, div.leading-none"
        )
        print(f"\nNumeric-like elements: {len(numbers)}")
        for n in numbers[:10]:
            text = n.text.strip()
            if text and text.isdigit():
                print(f"  Potential score: {text}")

        # 3. Look for odds table after O/U click
        print("\n=== LOOKING FOR ODDS TABLE ===")

        # Look for bookmaker rows
        bookmaker_rows = driver.find_elements(By.CSS_SELECTOR, "div.flex.border-b")
        print(f"Bookmaker row candidates: {len(bookmaker_rows)}")

        # Look for Bet365 or Pinnacle
        bet365 = driver.find_elements(
            By.XPATH, "//*[contains(text(), 'Bet365') or contains(text(), 'bet365')]"
        )
        pinnacle = driver.find_elements(By.XPATH, "//*[contains(text(), 'Pinnacle')]")
        print(f"Bet365 mentions: {len(bet365)}, Pinnacle: {len(pinnacle)}")

        # Look for total line values (e.g., 215.5)
        print("\n=== LOOKING FOR TOTAL LINES ===")
        total_lines = driver.find_elements(
            By.XPATH,
            "//*[contains(text(), '215') or contains(text(), '216') or contains(text(), '220')]",
        )
        for tl in total_lines[:10]:
            print(f"  Line element: {tl.text[:80] if tl.text else '(empty)'}")

        # 4. Get all visible text to understand structure
        print("\n=== PAGE STRUCTURE SAMPLE ===")
        body = driver.find_element(By.TAG_NAME, "body")
        body_text = body.text[:2000]
        print(body_text)

    finally:
        driver.quit()


if __name__ == "__main__":
    debug_game_page_v2()
