"""
NBA Composite Scrapers Module
-----------------------------
Provides specific scraper implementations for the CompositeNewsAggregator.
Focuses on "Truly Free" sources:
1. Rotowire (HTML Scraping) for injuries
2. RSS Feeds for general news
"""

import logging
import random
import time
from datetime import datetime
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base scraper with politeness policies."""

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    ]

    def _get_headers(self) -> dict:
        """Rotate user agents to avoid detection."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
        }

    def _sleep_random(self, min_sec=1, max_sec=3):
        """Random sleep to behave like a human."""
        time.sleep(random.uniform(min_sec, max_sec))


class ESPNInjuryScraper(BaseScraper):
    """Scrapes ESPN NBA Injury Report page."""

    URL = "https://www.espn.com/nba/injuries"

    def scrape(self) -> list:
        """
        Scrape current injuries from ESPN.
        """
        injuries = []
        try:
            self._sleep_random(1, 2)
            response = requests.get(self.URL, headers=self._get_headers(), timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # ESPN Structure:
            # Tables for each team.
            # Table class often 'Table' or 'responsive-table-wrap'

            tables = soup.find_all(
                "div", class_="Table__Title"
            )  # Team names often here
            # Actually ESPN often lists teams as headers then a table below.

            # Strategy: Find all tables, extract rows
            # Each row: Name, Status, Date, Comment

            # The structure is often:
            # <div class="ResponsiveTable"> ... <table> ...

            all_tables = soup.find_all("table", class_="Table")

            for table in all_tables:
                # Try to find team name preceding the table?
                # Sometimes easiest is just to parse the row which might not have team name inside
                # but usually ESPN groups by team.

                # Let's try to extract rows directly and infer team or just get player/status
                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    if not cols or len(cols) < 2:
                        continue

                    # Header row check
                    if cols[0].get_text(strip=True) == "NAME":
                        continue

                    try:
                        # Col 0: Name, Col 1: Status, Col 2: Date, Col 3: Comment
                        name = cols[0].get_text(strip=True)
                        status = cols[1].get_text(strip=True)
                        comment = cols[3].get_text(strip=True) if len(cols) > 3 else ""

                        # We might ignore team for now if hard to find, or try to lookup
                        # But 'UnifiedHybridPipeline' can map player name to team if needed
                        # Or we can return 'Unknown' team

                        injuries.append(
                            {
                                "source": "ESPN",
                                "player": name,
                                "team": "Unknown",  # ESPN structure makes team extraction tricky without more logic
                                "status": status,
                                "details": comment,
                                "scraped_at": datetime.now().isoformat(),
                            }
                        )
                    except Exception:
                        continue

            logger.info(f"✅ Scraped {len(injuries)} injuries from ESPN")
            return injuries

        except Exception as e:
            logger.error(f"❌ Failed to scrape ESPN: {e}")
            return []


class RotowireInjuryScraper(BaseScraper):
    """Scrapes Rotowire NBA Injury Report page."""

    URL = "https://www.rotowire.com/basketball/injury-report.php"

    def scrape(self) -> list:
        """
        Scrape current injuries.
        Returns list of dicts: {player, team, status, injury, details, updated}
        """
        injuries = []
        try:
            self._sleep_random(1, 2)
            response = requests.get(self.URL, headers=self._get_headers(), timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Rotowire injury table structure often uses specific classes
            # We look for the main injury table rows
            # Note: This selector is based on typical Rotowire structure.
            # May need adjustment if they redesign.

            # They typically have a div with class 'injury-report' or table
            # Let's try to find potential player boxes or table rows

            # Strategy: Find all elements that look like player entries
            # Rotowire typically uses .injury-report__player-name linked to players

            player_cells = soup.find_all(class_="injury-report__player")

            if not player_cells:
                # Fallback: simple table row search if div structure changed
                rows = soup.find_all("tr")
                pass  # Logic to parse TRs if needed

            # Iterate through found card/rows
            # Note: Rotowire structure is usually:
            # <div class="injury-report__player"> <a ...>Name</a> </div>
            # <div class="injury-report__team"> ATL </div>
            # <div class="injury-report__status"> GTD </div>
            # <div class="injury-report__info"> Injury details... </div>

            boxes = soup.find_all("div", class_="injury-report__row")

            if not boxes:
                # Fallback for table-based layout
                logger.warning(
                    "Rotowire structure mismatch (div.injury-report__row). Attempting fallback."
                )
                return []

            for box in boxes:
                try:
                    name_tag = box.find(class_="injury-report__player-name")
                    team_tag = box.find(class_="injury-report__team")
                    status_tag = box.find(class_="injury-report__status")
                    info_tag = box.find(
                        class_="injury-report__details"
                    )  # usually holds the injury text

                    if name_tag and team_tag:
                        player = name_tag.get_text(strip=True)
                        team = team_tag.get_text(strip=True)
                        status = (
                            status_tag.get_text(strip=True) if status_tag else "Unknown"
                        )
                        details = info_tag.get_text(strip=True) if info_tag else ""

                        injuries.append(
                            {
                                "source": "Rotowire",
                                "player": player,
                                "team": team,
                                "status": status,
                                "details": details,
                                "scraped_at": datetime.now().isoformat(),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue

            logger.info(f"✅ Scraped {len(injuries)} injuries from Rotowire")
            return injuries

        except Exception as e:
            logger.error(f"❌ Failed to scrape Rotowire: {e}")
            return []


class RSSNewsProvider:
    """Fetches news from RSS feeds."""

    DEFAULT_FEEDS = [
        "https://www.espn.com/espn/rss/nba/news",
        "https://sports.yahoo.com/nba/rss.xml",
    ]

    def fetch_all(self, feed_urls: Optional[list] = None) -> list:
        """
        Fetch and parse all RSS feeds.
        Returns list of dicts: {title, summary, link, published, source}
        """
        urls = feed_urls or self.DEFAULT_FEEDS
        news_items = []

        for url in urls:
            try:
                feed = feedparser.parse(url)
                source_name = "RSS"
                if "title" in feed.feed:
                    source_name = feed.feed.title

                for entry in feed.entries:
                    news_items.append(
                        {
                            "source": source_name,
                            "title": entry.title,
                            "summary": getattr(entry, "summary", ""),
                            "link": entry.link,
                            "published": getattr(
                                entry, "published", datetime.now().isoformat()
                            ),
                            "scraped_at": datetime.now().isoformat(),
                        }
                    )
            except Exception as e:
                logger.error(f"❌ Failed to parse RSS {url}: {e}")

        logger.info(f"✅ Fetched {len(news_items)} news items from RSS")
        return news_items
