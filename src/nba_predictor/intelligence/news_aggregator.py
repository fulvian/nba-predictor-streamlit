"""
NewsIntelligence Module
-----------------------
Aggregates real-time news from "Truly Free" sources (Rotowire Scraping, RSS Feeds).
Feeds into the NanoGPT Consensus Engine for "Thinking" analysis.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Implemented scrapers
try:
    from .scrapers import RotowireInjuryScraper, RSSNewsProvider, ESPNInjuryScraper
except ImportError:
    # Handle case where scrapers.py might not be importable yet during dev
    from src.nba_predictor.intelligence.scrapers import (
        RotowireInjuryScraper,
        RSSNewsProvider,
        ESPNInjuryScraper,
    )

logger = logging.getLogger(__name__)


class CompositeNewsAggregator:
    """
    Aggregates news from multiple free sources with caching and deduplication.
    """

    def __init__(self, cache_db_path: str = "data/persistent/news_cache.db"):
        self.cache_db_path = Path(cache_db_path)
        self.injury_scraper = RotowireInjuryScraper()
        self.espn_scraper = ESPNInjuryScraper()
        self.rss_provider = RSSNewsProvider()

        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for news and injuries."""
        try:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.cache_db_path)
            c = conn.cursor()

            # Table for injuries
            c.execute("""
                CREATE TABLE IF NOT EXISTS injuries (
                    player TEXT,
                    team TEXT,
                    status TEXT,
                    details TEXT,
                    source TEXT,
                    scraped_at TIMESTAMP,
                    PRIMARY KEY (player, team)
                )
            """)

            # Table for news
            c.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    link TEXT PRIMARY KEY,
                    title TEXT,
                    summary TEXT,
                    source TEXT,
                    published TIMESTAMP,
                    scraped_at TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Failed to init news cache: {e}")

    def get_latest_news(self, team_name: Optional[str] = None) -> list[dict]:
        """
        Get combined news/injuries.

        Args:
            team_name: Filter by team name (e.g., "Lakers", "Los Angeles Lakers")
        """
        news_items = []

        # 1. Fetch & Cache Data (if stale)
        self._refresh_data_if_needed()

        # 2. Query Cache
        try:
            conn = sqlite3.connect(self.cache_db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            # Injuries
            query = "SELECT * FROM injuries"
            params = []
            if team_name:
                query += " WHERE team LIKE ?"
                # Basic matching: "Lakers" matches "Lakers", "LAL" etc requires mapping
                # For now, simplistic partial match
                params.append(f"%{team_name}%")

            c.execute(query, params)
            for row in c.fetchall():
                news_items.append(
                    {
                        "type": "injury",
                        "player": row["player"],
                        "team": row["team"],
                        "status": row["status"],
                        "text": f"{row['player']} ({row['team']}) is {row['status']}: {row['details']}",
                        "source": row["source"],
                        "timestamp": row["scraped_at"],
                    }
                )

            # News
            query = "SELECT * FROM news ORDER BY published DESC LIMIT 50"
            c.execute(query)
            for row in c.fetchall():
                # Filter relevant global news or specific team news if logic allows
                # RSS feeds are often general. We include them for broad context.
                news_items.append(
                    {
                        "type": "news",
                        "title": row["title"],
                        "text": f"{row['title']} - {row['summary']}",
                        "source": row["source"],
                        "timestamp": row["published"],
                    }
                )

            conn.close()

        except Exception as e:
            logger.error(f"❌ Error reading news cache: {e}")

        return news_items

    def _refresh_data_if_needed(self):
        """Refresh data if cache is older than 1 hour."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            c = conn.cursor()

            # Check last update
            c.execute("SELECT MAX(scraped_at) FROM injuries")
            last_scrape = c.fetchone()[0]

            should_update = True
            if last_scrape:
                last_dt = datetime.fromisoformat(last_scrape)
                if datetime.now() - last_dt < timedelta(minutes=60):
                    should_update = False

            if should_update:
                logger.info("🔄 Refreshing News & Injury Data (Scraping)...")

                # 1. Injuries - Try Rotowire first, then ESPN
                injuries = self.injury_scraper.scrape()
                if not injuries:
                    logger.warning(
                        "⚠️ Rotowire scrape empty/failed. Attempting ESPN fallback..."
                    )
                    injuries = self.espn_scraper.scrape()

                if injuries:
                    # Upsert (Replace)
                    c.execute(
                        "DELETE FROM injuries"
                    )  # Simple strategy: replace all snapshot
                    c.executemany(
                        """
                        INSERT INTO injuries (player, team, status, details, source, scraped_at)
                        VALUES (:player, :team, :status, :details, :source, :scraped_at)
                    """,
                        injuries,
                    )

                # 2. RSS News
                news = self.rss_provider.fetch_all()
                if news:
                    # Insert ignore duplicates
                    c.executemany(
                        """
                        INSERT OR IGNORE INTO news (link, title, summary, source, published, scraped_at)
                        VALUES (:link, :title, :summary, :source, :published, :scraped_at)
                    """,
                        news,
                    )

                conn.commit()
                logger.info("✅ News data refreshed and cached.")

            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to refresh data: {e}")


# Backward compatibility alias
NewsAggregator = CompositeNewsAggregator
