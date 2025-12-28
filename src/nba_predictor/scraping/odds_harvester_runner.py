"""
OddsHarvester Runner - Wrapper for executing OddsHarvester CLI.

This module provides a Python wrapper around OddsHarvester for scraping
NBA Over/Under odds from OddsPortal.
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


class OddsHarvesterConfig:
    """Configuration for OddsHarvester execution."""

    def __init__(
        self,
        sport: str = "basketball",
        league: str = "usa-nba",
        markets: List[str] = None,
        bookmakers: Optional[List[str]] = None,
        output_format: str = "json",
        output_dir: Path = None,
        headless: bool = True,
        save_logs: bool = False,
        proxies: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
        concurrency: int = 3,
    ) -> None:
        """
        Initialize configuration.

        Args:
            sport: Sport type (basketball)
            league: League identifier (usa-nba)
            markets: List of markets to scrape (default: over/under)
            bookmakers: Target bookmakers (None = all)
            output_format: Output format (json or csv)
            output_dir: Directory for output files
            headless: Run browser in headless mode
            save_logs: Save browser logs to file
            proxies: List of proxies to use (e.g., ["http://user:pass@host:port"])
            max_pages: Maximum pages to scrape (None = all)
            concurrency: Number of concurrent tasks
        """
        self.sport = sport
        self.league = league
        self.markets = markets or ["over/under"]
        self.bookmakers = bookmakers
        self.output_format = output_format
        self.output_dir = output_dir or Path("data/odds_harvester_raw")
        self.headless = headless
        self.save_logs = save_logs
        self.proxies = proxies
        self.max_pages = max_pages
        self.concurrency = concurrency

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "OddsHarvesterConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        config = data.get("odds_harvester", {})
        return cls(
            sport=config.get("sport", "basketball"),
            league=config.get("league", "usa-nba"),
            markets=config.get("markets", ["over/under"]),
            bookmakers=config.get("bookmakers"),
            output_format=config.get("output", {}).get("format", "json"),
            output_dir=Path(
                config.get("output", {}).get("path", "data/odds_harvester_raw")
            ),
            headless=config.get("scraping", {}).get("headless", True),
            save_logs=config.get("scraping", {}).get("save_logs", False),
            proxies=config.get("scraping", {}).get("proxies"),
            max_pages=config.get("scraping", {}).get("max_pages"),
            concurrency=config.get("scraping", {}).get("concurrency", 3),
        )

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        config = {
            "odds_harvester": {
                "sport": self.sport,
                "league": self.league,
                "markets": self.markets,
                "bookmakers": self.bookmakers,
                "output": {
                    "format": self.output_format,
                    "path": str(self.output_dir),
                },
                "scraping": {
                    "headless": self.headless,
                    "save_logs": self.save_logs,
                    "proxies": self.proxies,
                    "max_pages": self.max_pages,
                    "concurrency": self.concurrency,
                },
            }
        }
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


class OddsHarvesterRunner:
    """
    Wrapper for executing OddsHarvester CLI for NBA Over/Under odds.

    OddsHarvester must be installed and available in PATH or specified path.
    Installation: pip install git+https://github.com/jordantete/OddsHarvester.git
    """

    def __init__(
        self,
        odds_harvester_path: Optional[Path] = None,
        config: Optional[OddsHarvesterConfig] = None,
    ) -> None:
        """
        Initialize runner.

        Args:
            odds_harvester_path: Path to OddsHarvester repo (if not in PATH)
            config: Configuration object (or use defaults)
        """
        self.odds_harvester_path = odds_harvester_path
        self.config = config or OddsHarvesterConfig()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_command(
        self,
        mode: str,  # "scrape_historic" or "scrape_upcoming"
        season: Optional[str] = None,
        date: Optional[str] = None,
        days_ahead: Optional[int] = None,
    ) -> List[str]:
        """
        Build CLI command for OddsHarvester.

        Args:
            mode: Scraping mode
            season: Season string for historic (e.g., "2023-2024" or "current")
            date: Specific date (YYYYMMDD)
            days_ahead: Days ahead for upcoming

        Returns:
            Command as list of strings
        """
        # Determine Python command
        if self.odds_harvester_path:
            python_cmd = ["python", str(self.odds_harvester_path / "src" / "main.py")]
        else:
            # Assume installed via pip or use uv
            python_cmd = ["uv", "run", "python", "-m", "odds_harvester"]

        cmd = python_cmd + [mode]

        # Add common arguments
        cmd.extend(["--sport", self.config.sport])
        cmd.extend(["--leagues", self.config.league])
        cmd.extend(["--markets", ",".join(self.config.markets)])
        cmd.extend(["--storage", "local"])

        # Output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.config.output_dir
            / f"nba_ou_{mode}_{timestamp}.{self.config.output_format}"
        )
        cmd.extend(["--file_path", str(output_file)])
        cmd.extend(["--format", self.config.output_format])

        # Mode-specific arguments
        if mode == "scrape_historic" and season:
            cmd.extend(["--season", season])
            if self.config.max_pages:
                cmd.extend(["--max_pages", str(self.config.max_pages)])
        elif mode == "scrape_upcoming" and date:
            cmd.extend(["--date", date])

        # Concurrency
        cmd.extend(["--concurrency_tasks", str(self.config.concurrency)])

        # Headless mode
        if self.config.headless:
            cmd.append("--headless")

        if self.config.save_logs:
            cmd.append("--save_logs")

        if self.config.proxies:
            cmd.append("--proxies")
            cmd.extend(self.config.proxies)

        # Target bookmaker if specified
        if self.config.bookmakers and len(self.config.bookmakers) == 1:
            cmd.extend(["--target_bookmaker", self.config.bookmakers[0]])

        return cmd, output_file

    def run_historic_scrape(
        self,
        season: str = "current",
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Execute historic scraping for a season.

        Args:
            season: Season string (e.g., "2023-2024", "2024-2025", "current")
            markets: Override markets (default: config markets)
            bookmakers: Override bookmakers (default: config bookmakers)
            dry_run: If True, print command without executing

        Returns:
            Path to output file, or None if failed
        """
        # Update config if overrides provided
        if markets:
            self.config.markets = markets
        if bookmakers:
            self.config.bookmakers = bookmakers

        cmd, output_file = self._build_command("scrape_historic", season=season)

        logger.info(f"Running OddsHarvester historic scrape for season {season}")
        logger.debug(f"Command: {' '.join(cmd)}")

        if dry_run:
            print(f"DRY RUN: {' '.join(cmd)}")
            return output_file

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"Scrape completed successfully, output: {output_file}")
                if output_file.exists():
                    return output_file
                else:
                    logger.warning(f"Output file not found after scrape: {output_file}")
                    return None
            else:
                logger.error(f"Scrape failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Scrape timed out after 1 hour")
            return None
        except FileNotFoundError:
            logger.error("OddsHarvester not found. Please install it first.")
            return None
        except Exception as e:
            logger.error(f"Scrape failed with error: {e}")
            return None

    def run_upcoming_scrape(
        self,
        days_ahead: int = 7,
        specific_date: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Execute upcoming matches scrape.

        Args:
            days_ahead: Number of days ahead to scrape
            specific_date: Specific date (YYYYMMDD format)
            dry_run: If True, print command without executing

        Returns:
            Path to output file, or None if failed
        """
        if specific_date:
            date_str = specific_date
        else:
            date_str = datetime.now().strftime("%Y%m%d")

        cmd, output_file = self._build_command("scrape_upcoming", date=date_str)

        logger.info(f"Running OddsHarvester upcoming scrape for date {date_str}")
        logger.debug(f"Command: {' '.join(cmd)}")

        if dry_run:
            print(f"DRY RUN: {' '.join(cmd)}")
            return output_file

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )

            if result.returncode == 0:
                logger.info(f"Scrape completed successfully, output: {output_file}")
                if output_file.exists():
                    return output_file
                else:
                    logger.warning(f"Output file not found after scrape: {output_file}")
                    return None
            else:
                logger.error(f"Scrape failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Scrape timed out after 30 minutes")
            return None
        except FileNotFoundError:
            logger.error("OddsHarvester not found. Please install it first.")
            return None
        except Exception as e:
            logger.error(f"Scrape failed with error: {e}")
            return None

    @staticmethod
    def install_odds_harvester() -> bool:
        """
        Install OddsHarvester from GitHub.

        Returns:
            True if installation successful
        """
        try:
            logger.info("Installing OddsHarvester from GitHub...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/jordantete/OddsHarvester.git",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("OddsHarvester installed successfully")
                return True
            else:
                logger.error(f"Installation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False

    @staticmethod
    def check_installation() -> bool:
        """Check if OddsHarvester is installed and accessible."""
        try:
            result = subprocess.run(
                ["python", "-c", "import odds_harvester"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False


if __name__ == "__main__":
    # Test runner
    logging.basicConfig(level=logging.INFO)

    runner = OddsHarvesterRunner()

    # Dry run test
    print("Testing historic scrape command (dry run):")
    runner.run_historic_scrape(season="2024-2025", dry_run=True)

    print("\nTesting upcoming scrape command (dry run):")
    runner.run_upcoming_scrape(dry_run=True)
