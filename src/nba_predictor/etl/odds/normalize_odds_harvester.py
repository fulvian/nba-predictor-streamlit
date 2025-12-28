"""
OddsHarvester Output Normalizer.

This module parses and normalizes the JSON/CSV output from OddsHarvester
into the nba_totals_odds schema format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


# Team name mapping from OddsPortal to NBA team IDs
ODDSPORTAL_TEAM_MAPPING = {
    # Standard names
    "Atlanta Hawks": {"nba_id": 1610612737, "abbreviation": "ATL"},
    "Boston Celtics": {"nba_id": 1610612738, "abbreviation": "BOS"},
    "Brooklyn Nets": {"nba_id": 1610612751, "abbreviation": "BKN"},
    "Charlotte Hornets": {"nba_id": 1610612766, "abbreviation": "CHA"},
    "Chicago Bulls": {"nba_id": 1610612741, "abbreviation": "CHI"},
    "Cleveland Cavaliers": {"nba_id": 1610612739, "abbreviation": "CLE"},
    "Dallas Mavericks": {"nba_id": 1610612742, "abbreviation": "DAL"},
    "Denver Nuggets": {"nba_id": 1610612743, "abbreviation": "DEN"},
    "Detroit Pistons": {"nba_id": 1610612765, "abbreviation": "DET"},
    "Golden State Warriors": {"nba_id": 1610612744, "abbreviation": "GSW"},
    "Houston Rockets": {"nba_id": 1610612745, "abbreviation": "HOU"},
    "Indiana Pacers": {"nba_id": 1610612754, "abbreviation": "IND"},
    "Los Angeles Clippers": {"nba_id": 1610612746, "abbreviation": "LAC"},
    "Los Angeles Lakers": {"nba_id": 1610612747, "abbreviation": "LAL"},
    "Memphis Grizzlies": {"nba_id": 1610612763, "abbreviation": "MEM"},
    "Miami Heat": {"nba_id": 1610612748, "abbreviation": "MIA"},
    "Milwaukee Bucks": {"nba_id": 1610612749, "abbreviation": "MIL"},
    "Minnesota Timberwolves": {"nba_id": 1610612750, "abbreviation": "MIN"},
    "New Orleans Pelicans": {"nba_id": 1610612740, "abbreviation": "NOP"},
    "New York Knicks": {"nba_id": 1610612752, "abbreviation": "NYK"},
    "Oklahoma City Thunder": {"nba_id": 1610612760, "abbreviation": "OKC"},
    "Orlando Magic": {"nba_id": 1610612753, "abbreviation": "ORL"},
    "Philadelphia 76ers": {"nba_id": 1610612755, "abbreviation": "PHI"},
    "Phoenix Suns": {"nba_id": 1610612756, "abbreviation": "PHX"},
    "Portland Trail Blazers": {"nba_id": 1610612757, "abbreviation": "POR"},
    "Sacramento Kings": {"nba_id": 1610612758, "abbreviation": "SAC"},
    "San Antonio Spurs": {"nba_id": 1610612759, "abbreviation": "SAS"},
    "Toronto Raptors": {"nba_id": 1610612761, "abbreviation": "TOR"},
    "Utah Jazz": {"nba_id": 1610612762, "abbreviation": "UTA"},
    "Washington Wizards": {"nba_id": 1610612764, "abbreviation": "WAS"},
    # Alternative names / abbreviations
    "LA Lakers": {"nba_id": 1610612747, "abbreviation": "LAL"},
    "L.A. Lakers": {"nba_id": 1610612747, "abbreviation": "LAL"},
    "LA Clippers": {"nba_id": 1610612746, "abbreviation": "LAC"},
    "L.A. Clippers": {"nba_id": 1610612746, "abbreviation": "LAC"},
    "New Jersey Nets": {"nba_id": 1610612751, "abbreviation": "BKN"},  # Historic
    "Charlotte Bobcats": {"nba_id": 1610612766, "abbreviation": "CHA"},  # Historic
    "New Orleans Hornets": {"nba_id": 1610612740, "abbreviation": "NOP"},  # Historic
    "Seattle SuperSonics": {"nba_id": 1610612760, "abbreviation": "OKC"},  # Historic
}


class OddsHarvesterNormalizer:
    def __init__(
        self, priority_bookmaker: str = "Pinnacle", fallback_to_average: bool = True
    ):
        self.priority_bookmaker = priority_bookmaker
        self.fallback_to_average = fallback_to_average
        self.debug_log_count = []


def map_oddsportal_team_name(raw_name: str) -> Optional[int]:
    """
    Map OddsPortal team name to NBA team ID.

    Args:
        raw_name: Team name from OddsPortal

    Returns:
        NBA team ID or None if not found
    """
    if not raw_name:
        return None

    # Direct lookup
    if raw_name in ODDSPORTAL_TEAM_MAPPING:
        return ODDSPORTAL_TEAM_MAPPING[raw_name]["nba_id"]

    # Case-insensitive lookup
    raw_lower = raw_name.lower().strip()
    for key, value in ODDSPORTAL_TEAM_MAPPING.items():
        if key.lower() == raw_lower:
            return value["nba_id"]

    # Partial match (last word = city or nickname)
    for key, value in ODDSPORTAL_TEAM_MAPPING.items():
        if raw_lower in key.lower() or key.lower() in raw_lower:
            return value["nba_id"]

    logger.warning(f"Unknown team name: {raw_name}")
    return None


def extract_season_from_date(game_date: str) -> str:
    """
    Extract NBA season from game date.

    NBA season runs October -> June, so:
    - Oct-Dec: season starts that year
    - Jan-Sep: season started previous year

    Args:
        game_date: Date string (YYYY-MM-DD)

    Returns:
        Season string (e.g., "2023-2024")
    """
    try:
        dt = datetime.strptime(game_date[:10], "%Y-%m-%d")
        if dt.month >= 10:
            start_year = dt.year
        else:
            start_year = dt.year - 1
        return f"{start_year}-{start_year + 1}"
    except (ValueError, TypeError):
        return "unknown"


class OddsHarvesterNormalizer:
    """Normalizes OddsHarvester output to nba_totals_odds schema."""

    def __init__(
        self, priority_bookmaker: str = "Pinnacle", fallback_to_average: bool = True
    ) -> None:
        """
        Initialize normalizer.

        Args:
            priority_bookmaker: Preferred bookmaker for closing odds
            fallback_to_average: If priority bookmaker unavailable, use average
        """
        self.priority_bookmaker = priority_bookmaker
        self.fallback_to_average = fallback_to_average

    def parse_json_output(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse OddsHarvester JSON output.

        Args:
            file_path: Path to JSON file

        Returns:
            List of raw match records
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # OddsHarvester output structure varies, handle common formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "matches" in data:
            return data["matches"]
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            return [data]

    def parse_csv_output(self, file_path: Path) -> pl.DataFrame:
        """
        Parse OddsHarvester CSV output.

        Args:
            file_path: Path to CSV file

        Returns:
            Raw Polars DataFrame
        """
        return pl.read_csv(file_path)

    def normalize_match_record(
        self, match: Dict[str, Any], scrape_datetime: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Normalize a single match record to schema format.

        OddsHarvester match structure (typical):
        {
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "date": "2024-01-15",
            "time": "19:30",
            "odds": {
                "over_under": {
                    "Pinnacle": {"line": 224.5, "over": 1.91, "under": 1.91},
                    "Bet365": {"line": 224.5, "over": 1.90, "under": 1.90},
                    ...
                }
            }
        }

        Args:
            match: Raw match dictionary from OddsHarvester
            scrape_datetime: When data was scraped

        Returns:
            List of normalized records (one per bookmaker)
        """
        scrape_dt = scrape_datetime or datetime.utcnow()

        # Extract basic match info
        home_team = match.get("home_team") or match.get("home")
        away_team = match.get("away_team") or match.get("away")
        game_date = (
            match.get("date") or match.get("game_date") or match.get("match_date")
        )

        if not home_team or not away_team:
            logger.warning(f"Missing team info in match: {match}")
            return []

        home_team_id = map_oddsportal_team_name(home_team)
        away_team_id = map_oddsportal_team_name(away_team)

        if not home_team_id or not away_team_id:
            logger.warning(f"Could not map teams: {home_team} vs {away_team}")
            return []

        # Generate pseudo game_id from date and teams
        game_id = self._generate_game_id(game_date, home_team_id, away_team_id)
        season = extract_season_from_date(game_date) if game_date else "unknown"
        stage = match.get("stage", "RS")  # Default to Regular Season

        records = []

        # Extract over/under odds
        odds_data = match.get("odds", {})
        ou_odds = (
            odds_data.get("over_under")
            or odds_data.get("totals")
            or match.get("over_under_market")  # Handle flat structure from preview mode
            or {}
        )

        if isinstance(ou_odds, list):
            # Handle preview mode output (list of submarkets)
            for submarket in ou_odds:
                sub_name = submarket.get("submarket_name", "")
                # Extract line from submarket name (e.g. "+210.5" -> "210.5")
                line_str = (
                    sub_name.replace("Over/Under", "").replace("+", "").strip()
                )  # FIXED line extraction?

                over = submarket.get("odds_over")
                under = submarket.get("odds_under")

                # Try to parse line
                try:
                    line_val = float(line_str)
                except ValueError:
                    # logger.debug(f"Skipping malformed line: {line_str}")
                    continue

                if over and under and over != "-" and under != "-":
                    try:
                        over_val = float(over)
                        under_val = float(under)

                        records.append(
                            {
                                "game_id": game_id,
                                "game_date": game_date,
                                "season": season,
                                "stage": stage,
                                "home_team_id": home_team_id,
                                "away_team_id": away_team_id,
                                "bookmaker": "average",  # Preview mode represents visible aggregated odds
                                "total_points_line": line_val,
                                "odds_over_decimal": over_val,
                                "odds_under_decimal": under_val,
                                "scrape_datetime": scrape_dt,
                                "source": "oddsharvester",
                                "is_closing": False,
                            }
                        )
                    except ValueError:
                        continue
        else:
            # Standard mode (Dict[Bookmaker, Odds])
            for bookmaker, book_odds in ou_odds.items():
                if isinstance(book_odds, dict):
                    line = book_odds.get("line") or book_odds.get("total")
                    over = book_odds.get("over") or book_odds.get("over_odds")
                    under = book_odds.get("under") or book_odds.get("under_odds")

                    if line and over and under:
                        records.append(
                            {
                                "game_id": game_id,
                                "game_date": game_date,
                                "season": season,
                                "stage": stage,
                                "home_team_id": home_team_id,
                                "away_team_id": away_team_id,
                                "bookmaker": bookmaker,
                                "total_points_line": float(line),
                                "odds_over_decimal": float(over),
                                "odds_under_decimal": float(under),
                                "scrape_datetime": scrape_dt,
                                "source": "oddsharvester",
                                "is_closing": False,
                            }
                        )

        return records

    def _generate_game_id(
        self, game_date: str, home_team_id: int, away_team_id: int
    ) -> str:
        """
        Generate a pseudo game_id for OddsHarvester data.

        Format: OH_YYYYMMDD_HOME_AWAY

        Args:
            game_date: Date string
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID

        Returns:
            Generated game ID
        """
        try:
            date_part = game_date.replace("-", "")[:8]
        except (AttributeError, TypeError):
            date_part = "00000000"

        return f"OH_{date_part}_{home_team_id}_{away_team_id}"

    def normalize_to_dataframe(
        self, file_path: Path, scrape_datetime: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Normalize OddsHarvester output file to Polars DataFrame.

        Args:
            file_path: Path to OddsHarvester output (JSON or CSV)
            scrape_datetime: When data was scraped

        Returns:
            Normalized Polars DataFrame
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".json":
            raw_data = self.parse_json_output(file_path)
            all_records = []
            for match in raw_data:
                all_records.extend(self.normalize_match_record(match, scrape_datetime))
        elif file_path.suffix.lower() == ".csv":
            raw_df = self.parse_csv_output(file_path)
            all_records = self._normalize_csv_to_records(raw_df, scrape_datetime)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if not all_records:
            logger.warning(f"No records extracted from {file_path}")
            return pl.DataFrame()

        return pl.DataFrame(all_records)

    def _normalize_csv_to_records(
        self, df: pl.DataFrame, scrape_datetime: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Normalize CSV DataFrame to records.

        Args:
            df: Raw CSV DataFrame
            scrape_datetime: When data was scraped

        Returns:
            List of normalized records
        """
        scrape_dt = scrape_datetime or datetime.utcnow()
        records = []

        # Try to identify column mapping (OddsHarvester CSV format varies)
        # Common formats: home, away, date, bookmaker, line, over, under

        for row in df.iter_rows(named=True):
            home_team = row.get("home_team") or row.get("home")
            away_team = row.get("away_team") or row.get("away")
            game_date = row.get("date") or row.get("game_date")
            bookmaker = row.get("bookmaker") or row.get("book")
            line = row.get("line") or row.get("total")
            over = row.get("over") or row.get("over_odds")
            under = row.get("under") or row.get("under_odds")

            if not all([home_team, away_team, line, over, under]):
                continue

            home_team_id = map_oddsportal_team_name(home_team)
            away_team_id = map_oddsportal_team_name(away_team)

            if not home_team_id or not away_team_id:
                continue

            game_id = self._generate_game_id(game_date, home_team_id, away_team_id)
            season = extract_season_from_date(game_date) if game_date else "unknown"

            records.append(
                {
                    "game_id": game_id,
                    "game_date": game_date,
                    "season": season,
                    "stage": "RS",
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "bookmaker": bookmaker or "unknown",
                    "total_points_line": float(line),
                    "odds_over_decimal": float(over),
                    "odds_under_decimal": float(under),
                    "scrape_datetime": scrape_dt,
                    "source": "oddsharvester",
                    "is_closing": False,
                }
            )

        return records

    def select_priority_odds(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Select priority bookmaker odds with fallback to average.

        Args:
            df: DataFrame with multiple bookmaker odds per game

        Returns:
            DataFrame with one row per game (priority bookmaker or average)
        """
        if df.is_empty():
            return df

        # Check if priority bookmaker exists
        if self.priority_bookmaker in df["bookmaker"].unique().to_list():
            priority_df = df.filter(pl.col("bookmaker") == self.priority_bookmaker)
            logger.info(f"Using {self.priority_bookmaker} as priority bookmaker")
            return priority_df

        if self.fallback_to_average:
            # Calculate average odds per game
            avg_df = df.group_by(
                [
                    "game_id",
                    "game_date",
                    "season",
                    "stage",
                    "home_team_id",
                    "away_team_id",
                    "total_points_line",
                ]
            ).agg(
                [
                    pl.col("odds_over_decimal").mean(),
                    pl.col("odds_under_decimal").mean(),
                    pl.col("scrape_datetime").max(),
                    pl.lit("average").alias("bookmaker"),
                    pl.first("source"),
                    pl.first("is_closing"),
                ]
            )
            logger.info("Using average odds as fallback")
            return avg_df

        # Return first available bookmaker
        first_book = df["bookmaker"].unique().to_list()[0]
        logger.info(f"Using {first_book} as only available bookmaker")
        return df.filter(pl.col("bookmaker") == first_book)


# Export team mapping for external use
def save_team_mapping(output_path: Path) -> None:
    """Save team mapping to JSON file."""
    import json

    with open(output_path, "w") as f:
        json.dump(ODDSPORTAL_TEAM_MAPPING, f, indent=2)
    logger.info(f"Saved team mapping to {output_path}")
