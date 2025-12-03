import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class TeamNameNormalizer:
    """
    Centralized utility for normalizing NBA team names and generating canonical IDs.
    Ensures consistent matching across different data sources (BallDontLie, NBA API, etc.).
    """

    # Mapping of various team name formats to a standard 3-letter abbreviation
    TEAM_MAPPING = {
        "Atlanta Hawks": "ATL",
        "Hawks": "ATL",
        "Atlanta": "ATL",
        "Boston Celtics": "BOS",
        "Celtics": "BOS",
        "Boston": "BOS",
        "Brooklyn Nets": "BKN",
        "Nets": "BKN",
        "Brooklyn": "BKN",
        "Charlotte Hornets": "CHA",
        "Hornets": "CHA",
        "Charlotte": "CHA",
        "Chicago Bulls": "CHI",
        "Bulls": "CHI",
        "Chicago": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Cavaliers": "CLE",
        "Cleveland": "CLE",
        "Dallas Mavericks": "DAL",
        "Mavericks": "DAL",
        "Dallas": "DAL",
        "Denver Nuggets": "DEN",
        "Nuggets": "DEN",
        "Denver": "DEN",
        "Detroit Pistons": "DET",
        "Pistons": "DET",
        "Detroit": "DET",
        "Golden State Warriors": "GSW",
        "Warriors": "GSW",
        "Golden State": "GSW",
        "Houston Rockets": "HOU",
        "Rockets": "HOU",
        "Houston": "HOU",
        "Indiana Pacers": "IND",
        "Pacers": "IND",
        "Indiana": "IND",
        "Los Angeles Clippers": "LAC",
        "Clippers": "LAC",
        "L.A. Clippers": "LAC",
        "LA Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Lakers": "LAL",
        "L.A. Lakers": "LAL",
        "LA Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Grizzlies": "MEM",
        "Memphis": "MEM",
        "Miami Heat": "MIA",
        "Heat": "MIA",
        "Miami": "MIA",
        "Milwaukee Bucks": "MIL",
        "Bucks": "MIL",
        "Milwaukee": "MIL",
        "Minnesota Timberwolves": "MIN",
        "Timberwolves": "MIN",
        "Minnesota": "MIN",
        "New Orleans Pelicans": "NOP",
        "Pelicans": "NOP",
        "New Orleans": "NOP",
        "New York Knicks": "NYK",
        "Knicks": "NYK",
        "New York": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Thunder": "OKC",
        "Oklahoma City": "OKC",
        "Orlando Magic": "ORL",
        "Magic": "ORL",
        "Orlando": "ORL",
        "Philadelphia 76ers": "PHI",
        "76ers": "PHI",
        "Philadelphia": "PHI",
        "Sixers": "PHI",
        "Phoenix Suns": "PHX",
        "Suns": "PHX",
        "Phoenix": "PHX",
        "Portland Trail Blazers": "POR",
        "Trail Blazers": "POR",
        "Portland": "POR",
        "Sacramento Kings": "SAC",
        "Kings": "SAC",
        "Sacramento": "SAC",
        "San Antonio Spurs": "SAS",
        "Spurs": "SAS",
        "San Antonio": "SAS",
        "Toronto Raptors": "TOR",
        "Raptors": "TOR",
        "Toronto": "TOR",
        "Utah Jazz": "UTA",
        "Jazz": "UTA",
        "Utah": "UTA",
        "Washington Wizards": "WAS",
        "Wizards": "WAS",
        "Washington": "WAS",
    }

    @staticmethod
    def normalize_team(team_name: str) -> str:
        """
        Normalizes a team name to its 3-letter abbreviation.

        Args:
            team_name: The team name to normalize (e.g., "Los Angeles Lakers", "Lakers")

        Returns:
            str: The 3-letter abbreviation (e.g., "LAL") or "UNK" if not found.
        """
        if not team_name:
            return "UNK"

        # Clean up the input
        clean_name = str(team_name).strip()

        # Direct lookup
        if clean_name in TeamNameNormalizer.TEAM_MAPPING:
            return TeamNameNormalizer.TEAM_MAPPING[clean_name]

        # Check if it's already a valid abbreviation (value in the map)
        if clean_name in TeamNameNormalizer.TEAM_MAPPING.values():
            return clean_name

        # Try case-insensitive lookup
        for key, value in TeamNameNormalizer.TEAM_MAPPING.items():
            if key.lower() == clean_name.lower():
                return value

        logger.warning(f"Could not normalize team name: {team_name}")
        return "UNK"

    @staticmethod
    def generate_match_id(game_date: date, home_team: str, away_team: str) -> str:
        """
        Generates a deterministic canonical ID for a game.
        Format: YYYYMMDD_AWAY_HOME (e.g., 20251202_OKC_GSW)

        Args:
            game_date: The date of the game
            home_team: Home team name
            away_team: Away team name

        Returns:
            str: The canonical match ID
        """
        try:
            date_str = game_date.strftime("%Y%m%d")
            home_abbr = TeamNameNormalizer.normalize_team(home_team)
            away_abbr = TeamNameNormalizer.normalize_team(away_team)

            return f"{date_str}_{away_abbr}_{home_abbr}"
        except Exception as e:
            logger.error(f"Error generating match ID: {e}")
            return f"ERROR_{datetime.now().timestamp()}"
