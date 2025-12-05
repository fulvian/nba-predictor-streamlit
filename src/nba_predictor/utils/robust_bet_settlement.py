"""
🎯 Robust NBA Bet Settlement System - Production Ready Solution

Sistema robusto di settlement per scommesse NBA che risolve tutti i problemi identificati:
1. Database schema issues
2. Game ID mismatch
3. API NBA scores issues
4. Team name mapping
5. Missing error recovery

Basato su: BETTING_SETTLEMENT_ANALYSIS_REPORT.md
"""

import logging
import json
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameMatch:
    """Rappresenta un match tra scommessa e partita NBA."""

    bet_id: str
    game_id_from_bet: str
    nba_game_id: Optional[str]
    home_team: str
    away_team: str
    game_date: datetime
    final_score: Optional[Tuple[int, int]] = None
    match_confidence: float = 0.0


class NBATeamMapper:
    """Sistema robusto di mapping tra team names e NBA game IDs."""

    # Mappatura completa dei team NBA → abbreviazioni ufficiali
    TEAM_MAPPINGS = {
        # Eastern Conference - Atlantic
        "Boston Celtics": "BOS",
        "Celtics": "BOS",
        "Boston": "BOS",
        "Brooklyn Nets": "BRK",
        "Nets": "BRK",
        "Brooklyn": "BRK",
        "New York Knicks": "NYK",
        "Knicks": "NYK",
        "New York": "NYK",
        "Philadelphia 76ers": "PHI",
        "Sixers": "PHI",
        "Philadelphia": "PHI",
        "76ers": "PHI",
        "Toronto Raptors": "TOR",
        "Raptors": "TOR",
        "Toronto": "TOR",
        # Eastern Conference - Central
        "Chicago Bulls": "CHI",
        "Bulls": "CHI",
        "Chicago": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Cavaliers": "CLE",
        "Cleveland": "CLE",
        "Cavs": "CLE",
        "Detroit Pistons": "DET",
        "Pistons": "DET",
        "Detroit": "DET",
        "Indiana Pacers": "IND",
        "Pacers": "IND",
        "Indiana": "IND",
        "Milwaukee Bucks": "MIL",
        "Bucks": "MIL",
        "Milwaukee": "MIL",
        # Eastern Conference - Southeast
        "Atlanta Hawks": "ATL",
        "Hawks": "ATL",
        "Atlanta": "ATL",
        "Charlotte Hornets": "CHA",
        "Hornets": "CHA",
        "Charlotte": "CHA",
        "Miami Heat": "MIA",
        "Heat": "MIA",
        "Miami": "MIA",
        "Orlando Magic": "ORL",
        "Magic": "ORL",
        "Orlando": "ORL",
        "Washington Wizards": "WAS",
        "Wizards": "WAS",
        "Washington": "WAS",
        # Western Conference - Northwest
        "Denver Nuggets": "DEN",
        "Nuggets": "DEN",
        "Denver": "DEN",
        "Minnesota Timberwolves": "MIN",
        "Timberwolves": "MIN",
        "Minnesota": "MIN",
        "T-Wolves": "MIN",
        "Oklahoma City Thunder": "OKC",
        "Thunder": "OKC",
        "Oklahoma City": "OKC",
        "Portland Trail Blazers": "POR",
        "Trail Blazers": "POR",
        "Portland": "POR",
        "Blazers": "POR",
        "Utah Jazz": "UTA",
        "Jazz": "UTA",
        "Utah": "UTA",
        # Western Conference - Pacific
        "Golden State Warriors": "GSW",
        "Warriors": "GSW",
        "Golden State": "GSW",
        "Los Angeles Clippers": "LAC",
        "Clippers": "LAC",
        "LA Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Lakers": "LAL",
        "LA Lakers": "LAL",
        "Phoenix Suns": "PHX",
        "Suns": "PHX",
        "Phoenix": "PHX",
        "Sacramento Kings": "SAC",
        "Kings": "SAC",
        "Sacramento": "SAC",
        # Western Conference - Southwest
        "Dallas Mavericks": "DAL",
        "Mavericks": "DAL",
        "Dallas": "DAL",
        "Mavs": "DAL",
        "Houston Rockets": "HOU",
        "Rockets": "HOU",
        "Houston": "HOU",
        "Memphis Grizzlies": "MEM",
        "Grizzlies": "MEM",
        "Memphis": "MEM",
        "New Orleans Pelicans": "NOP",
        "Pelicans": "NOP",
        "New Orleans": "NOP",
        "San Antonio Spurs": "SAS",
        "Spurs": "SAS",
        "San Antonio": "SAS",
    }

    @classmethod
    def normalize_team_name(cls, team_name: str) -> str:
        """Normalizza il nome del team all'abbreviazione ufficiale NBA."""
        if not team_name:
            return ""

        # Rimuovi caratteri speciali e converti a case insensitive
        clean_name = re.sub(r"[^\w\s]", "", team_name.strip()).upper()

        # Cerca match diretto nelle mappature
        for full_name, abbreviation in cls.TEAM_MAPPINGS.items():
            if clean_name == full_name.upper() or clean_name == abbreviation.upper():
                return abbreviation

        # Cerca match parziale
        for key, value in cls.TEAM_MAPPINGS.items():
            if clean_name in key.upper() or key.upper() in clean_name:
                return value

        return team_name  # Return original if no match

    @classmethod
    def calculate_match_confidence(
        cls, bet_home: str, bet_away: str, api_home: str, api_away: str
    ) -> float:
        """Calcola il confidence score del matching tra team."""
        bet_home_norm = cls.normalize_team_name(bet_home)
        bet_away_norm = cls.normalize_team_name(bet_away)
        api_home_norm = cls.normalize_team_name(api_home)
        api_away_norm = cls.normalize_team_name(api_away)

        # Direct match (perfect score)
        if (bet_home_norm == api_home_norm and bet_away_norm == api_away_norm) or (
            bet_home_norm == api_away_norm and bet_away_norm == api_home_norm
        ):
            return 1.0

        # Partial match
        home_match = bet_home_norm in api_home_norm or api_home_norm in bet_home_norm
        away_match = bet_away_norm in api_away_norm or api_away_norm in bet_away_norm

        if home_match and away_match:
            return 0.8
        elif home_match or away_match:
            return 0.5
        else:
            return 0.0


class NBABoxscoreAPI:
    """API client robusto per recuperare punteggi finali NBA."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        }

    def get_game_boxscore(self, game_id: str) -> Optional[Tuple[int, int]]:
        """
        Recupera il punteggio finale usando Boxscore API (più affidabile).

        Args:
            game_id: NBA game ID (es. "0022400001")

        Returns:
            Tuple (home_score, away_score) o None se non disponibile
        """
        try:
            from nba_api.stats.endpoints import BoxScoreTraditionalV2

            # Usa nba_api per ottenere boxscore completo
            boxscore = BoxScoreTraditionalV2(game_id=game_id)
            data = boxscore.get_normalized_dict()

            if "PlayerStats" in data and data["PlayerStats"]:
                player_stats = data["PlayerStats"]
                if player_stats:
                    # Estrai punteggi dal primo giocatore (ogni riga ha i punteggi totali)
                    home_score = 0
                    away_score = 0

                    for row in player_stats:
                        if row.get("PTS"):
                            team_id = row.get("TEAM_ID")
                            if team_id and len(player_stats) > 0:
                                # Ottieni il primo punteggio di ogni team
                                if row.get("TEAM_ID") == player_stats[0].get("TEAM_ID"):
                                    home_score = max(home_score, int(row.get("PTS", 0)))
                                else:
                                    away_score = max(away_score, int(row.get("PTS", 0)))

                    if home_score > 0 and away_score > 0:
                        logger.info(
                            f"✅ Boxscore API found: {game_id} → {away_score}-{home_score}"
                        )
                        return (home_score, away_score)

        except Exception as e:
            logger.debug(f"Boxscore API failed for {game_id}: {e}")

        # Fallback a NBA.com stats API
        return self._get_fallback_score(game_id)

    def _get_fallback_score(self, game_id: str) -> Optional[Tuple[int, int]]:
        """Metodo fallback usando NBA.com API."""
        try:
            # Prova score endpoint per data specifica
            date_str = f"20{game_id[1:3]}-{game_id[3:5]}-{game_id[5:7]}"  # Converti game_id a data
            url = "https://stats.nba.com/stats/scoreboardv2"
            params = {"LeagueID": "00", "GameDate": date_str}

            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if "resultSets" in data:
                    for rs in data["resultSets"]:
                        if rs["name"] == "Scoreboard":
                            for row in rs["rowSet"]:
                                if str(row[0]) == game_id:  # GAME_ID match
                                    home_score = row[21] if len(row) > 21 else 0
                                    away_score = row[22] if len(row) > 22 else 0
                                    game_status = row[3] if len(row) > 3 else ""

                                    if (
                                        "Final" in game_status
                                        and home_score > 0
                                        and away_score > 0
                                    ):
                                        logger.info(
                                            f"✅ Fallback API found: {game_id} → {away_score}-{home_score}"
                                        )
                                        return (home_score, away_score)
        except Exception as e:
            logger.debug(f"Fallback API failed for {game_id}: {e}")

        return None

    def find_game_by_teams_and_date(
        self, home_team: str, away_team: str, game_date: datetime
    ) -> Optional[str]:
        """
        Trova NBA game ID usando team names e data.
        Uses shared utility for robust parsing and fallback strategies.
        """
        try:
            from nba_predictor.utils.nba_timezone_utils import (
                get_nba_games_official_api,
            )

            # Use robust shared utility
            games_list = get_nba_games_official_api(
                game_date.date() if isinstance(game_date, datetime) else game_date
            )

            if not games_list:
                return None

            best_match_id = None
            best_confidence = 0.0

            for game in games_list:
                api_home = game.get("home_team", "")
                api_away = game.get("away_team", "")
                api_game_id = str(game.get("game_id", ""))

                # Calcola confidence del matching
                confidence = NBATeamMapper.calculate_match_confidence(
                    home_team, away_team, api_home, api_away
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match_id = api_game_id

            if best_confidence >= 0.8:
                logger.info(f"✅ Game found: {best_match_id} (Conf: {best_confidence})")
                return best_match_id

            return None

        except Exception as e:
            logger.error(f"Error finding game: {e}")
            return None


class RobustBetSettlement:
    """
    Sistema robusto di settlement che risolve tutti i problemi identificati.
    """

    def __init__(self, betting_db_manager):
        """
        Inizializza il sistema robusto di settlement.

        Args:
            betting_db_manager: Istanza del BettingDatabaseManager
        """
        self.betting_db = betting_db_manager
        self.boxscore_api = NBABoxscoreAPI()
        self.team_mapper = NBATeamMapper()
        self.logger = logger

    def analyze_pending_bets(self) -> List[GameMatch]:
        """
        Analizza le scommesse pendenti e crea matches con partite NBA.

        Returns:
            Lista di GameMatch con confidence scores
        """
        try:
            pending_bets = self.betting_db.get_pending_bets()
            game_matches = []

            for bet in pending_bets:
                # Estrai team names dal bet
                home_team = getattr(bet, "home_team", "")
                away_team = getattr(bet, "away_team", "")
                game_date = getattr(bet, "placed_at", datetime.now())

                if not home_team or not away_team:
                    self.logger.warning(f"⚠️ Bet {bet.bet_id} missing team names")
                    continue

                # Cerca NBA game ID
                nba_game_id = self.boxscore_api.find_game_by_teams_and_date(
                    home_team, away_team, game_date
                )

                game_match = GameMatch(
                    bet_id=bet.bet_id,
                    game_id_from_bet=bet.game_id,
                    nba_game_id=nba_game_id,
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_date,
                    match_confidence=1.0 if nba_game_id else 0.0,
                )

                game_matches.append(game_match)

            self.logger.info(f"📊 Analyzed {len(game_matches)} pending bets")
            return game_matches

        except Exception as e:
            self.logger.error(f"❌ Error analyzing pending bets: {e}")
            return []

    def get_final_scores(self, game_matches: List[GameMatch]) -> List[GameMatch]:
        """
        Recupera punteggi finali per i game matches.

        Args:
            game_matches: Lista di GameMatch

        Returns:
            Lista aggiornata di GameMatch con punteggi finali
        """
        settled_matches = []

        for match in game_matches:
            if not match.nba_game_id:
                self.logger.warning(f"⚠️ No NBA game ID for bet {match.bet_id}")
                continue

            try:
                # Recupera punteggio finale
                final_score = self.boxscore_api.get_game_boxscore(match.nba_game_id)

                if final_score:
                    match.final_score = final_score
                    self.logger.info(
                        f"✅ Got final score for {match.bet_id}: {final_score[1]}-{final_score[0]}"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ No final score found for game {match.nba_game_id}"
                    )

                settled_matches.append(match)

            except Exception as e:
                self.logger.error(
                    f"❌ Error getting final score for {match.bet_id}: {e}"
                )
                settled_matches.append(match)  # Keep even if failed

        return settled_matches

    def settle_bets_with_scores(self, game_matches: List[GameMatch]) -> Dict[str, any]:
        """
        Esegue il settlement delle scommesse con i punteggi finali.

        Args:
            game_matches: Lista di GameMatch con punteggi finali

        Returns:
            Report del settlement
        """
        settlement_report = {
            "total_processed": len(game_matches),
            "successful_settlements": 0,
            "failed_settlements": 0,
            "details": [],
        }

        for match in game_matches:
            try:
                if not match.final_score or not match.nba_game_id:
                    settlement_report["failed_settlements"] += 1
                    settlement_report["details"].append(
                        {
                            "bet_id": match.bet_id,
                            "result": "failed",
                            "reason": "No final score or NBA game ID",
                        }
                    )
                    continue

                home_score, away_score = match.final_score

                # Usa il metodo esistente del database manager
                settled_count = self.betting_db.update_game_results_from_scores(
                    match.nba_game_id, home_score, away_score
                )

                if settled_count > 0:
                    settlement_report["successful_settlements"] += settled_count
                    settlement_report["details"].append(
                        {
                            "bet_id": match.bet_id,
                            "nba_game_id": match.nba_game_id,
                            "result": "settled",
                            "final_score": f"{away_score}-{home_score}",
                            "bets_settled": settled_count,
                        }
                    )

                    self.logger.info(
                        f"✅ Settled {settled_count} bets for {match.bet_id}: {away_score}-{home_score}"
                    )
                else:
                    settlement_report["failed_settlements"] += 1
                    settlement_report["details"].append(
                        {
                            "bet_id": match.bet_id,
                            "result": "no_pending_bets",
                            "reason": "No pending bets found for this game",
                        }
                    )

            except Exception as e:
                self.logger.error(f"❌ Error settling bet {match.bet_id}: {e}")
                settlement_report["failed_settlements"] += 1
                settlement_report["details"].append(
                    {"bet_id": match.bet_id, "result": "error", "error": str(e)}
                )

        return settlement_report

    def execute_robust_settlement(self) -> Dict[str, any]:
        """
        Esegue il processo completo di robust settlement.

        Returns:
            Report completo del settlement
        """
        self.logger.info("🚀 Starting Robust Bet Settlement Process...")

        # Step 1: Analyza scommesse pendenti
        self.logger.info("🔍 Analyzing pending bets and finding NBA games...")
        game_matches = self.analyze_pending_bets()

        if not game_matches:
            return {
                "success": True,
                "total_pending": 0,
                "settled_bets": 0,
                "message": "No pending bets found",
                "details": [],
            }

        # Step 2: Recupera punteggi finali
        self.logger.info("🏀 Retrieving final scores from NBA API...")
        game_matches_with_scores = self.get_final_scores(game_matches)

        # Step 3: Esegue settlement
        self.logger.info("💰 Settling bets with final scores...")
        settlement_report = self.settle_bets_with_scores(game_matches_with_scores)

        # Compose final report
        success_rate = (
            (
                settlement_report["successful_settlements"]
                / settlement_report["total_processed"]
            )
            * 100
            if settlement_report["total_processed"] > 0
            else 0
        )

        final_report = {
            "success": settlement_report["successful_settlements"] > 0,
            "total_pending": settlement_report["total_processed"],
            "settled_bets": settlement_report["successful_settlements"],
            "failed_settlements": settlement_report["failed_settlements"],
            "success_rate": success_rate,
            "message": f"Settled {settlement_report['successful_settlements']} of {settlement_report['total_processed']} pending bets ({success_rate:.1f}%)",
            "details": settlement_report["details"],
        }

        self.logger.info(
            f"🎉 Robust Settlement Complete: {final_report['settled_bets']}/{final_report['total_pending']} bets settled"
        )

        return final_report


def create_robust_settlement_system(betting_db_manager) -> RobustBetSettlement:
    """
    Factory function per creare il sistema robusto di settlement.

    Args:
        betting_db_manager: Istanza del BettingDatabaseManager

    Returns:
        Istanza di RobustBetSettlement
    """
    return RobustBetSettlement(betting_db_manager)
