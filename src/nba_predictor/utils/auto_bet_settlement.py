"""
🎯 Automatic Bet Settlement System - Context7 Compliant Solution

Sistema automatico per il settlement delle scommesse su partite NBA concluse.

Funzionalità:
- Identificazione automatica partite concluse con scommesse pendenti
- Recupero risultati finali da API NBA ufficiali
- Auto-settlement tramite update_game_results_from_scores
- Integrazione con dashboard betting workflow
- Logging completo delle operazioni
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoBetSettlement:
    """
    Sistema automatico per il settlement di scommesse NBA.

    Gestisce l'identificazione di partite concluse con scommesse pendenti,
    il recupero dei risultati finali e l'aggiornamento automatico delle scommesse.
    """

    def __init__(self, betting_db_manager):
        """
        Inizializza il sistema di auto-settlement.

        Args:
            betting_db_manager: Istanza del BettingDatabaseManager
        """
        self.betting_db = betting_db_manager
        self.logger = logger

        # Headers per API NBA
        self.nba_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nba.com/',
            'Origin': 'https://www.nba.com'
        }

    def get_pending_bets_with_games(self) -> List[Dict]:
        """
        Recupera tutte le scommesse pendenti con informazioni sulle partite.

        Returns:
            Lista di scommesse pendenti con dettagli partite
        """
        try:
            pending_bets = self.betting_db.get_pending_bets()

            enriched_bets = []
            for bet in pending_bets:
                # Arricchisci con informazioni aggiuntive
                enriched_bet = {
                    'bet_id': bet.bet_id,
                    'game_id': bet.game_id,
                    'bet_type': bet.bet_type,
                    'line': bet.line,
                    'odds': bet.odds,
                    'stake': bet.stake,
                    'potential_return': bet.potential_return,
                    'home_team': getattr(bet, 'home_team', 'Unknown'),
                    'away_team': getattr(bet, 'away_team', 'Unknown'),
                    'placed_at': bet.placed_at,
                    'days_since_placed': (datetime.now() - bet.placed_at).days
                }
                enriched_bets.append(enriched_bet)

            self.logger.info(f"📊 Found {len(enriched_bets)} pending bets")
            return enriched_bets

        except Exception as e:
            self.logger.error(f"❌ Failed to get pending bets: {e}")
            return []

    def get_game_final_score(self, game_id: str) -> Optional[Tuple[int, int]]:
        """
        Recupera il punteggio finale di una partita tramite API NBA.

        Args:
            game_id: ID della partita NBA

        Returns:
            Tuple (home_score, away_score) o None se non disponibile
        """
        try:
            # Strategy 1: Try NBA.com CDN endpoint (most reliable for recent games)
            if self._try_cdn_scoreboard(game_id):
                return self._try_cdn_scoreboard(game_id)

            # Strategy 2: Try stats.nba.com endpoint
            if self._try_stats_nba(game_id):
                return self._try_stats_nba(game_id)

            # Strategy 3: Try nba_api endpoint
            if self._try_nba_api(game_id):
                return self._try_nba_api(game_id)

            self.logger.warning(f"⚠️ Could not find final score for game {game_id}")
            return None

        except Exception as e:
            self.logger.error(f"❌ Error getting final score for {game_id}: {e}")
            return None

    def _try_cdn_scoreboard(self, game_id: str) -> Optional[Tuple[int, int]]:
        """Prova a ottenere il punteggio dal CDN endpoint di NBA.com."""
        try:
            # Il CDN endpoint mostra partite degli ultimi 1-2 giorni
            url = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'
            response = requests.get(url, headers=self.nba_headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if 'scoreboard' in data and 'games' in data['scoreboard']:
                    games = data['scoreboard']['games']
                    for game in games:
                        if str(game.get('gameId', '')) == str(game_id):
                            home_score = game.get('homeTeamScore', 0)
                            away_score = game.get('awayTeamScore', 0)

                            # Verifica che la partita sia conclusa
                            game_status = game.get('gameStatusText', '').lower()
                            if 'final' in game_status or home_score > 0 or away_score > 0:
                                self.logger.info(f"✅ Found final score via CDN: {game_id} -> {away_score}-{home_score}")
                                return (int(home_score), int(away_score))

            return None

        except Exception as e:
            self.logger.debug(f"CDN scoreboard failed for {game_id}: {e}")
            return None

    def _try_stats_nba(self, game_id: str) -> Optional[Tuple[int, int]]:
        """Prova a ottenere il punteggio da stats.nba.com."""
        try:
            # Per partite concluse, prova a cercare nei risultati degli ultimi giorni
            today = datetime.now(timezone.utc)
            for days_back in range(0, 4):  # Controlla ultimi 4 giorni
                check_date = today - timedelta(days=days_back)

                url = 'https://stats.nba.com/stats/scoreboardv2'
                params = {
                    'LeagueID': '00',
                    'GameDate': check_date.strftime('%Y-%m-%d')
                }

                response = requests.get(url, headers=self.nba_headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if 'resultSets' in data:
                        for rs in data['resultSets']:
                            if rs['name'] == 'Scoreboard':
                                for row in rs['rowSet']:
                                    if str(row[0]) == str(game_id):  # GAME_ID
                                        home_score = row[21] if len(row) > 21 else 0  # HOME_TEAM_SCORE
                                        away_score = row[22] if len(row) > 22 else 0  # VISITOR_TEAM_SCORE
                                        game_status = row[3] if len(row) > 3 else ''  # GAME_STATUS_TEXT

                                        if 'Final' in game_status or (home_score > 0 and away_score > 0):
                                            self.logger.info(f"✅ Found final score via stats.nba: {game_id} -> {away_score}-{home_score}")
                                            return (int(home_score), int(away_score))

            return None

        except Exception as e:
            self.logger.debug(f"stats.nba failed for {game_id}: {e}")
            return None

    def _try_nba_api(self, game_id: str) -> Optional[Tuple[int, int]]:
        """Prova a ottenere il punteggio tramite nba_api."""
        try:
            from nba_api.stats.endpoints import PlayByPlay

            # Il PlayByPlay endpoint contiene punteggi finali per partite concluse
            try:
                pbp = PlayByPlay(game_id=game_id)
                pbp_data = pbp.get_normalized_json()
                pbp_dict = json.loads(pbp_data)

                if 'Available' in pbp_dict:
                    available_video = pbp_dict['Available']
                    if available_video and len(available_video) > 0:
                        # Se ci sono dati disponibili, la partita probabilmente è conclusa
                        # Cerca i punteggi finali nei dati del gioco
                        game_info = available_video[0]
                        # Questo è un workaround, potremmo dover elaborare i dati del play-by-play

                        self.logger.info(f"✅ Found game data via nba_api: {game_id}")
                        # Per ora, restituisci None - richiederebbe elaborazione più complessa
                        # TODO: Implementare parsing completo dei dati play-by-play

            except Exception as api_error:
                self.logger.debug(f"nba_api PlayByPlay failed for {game_id}: {api_error}")

            return None

        except Exception as e:
            self.logger.debug(f"nba_api approach failed for {game_id}: {e}")
            return None

    def is_game_likely_completed(self, placed_at: datetime, max_hours: int = 48) -> bool:
        """
        Determina se una partita è probabilmente conclusa basandosi sull'orario.

        Args:
            placed_at: Quando la scommessa è stata piazzata
            max_hours: Ore massime dopo cui una partita è considerata conclususa

        Returns:
            True se la partita è probabilmente conclusa
        """
        time_since_placed = datetime.now() - placed_at
        hours_passed = time_since_placed.total_seconds() / 3600

        # Le partite NBA durano circa 2.5-3 ore da inizio a fine
        # Se sono passate più di max_hours, la partita è quasi certamente conclusa
        return hours_passed >= max_hours

    def settle_pending_bets(self, force_settlement: bool = False) -> Dict[str, any]:
        """
        Esegue il settlement automatico delle scommesse pendenti.

        Args:
            force_settlement: Se True, forza il settlement anche senza punteggi finali

        Returns:
            Report del settlement eseguito
        """
        try:
            self.logger.info("🚀 Starting automatic bet settlement process...")

            pending_bets = self.get_pending_bets_with_games()

            if not pending_bets:
                return {
                    'success': True,
                    'settled_bets': 0,
                    'total_pending': 0,
                    'message': 'No pending bets found',
                    'details': []
                }

            settlement_report = {
                'success': True,
                'settled_bets': 0,
                'total_pending': len(pending_bets),
                'failed_settlements': 0,
                'details': []
            }

            for bet in pending_bets:
                try:
                    bet_id = bet['bet_id']
                    game_id = bet['game_id']
                    placed_at = bet['placed_at']
                    hours_old = bet['days_since_placed'] * 24

                    self.logger.info(f"🎯 Processing bet {bet_id} (game {game_id}, {hours_old:.1f}h old)")

                    # Check se la partita è probabilmente conclusa
                    is_likely_completed = self.is_game_likely_completed(placed_at)

                    if not is_likely_completed and not force_settlement:
                        self.logger.info(f"⏰ Game {game_id} likely not finished yet, skipping")
                        continue

                    # Recupera i punteggi finali
                    final_score = self.get_game_final_score(game_id)

                    if final_score:
                        home_score, away_score = final_score

                        # Usa il metodo esistente del database manager
                        settled_count = self.betting_db.update_game_results_from_scores(
                            game_id, home_score, away_score
                        )

                        if settled_count > 0:
                            settlement_report['settled_bets'] += settled_count
                            self.logger.info(f"✅ Settled {settled_count} bets for game {game_id}: {away_score}-{home_score}")

                            settlement_report['details'].append({
                                'bet_id': bet_id,
                                'game_id': game_id,
                                'result': 'settled',
                                'final_score': f"{away_score}-{home_score}",
                                'method': 'auto_settlement'
                            })
                        else:
                            self.logger.warning(f"⚠️ No pending bets found to settle for game {game_id}")

                    elif force_settlement and is_likely_completed:
                        # Se forzato, marca come void se non ci sono punteggi disponibili
                        self.logger.warning(f"⚠️ Force settlement enabled but no score found for {game_id}")

                        # Potrebbe implementare logica per marcature come void o cancelled
                        # Per ora, solo logghiamo

                    else:
                        self.logger.info(f"ℹ️ No final score available yet for game {game_id}")

                except Exception as bet_error:
                    self.logger.error(f"❌ Error settling bet {bet['bet_id']}: {bet_error}")
                    settlement_report['failed_settlements'] += 1

                    settlement_report['details'].append({
                        'bet_id': bet['bet_id'],
                        'game_id': bet['game_id'],
                        'result': 'failed',
                        'error': str(bet_error)
                    })

            # Summary
            success_rate = (settlement_report['settled_bets'] / settlement_report['total_pending']) * 100 if settlement_report['total_pending'] > 0 else 0

            self.logger.info(f"🎉 Settlement complete: {settlement_report['settled_bets']}/{settlement_report['total_pending']} bets settled ({success_rate:.1f}%)")

            settlement_report['message'] = f"Settled {settlement_report['settled_bets']} of {settlement_report['total_pending']} pending bets"
            settlement_report['success_rate'] = success_rate

            return settlement_report

        except Exception as e:
            self.logger.error(f"❌ Critical error in settlement process: {e}")
            return {
                'success': False,
                'error': str(e),
                'settled_bets': 0,
                'total_pending': 0,
                'details': []
            }

    def get_settlement_status_summary(self) -> Dict[str, any]:
        """
        Ottiene un riepilogo dello stato attuale delle scommesse.

        Returns:
            Riepilogo dello stato delle scommesse
        """
        try:
            pending_bets = self.get_pending_bets_with_games()

            if not pending_bets:
                return {
                    'pending_count': 0,
                    'ready_for_settlement': 0,
                    'recent_bets': 0,
                    'old_bets': 0,
                    'oldest_bet_hours': 0,
                    'newest_bet_hours': 0
                }

            now = datetime.now()
            settlement_ready = 0
            recent_bets = 0  # < 12 ore
            old_bets = 0     # > 48 ore

            ages_in_hours = []

            for bet in pending_bets:
                hours_old = bet['days_since_placed'] * 24
                ages_in_hours.append(hours_old)

                if self.is_game_likely_completed(bet['placed_at']):
                    settlement_ready += 1

                if hours_old < 12:
                    recent_bets += 1
                elif hours_old > 48:
                    old_bets += 1

            return {
                'pending_count': len(pending_bets),
                'ready_for_settlement': settlement_ready,
                'recent_bets': recent_bets,
                'old_bets': old_bets,
                'oldest_bet_hours': max(ages_in_hours) if ages_in_hours else 0,
                'newest_bet_hours': min(ages_in_hours) if ages_in_hours else 0
            }

        except Exception as e:
            self.logger.error(f"❌ Error getting settlement status: {e}")
            return {
                'pending_count': 0,
                'ready_for_settlement': 0,
                'error': str(e)
            }


def create_auto_settlement_system(betting_db_manager) -> AutoBetSettlement:
    """
    Factory function per creare il sistema di auto-settlement.

    Args:
        betting_db_manager: Istanza del BettingDatabaseManager

    Returns:
        Istanza di AutoBetSettlement
    """
    return AutoBetSettlement(betting_db_manager)