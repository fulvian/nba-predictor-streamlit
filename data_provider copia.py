"""
Modulo per la gestione dei dati NBA usando nba-api.
Versione 2.0 - Completamente basato su nba-api invece di api-sports
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, date, timedelta
import time
from typing import Dict, List, Optional, Any

# Importazioni da nba_api
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import (
    teamdashboardbygeneralsplits,
    teamgamelog,
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    boxscoresummaryv2,
    commonteamroster,
    playergamelog,
    playerestimatedmetrics,
    teamestimatedmetrics,
    leaguestandings,
    scoreboardv2,
    teamyearbyyearstats,
    leaguedashteamstats,
    teamplayerdashboard,
    playercareerstats,
    commonplayerinfo,
    leaguegamelog
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

# Per The Odds API
import requests
from dotenv import load_dotenv

# Importazioni locali
from player_impact_analyzer import PlayerImpactAnalyzer
from injury_reporter import InjuryReporter

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione
NBA_API_REQUEST_DELAY = 0.6  # Secondi tra le richieste all'API NBA
ODDS_API_REQUEST_DELAY = 1.0  # Secondi tra le richieste all'API Odds

# Directory di base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_BASE_DIR = os.path.join(BASE_DIR, 'models')
SETTINGS_FILE = os.path.join(BASE_DIR, 'settings.json')

# Chiavi API
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# Crea le directory se non esistono
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

class NBADataProvider:
    """
    Classe per il recupero e la gestione dei dati NBA usando nba-api.
    """
    
    def __init__(self):
        """Inizializza il provider di dati NBA con le cache necessarie."""
        self.team_cache = {}
        self.team_data_cache = {}
        self.game_results_cache = {}
        self.player_stats_cache = {}
        self.h2h_cache = {}
        
        # Ottieni informazioni statiche su squadre e giocatori
        self.nba_teams_info = nba_teams.get_teams()
        self.nba_players_info = nba_players.get_players()
        
        # Crea mapping per accesso rapido
        self.team_id_to_info = {team['id']: team for team in self.nba_teams_info}
        self.team_name_to_info = {team['full_name']: team for team in self.nba_teams_info}
        self.team_abbreviation_to_info = {team['abbreviation']: team for team in self.nba_teams_info}
        
        # Inizializza gli analizzatori
        self.player_impact_analyzer = PlayerImpactAnalyzer()
        self.injury_reporter = InjuryReporter(self)
        
        # Headers personalizzati per nba-api
        self.headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Connection': 'keep-alive',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com'
        }
        
        print("✅ NBADataProvider inizializzato con nba-api")
        print(f"   📊 Caricate {len(self.nba_teams_info)} squadre NBA")
        print(f"   👥 Caricati {len(self.nba_players_info)} giocatori NBA")
    
    def _get_season_str_for_nba_api(self, for_date: date) -> str:
        """
        Restituisce la stringa della stagione nel formato richiesto dall'API NBA (es. "2024-25").
        
        Args:
            for_date (date): Data per cui determinare la stagione
            
        Returns:
            str: Stringa della stagione nel formato "YYYY-YY"
        """
        year = for_date.year
        month = for_date.month
        
        # La stagione NBA inizia a ottobre e termina a giugno dell'anno successivo
        if month >= 10:  # Da ottobre a dicembre
            return f"{year}-{str(year + 1)[-2:]}"
        else:  # Da gennaio a settembre
            return f"{year - 1}-{str(year)[-2:]}"
    
    def _find_team_by_name(self, team_name: str) -> Optional[Dict]:
        """
        Trova una squadra NBA per nome (supporta match parziali).
        
        Args:
            team_name (str): Nome della squadra
            
        Returns:
            Optional[Dict]: Informazioni sulla squadra o None
        """
        # Prima prova match esatto
        if team_name in self.team_name_to_info:
            return self.team_name_to_info[team_name]
        
        # Prova con abbreviazione
        if team_name.upper() in self.team_abbreviation_to_info:
            return self.team_abbreviation_to_info[team_name.upper()]
        
        # Prova match parziale
        team_name_lower = team_name.lower()
        for team in self.nba_teams_info:
            if (team_name_lower in team['full_name'].lower() or 
                team['nickname'].lower() in team_name_lower or
                team_name_lower in team['nickname'].lower()):
                return team
        
        return None
    
    def get_scheduled_games_from_odds_api(self, days_ahead=3):
        """
        Recupera le partite in programma per i prossimi N giorni da The Odds API.
        (Mantiene la compatibilità con il sistema esistente)
        """
        if not ODDS_API_KEY:
            print("⚠️ ODDS_API_KEY non configurata. Impossibile recuperare partite da The Odds API.")
            return []
            
        scheduled_games = []
        base_date = date.today()
        end_date = base_date + timedelta(days=days_ahead - 1)
        
        print(f"🔍 Cercando partite NBA su The Odds API dal {base_date.strftime('%Y-%m-%d')} al {end_date.strftime('%Y-%m-%d')}...")
        
        try:
            # Parametri per la richiesta
            params = {
                'apiKey': ODDS_API_KEY,
                'sport': 'basketball_nba',
                'regions': 'us,eu,uk,au',
                'markets': 'h2h,totals',
                'dateFormat': 'iso',
                'oddsFormat': 'decimal',
                'bookmakers': 'fanduel,draftkings,williamhill,betfair'
            }
            
            response = requests.get(
                'https://api.the-odds-api.com/v4/sports/basketball_nba/odds',
                params=params
            )
            
            if response.status_code == 200:
                games = response.json()
                print(f"✅ Trovate {len(games)} partite su The Odds API")
                
                for game in games:
                    try:
                        # Trova gli ID delle squadre NBA
                        home_team_info = self._find_team_by_name(game.get('home_team', ''))
                        away_team_info = self._find_team_by_name(game.get('away_team', ''))
                        
                        if not home_team_info or not away_team_info:
                            print(f"⚠️ Squadre non trovate: {game.get('home_team')} vs {game.get('away_team')}")
                            continue
                        
                        # Estrai le quote over/under
                        over_under = {}
                        odds = []
                        
                        for bookmaker in game.get('bookmakers', []):
                            for market in bookmaker.get('markets', []):
                                if market.get('key') == 'totals':
                                    for outcome in market.get('outcomes', []):
                                        if outcome.get('name') == 'Over':
                                            over_under['over'] = outcome.get('point', 0)
                                            over_under['over_odds'] = outcome.get('price', 0)
                                        elif outcome.get('name') == 'Under':
                                            over_under['under'] = outcome.get('point', 0)
                                            over_under['under_odds'] = outcome.get('price', 0)
                        
                        if over_under.get('over') and over_under.get('over_odds') and over_under.get('under_odds'):
                            odds.append({
                                'line': over_under['over'],
                                'over_quote': over_under['over_odds'],
                                'under_quote': over_under['under_odds']
                            })
                        
                        game_info = {
                            'date': game.get('commence_time', '').split('T')[0],
                            'time': game.get('commence_time', '').split('T')[1].split('.')[0],
                            'home_team': home_team_info['full_name'],
                            'away_team': away_team_info['full_name'],
                            'home_team_id': home_team_info['id'],
                            'away_team_id': away_team_info['id'],
                            'odds': odds,
                            'api_id': game.get('id'),
                            'team_stats': {
                                'home': {
                                    'name': home_team_info['full_name'],
                                    'id': home_team_info['id'],
                                    'team_id_nba': home_team_info['id'],
                                    'abbreviation': home_team_info['abbreviation'],
                                    'city': home_team_info['city'],
                                    'full_name': home_team_info['full_name']
                                },
                                'away': {
                                    'name': away_team_info['full_name'],
                                    'id': away_team_info['id'],
                                    'team_id_nba': away_team_info['id'],
                                    'abbreviation': away_team_info['abbreviation'],
                                    'city': away_team_info['city'],
                                    'full_name': away_team_info['full_name']
                                }
                            }
                        }
                        
                        scheduled_games.append(game_info)
                        
                    except Exception as e:
                        print(f"   ⚠️ Errore nell'elaborazione di una partita: {e}")
                        continue
                
                return scheduled_games
                
            else:
                print(f"❌ Errore nella richiesta a The Odds API: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Errore durante il recupero delle partite da The Odds API: {e}")
            return []

    def get_scheduled_games(self, days_ahead=3):
        """
        Recupera le partite in programma usando nba-api.
        """
        # Prima prova con The Odds API per avere anche le quote
        odds_games = self.get_scheduled_games_from_odds_api(days_ahead)
        if odds_games:
            return odds_games
        
        # Altrimenti usa nba-api
        print("⚠️ Nessuna partita trovata su The Odds API. Uso nba-api...")
        
        scheduled_games = []
        base_date = date.today()
        
        try:
            # Per ogni giorno richiesto
            for days_offset in range(days_ahead):
                current_date = base_date + timedelta(days=days_offset)
                date_str = current_date.strftime('%Y-%m-%d')
                
                print(f"📅 Cercando partite per il {date_str}...")
                
                # Usa ScoreboardV2 per ottenere le partite del giorno
                try:
                    time.sleep(NBA_API_REQUEST_DELAY)
                    scoreboard = scoreboardv2.ScoreboardV2(
                        game_date=date_str,
                        league_id='00',
                        headers=self.headers
                    )
                    
                    games = scoreboard.game_header.get_data_frame()
                    
                    if not games.empty:
                        for _, game in games.iterrows():
                            # Trova informazioni squadre
                            home_team_info = self.team_id_to_info.get(game['HOME_TEAM_ID'])
                            away_team_info = self.team_id_to_info.get(game['VISITOR_TEAM_ID'])
                            
                            if home_team_info and away_team_info:
                                game_info = {
                                    'date': date_str,
                                    'time': game.get('GAME_STATUS_TEXT', ''),
                                    'home_team': home_team_info['full_name'],
                                    'away_team': away_team_info['full_name'],
                                    'home_team_id': home_team_info['id'],
                                    'away_team_id': away_team_info['id'],
                                    'game_id': game['GAME_ID'],
                                    'game_sequence': game.get('GAME_SEQUENCE', 0),
                                    'game_status_id': game.get('GAME_STATUS_ID', 1),
                                    'odds': [],  # nba-api non fornisce quote
                                    'team_stats': {
                                        'home': {
                                            'name': home_team_info['full_name'],
                                            'id': home_team_info['id'],
                                            'team_id_nba': home_team_info['id'],
                                            'abbreviation': home_team_info['abbreviation'],
                                            'city': home_team_info['city'],
                                            'full_name': home_team_info['full_name']
                                        },
                                        'away': {
                                            'name': away_team_info['full_name'],
                                            'id': away_team_info['id'],
                                            'team_id_nba': away_team_info['id'],
                                            'abbreviation': away_team_info['abbreviation'],
                                            'city': away_team_info['city'],
                                            'full_name': away_team_info['full_name']
                                        }
                                    }
                                }
                                
                                scheduled_games.append(game_info)
                                print(f"   ✅ {away_team_info['full_name']} @ {home_team_info['full_name']}")
                
                except Exception as e:
                    print(f"   ⚠️ Errore recupero partite per {date_str}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ Errore generale nel recupero partite: {e}")
        
        return scheduled_games
    
    def get_injury_adjusted_data_for_game(self, game_details_input, manual_overrides=None):
        """
        Recupera i dati di una partita con le informazioni sugli infortuni.
        """
        if not game_details_input:
            return None
            
        # Crea una copia profonda per non modificare l'originale
        import copy
        from datetime import datetime
        game_details = copy.deepcopy(game_details_input)
        
        # Assicurati che le quote siano nel formato corretto
        saved_odds = game_details.get('odds', []).copy() if 'odds' in game_details else []
        
        # Chiama il metodo base per ottenere i dati della partita
        game_data = self.get_data_for_game(game_details, manual_overrides)
        
        if not game_data:
            return None
            
        # Ripristina le quote
        if saved_odds:
            game_data['odds'] = saved_odds
        
        # Controlla se la partita è già completata
        if game_data.get('game_info', {}).get('status') in ['COMPLETED', 'PENDING_RESULT']:
            return game_data
        
        try:
            # Salva una copia dei dati originali della partita
            original_game_data = copy.deepcopy(game_data)
            
            # Usa l'istanza di InjuryReporter per ottenere i dati sugli infortuni
            injury_data = self.injury_reporter.get_injury_adjusted_data(game_details_input, game_data)

            # Se abbiamo dati sugli infortuni, uniscili mantenendo le informazioni originali
            if injury_data:
                # Assicurati che le chiavi importanti non vengano sovrascritte
                for key in ['game_info', 'team_stats', 'odds', 'settings']:
                    # Se la chiave esiste in original_game_data e non in injury_data,
                    # o se il valore in injury_data è vuoto/None, usa quello da original_game_data
                    if key in original_game_data and (key not in injury_data or not injury_data[key]):
                        injury_data[key] = original_game_data[key]
                    # Se la chiave esiste in entrambi, ma original_game_data[key] è più completo
                    # (es. un dizionario non vuoto vs. uno vuoto), preferisci original_game_data.
                    # Questa logica potrebbe necessitare di affinamento a seconda dei casi specifici.
                    if key in original_game_data and key in injury_data:
                        injury_data[key] = original_game_data[key]
                
                # Aggiorna i dati del gioco con le informazioni sugli infortuni
                game_data.update(injury_data)

            # Assicurati che injury_reports esista e abbia la struttura base
            if 'injury_reports' not in game_data:
                game_data['injury_reports'] = {}
            if 'home_nba' not in game_data['injury_reports']:
                game_data['injury_reports']['home_nba'] = {'players': []}
            if 'away_nba' not in game_data['injury_reports']:
                game_data['injury_reports']['away_nba'] = {'players': []}
            if 'impact_analysis' not in game_data['injury_reports']:
                game_data['injury_reports']['impact_analysis'] = {}

            # Popola game_data['injury_reports'][team_key]['players'] con i roster da injury_data
            # Questo ora viene gestito da InjuryReporter che restituisce i roster in 'roster_data'
            # e get_injury_adjusted_data dovrebbe popolare 'home_nba' e 'away_nba'
            # all'interno di game_data['injury_reports']
            # La logica qui sotto è stata spostata/integrata in InjuryReporter e nel flusso di _calculate_advanced_injury_impact

            if 'injury_reports' in game_data and 'game_info' in game_data and 'team_stats' in game_data:
                injury_home_nba = game_data['injury_reports'].get('home_nba', {'players': []})
                injury_away_nba = game_data['injury_reports'].get('away_nba', {'players': []})
                
                home_team_name = game_data['game_info'].get('home_team')
                away_team_name = game_data['game_info'].get('away_team')
                
                if not home_team_name or not away_team_name:
                    print("⚠️ Nomi delle squadre mancanti in game_info")
                    return game_data
                
                team_stats_home = game_data['team_stats'].get('home', {})
                team_stats_away = game_data['team_stats'].get('away', {})
                
                # Calcola l'impatto avanzato
                injury_impact = self._calculate_advanced_injury_impact(
                    injury_home_nba, 
                    injury_away_nba,
                    team_stats_home,
                    team_stats_away,
                    datetime.strptime(game_data['game_info']['date'], '%Y-%m-%d').date(),
                    home_team_name,
                    away_team_name
                )
                
                # Aggiorna i dati con l'analisi di impatto
                game_data['injury_reports']['impact_analysis'] = injury_impact
                
                # Imposta l'impatto combinato
                if injury_impact and 'home_team_impact' in injury_impact:
                    game_data['home_team_combined_impact'] = injury_impact['home_team_impact']
                else:
                    game_data['home_team_combined_impact'] = 0.0
            
            return game_data
            
        except Exception as e:
            print(f"⚠️ Errore durante il recupero del report infortuni: {e}")
            # In caso di errore, restituisci i dati originali senza informazioni sugli infortuni
            game_data['injury_reports'] = {
                'home_nba': {'players': []}, 
                'away_nba': {'players': []},
                'impact_analysis': {}
            }
            game_data['home_team_combined_impact'] = 0.0
            return game_data
    
    def _get_team_stats(self, team_name, is_home=True):
        """
        Recupera le statistiche di una squadra NBA usando nba-api.
        """
        # Inizializza le statistiche di default
        stats = {
            'team_name': team_name,
            'is_home': is_home,
            'games_played': 0,
            'points_per_game': 0,
            'points_against_per_game': 0,
            'offensive_rating': 0,
            'defensive_rating': 0,
            'pace': 0,
            'win_percentage': 0,
            'last_10_games': {'wins': 0, 'losses': 0},
            'has_data': False
        }
        
        try:
            # Genera la chiave per la cache
            cache_key = f"{team_name}_{date.today().strftime('%Y-%m-%d')}"
            
            # Controlla se i dati sono già in cache
            if cache_key in self.team_data_cache:
                print(f"📦 Statistiche per {team_name} recuperate dalla cache")
                return self.team_data_cache[cache_key]
            
            # Trova l'ID della squadra
            team_info = self._find_team_by_name(team_name)
            if not team_info:
                print(f"⚠️ Squadra non trovata: {team_name}")
                return stats
            
            team_id = team_info['id']
            current_season = self._get_season_str_for_nba_api(date.today())
            
            print(f"🔍 Richiesta statistiche NBA per {team_name} (ID: {team_id}) - Stagione {current_season}")
            
            # Ottieni dashboard della squadra con statistiche avanzate
            time.sleep(NBA_API_REQUEST_DELAY)
            
            try:
                # Prova prima con la stagione corrente
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                    team_id=team_id,
                    season=current_season,
                    season_type_all_star='Regular Season',
                    measure_type_detailed_defense='Advanced',
                    headers=self.headers
                )
                overall_stats = dashboard.overall_team_dashboard.get_data_frame()
                
                # Se vuoto, prova stagione precedente
                if overall_stats.empty:
                    prev_season = f"{int(current_season[:4])-1}-{str(int(current_season[:4]))[-2:]}"
                    print(f"   ⚠️ Nessun dato per {current_season}, provo con {prev_season}")
                    
                    time.sleep(NBA_API_REQUEST_DELAY)
                    dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                        team_id=team_id,
                        season=prev_season,
                        season_type_all_star='Regular Season',
                        measure_type_detailed_defense='Advanced',
                        headers=self.headers
                    )
                    overall_stats = dashboard.overall_team_dashboard.get_data_frame()
                
                if not overall_stats.empty:
                    row = overall_stats.iloc[0]
                    
                    # Estrai statistiche avanzate
                    stats.update({
                        'games_played': int(row.get('GP', 0)),
                        'games': int(row.get('GP', 0)),
                        'gp': int(row.get('GP', 0)),
                        'wins': int(row.get('W', 0)),
                        'losses': int(row.get('L', 0)),
                        'win_percentage': float(row.get('W_PCT', 0)),
                        'win_rate': float(row.get('W_PCT', 0)),
                        'win_pct': float(row.get('W_PCT', 0)),
                        'points_per_game': float(row.get('PTS', 0)) / max(int(row.get('GP', 1)), 1),
                        'points': float(row.get('PTS', 0)),
                        'pts': float(row.get('PTS', 0)),
                        'offensive_rating': float(row.get('OFF_RATING', 110)),
                        'off_rating': float(row.get('OFF_RATING', 110)),
                        'off_eff': float(row.get('OFF_RATING', 110)),
                        'defensive_rating': float(row.get('DEF_RATING', 110)),
                        'def_rating': float(row.get('DEF_RATING', 110)),
                        'def_eff': float(row.get('DEF_RATING', 110)),
                        'net_rating': float(row.get('NET_RATING', 0)),
                        'pace': float(row.get('PACE', 100)),
                        'pace_rating': float(row.get('PACE', 100)),
                        'pie': float(row.get('PIE', 0.5)),
                        'fg_pct': float(row.get('FG_PCT', 0)),
                        'fg3_pct': float(row.get('FG3_PCT', 0)),
                        'ft_pct': float(row.get('FT_PCT', 0)),
                        'efg_pct': float(row.get('EFG_PCT', 0)),
                        'ts_pct': float(row.get('TS_PCT', 0)),
                        'rebounds': float(row.get('REB', 0)),
                        'assists': float(row.get('AST', 0)),
                        'steals': float(row.get('STL', 0)),
                        'blocks': float(row.get('BLK', 0)),
                        'turnovers': float(row.get('TOV', 0)),
                        'pf': float(row.get('PF', 0)),
                        'has_data': True
                    })
                    
                    # Calcola punti subiti per partita (stima)
                    if 'points_per_game' in stats and stats['games_played'] > 0:
                        # Usa il defensive rating per stimare i punti subiti
                        stats['points_against_per_game'] = (stats['defensive_rating'] * stats['pace']) / 100
                        stats['points_against'] = stats['points_against_per_game'] * stats['games_played']
                        stats['opp_pts'] = stats['points_against']
                        stats['opp_points'] = stats['points_against']
                    
                    # Ottieni statistiche ultime 10 partite
                    try:
                        time.sleep(NBA_API_REQUEST_DELAY)
                        gamelog = teamgamelog.TeamGameLog(
                            team_id=team_id,
                            season=current_season,
                            season_type_all_star='Regular Season',
                            headers=self.headers
                        )
                        recent_games = gamelog.get_data_frames()[0].head(10)
                        
                        if not recent_games.empty:
                            wins_last_10 = len(recent_games[recent_games['WL'] == 'W'])
                            losses_last_10 = len(recent_games[recent_games['WL'] == 'L'])
                            
                            stats['last_10_games'] = {'wins': wins_last_10, 'losses': losses_last_10}
                            stats['last10'] = {'wins': wins_last_10, 'losses': losses_last_10}
                            stats['last_10'] = {'wins': wins_last_10, 'losses': losses_last_10}
                    except:
                        # Stima basata su win percentage
                        wins_est = round(10 * stats['win_percentage'])
                        stats['last_10_games'] = {'wins': wins_est, 'losses': 10 - wins_est}
                    
                    print(f"✅ Statistiche NBA complete recuperate per {team_name}")
                else:
                    print(f"⚠️ Nessuna statistica disponibile per {team_name}")
                    
            except Exception as e:
                print(f"⚠️ Errore NBA API per {team_name}: {str(e)}")
            
            # Salva in cache
            self.team_data_cache[cache_key] = stats
            return stats
                
        except Exception as e:
            print(f"❌ Errore critico nel recupero statistiche per {team_name}: {e}")
            import traceback
            traceback.print_exc()
            return stats

    # NUOVO METODO DA AGGIUNGERE A data_provider.py
    def get_team_stats_for_game(self, home_team_name: str, away_team_name: str) -> Optional[Dict]:
        """
        Recupera le statistiche per entrambe le squadre di una partita.

        Args:
            home_team_name (str): Nome della squadra di casa.
            away_team_name (str): Nome della squadra ospite.

        Returns:
            Optional[Dict]: Un dizionario contenente le statistiche per 'home' e 'away'.
        """
        print(f"🔍 Recupero statistiche per la partita: {away_team_name} @ {home_team_name}")
        home_stats = self._get_team_stats(home_team_name, is_home=True)
        away_stats = self._get_team_stats(away_team_name, is_home=False)

        if home_stats and home_stats.get('has_data') and away_stats and away_stats.get('has_data'):
            return {
                'home': home_stats,
                'away': away_stats
            }
        else:
            print("❌ Impossibile recuperare le statistiche per una o entrambe le squadre.")
            return None
    
    def get_data_for_game(self, game_details_input, manual_overrides=None):
        """
        Recupera i dati di base per una partita NBA.
        """
        try:
            # Inizializza la struttura dati di ritorno
            game_data = {
                'game_info': {},
                'team_stats': {
                    'home': {},
                    'away': {}
                },
                'player_stats': {
                    'home': [],
                    'away': []
                },
                'injury_reports': {
                    'home_nba': {'players': []},
                    'away_nba': {'players': []},
                    'impact_analysis': {}
                },
                'predictions': {}
            }
            
            # Copia i dettagli della partita
            if isinstance(game_details_input, dict):
                game_data['game_info'].update(game_details_input)
            
            # Applica gli override manuali
            if manual_overrides:
                game_data['game_info'].update(manual_overrides)
            
            # Imposta lo stato di default se non specificato
            if 'status' not in game_data['game_info']:
                game_data['game_info']['status'] = 'SCHEDULED'
            
            # Recupera le statistiche delle squadre
            if 'home_team' in game_data['game_info'] and 'away_team' in game_data['game_info']:
                game_data['team_stats']['home'] = self._get_team_stats(
                    game_data['game_info']['home_team'], 
                    is_home=True
                )
                game_data['team_stats']['away'] = self._get_team_stats(
                    game_data['game_info']['away_team'], 
                    is_home=False
                )
                
                # Se abbiamo un game_id, prova a recuperare statistiche più dettagliate
                if 'game_id' in game_data['game_info']:
                    self._enrich_with_game_specific_data(game_data)
            
            return game_data
            
        except Exception as e:
            print(f"⚠️ Errore nel recupero dei dati della partita: {e}")
            return None
    
    def _enrich_with_game_specific_data(self, game_data):
        """
        Arricchisce i dati della partita con informazioni specifiche se disponibili.
        """
        try:
            game_id = game_data['game_info']['game_id']
            
            # Prova a ottenere il boxscore se la partita è iniziata/completata
            try:
                time.sleep(NBA_API_REQUEST_DELAY)
                boxscore = boxscoresummaryv2.BoxScoreSummaryV2(
                    game_id=game_id,
                    headers=self.headers
                )
                
                game_summary = boxscore.game_summary.get_data_frame()
                if not game_summary.empty:
                    summary = game_summary.iloc[0]
                    
                    # Aggiorna stato partita
                    if summary.get('GAME_STATUS_ID') == 3:
                        game_data['game_info']['status'] = 'COMPLETED'
                        
                        # Aggiungi risultato finale
                        line_score = boxscore.line_score.get_data_frame()
                        if not line_score.empty:
                            home_score = line_score[line_score['TEAM_ID'] == game_data['game_info']['home_team_id']]['PTS'].iloc[0]
                            away_score = line_score[line_score['TEAM_ID'] == game_data['game_info']['away_team_id']]['PTS'].iloc[0]
                            
                            game_data['game_info']['result'] = {
                                'home_score': int(home_score),
                                'away_score': int(away_score),
                                'total_score': int(home_score + away_score)
                            }
                    
                    elif summary.get('GAME_STATUS_ID') == 2:
                        game_data['game_info']['status'] = 'IN_PROGRESS'
                        
            except Exception as e:
                # La partita potrebbe non essere ancora iniziata
                pass
                
        except Exception as e:
            print(f"⚠️ Errore nell'arricchimento dati partita: {e}")
    
    def _calculate_advanced_injury_impact(self, injury_data_home, injury_data_away, 
                                         team_stats_home, team_stats_away, 
                                         game_date, home_team_name=None, away_team_name=None):
        """
        Calcola l'impatto avanzato degli infortuni utilizzando PlayerImpactAnalyzer.
        """
        try:
            # Inizializza il risultato
            impact_result = {
                'home_team_impact': 0.0,
                'away_team_impact': 0.0,
                'home_injury_impact': 0.0,
                'away_injury_impact': 0.0,
                'home_players_out': [],
                'away_players_out': [],
                'home_players_questionable': [],
                'away_players_questionable': []
            }
            
            # Calcola l'impatto per la squadra di casa
            if injury_data_home and 'players' in injury_data_home:
                for player in injury_data_home['players']:
                    player_impact = self.player_impact_analyzer.calculate_player_impact(
                        player, 
                        team_stats_home, 
                        game_date
                    )
                    
                    if player['status'] == 'Out':
                        impact_result['home_team_impact'] -= player_impact
                        impact_result['home_injury_impact'] += player_impact
                        impact_result['home_players_out'].append({
                            'name': player['name'],
                            'impact': player_impact
                        })
                    elif player['status'] == 'Questionable':
                        # Impatto ridotto per i giocatori dubbii
                        reduced_impact = player_impact * 0.5
                        impact_result['home_team_impact'] -= reduced_impact
                        impact_result['home_injury_impact'] += reduced_impact
                        impact_result['home_players_questionable'].append({
                            'name': player['name'],
                            'impact': reduced_impact
                        })
            
            # Calcola l'impatto per la squadra in trasferta
            if injury_data_away and 'players' in injury_data_away:
                for player in injury_data_away['players']:
                    player_impact = self.player_impact_analyzer.calculate_player_impact(
                        player, 
                        team_stats_away, 
                        game_date
                    )
                    
                    if player['status'] == 'Out':
                        impact_result['away_team_impact'] -= player_impact
                        impact_result['away_injury_impact'] += player_impact
                        impact_result['away_players_out'].append({
                            'name': player['name'],
                            'impact': player_impact
                        })
                    elif player['status'] == 'Questionable':
                        # Impatto ridotto per i giocatori dubbii
                        reduced_impact = player_impact * 0.5
                        impact_result['away_team_impact'] -= reduced_impact
                        impact_result['away_injury_impact'] += reduced_impact
                        impact_result['away_players_questionable'].append({
                            'name': player['name'],
                            'impact': reduced_impact
                        })
            
            # Log dei risultati
            if home_team_name and away_team_name:
                print(f"\n🏥 Impatto infortuni per {home_team_name} vs {away_team_name}:")
                print(f"   🏠 {home_team_name}: {impact_result['home_team_impact']:+.1f} punti")
                print(f"   🚌 {away_team_name}: {impact_result['away_team_impact']:+.1f} punti")
                
                if impact_result['home_players_out']:
                    print("\n   🚫 Giocatori fuori (casa):")
                    for player in impact_result['home_players_out']:
                        print(f"      • {player['name']} (-{player['impact']:.1f} pts)")
                        
                if impact_result['away_players_out']:
                    print("\n   🚫 Giocatori fuori (trasferta):")
                    for player in impact_result['away_players_out']:
                        print(f"      • {player['name']} (-{player['impact']:.1f} pts)")
            
            return impact_result
            
        except Exception as e:
            print(f"⚠️ Errore nel calcolo dell'impatto avanzato degli infortuni: {e}")
            return {
                'home_team_impact': 0.0,
                'away_team_impact': 0.0,
                'home_injury_impact': 0.0,
                'away_injury_impact': 0.0,
                'home_players_out': [],
                'away_players_out': [],
                'home_players_questionable': [],
                'away_players_questionable': []
            }
    
    def get_head_to_head_stats(self, team1_name, team2_name, seasons_back=3):
        """
        Recupera statistiche head-to-head tra due squadre.
        """
        cache_key = f"h2h_{team1_name}_{team2_name}_{seasons_back}"
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        try:
            # Trova gli ID delle squadre
            team1_info = self._find_team_by_name(team1_name)
            team2_info = self._find_team_by_name(team2_name)
            
            if not team1_info or not team2_info:
                print(f"⚠️ Squadre non trovate per H2H: {team1_name} o {team2_name}")
                return None
            
            print(f"🔍 Recupero H2H tra {team1_name} e {team2_name}...")
            
            # Usa LeagueGameFinder per trovare le partite tra le due squadre
            current_year = datetime.now().year
            all_games = []
            
            for i in range(seasons_back):
                season_year = current_year - i
                season_str = f"{season_year-1}-{str(season_year)[-2:]}"
                
                try:
                    time.sleep(NBA_API_REQUEST_DELAY)
                    
                    # Cerca partite dove team1 è in casa
                    gamefinder = leaguegamefinder.LeagueGameFinder(
                        team_id_nullable=team1_info['id'],
                        vs_team_id_nullable=team2_info['id'],
                        season_nullable=season_str,
                        season_type_nullable='Regular Season',
                        headers=self.headers
                    )
                    games_df = gamefinder.get_data_frames()[0]
                    
                    if not games_df.empty:
                        all_games.append(games_df)
                        
                except Exception as e:
                    print(f"   ⚠️ Errore recupero H2H per stagione {season_str}: {e}")
                    continue
            
            if all_games:
                combined_df = pd.concat(all_games, ignore_index=True)
                
                # Calcola statistiche H2H
                h2h_stats = {
                    'games': len(combined_df),
                    'team1_wins': len(combined_df[combined_df['WL'] == 'W']),
                    'team2_wins': len(combined_df[combined_df['WL'] == 'L']),
                    'avg_total_points': (combined_df['PTS'].mean() + combined_df['PTS'].mean()),
                    'team1_avg_points': combined_df['PTS'].mean(),
                    'team2_avg_points': combined_df['PTS'].mean(),
                    'last_5_games': []
                }
                
                # Dettagli ultimi 5 incontri
                for idx, game in combined_df.head(5).iterrows():
                    h2h_stats['last_5_games'].append({
                        'date': game['GAME_DATE'],
                        'team1_pts': game['PTS'],
                        'team2_pts': abs(game['PLUS_MINUS'] - game['PTS']),
                        'total': game['PTS'] + abs(game['PLUS_MINUS'] - game['PTS']),
                        'winner': team1_name if game['WL'] == 'W' else team2_name
                    })
                
                print(f"   ✅ Trovate {h2h_stats['games']} partite H2H")
                
                # Salva in cache
                self.h2h_cache[cache_key] = h2h_stats
                return h2h_stats
            
            return None
            
        except Exception as e:
            print(f"❌ Errore recupero H2H: {e}")
            return None
    
    def get_player_stats(self, player_name, season=None):
        """
        Recupera statistiche di un giocatore.
        """
        try:
            # Cerca il giocatore
            matching_players = nba_players.find_players_by_full_name(player_name)
            
            if not matching_players:
                print(f"⚠️ Giocatore non trovato: {player_name}")
                return None
            
            player = matching_players[0]
            player_id = player['id']
            
            if not season:
                season = self._get_season_str_for_nba_api(date.today())
            
            # Recupera statistiche carriera
            time.sleep(NBA_API_REQUEST_DELAY)
            career_stats = playercareerstats.PlayerCareerStats(
                player_id=player_id,
                headers=self.headers
            )
            
            season_totals = career_stats.season_totals_regular_season.get_data_frame()
            
            # Filtra per la stagione richiesta
            if not season_totals.empty:
                season_stats = season_totals[season_totals['SEASON_ID'] == season]
                
                if not season_stats.empty:
                    stats = season_stats.iloc[0]
                    
                    return {
                        'player_id': player_id,
                        'player_name': player['full_name'],
                        'season': season,
                        'games_played': int(stats.get('GP', 0)),
                        'minutes': float(stats.get('MIN', 0)),
                        'points': float(stats.get('PTS', 0)),
                        'rebounds': float(stats.get('REB', 0)),
                        'assists': float(stats.get('AST', 0)),
                        'steals': float(stats.get('STL', 0)),
                        'blocks': float(stats.get('BLK', 0)),
                        'turnovers': float(stats.get('TOV', 0)),
                        'fg_pct': float(stats.get('FG_PCT', 0)),
                        'fg3_pct': float(stats.get('FG3_PCT', 0)),
                        'ft_pct': float(stats.get('FT_PCT', 0)),
                        'ppg': float(stats.get('PTS', 0)) / max(int(stats.get('GP', 1)), 1),
                        'rpg': float(stats.get('REB', 0)) / max(int(stats.get('GP', 1)), 1),
                        'apg': float(stats.get('AST', 0)) / max(int(stats.get('GP', 1)), 1)
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Errore recupero statistiche giocatore {player_name}: {e}")
            return None
            
    def get_player_game_logs(self, player_id, season=None, last_n_games=10):
        """
        Recupera i log delle partite di un giocatore in modo sicuro.
        """
        # --- AGGIUNTO CONTROLLO DI SICUREZZA ---
        if player_id is None:
            print("⚠️ Tentativo di recuperare log per player_id nullo. Operazione saltata.")
            return None
        # --- FINE CONTROLLO ---

        if not season:
            season = self._get_season_str_for_nba_api(date.today())

        try:
            # Rimosso il DEBUG per pulizia log
            # print(f"🔍 [DEBUG] Recupero log partite per giocatore {player_id}, stagione {season}")


            # Usa il modulo playergamelog di nba_api
            from nba_api.stats.endpoints import playergamelog

            game_log = playergamelog.PlayerGameLog(
                player_id=str(player_id),
                season=season,
                headers=self.headers
            )

            df = game_log.get_data_frames()[0]

            if df.empty:
                return None  # Restituisce None se non ci sono dati

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)

            return df.head(last_n_games) if last_n_games else df

        except Exception as e:
            # L'errore JSONDecodeError verrà catturato qui
            print(f"❌ Errore nel recupero dei log partite per il giocatore {player_id}: {e}")
            return None
    
    def get_team_roster(self, team_name, season=None):
        """
        Recupera il roster di una squadra con informazioni dettagliate sui giocatori.
        
        Args:
            team_name (str): Nome della squadra
            season (str, optional): Stagione nel formato 'YYYY-YY'. Default: stagione corrente
            
        Returns:
            list: Lista di dizionari con le informazioni dettagliate sui giocatori
        """
        if not season:
            season = self._get_season_str_for_nba_api(date.today())
            
        try:
            print(f"🔍 [DEBUG] Recupero roster per {team_name}, stagione {season}")
            
            # Trova l'ID della squadra
            team = self._find_team_by_name(team_name)
            if not team:
                print(f"⚠️ Squadra non trovata: {team_name}")
                return []
                
            team_id = team['id']
            
            # Usa il modulo commonteamroster di nba_api
            from nba_api.stats.endpoints import commonteamroster
            
            roster_data = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                headers=self.headers
            )
            
            # Ottieni sia il roster che i dettagli dei giocatori
            roster_df = roster_data.get_data_frames()[0]
            
            if roster_df.empty:
                print(f"⚠️ Nessun giocatore trovato per la squadra {team_name} nella stagione {season}")
                return []
            
            # Ottieni le statistiche dettagliate dei giocatori
            from nba_api.stats.endpoints import playercareerstats
            
            roster = []
            missing_players = []
            
            for _, row in roster_df.iterrows():
                player_id = row['PLAYER_ID']
                player_name = row['PLAYER']
                
                try:
                    # Ottieni le statistiche della carriera del giocatore
                    player_stats = playercareerstats.PlayerCareerStats(
                        player_id=player_id,
                        per_mode36='PerGame',
                        headers=self.headers
                    )
                    
                    stats_df = player_stats.get_data_frames()[0]
                    
                    # Se non ci sono statistiche, usa valori di default
                    if stats_df.empty:
                        print(f"⚠️ Nessuna statistica trovata per {player_name} (ID: {player_id})")
                        missing_players.append(player_name)
                        continue
                    
                    # Prendi le statistiche più recenti
                    latest_stats = stats_df.iloc[0].to_dict()
                    
                    # Calcola i minuti per partita
                    gp = latest_stats.get('GP', 1)
                    min_per_game = latest_stats.get('MIN', 0) / max(gp, 1)
                    
                    # Determina lo stato del giocatore in base ai minuti giocati
                    if min_per_game > 25:
                        rotation_status = 'STARTER'
                    elif min_per_game > 15:
                        rotation_status = 'BENCH'
                    else:
                        rotation_status = 'RESERVE'
                    
                    # Crea il dizionario con le informazioni del giocatore
                    player_info = {
                        'PLAYER_ID': player_id,
                        'PLAYER_NAME': player_name,
                        'POSITION': row.get('POSITION', 'G-F'),  # Default a guardia-ala
                        'HEIGHT': row.get('HEIGHT', '6-6'),
                        'WEIGHT': row.get('WEIGHT', 200),
                        'SEASON_EXP': row.get('SEASON_EXP', 1),
                        'JERSEY_NUMBER': row.get('NUM', '00'),
                        'TEAM_ID': team_id,
                        'TEAM_NAME': team_name,
                        'ROTATION_STATUS': rotation_status,
                        'MIN': min_per_game,
                        'PTS': latest_stats.get('PTS', 0),
                        'REB': latest_stats.get('REB', 0),
                        'AST': latest_stats.get('AST', 0),
                        'STL': latest_stats.get('STL', 0),
                        'BLK': latest_stats.get('BLK', 0),
                        'FG_PCT': latest_stats.get('FG_PCT', 0.45),
                        'FG3_PCT': latest_stats.get('FG3_PCT', 0.35),
                        'FT_PCT': latest_stats.get('FT_PCT', 0.75),
                        'TOV': latest_stats.get('TOV', 0),
                        'PLUS_MINUS': latest_stats.get('PLUS_MINUS', 0)
                    }
                    
                    roster.append(player_info)
                    
                except Exception as e:
                    print(f"⚠️ Errore nel recupero statistiche per {player_name}: {str(e)}")
                    missing_players.append(player_name)
            
            if missing_players:
                print(f"⚠️ Giocatori senza statistiche complete: {', '.join(missing_players)}")
            
            print(f"✅ Trovati {len(roster)} giocatori per {team_name}")
            if roster:
                print(f"   Primi 3 giocatori: {', '.join([p['PLAYER_NAME'] for p in roster[:3]])}")
                
            return roster
            
        except Exception as e:
            print(f"❌ Errore recupero roster {team_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []