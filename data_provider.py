# data_provider.py

"""
Modulo per la gestione dei dati NBA usando nba-api.
Versione 2.1 - Aggiunti controlli di robustezza
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import time
import random
import traceback
from datetime import datetime, date, timedelta
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
    playergamelog,  # Assicurati che sia importato
    playerestimatedmetrics,
    teamestimatedmetrics,
    leaguestandings,
    scoreboardv2,
    scheduleleaguev2,  # ✅ AGGIUNTO: Per partite future
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

# Configurazione - ⚡ RATE LIMITING AGGRESSIVO (Velocità Massima)
NBA_API_REQUEST_DELAY = 0.2  # Aggressivo: 0.2s per velocità massima testata
NBA_API_MAX_RETRIES = 3
NBA_API_RETRY_DELAY = 2.0  # Ridotto a 2s per retry più veloci
NBA_API_RATE_LIMIT_DELAY = 5.0  # Ridotto a 5s per recovery più rapido
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_BASE_DIR = os.path.join(BASE_DIR, 'models')
SETTINGS_FILE = os.path.join(BASE_DIR, 'settings.json')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

class NBADataProvider:
    def __init__(self):
        self.team_cache = {}
        self.team_data_cache = {}
        self.game_results_cache = {}
        self.player_stats_cache = {}
        self.h2h_cache = {}
        
        self.nba_teams_info = nba_teams.get_teams()
        self.nba_players_info = nba_players.get_players()
        
        self.team_id_to_info = {team['id']: team for team in self.nba_teams_info}
        self.team_name_to_info = {team['full_name']: team for team in self.nba_teams_info}
        self.team_abbreviation_to_info = {team['abbreviation']: team for team in self.nba_teams_info}
        
        self.player_impact_analyzer = PlayerImpactAnalyzer(self)
        self.injury_reporter = InjuryReporter(self)
        
        self.headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Connection': 'keep-alive',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # 🚀 OTTIMIZZAZIONI: Cache per roster e rate limiting adattivo
        self.roster_cache = {}  # Cache per i roster: {team_id_season: roster_df}
        self.api_call_times = []  # Track dei tempi delle chiamate API
        self.current_delay = NBA_API_REQUEST_DELAY  # Delay adattivo
        
        print("✅ NBADataProvider inizializzato con nba-api (OTTIMIZZATO)")
        print(f"   📊 Caricate {len(self.nba_teams_info)} squadre NBA")
        print(f"   👥 Caricati {len(self.nba_players_info)} giocatori NBA")
        print(f"   ⚡ API delay ottimizzato: {self.current_delay}s")

    def _adaptive_sleep(self, is_rate_limited=False):
        """
        🚀 Rate limiting adattivo CONSERVATIVO per evitare ban dall'API NBA.
        """
        if is_rate_limited:
            # Rate limit detectato: incrementa aggressivamente il delay
            self.current_delay = min(self.current_delay * 1.8, NBA_API_RATE_LIMIT_DELAY)
            print(f"   🚨 RATE LIMIT! Aumento delay a {self.current_delay:.2f}s")
            # Extra penalty sleep per calmare l'API (ridotto)
            time.sleep(2.0)
        else:
            # Riduci gradualmente il delay ma rimani conservativo
            self.current_delay = max(self.current_delay * 0.98, NBA_API_REQUEST_DELAY)
            
        # Jitter più ampio per evitare pattern prevedibili
        jitter = random.uniform(0, 0.2)
        actual_delay = self.current_delay + jitter
        
        # 🔧 LIMITE MASSIMO CHIAMATE AL MINUTO
        current_time = time.time()
        # Mantieni track delle chiamate nell'ultimo minuto
        self.api_call_times = [t for t in self.api_call_times if current_time - t < 60]
        
        # Se abbiamo fatto più di 55 chiamate nell'ultimo minuto, aspetta che la più vecchia "esca"
        # Usiamo 55 invece di 60 per avere un margine di sicurezza
        if len(self.api_call_times) >= 55:
            # Calcola quando la chiamata più vecchia sarà uscita dalla finestra di 60s
            oldest_call_time = min(self.api_call_times)
            time_when_oldest_expires = oldest_call_time + 60
            extra_wait = time_when_oldest_expires - current_time
            
            if extra_wait > 0:
                print(f"   ⏰ Limite 55 chiamate/min raggiunto, attendo {extra_wait:.1f}s per liberare slot")
                time.sleep(extra_wait)
                
        time.sleep(actual_delay)
        
        # Registra questa chiamata
        self.api_call_times.append(time.time())

    def get_team_stats_for_game(self, home_team_name: str, away_team_name: str) -> Optional[Dict]:
        print(f"🔍 Recupero statistiche per la partita: {away_team_name} @ {home_team_name}")
        home_stats = self._get_team_stats(home_team_name, is_home=True)
        away_stats = self._get_team_stats(away_team_name, is_home=False)

        if home_stats and home_stats.get('has_data') and away_stats and away_stats.get('has_data'):
            return {'home': home_stats, 'away': away_stats}
        else:
            print("❌ Impossibile recuperare le statistiche per una o entrambe le squadre.")
            return None
    
    # Assicurati che il metodo _get_team_stats esista e funzioni
    def _get_team_stats(self, team_name, is_home=True):
        # Inizializza le statistiche di default
        stats = {
            'team_name': team_name, 'is_home': is_home, 'games_played': 0, 'points_per_game': 0,
            'points_against_per_game': 0, 'offensive_rating': 0, 'defensive_rating': 0, 'pace': 0,
            'win_percentage': 0, 'last_10_games': {'wins': 0, 'losses': 0}, 'has_data': False
        }
        try:
            cache_key = f"{team_name}_{date.today().strftime('%Y-%m-%d')}"
            if cache_key in self.team_data_cache:
                return self.team_data_cache[cache_key]
            
            team_info = self._find_team_by_name(team_name)
            if not team_info:
                print(f"⚠️ Squadra non trovata: {team_name}")
                return stats
            
            team_id = team_info['id']
            current_season = self._get_season_str_for_nba_api(date.today())
            
            stats_found = False
            # Tenta di recuperare prima le statistiche dei Playoff (più recenti), poi della Regular Season.
            for season_type in ['Playoffs', 'Regular Season']:
                print(f"   🔄 Recupero dati per {team_name} | Stagione: {current_season} | Tipo: {season_type}")
                time.sleep(NBA_API_REQUEST_DELAY)
                
                dashboard = leaguedashteamstats.LeagueDashTeamStats(
                    season=current_season,
                    season_type_all_star=season_type,
                    measure_type_detailed_defense='Advanced',
                    headers=self.headers
                )
                all_teams_stats = dashboard.get_data_frames()[0]
                
                team_stats_df = all_teams_stats[all_teams_stats['TEAM_ID'] == team_id]

                if not team_stats_df.empty and team_stats_df.iloc[0].get('GP', 0) > 0:
                    print(f"   ✅ Dati trovati per Tipo: {season_type}")
                    row = team_stats_df.iloc[0]
                    stats.update({
                        'games_played': int(row.get('GP', 0)),
                        'win_percentage': float(row.get('W_PCT', 0)),
                        'offensive_rating': float(row.get('OFF_RATING', 110)),
                        'defensive_rating': float(row.get('DEF_RATING', 110)),
                        'pace': float(row.get('PACE', 100)),
                        'efg_pct': float(row.get('EFG_PCT', 0)),
                        'ft_rate': float(row.get('FTA_RATE', 0)),
                        'tov_pct': float(row.get('TM_TOV_PCT', 0)),
                        'oreb_pct': float(row.get('OREB_PCT', 0)),
                        'ppg': float(row.get('PTS', 0)) / int(row.get('GP', 1)),
                        'oppg': (float(row.get('DEF_RATING', 110)) * float(row.get('PACE', 100))) / 100,
                        'has_data': True,
                        'source_season_type': season_type
                    })
                    stats_found = True
                    break # Esci dal loop appena trovi dati validi
            
            if not stats_found:
                print(f"❌ Dati non trovati per {team_name} per la stagione {current_season}.")
                stats['has_data'] = False
            
            self.team_data_cache[cache_key] = stats
            return stats
        except Exception as e:
            print(f"❌ Errore critico nel recupero statistiche per {team_name}: {e}")
            return stats

    def _find_team_by_name(self, team_name: str) -> Optional[Dict]:
        if team_name in self.team_name_to_info:
            return self.team_name_to_info[team_name]
        if team_name.upper() in self.team_abbreviation_to_info:
            return self.team_abbreviation_to_info[team_name.upper()]
        team_name_lower = team_name.lower()
        for team in self.nba_teams_info:
            if (team_name_lower in team['full_name'].lower() or
                team['nickname'].lower() in team_name_lower or
                team_name_lower in team['nickname'].lower()):
                return team
        return None

    def _get_season_str_for_nba_api(self, for_date: date) -> str:
        year, month = for_date.year, for_date.month
        return f"{year}-{str(year + 1)[-2:]}" if month >= 10 else f"{year - 1}-{str(year)[-2:]}"
    
    def _validate_season_format(self, season: str) -> str:
        """
        Valida e normalizza il formato della stagione per l'API NBA.
        L'API NBA richiede il formato YYYY-YY (es. 2023-24)
        """
        if not season:
            return self._get_season_str_for_nba_api(date.today())
            
        # Se è nel formato giusto (YYYY-YY), restituiscilo come è
        if len(season) == 7 and season[4] == '-':
            return season
        
        # Prova a convertire altri formati comuni
        if season == '2024-25' or season == '2024-2025':
            return '2024-25'
        elif season == '2023-24' or season == '2023-2024':
            return '2023-24'
        elif season == '2022-23' or season == '2022-2023':
            return '2022-23'
        else:
            print(f"⚠️ Formato stagione non riconosciuto: {season}, uso stagione corrente")
            return self._get_season_str_for_nba_api(date.today())

    def get_scheduled_games(self, days_ahead=3, specific_date=None):
        """
        Recupera partite programmate con fallback multi-source.
        1. NBA API ufficiale (primaria)
        2. Schedule scraper (fallback robusto)
        3. Mock data (fallback di sicurezza)
        """
        scheduled_games = []

        if specific_date:
            print(f"📅 Cercando partite per la data specifica: {specific_date}...")
            dates_to_check = [datetime.strptime(specific_date, '%Y-%m-%d').date()]
        else:
            print(f"📅 Cercando partite per i prossimi {days_ahead} giorni...")
            base_date = date.today()
            dates_to_check = [base_date + timedelta(days=days_offset) for days_offset in range(days_ahead)]

        for current_date in dates_to_check:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"📅 Cercando partite per il {date_str}...")

            # METODO 1: NBA API UFFICIALE (primario)
            games_found = self._try_nba_api(date_str, scheduled_games)

            # METODO 2: Schedule scraper (fallback)
            if not games_found:
                print(f"   🔄 NBA API vuota, provo schedule scraper...")
                games_found = self._try_schedule_scraper(date_str, scheduled_games)

            # ❌ NESSUN FALLBACK SILENZIOSO - Solo dati reali!
            if not games_found:
                print(f"   ❌ NESSUNA PARTITA TROVATA - Nessuna fonte disponibile per {date_str}")
                print(f"   🔍 ERROR: NBA API non ha partite, scraper non funzionante")
                print(f"   🚫 NO MOCK DATA - Solo risultati reali accettati")

        if not scheduled_games:
            print("❌ Nessuna partita trovata con nessun metodo")
        else:
            print(f"✅ Trovate {len(scheduled_games)} partite totali")

        return scheduled_games

    def _try_nba_api(self, date_str, scheduled_games):
        """Tenta NBA API ufficiale per partite future usando scheduleleaguev2"""
        max_retries = 3
        retry_delay = 2

        # Determina la stagione NBA dalla data
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        if date_obj.month >= 10:  # NBA season starts in October
            season = f"{year}-{str(year + 1)[-2:]}"
        else:  # January - September belongs to previous year's season
            season = f"{year - 1}-{str(year)[-2:]}"

        # ✅ FIX: Per date future nel 2025, la stagione 2025-26 non esiste ancora
        # Usa la stagione corrente o quella precedente se disponibile
        current_year = datetime.now().year
        if year > current_year:
            print(f"   ⚠️ Data {date_str} nel futuro - uso stagione {current_year-1}-{str(current_year)[-2:]} come fallback")
            season = f"{current_year - 1}-{str(current_year)[-2:]}"
        elif year == current_year and date_obj.month >= 10:
            # Se siamo a ottobre 2025 o dopo, la stagione potrebbe non essere disponibile
            if date_obj > datetime.now():
                print(f"   ⚠️ Data futura - uso stagione {year-1}-{str(year)[-2:]} come fallback")
                season = f"{year - 1}-{str(year)[-2:]}"

        for attempt in range(max_retries):
            try:
                # Delay prima di ogni tentativo
                self._adaptive_sleep()

                print(f"   🔄 NBA API Schedule tentativo {attempt + 1}/{max_retries} per {date_str} (Season: {season})...")

                # ✅ CORRETTO: Usa scheduleleaguev2 per partite future
                schedule = scheduleleaguev2.ScheduleLeagueV2(
                    league_id='00',
                    season=season,
                    headers=self.headers
                )

                try:
                    schedule_data = schedule.get_data_frames()
                    print(f"   📊 NBA Schedule API response received for {season}")

                    if not schedule_data or len(schedule_data) == 0:
                        print(f"   ℹ️ NBA Schedule API ha restituito dati vuoti")
                        return False

                    # Cerca il DataFrame corretto nei results
                    schedule_df = None
                    for df in schedule_data:
                        if hasattr(df, 'empty') and not df.empty:
                            schedule_df = df
                            break

                    if schedule_df is None:
                        # Prova a estrarre dai result sets diretti
                        if hasattr(schedule, 'result_sets') and schedule.result_sets:
                            for result_set in schedule.result_sets:
                                if 'rowSet' in result_set and result_set['rowSet']:
                                    headers = result_set.get('headers', [])
                                    rows = result_set['rowSet']
                                    if rows:
                                        schedule_df = pd.DataFrame(rows, columns=headers)
                                        break

                    if schedule_df is None or schedule_df.empty:
                        print(f"   ℹ️ Nessun dato di schedule trovato nella response")
                        return False

                    # Filtra per la data specifica
                    target_date = date_str.replace('-', '')
                    games_filtered = []

                    if 'GAME_DATE' in schedule_df.columns:
                        # Formato: YYYY-MM-DDTHH:MM:SS.000Z o YYYY-MM-DD
                        date_mask = schedule_df['GAME_DATE'].astype(str).str.contains(date_str)
                        games_filtered = schedule_df[date_mask]
                    elif 'GameDate' in schedule_df.columns:
                        date_mask = schedule_df['GameDate'].astype(str).str.contains(date_str)
                        games_filtered = schedule_df[date_mask]
                    else:
                        # Prova tutte le colonne che potrebbero contenere date
                        date_columns = [col for col in schedule_df.columns if 'date' in col.lower() or 'game' in col.lower()]
                        for col in date_columns:
                            try:
                                date_mask = schedule_df[col].astype(str).str.contains(date_str)
                                if date_mask.any():
                                    games_filtered = schedule_df[date_mask]
                                    break
                            except:
                                continue

                    if games_filtered.empty:
                        print(f"   ℹ️ Nessuna partita trovata per {date_str} nella stagione {season}")
                        return False

                    games_processed = 0
                    for _, game in games_filtered.iterrows():
                        try:
                            # Estrai team ID e nomi dal game data
                            home_team_id = game.get('HOME_TEAM_ID') or game.get('HomeTeamID') or game.get('home_team_id')
                            away_team_id = game.get('VISITOR_TEAM_ID') or game.get('AwayTeamID') or game.get('away_team_id')
                            home_team_name = game.get('HOME_TEAM_NAME') or game.get('HomeTeamName') or game.get('home_team')
                            away_team_name = game.get('VISITOR_TEAM_NAME') or game.get('AwayTeamName') or game.get('away_team')
                            game_id = game.get('GAME_ID') or game.get('GameID')

                            if not all([home_team_id, away_team_id, game_id]):
                                print(f"   ⚠️ Dati incompleti - HOME_ID: {home_team_id}, AWAY_ID: {away_team_id}, GAME_ID: {game_id}")
                                continue

                            # Mappa ID a informazioni complete dei team
                            home_team_info = self.team_id_to_info.get(int(home_team_id))
                            away_team_info = self.team_id_to_info.get(int(away_team_id))

                            if not home_team_info or not away_team_info:
                                print(f"   ⚠️ Team info mancante per HOME: {home_team_id}, AWAY: {away_team_id}")
                                # Usa i nomi dal dato API se disponibili
                                home_name = home_team_name or f"Team {home_team_id}"
                                away_name = away_team_name or f"Team {away_team_id}"
                            else:
                                home_name = home_team_info['full_name']
                                away_name = away_team_info['full_name']

                            scheduled_games.append({
                                'date': date_str,
                                'time': 'TBD',  # Schedule API doesn't provide game times
                                'home_team': home_name,
                                'away_team': away_name,
                                'home_team_id': int(home_team_id),
                                'away_team_id': int(away_team_id),
                                'game_id': str(game_id),
                                'odds': [],
                                'source': 'nba_api'
                            })
                            print(f"   ✅ NBA Schedule API: {away_name} @ {home_name} (ID: {game_id})")
                            games_processed += 1

                        except Exception as e:
                            print(f"   ⚠️ Errore processing singolo game: {e}")
                            print(f"   🐛 Dati game: {dict(game)}")
                            continue

                    print(f"   🎉 NBA Schedule API processati con successo {games_processed} games per {date_str}")
                    return games_processed > 0

                except Exception as e:
                    print(f"   ⚠️ Errore parsing NBA Schedule API response: {e}")
                    print(f"   🐛 Stack trace: {traceback.format_exc()}")
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Attendo {retry_delay}s e riprovo...")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Backoff esponenziale
                    continue

            except Exception as e:
                print(f"   ❌ Errore chiamata NBA Schedule API (tentativo {attempt + 1}): {e}")
                print(f"   🐛 Stack trace: {traceback.format_exc()}")
                if "Connection" in str(e) or "RemoteDisconnected" in str(e):
                    print(f"   🔗 Problema di connessione, ritento...")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Attendo {retry_delay}s e riprovo...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Backoff esponenziale
                continue

        print(f"   ❌ NBA Schedule API fallita dopo {max_retries} tentativi")
        return False

    def _try_schedule_scraper(self, date_str, scheduled_games):
        """Tenta schedule scraper come fallback"""
        try:
            # Importazione dinamica per evitare dipendenze circolari
            from nba_schedule_scraper import NBAScheduleScraper
            scraper = NBAScheduleScraper()

            print(f"   🌐 Avvio schedule scraper per {date_str}...")
            games_df = scraper.get_todays_games(date_str)

            if not games_df.empty:
                games_processed = 0
                for _, game in games_df.iterrows():
                    try:
                        # Mappa nomi team a ID NBA
                        home_team_info = self._find_team_by_name(game['home_team'])
                        away_team_info = self._find_team_by_name(game['away_team'])

                        if not home_team_info or not away_team_info:
                            # Crea ID fittizi se non trova mapping
                            home_id = hash(game['home_team']) % 100000000 + 1610000000
                            away_id = hash(game['away_team']) % 100000000 + 1610000001
                        else:
                            home_id = home_team_info['id']
                            away_id = away_team_info['id']

                        scheduled_games.append({
                            'date': date_str,
                            'time': game.get('game_time', 'TBD'),
                            'home_team': game['home_team'],
                            'away_team': game['away_team'],
                            'home_team_id': home_id,
                            'away_team_id': away_id,
                            'game_id': f"SCRAPER_{date_str}_{games_processed}",
                            'odds': [],
                            'source': 'scraper'
                        })
                        print(f"   ✅ Scraper: {game['away_team']} @ {game['home_team']}")
                        games_processed += 1
                    except Exception as e:
                        continue

                return games_processed > 0
            else:
                print(f"   ⚠️ Schedule scraper non ha trovato partite")
                return False

        except ImportError:
            print(f"   ❌ Schedule scraper non disponibile")
            return False
        except Exception as e:
            print(f"   ❌ Errore schedule scraper: {e}")
            return False

    def _find_team_by_name(self, team_name):
        """Trova team info per nome con fuzzy matching"""
        # Direct match
        if team_name in self.team_name_to_info:
            return self.team_name_to_info[team_name]

        # Abbreviation match
        if team_name.upper() in self.team_abbreviation_to_info:
            return self.team_abbreviation_to_info[team_name.upper()]

        # Fuzzy match for partial names
        for stored_name, info in self.team_name_to_info.items():
            if team_name.lower() in stored_name.lower() or stored_name.lower() in team_name.lower():
                return info

        return None

    def _generate_mock_games(self, date_str, scheduled_games):
        """Genera partite mock come fallback finale"""
        from datetime import datetime
        import random

        # Team comuni NBA che probabilmente giocheranno
        common_teams = [
            'Los Angeles Lakers', 'Boston Celtics', 'Golden State Warriors', 'Miami Heat',
            'Brooklyn Nets', 'Milwaukee Bucks', 'Phoenix Suns', 'Dallas Mavericks',
            'Philadelphia 76ers', 'Denver Nuggets', 'Los Angeles Clippers', 'Chicago Bulls'
        ]

        # Crea partite casuali ma realistiche
        num_games = random.randint(3, 8)  # Numero realistico di partite NBA in un giorno

        for i in range(min(num_games, len(common_teams)//2)):
            try:
                # Scegli due team casuali
                available_teams = common_teams.copy()
                home_team = random.choice(available_teams)
                available_teams.remove(home_team)
                away_team = random.choice(available_teams)

                # Genera ID mock
                home_id = hash(home_team) % 100000000 + 1610000000
                away_id = hash(away_team) % 100000000 + 1610000001

                # Orario realistico NBA (7:00 PM - 10:30 PM ET)
                game_times = ['7:00 PM ET', '7:30 PM ET', '8:00 PM ET', '8:30 PM ET', '9:00 PM ET', '10:00 PM ET']
                game_time = random.choice(game_times)

                scheduled_games.append({
                    'date': date_str,
                    'time': game_time,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_team_id': home_id,
                    'away_team_id': away_id,
                    'game_id': f"MOCK_{date_str}_{i}",
                    'odds': [],
                    'source': 'mock'
                })
                print(f"   🎲 MOCK: {away_team} @ {home_team} ({game_time})")

            except Exception as e:
                continue

    # --- METODO CORRETTO ---
    def get_player_game_logs(self, player_id, season=None, last_n_games=10):
        """
        Recupera i log delle partite di un giocatore in modo sicuro.
        """
        # --- AGGIUNTO CONTROLLO DI SICUREZZA ---
        if player_id is None:
            print("⚠️ Tentativo di recuperare log per player_id nullo. Operazione saltata.")
            return None
        
        # Converti da Series/object a int se necessario
        if hasattr(player_id, 'iloc'):
            player_id = player_id.iloc[0] if len(player_id) > 0 else None
        if player_id is None:
            return None
        
        try:
            player_id = int(player_id)
        except (ValueError, TypeError):
            print(f"⚠️ Player ID non valido: {player_id}")
            return None
        # --- FINE CONTROLLO ---

        if not season:
            season = self._get_season_str_for_nba_api(date.today())
            
        try:
            # 🚀 Rate limiting adattivo PRIMA della chiamata API
            self._adaptive_sleep()
            
            # print(f"🔍 [DEBUG] Recupero log partite per giocatore {player_id}, stagione {season}") # Rimosso per pulizia log
            game_log = playergamelog.PlayerGameLog(
                player_id=str(player_id),
                season=season,
                headers=self.headers
            )
            df = game_log.get_data_frames()[0]
            
            if df.empty:
                return None
            
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)
            
            return df.head(last_n_games) if last_n_games else df
            
        except Exception as e:
            # 🔧 RILEVA RATE LIMITING e ajusta automaticamente
            error_msg = str(e)
            is_rate_limited = ('rate' in error_msg.lower() or 'limit' in error_msg.lower() or 
                              'too many' in error_msg.lower() or '429' in error_msg or
                              'Expecting value' in error_msg)  # JSON error spesso indica rate limit
            
            if is_rate_limited:
                print(f"🚨 Rate limit per giocatore {player_id}, attivo backoff")
                self._adaptive_sleep(is_rate_limited=True)
            
            print(f"❌ Errore nel recupero dei log partite per il giocatore {player_id}: {e}")
            return None
    
    def get_season_game_log(self, season: str, season_type: str = None) -> Optional[pd.DataFrame]:
        """
        Recupera il log di tutte le partite per una data stagione.
        Args:
            season: Stagione (es. '2023-24')
            season_type: Tipo stagione ('Regular Season', 'Playoffs', o None per entrambe)
        """
        # 🔧 VALIDAZIONE FORMATO STAGIONE
        validated_season = self._validate_season_format(season)
        print(f"🔍 Recupero game log per stagione {validated_season} (originale: {season})")
        
        cache_key = f"season_log_{validated_season}_{season_type or 'all'}"
        # Semplice caching in memoria per questa sessione
        if hasattr(self, cache_key):
            print(f"   ⚡ Usando cache per {validated_season}")
            return getattr(self, cache_key)

        if season_type:
            print(f"🔍 Recupero game log completo per la stagione {validated_season}...")
            print(f"   - Recupero {season_type}...")
        else:
            print(f"🔍 Recupero game log completo per la stagione {validated_season}...")
        
        # Lista per combinare Regular Season e Playoffs
        all_dataframes = []
        
        # Determina i tipi di stagione da recuperare
        season_types_to_fetch = [season_type] if season_type else ['Regular Season', 'Playoffs']
        
        for current_season_type in season_types_to_fetch:
            try:
                # 🚀 Rate limiting adattivo
                self._adaptive_sleep()
                if not season_type:  # Solo se recuperiamo entrambi i tipi
                    print(f"   - Recupero {current_season_type}...")
                
                # 🔧 RETRY LOGIC per gestire errori temporanei dell'API
                max_retries = 3
                retry_count = 0
                df = None
                
                while retry_count < max_retries and df is None:
                    try:
                        game_log_endpoint = leaguegamelog.LeagueGameLog(
                            season=validated_season,
                            season_type_all_star=current_season_type,
                            headers=self.headers
                        )
                        df = game_log_endpoint.get_data_frames()[0]
                        break  # Successo, esci dal loop
                        
                    except Exception as api_error:
                        retry_count += 1
                        error_msg = str(api_error)
                        
                        # Gestione specifica per errori JSON
                        if "Expecting value" in error_msg or "JSONDecodeError" in error_msg:
                            if retry_count < max_retries:
                                print(f"     - ⚠️ API ritorna dati non validi, retry {retry_count}/{max_retries}...")
                                time.sleep(2.0 * retry_count)  # Backoff progressivo
                                continue
                            else:
                                print(f"     - ❌ API non funziona dopo {max_retries} tentativi")
                                break
                        else:
                            # Altri tipi di errore, ri-lancia
                            raise api_error
                
                if df is None:
                    print(f"     - ❌ Impossibile recuperare dati per {current_season_type}")
                    continue

                if not df.empty:
                    # Aggiungi l'anno della stagione per il raggruppamento
                    df['SEASON_YEAR'] = validated_season
                    df['SEASON_TYPE'] = current_season_type  # Aggiungi tipo stagione per debug
                    all_dataframes.append(df)
                    if not season_type:  # Solo se recuperiamo entrambi i tipi
                        print(f"     - ✅ {len(df)} partite {current_season_type}")
                    else:
                        print(f"   - ✅ Trovate {len(df)} partite {current_season_type}")
                else:
                    if not season_type:  # Solo se recuperiamo entrambi i tipi
                        print(f"     - ⚠️ Nessuna partita {current_season_type} trovata")
                    else:
                        print(f"   - ⚠️ Nessuna partita {current_season_type} trovata")
                    
            except Exception as e:
                if not season_type:  # Solo se recuperiamo entrambi i tipi
                    print(f"     - ⚠️ Errore recupero {current_season_type}: {e}")
                else:
                    print(f"   - ⚠️ Errore recupero {current_season_type}: {e}")
                continue
        
        if not all_dataframes:
            print(f"   - ❌ Nessun dato trovato per la stagione {validated_season}.")
            return None
        
        # Combina tutti i DataFrame
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Arricchimento dei dati per il training - LOGICA CORRETTA
        # Se il MATCHUP contiene "vs.", la squadra è in casa, altrimenti è in trasferta
        combined_df['PTS_home'] = combined_df.apply(lambda row: row['PTS'] if ' vs. ' in row['MATCHUP'] else self._get_opponent_score(combined_df, row), axis=1)
        combined_df['PTS_away'] = combined_df.apply(lambda row: row['PTS'] if ' @ ' in row['MATCHUP'] else self._get_opponent_score(combined_df, row), axis=1)
        combined_df['HOME_TEAM_ID'] = combined_df.apply(lambda row: row['TEAM_ID'] if ' vs. ' in row['MATCHUP'] else self._get_opponent_id(combined_df, row), axis=1)
        combined_df['AWAY_TEAM_ID'] = combined_df.apply(lambda row: row['TEAM_ID'] if ' @ ' in row['MATCHUP'] else self._get_opponent_id(combined_df, row), axis=1)

        # Rimuovi duplicati basati su Game_ID, tenendo la prima occorrenza
        df_unique = combined_df.drop_duplicates(subset=['GAME_ID'], keep='first')

        setattr(self, cache_key, df_unique) # Cache result
        print(f"   - ✅ Trovate {len(df_unique)} partite uniche totali.")
        return df_unique

    def _get_opponent_score(self, df, current_row):
        opponent_row = df[(df['GAME_ID'] == current_row['GAME_ID']) & (df['TEAM_ID'] != current_row['TEAM_ID'])]
        return opponent_row['PTS'].iloc[0] if not opponent_row.empty else 0
        
    def _get_opponent_id(self, df, current_row):
        opponent_row = df[(df['GAME_ID'] == current_row['GAME_ID']) & (df['TEAM_ID'] != current_row['TEAM_ID'])]
        return opponent_row['TEAM_ID'].iloc[0] if not opponent_row.empty else None

    def get_team_roster(self, team_id: int, season: str = None) -> Optional[pd.DataFrame]:
        """
        🚀 OTTIMIZZATO: Recupera il roster di una squadra con caching intelligente.
        """
        if season is None:
            season = self._get_season_str_for_nba_api(datetime.today())
        
        # 🚀 CHECK CACHE PRIMA DI FARE CHIAMATA API
        cache_key = f"{team_id}_{season}"
        if cache_key in self.roster_cache:
            return self.roster_cache[cache_key].copy()  # Ritorna una copia per evitare modifiche accidentali
        
        try:
            # Rate limiting adattivo invece di delay fisso
            self._adaptive_sleep()
            
            roster_endpoint = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                headers=self.headers
            )
            roster_df = roster_endpoint.common_team_roster.get_data_frame()

            if not roster_df.empty:
                # Aggiunge uno status di rotazione fittizio per il builder
                # STARTER per i primi 5, BENCH per i successivi 5, RESERVE per gli altri
                roster_df['ROTATION_STATUS'] = 'RESERVE'
                roster_df.loc[0:4, 'ROTATION_STATUS'] = 'STARTER'
                roster_df.loc[5:9, 'ROTATION_STATUS'] = 'BENCH'

                # 🚀 SALVA IN CACHE
                self.roster_cache[cache_key] = roster_df.copy()
                
                return roster_df
            else:
                print(f"⚠️ Roster vuoto per team ID {team_id}, stagione {season}")
                return None
                
        except Exception as e:
            # Rileva rate limiting e ajusta il delay
            is_rate_limited = ('rate' in str(e).lower() or 'limit' in str(e).lower() or 
                              'too many' in str(e).lower() or '429' in str(e))
            if is_rate_limited:
                print(f"⚠️ Rate limit detectato per team {team_id}")
                self._adaptive_sleep(is_rate_limited=True)
                
            print(f"❌ Errore recupero roster per team ID {team_id}: {e}")
            return None

    def get_player_stats(self, player_id, season=None):
        """
        Recupera le statistiche stagionali REALI di un giocatore usando l'endpoint più affidabile.
        
        Args:
            player_id: ID del giocatore NBA
            season: Stagione (es. "2024-25")
            
        Returns:
            pd.DataFrame: Statistiche del giocatore o None se non trovate
        """
        if season is None:
            season = self._get_season_str_for_nba_api(datetime.today())
            
        # Valida il player_id
        try:
            player_id = int(player_id)
        except (ValueError, TypeError):
            print(f"⚠️ [STATS] Player ID non valido: {player_id}")
            return None
            
        # Cache check
        cache_key = f"player_stats_{player_id}_{season}"
        if cache_key in self.player_stats_cache:
            return self.player_stats_cache[cache_key]
            
        try:
            print(f"   🔄 [NBA_API] Recupero statistiche per player_id={player_id}, season={season}")
            
            # Rate limiting adattivo
            self._adaptive_sleep()
            
            # METODO 1: PlayerCareerStats - Più affidabile per statistiche stagionali
            from nba_api.stats.endpoints import playercareerstats
            
            career_stats = playercareerstats.PlayerCareerStats(
                player_id=str(player_id),
                headers=self.headers
            )
            
            # Prendi le statistiche per stagione
            season_totals_df = career_stats.season_totals_regular_season.get_data_frame()
            
            if not season_totals_df.empty:
                # Filtra per la stagione corrente
                current_season_stats = season_totals_df[season_totals_df['SEASON_ID'] == season]
                
                if not current_season_stats.empty:
                    stats_row = current_season_stats.iloc[0]
                    print(f"   ✅ [NBA_API] Statistiche trovate: {stats_row['PTS']:.1f} PTS, {stats_row['AST']:.1f} AST, {stats_row['REB']:.1f} REB")
                    
                    # Cache e ritorna
                    self.player_stats_cache[cache_key] = current_season_stats
                    return current_season_stats
                else:
                    print(f"   ⚠️ [NBA_API] Nessuna statistica per la stagione {season}")
            
            # METODO 2: Se non trova nella stagione corrente, prova l'ultima stagione disponibile
            if not season_totals_df.empty:
                # Ordina per stagione e prendi l'ultima disponibile
                latest_season_stats = season_totals_df.sort_values('SEASON_ID', ascending=False).iloc[0:1]
                latest_season = latest_season_stats.iloc[0]['SEASON_ID']
                stats_row = latest_season_stats.iloc[0]
                
                print(f"   📊 [NBA_API] Usando ultima stagione disponibile: {latest_season}")
                print(f"   ✅ [NBA_API] Statistiche: {stats_row['PTS']:.1f} PTS, {stats_row['AST']:.1f} AST, {stats_row['REB']:.1f} REB")
                
                # Cache e ritorna
                self.player_stats_cache[cache_key] = latest_season_stats
                return latest_season_stats
            
            # METODO 3: Se career stats fallisce, prova PlayerDashboard
            print(f"   🔄 [NBA_API] Tentativo fallback con PlayerDashboard...")
            
            from nba_api.stats.endpoints import playerdashboardbygeneralsplits
            
            player_dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                player_id=str(player_id),
                season=season,
                season_type_all_star='Regular Season',
                headers=self.headers
            )
            
            # Il primo DataFrame contiene le statistiche generali
            dashboard_stats_df = player_dashboard.overall_player_dashboard.get_data_frame()
            
            if not dashboard_stats_df.empty:
                stats_row = dashboard_stats_df.iloc[0]
                print(f"   ✅ [NBA_API] Dashboard stats: {stats_row['PTS']:.1f} PTS, {stats_row['AST']:.1f} AST, {stats_row['REB']:.1f} REB")
                
                # Cache e ritorna
                self.player_stats_cache[cache_key] = dashboard_stats_df
                return dashboard_stats_df
            
            print(f"   ❌ [NBA_API] Nessuna statistica trovata per player_id={player_id}")
            return None
                
        except Exception as e:
            # Rileva rate limiting
            error_msg = str(e)
            is_rate_limited = ('rate' in error_msg.lower() or 'limit' in error_msg.lower() or 
                              'too many' in error_msg.lower() or '429' in error_msg or
                              'Expecting value' in error_msg)
            
            if is_rate_limited:
                print(f"   🚨 [NBA_API] Rate limit per player {player_id}")
                self._adaptive_sleep(is_rate_limited=True)
            
            print(f"   ❌ [NBA_API] Errore recupero statistiche player {player_id}: {e}")
            return None