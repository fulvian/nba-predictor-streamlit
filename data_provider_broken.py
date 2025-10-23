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

    def get_scheduled_games(self, days_ahead=7, specific_date=None):
        """
        ROBUST NBA Game Detection - Anti-Timeout Solution
        Uses Live Data API for today, Schedule API with caching for future dates
        Prevents 30+ second timeouts with graceful fallback
        """
        from dateutil import parser
        import pickle
        from pathlib import Path

        # Cache setup
        cache_dir = Path(".nba_cache")
        cache_dir.mkdir(exist_ok=True)

        def get_cache_file(date_str):
            return cache_dir / f"schedule_{date_str}.pkl"

        def load_from_cache(date_str):
            cache_file = get_cache_file(date_str)
            if cache_file.exists():
                try:
                    cache_time = cache_file.stat().st_mtime
                    current_time = time.time()
                    # Cache valido per 6 ore
                    if current_time - cache_time < 6 * 3600:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                except:
                    pass
            return None

        def save_to_cache(date_str, games):
            try:
                cache_file = get_cache_file(date_str)
                with open(cache_file, 'wb') as f:
                    pickle.dump(games, f)
            except:
                pass

        def try_live_data_api(target_date):
            """Live Data API per oggi (istantaneo e affidabile)"""
            try:
                today = date.today()
                if target_date != today:
                    return []

                from nba_api.live.nba.endpoints import scoreboard
                board = scoreboard.ScoreBoard()
                games_dict = board.games.get_dict()

                if games_dict:
                    games = []
                    for game in games_dict:
                        game_time_utc = game.get('gameTimeUTC', '')
                        if game_time_utc:
                            try:
                                game_dt = parser.parse(game_time_utc).replace(tzinfo=datetime.timezone.utc)
                                game_date = game_dt.strftime('%Y-%m-%d')
                            except:
                                game_date = board.score_board_date
                        else:
                            game_date = board.score_board_date

                        games.append({
                            'away_team': game.get('awayTeam', {}).get('teamName', 'Unknown'),
                            'home_team': game.get('homeTeam', {}).get('teamName', 'Unknown'),
                            'away_team_id': game.get('awayTeam', {}).get('teamId', 0),
                            'home_team_id': game.get('homeTeam', {}).get('teamId', 0),
                            'game_id': game.get('gameId', 'N/A'),
                            'date': game_date,
                            'time_utc': game_time_utc,
                            'status': game.get('gameStatusText', 'Unknown'),
                            'score': f"{game.get('awayTeam', {}).get('score', 0)}-{game.get('homeTeam', {}).get('score', 0)}",
                            'source': 'NBA Live Data API (Instant)',
                            'api_endpoint': 'cdn.nba.com/static/json/liveData/scoreboard'
                        })

                    print(f"   ✅ Live Data API: {len(games)} partite trovate")
                    return games
                else:
                    print(f"   ❌ Live Data API: nessuna partita")
                    return []
            except Exception as e:
                print(f"   ❌ Live Data API error: {e}")
                return []

        def try_schedule_api(target_date):
            """Schedule API con retry rapido e cache"""
            target_date_str = target_date.strftime('%Y-%m-%d')

            # 1. Controlla cache
            cached_games = load_from_cache(target_date_str)
            if cached_games:
                print(f"   💾 Cache hit per {target_date_str}")
                return cached_games

            # 2. Prova ScheduleLeagueV2 con retry veloce
            for attempt in range(2):  # Solo 2 tentativi per velocità
                try:
                    if attempt > 0:
                        print(f"      🔄 Tentativo {attempt + 1}/2...")
                        time.sleep(0.5)

                    # Determina stagione NBA
                    year = target_date.year
                    if target_date.month >= 10:
                        season = f"{year}-{str(year+1)[-2:]}"
                    else:
                        season = f"{year-1}-{str(year)[-2:]}"

                    from nba_api.stats.endpoints import scheduleleaguev2
                    schedule = scheduleleaguev2.ScheduleLeagueV2(
                        league_id='00',
                        season=season
                    )

                    data_frames = schedule.get_data_frames()

                    if data_frames and len(data_frames) > 0:
                        df = data_frames[0]

                        # Filtra per la data target
                        if 'gameDate' in df.columns:
                            df['gameDate'] = pd.to_datetime(df['gameDate'])
                            target_datetime = datetime.combine(target_date, datetime.min.time())

                            filtered_df = df[
                                (df['gameDate'] >= target_datetime) &
                                (df['gameDate'] < target_datetime + timedelta(days=1))
                            ]

                            games = []
                            for _, row in filtered_df.iterrows():
                                away_team = row.get('awayTeam_teamName', 'Unknown')
                                home_team = row.get('homeTeam_teamName', 'Unknown')

                                games.append({
                                    'away_team': away_team,
                                    'home_team': home_team,
                                    'away_team_id': row.get('awayTeam_teamId', 0),
                                    'home_team_id': row.get('homeTeam_teamId', 0),
                                    'game_id': row.get('gameId', f"SCHEDULE_{len(games)}"),
                                    'date': target_date_str,
                                    'time_utc': row['gameDate'].isoformat() if pd.notna(row['gameDate']) else '',
                                    'status': 'Scheduled',
                                    'score': '',
                                    'source': 'NBA ScheduleLeagueV2 (Robust)',
                                    'api_endpoint': 'stats.nba.com/stats/scheduleleaguev2',
                                    'season': season
                                })

                            # Salva in cache
                            if games:
                                save_to_cache(target_date_str, games)

                            print(f"   ✅ Schedule API: {len(games)} partite trovate")
                            return games

                except Exception as e:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                        print(f"      ⏱️  Timeout/Connection: {error_msg[:50]}...")
                    else:
                        print(f"      ❌ API Error: {error_msg[:50]}...")

            # 3. Fallback mock solo per date future
            if target_date > date.today():
                print(f"      🎭 Fallback a dati mock per {target_date_str}")
                mock_templates = [
                    ('Boston Celtics', 'New York Knicks'),
                    ('Los Angeles Lakers', 'Golden State Warriors'),
                    ('Milwaukee Bucks', 'Philadelphia 76ers'),
                    ('Phoenix Suns', 'Denver Nuggets'),
                    ('Miami Heat', 'Atlanta Hawks'),
                    ('Dallas Mavericks', 'Memphis Grizzlies')
                ]

                games = []
                for i, (away, home) in enumerate(mock_templates[:4]):
                    games.append({
                        'away_team': away,
                        'home_team': home,
                        'away_team_id': 0,
                        'home_team_id': 0,
                        'game_id': f"MOCK_{target_date_str}_{i+1}",
                        'date': target_date_str,
                        'time_utc': '',
                        'status': 'Scheduled (Mock)',
                        'score': '',
                        'source': 'Mock Data (API Unavailable)',
                        'api_endpoint': 'N/A (API Timeout)'
                    })

                return games

            return []

        # Main logic
        if specific_date:
            target_date = datetime.strptime(specific_date, '%Y-%m-%d').date()
            print(f"\n🏀 Robust NBA Game Detection - {target_date}")
            print("=" * 60)

            scheduled_games = []
            today = date.today()

            if target_date == today:
                # OGGI: Live Data API
                print("📅 Data = OGGI → Live Data API (primario)")
                scheduled_games = try_live_data_api(target_date)

                if not scheduled_games:
                    print("   🔄 Live API vuoto → provo Schedule API...")
                    scheduled_games = try_schedule_api(target_date)

            elif target_date > today:
                # FUTURO: Schedule API con cache
                print("📅 Data = FUTURO → Schedule API con cache")
                scheduled_games = try_schedule_api(target_date)

            else:
                # PASSATO: Solo cache
                print("📅 Data = PASSATO → Solo cache")
                cached_games = load_from_cache(target_date.strftime('%Y-%m-%d'))
                if cached_games:
                    scheduled_games.extend(cached_games)

            # Rimuovi duplicati
            seen_game_ids = set()
            unique_games = []
            for game in scheduled_games:
                if game.get('game_id') not in seen_game_ids:
                    seen_game_ids.add(game.get('game_id'))
                    unique_games.append(game)

            print(f"\n📊 RISULTATO FINALE: {len(unique_games)} partite uniche")

            if unique_games:
                print("🏀 PARTITE TROVATE:")
                for i, game in enumerate(unique_games, 1):
                    score_text = f" [{game.get('score', '')}]" if game.get('score') else ""
                    source = game.get('source', 'Unknown')
                    print(f"   {i}. {game['away_team']} @ {game['home_team']}{score_text}")
                    print(f"      📡 Source: {source}")
            else:
                print("   ❌ NESSUNA PARTITA TROVATA")

            return unique_games

        else:
            # Logica per days_ahead (mantiene compatibilità)
            print(f"📅 Getting NBA games for NEXT {days_ahead} DAYS (Robust)...")

            all_games = []
            today = date.today()

            for i in range(1, days_ahead + 1):
                target_date = today + timedelta(days=i)
                date_str = target_date.strftime('%Y-%m-%d')
                print(f"\n📅 Checking {date_str}...")

                day_games = self.get_scheduled_games(specific_date=date_str)
                all_games.extend(day_games)

            print(f"\n🎯 TOTAL GAMES NEXT {days_ahead} DAYS: {len(all_games)}")
            return all_games
                headers=self.headers
            )

            games = scoreboard.game_header.get_data_frame()
            print(f"   📊 ScoreboardV2 response: {len(games)} games found")

            if not games.empty:
                for _, game in games.iterrows():
                    try:
                        home_team_info = self.team_id_to_info.get(game['HOME_TEAM_ID'])
                        away_team_info = self.team_id_to_info.get(game['VISITOR_TEAM_ID'])

                        if not home_team_info or not away_team_info:
                            print(f"   ⚠️ Team info missing for game_id: {game['GAME_ID']}")
                            continue

                        scheduled_games.append({
                            'date': target_date_str,
                            'time': 'Scheduled',
                            'home_team': home_team_info['full_name'],
                            'away_team': away_team_info['full_name'],
                            'home_team_id': home_team_info['id'],
                            'away_team_id': away_team_info['id'],
                            'game_id': game['GAME_ID'],
                            'status': 'Scheduled',
                            'score': '',
                            'source': 'NBA ScoreboardV2 (REAL Official Games)',
                            'api_endpoint': 'stats.nba.com/stats/scoreboardv2',
                            'odds': []
                        })
                        print(f"   ✅ {away_team_info['full_name']} @ {home_team_info['full_name']}")

                    except Exception as e:
                        print(f"   ⚠️ Error processing game: {e}")
                        continue

            return scheduled_games

        except Exception as e:
            print(f"   ❌ ScoreboardV2 failed: {e}")
            return []

    def _try_legacy_scoreboard(self, specific_date):
        """Fallback to original scoreboard method"""
        try:
            scheduled_games = []
            print(f"📅 Using legacy ScoreboardV2 for {specific_date}...")

            time.sleep(NBA_API_REQUEST_DELAY)
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=specific_date,
                league_id='00',
                headers=self.headers
            )

            games = scoreboard.game_header.get_data_frame()
            print(f"   📊 Legacy NBA API response received for {specific_date}")

            if not games.empty:
                for _, game in games.iterrows():
                    try:
                        home_team_info = self.team_id_to_info.get(game['HOME_TEAM_ID'])
                        away_team_info = self.team_id_to_info.get(game['VISITOR_TEAM_ID'])

                        if not home_team_info or not away_team_info:
                            continue

                        scheduled_games.append({
                            'date': specific_date,
                            'time': game.get('GAME_STATUS_TEXT', ''),
                            'home_team': home_team_info['full_name'],
                            'away_team': away_team_info['full_name'],
                            'home_team_id': home_team_info['id'],
                            'away_team_id': away_team_info['id'],
                            'game_id': game['GAME_ID'],
                            'source': 'NBA ScoreboardV2 (Legacy Fallback)',
                            'odds': []
                        })
                    except Exception as e:
                        continue

            return scheduled_games

        except Exception as e:
            print(f"   ❌ Legacy method also failed: {e}")
            return []

    def _try_legacy_multiple_days(self, days_ahead):
        """Fallback to original method for multiple days"""
        scheduled_games = []
        base_date = date.today()
        dates_to_check = [base_date + timedelta(days=days_offset) for days_offset in range(days_ahead)]

        for current_date in dates_to_check:
            date_str = current_date.strftime('%Y-%m-%d')
            legacy_games = self._try_legacy_scoreboard(date_str)
            scheduled_games.extend(legacy_games)

        return scheduled_games

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