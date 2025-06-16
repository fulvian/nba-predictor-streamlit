"""
Module for handling NBA injury reports using nba-api and web scraping.
"""

print("🔄 Modulo injury_reporter caricato con successo!")

import time
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

# Import from nba_api
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams

# Local imports
from config import NBA_API_REQUEST_DELAY

# Mapping dei nomi delle squadre per Rotowire
TEAM_NAME_MAPPING = {
    'ATL': 'atl', 'BOS': 'bos', 'BKN': 'bkn', 'CHA': 'cha', 'CHI': 'chi',
    'CLE': 'cle', 'DAL': 'dal', 'DEN': 'den', 'DET': 'det', 'GSW': 'gs',
    'HOU': 'hou', 'IND': 'ind', 'LAC': 'lac', 'LAL': 'lal', 'MEM': 'mem',
    'MIA': 'mia', 'MIL': 'mil', 'MIN': 'min', 'NOP': 'no', 'NYK': 'ny',
    'OKC': 'okc', 'ORL': 'orl', 'PHI': 'phi', 'PHX': 'pho', 'POR': 'por',
    'SAC': 'sac', 'SAS': 'sa', 'TOR': 'tor', 'UTA': 'utah', 'WAS': 'wsh',
    'BRO': 'bkn', 'NET': 'bkn', 'PHO': 'pho', 'NY': 'ny', 'NO': 'no',
    'SA': 'sa', 'GS': 'gs', 'WSH': 'wsh', 'UTH': 'utah', 'UTAH': 'utah'
}

# Headers per le richieste HTTP
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
    'TE': 'Trailers',
}

class InjuryReporter:
    """
    Handles retrieval and processing of injury reports using nba-api.
    """
    
    def __init__(self, nba_data_provider=None):
        """
        Initialize the InjuryReporter with an optional NBADataProvider instance.
        
        Args:
            nba_data_provider: Optional instance of NBADataProvider for making API calls
        """
        self.nba_data_provider = nba_data_provider
        
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
        
        # Injury status mapping for NBA API
        self.NBA_INJURY_STATUS_MAP = {
            'Out': 'out',
            'Questionable': 'questionable',
            'Probable': 'probable',
            'Day To Day': 'day-to-day',
            'Gtd': 'game-time-decision',
            'Injured': 'injured',
            'Inactive': 'inactive',
            'Not With Team': 'not-with-team',
            'Two-Way': 'two-way',
            'Active': 'active',
            'Rest': 'rest',
            'Not Injury Related': 'not-injury-related',
            'Personal': 'personal',
            'Suspended': 'suspended',
            'Health And Safety Protocols': 'health-and-safety-protocols',
            'Injury': 'injury',
            'Paternity Leave': 'paternity-leave',
            'G League': 'g-league',
            'G League - Two-Way': 'g-league-two-way',
            'G League - On Assignment': 'g-league-on-assignment',
            'G League - Two-Way On Assignment': 'g-league-two-way-on-assignment'
        }
        
        # Cache per gli infortuni
        self.injury_cache = {}
        self.cache_expiry = 3600  # 1 ora di cache
        
        # URL di base per Rotowire
        self.ROTOWIRE_BASE_URL = "https://www.rotowire.com/basketball/team/"
        
        # Mappatura degli stati di infortunio da Rotowire a standard
        self.ROTOWIRE_STATUS_MAP = {
            'out': 'out',
            'questionable': 'questionable',
            'doubtful': 'doubtful',
            'probable': 'probable',
            'gtd': 'game-time-decision',
            'suspended': 'suspended',
            'not with team': 'not-with-team',
            'injury': 'injury',
            'rest': 'rest',
            'personal': 'personal',
            'health and safety protocols': 'health-and-safety-protocols'
        }

    # In injury_reporter.py

    def _get_team_abbreviation(self, team_id):
        """Get team abbreviation from team ID."""
        team = next((t for t in nba_teams.get_teams() if t['id'] == int(team_id)), None)
        # Restituisce l'abbreviazione esatta (es. OKC) che useremo nell'URL
        return team['abbreviation'] if team else None

    def _fetch_rotowire_injuries(self, team_abbr):
        """Fetch injury data from Rotowire for a specific team."""
        try:
            if not team_abbr:
                return []

            # Mappatura abbreviazioni squadre a nomi completi per Rotowire
            team_name_map = {
                'ATL': 'atlanta-hawks',
                'BOS': 'boston-celtics',
                'BKN': 'brooklyn-nets',
                'CHA': 'charlotte-hornets',
                'CHI': 'chicago-bulls',
                'CLE': 'cleveland-cavaliers',
                'DAL': 'dallas-mavericks',
                'DEN': 'denver-nuggets',
                'DET': 'detroit-pistons',
                'GSW': 'golden-state-warriors',
                'HOU': 'houston-rockets',
                'IND': 'indiana-pacers',
                'LAC': 'la-clippers',
                'LAL': 'los-angeles-lakers',
                'MEM': 'memphis-grizzlies',
                'MIA': 'miami-heat',
                'MIL': 'milwaukee-bucks',
                'MIN': 'minnesota-timberwolves',
                'NOP': 'new-orleans-pelicans',
                'NYK': 'new-york-knicks',
                'OKC': 'oklahoma-city-thunder',
                'ORL': 'orlando-magic',
                'PHI': 'philadelphia-76ers',
                'PHX': 'phoenix-suns',
                'POR': 'portland-trail-blazers',
                'SAC': 'sacramento-kings',
                'SAS': 'san-antonio-spurs',
                'TOR': 'toronto-raptors',
                'UTA': 'utah-jazz',
                'WAS': 'washington-wizards'
            }
            team_name = team_name_map.get(team_abbr.upper(), team_abbr.lower())
            url = f"{self.ROTOWIRE_BASE_URL}{team_name}/"
            print(f"   [INJURY_SCRAPER] Fetching injuries from: {url}")
            
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Nuova logica di scraping per la struttura aggiornata di Rotowire
            injury_table = soup.find('table', {'class': 'lineup is-mobile'})
            if injury_table is None:
                print(f"   [INJURY_SCRAPER] Nessuna tabella infortuni trovata su {url}.")
                return []
            
            injuries = []
            for row in injury_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue
                player = cells[0].text.strip()
                position = cells[1].text.strip()
                status = cells[2].text.strip()
                injury = cells[3].text.strip()
                
                # Mappa lo stato di Rotowire al nostro standard
                status = self.ROTOWIRE_STATUS_MAP.get(status, status)
                
                injuries.append({
                    'player': player,
                    'position': position,
                    'status': status,
                    'injury': injury,
                    'source': 'rotowire',
                    'team_abbr': team_abbr
                })
            
            if injuries:
                print(f"   [INJURY_SCRAPER] Trovati {len(injuries)} giocatori infortunati per {team_abbr}.")

            return injuries
            
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                print(f"   [INJURY_SCRAPER] ❌ Errore 404: La pagina per {team_abbr} non esiste all'URL: {url}")
            else:
                print(f"   [INJURY_SCRAPER] ❌ Errore HTTP: {http_err}")
            return []
        except Exception as e:
            print(f"   [INJURY_SCRAPER] ❌ Errore generico nel recupero da Rotowire: {e}")
            return []
    
    def get_team_injuries(self, team_id, use_cache=True):
        """
        Get injury report for a specific team.
        
        Args:
            team_id: NBA team ID (int or str)
            use_cache: Whether to use cached data if available
            
        Returns:
            list: List of injury reports for the team
        """
        try:
            # Get team abbreviation
            team_abbr = self._get_team_abbreviation(team_id)
            if not team_abbr:
                print(f"⚠️ Impossibile trovare l'abbreviazione per il team ID: {team_id}")
                return []
                
            # Check cache
            cache_key = f"injuries_{team_abbr}"
            if use_cache and cache_key in self.injury_cache:
                cached_data = self.injury_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(seconds=self.cache_expiry):
                    return cached_data['data']
            
            # Fetch from Rotowire
            injuries = self._fetch_rotowire_injuries(team_abbr)
            
            # Update cache
            self.injury_cache[cache_key] = {
                'data': injuries,
                'timestamp': datetime.now().isoformat()
            }
            
            return injuries
            
        except Exception as e:
            print(f"⚠️ Errore nel recupero degli infortuni: {e}")
            return []
    
    def get_team_roster(self, team_id, season=None):
        """
        Get the current roster for a team with integrated injury status from Rotowire.
        
        Args:
            team_id: NBA team ID (int or str)
            season: Optional season in format 'YYYY-YY' (default: current season)
            
        Returns:
            list: List of player dictionaries with injury status
        """
        try:
            # Ensure team_id is an integer
            if isinstance(team_id, dict):
                print(f"🔍 [INJURY_REPORTER] Ricevuto dizionario come team_id, estraggo l'ID corretto")
                team_id = team_id.get('home_team_id') or team_id.get('away_team_id')
                if team_id is None:
                    print("⚠️ [INJURY_REPORTER] Impossibile estrarre l'ID della squadra dal dizionario")
                    return []
            
            team_id = int(team_id)  # Ensure team_id is an integer
            
            if season is None:
                # Get current season
                current_year = datetime.now().year
                if datetime.now().month >= 10:  # NBA season starts in October
                    season = f"{current_year}-{str(current_year + 1)[2:]}"
                else:
                    season = f"{current_year - 1}-{str(current_year)[2:]}"
            
            print(f"🔍 [INJURY_REPORTER] Recupero roster per team_id: {team_id}, season: {season}")
            
            # Get team abbreviation for Rotowire
            team_info = next((t for t in nba_teams.get_teams() if t['id'] == team_id), None)
            if not team_info:
                print(f"⚠️ [INJURY_REPORTER] Team ID {team_id} non trovato")
                return []
                
            team_abbr = team_info['abbreviation']
            team_slug = TEAM_NAME_MAPPING.get(team_abbr)
            
            # Fetch injuries from Rotowire and create a mapping
            injury_list = self._fetch_rotowire_injuries(team_slug) if team_slug else []
            injury_map = {player['player'].lower(): player for player in injury_list}
            
            if injury_list:
                print(f"   ✅ Trovati {len(injury_list)} report infortuni da Rotowire per {team_abbr}.")
            
            # Get base team roster from nba-api
            roster_data = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                headers=self.headers
            )
            
            # Convert to DataFrame
            roster_df = roster_data.get_data_frames()[0]
            
            # Integrate injury status into the roster
            roster_with_injuries = []
            for _, row in roster_df.iterrows():
                player_name = row['PLAYER']
                player_name_lower = player_name.lower()
                player_injury_data = injury_map.get(player_name_lower)
                
                # Default values
                status = 'active'
                injury_details = 'N/A'
                
                # Update status based on injury data
                if player_injury_data:
                    status = self.ROTOWIRE_STATUS_MAP.get(player_injury_data.get('status', '').lower(), 'questionable')
                    injury_details = player_injury_data.get('injury', 'N/A')
                
                # Create player dictionary with all relevant fields
                player = {
                    'id': row['PLAYER_ID'],
                    'name': player_name,
                    'jersey': row['NUM'],
                    'position': row['POSITION'],
                    'height': row['HEIGHT'],
                    'weight': row['WEIGHT'],
                    'birth_date': row['BIRTH_DATE'],
                    'age': row['AGE'],
                    'experience': row['EXP'],
                    'school': row['SCHOOL'],
                    'status': status,  # Updated with injury status
                    'injury': injury_details,
                    'team_id': team_id,
                    'first_name': player_name.split(' ')[0] if ' ' in player_name else player_name,
                    'last_name': ' '.join(player_name.split(' ')[1:]) if ' ' in player_name else '',
                    'min': 0  # Will be updated with actual minutes if available
                }
                roster_with_injuries.append(player)
            
            print(f"✅ [INJURY_REPORTER] Trovati e processati {len(roster_with_injuries)} giocatori per il team {team_abbr} ({team_id})")
            return roster_with_injuries
            
        except Exception as e:
            print(f"⚠️ [INJURY_REPORTER] Errore durante il recupero del roster per il team {team_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_injury_adjusted_data(self, team_id, game_date=None):
        """
        Recupera i dati infortuni per un team specifico dalla NBA API.
        """
        # Ottieni la stagione corrente in base alla data
        season = self._get_season(game_date)
        
        # Costruisci l'URL per l'endpoint infortuni della NBA
        url = f"http://data.nba.net/prod/v1/{season}/leagueinjury.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Filtra per team_id e verifica che il giocatore sia ancora infortunato
            injuries = []
            for player in data.get('league', {}).get('standard', []):
                if player.get('teamId') == team_id and player.get('status') not in ["ACTIVE", "RETURNING"]:
                    # Mappa i dati al nostro formato
                    injury = {
                        'player': player.get('firstName') + ' ' + player.get('lastName'),
                        'position': player.get('pos'),
                        'status': player.get('status'),
                        'injury': player.get('injuryDescription', ''),
                        'source': 'nba_api',
                    }
                    injuries.append(injury)
            
            return injuries
            
        except Exception as e:
            print(f"Errore nel recupero infortuni da NBA API: {e}")
            return []

    def _get_season(self, game_date):
        """
        Ottieni la stagione corrente in base alla data.
        
        Args:
            game_date: Data della partita (str)
        
        Returns:
            str: Stagione corrente in formato 'YYYY-YY'
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ottieni l'anno e il mese della data
        year, month, _ = game_date.split('-')
        year, month = int(year), int(month)
        
        # Determina la stagione in base al mese
        if month >= 10:  # NBA season starts in October
            season = f"{year}-{str(year + 1)[2:]}"
        else:
            season = f"{year - 1}-{str(year)[2:]}"
        
        return season

# Import the PlayerImpactAnalyzer class from the new module
from player_impact_analyzer import PlayerImpactAnalyzer

def create_player_impact_analyzer(nba_data_provider=None):
    """
    Factory function to create a PlayerImpactAnalyzer instance.
    
    Args:
        nba_data_provider: Optional instance of NBADataProvider
        
    Returns:
        PlayerImpactAnalyzer: A properly initialized instance
    """
    print("\n🔍 [DEBUG] Creazione di una nuova istanza di PlayerImpactAnalyzer...")
    analyzer = PlayerImpactAnalyzer(nba_data_provider)
    
    # Verifica che i metodi siano disponibili
    if not hasattr(analyzer, 'calculate_player_impact'):
        print("⚠️ [WARNING] Il metodo calculate_player_impact non è disponibile dopo l'inizializzazione")
    else:
        print("✅ [DEBUG] Il metodo calculate_player_impact è disponibile dopo l'inizializzazione")
    
    return analyzer
