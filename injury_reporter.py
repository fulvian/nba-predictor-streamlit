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
# from config import NBA_API_REQUEST_DELAY
NBA_API_REQUEST_DELAY = 0.6  # Definita localmente per evitare dipendenza da config

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
        """METODO RIMOSSO - Rotowire non più utilizzato nel sistema semplificato."""
        return []

    def _fetch_nba_official_injuries(self, team_abbr):
        """METODO RIMOSSO - NBA Official non più utilizzato nel sistema semplificato."""
        return []

    def _fetch_cbs_sports_injuries(self, team_abbr):
        """Fetch injury data da CBS Sports - fonte con tabelle ben strutturate."""
        try:
            if not team_abbr:
                return []
                
            # CBS Sports injury URL
            cbs_url = "https://www.cbssports.com/nba/injuries/"
            print(f"   [CBS_SPORTS] Tentativo CBS Sports: {cbs_url}")
            
            response = requests.get(cbs_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            injuries = []
            
            # CBS Sports usa tabelle con struttura ben definita
            # Cerca tabelle con injury data
            tables = soup.find_all('table')
            
            for table in tables:
                # Cerca header che indica è una injury table
                headers = table.find_all(['th', 'td'])
                header_text = ' '.join([h.get_text() for h in headers[:5]]).lower()
                
                if any(keyword in header_text for keyword in ['player', 'injury', 'status', 'position']):
                    # Cerca il team section che ci interessa
                    team_section = table.find_parent()
                    while team_section and team_section.name != 'body':
                        section_text = team_section.get_text().lower()
                        
                        # Verifica se questa sezione è per il nostro team
                        team_names = {
                            'IND': ['indiana', 'pacers'],
                            'OKC': ['oklahoma', 'thunder'],
                            'LAL': ['lakers', 'los angeles'],
                            'GSW': ['warriors', 'golden state']
                        }
                        
                        team_keywords = team_names.get(team_abbr.upper(), [team_abbr.lower()])
                        if any(keyword in section_text for keyword in team_keywords):
                            
                            # Parse della tabella
                            rows = table.find_all('tr')[1:]  # Skip header
                            for row in rows:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 4:
                                    # CBS format: Player | Position | Updated | Injury | Status  
                                    player_cell = cells[0]
                                    position_cell = cells[1] if len(cells) > 1 else None
                                    injury_cell = cells[-2] if len(cells) > 3 else cells[-1]
                                    status_cell = cells[-1]
                                    
                                    player_name = player_cell.get_text(strip=True)
                                    position = position_cell.get_text(strip=True) if position_cell else ''
                                    injury_desc = injury_cell.get_text(strip=True)
                                    status = status_cell.get_text(strip=True).lower()
                                    
                                    # Normalizza status
                                    if 'out' in status:
                                        status = 'out'
                                    elif 'questionable' in status:
                                        status = 'questionable'
                                    elif 'doubtful' in status:
                                        status = 'doubtful'
                                    elif 'probable' in status:
                                        status = 'probable'
                                    
                                    if player_name and status and player_name not in ['Player', 'Name']:
                                        injuries.append({
                                            'player': player_name,
                                            'position': position,
                                            'status': status,
                                            'injury': injury_desc,
                                            'source': 'cbs_sports',
                                            'team_abbr': team_abbr
                                        })
                            break
                        
                        team_section = team_section.find_parent()
            
            if injuries:
                print(f"   [CBS_SPORTS] ✅ Trovati {len(injuries)} injury reports CBS Sports per {team_abbr}")
            else:
                print(f"   [CBS_SPORTS] ⚠️ Nessun injury report CBS Sports per {team_abbr}")
                
            return injuries
            
        except Exception as e:
            print(f"   [CBS_SPORTS] ❌ Errore CBS Sports scraping: {e}")
            return []

    def _fetch_espn_injuries(self, team_abbr):
        """Fetch injury data da ESPN - fonte mainstream affidabile."""
        try:
            if not team_abbr:
                return []
                
            # ESPN injuries URL
            espn_url = "https://www.espn.com/nba/injuries"
            print(f"   [ESPN_SCRAPER] Tentativo ESPN: {espn_url}")
            
            response = requests.get(espn_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            injuries = []
            
            # ESPN usa diverse strutture, cerchiamo tabelle e sezioni
            # Pattern 1: Tabelle con classe Table
            tables = soup.find_all('table', class_=lambda x: x and 'table' in x.lower())
            
            # Pattern 2: Div con injury content
            if not tables:
                tables = soup.find_all(['div', 'section'], class_=lambda x: x and any(
                    keyword in x.lower() for keyword in ['injury', 'report', 'player']
                ))
            
            for table in tables:
                # Cerca contenuto relativo al nostro team
                table_text = table.get_text().lower()
                
                team_indicators = {
                    'IND': ['indiana', 'pacers', 'ind'],
                    'OKC': ['oklahoma', 'thunder', 'okc'],
                    'LAL': ['lakers', 'angeles', 'lal'],
                    'GSW': ['warriors', 'golden', 'gsw']
                }
                
                indicators = team_indicators.get(team_abbr.upper(), [team_abbr.lower()])
                
                if any(indicator in table_text for indicator in indicators):
                    # Parse injury data dalla tabella/sezione
                    rows = table.find_all(['tr', 'div'])
                    
                    for row in rows:
                        row_text = row.get_text()
                        
                        # Cerca pattern injury nei singoli row
                        if any(keyword in row_text.lower() for keyword in ['out', 'questionable', 'doubtful', 'injury']):
                            # Estrai informazioni base
                            cells = row.find_all(['td', 'span', 'div'])
                            
                            if len(cells) >= 2:
                                # ESPN structure varia, cerca pattern comuni
                                player_text = cells[0].get_text(strip=True)
                                status_text = ' '.join([c.get_text(strip=True) for c in cells[1:]])
                                
                                # Determina status
                                status = 'questionable'  # default
                                if 'out' in status_text.lower():
                                    status = 'out'
                                elif 'doubtful' in status_text.lower():
                                    status = 'doubtful'
                                elif 'probable' in status_text.lower():
                                    status = 'probable'
                                
                                if player_text and len(player_text) > 1:
                                    injuries.append({
                                        'player': player_text,
                                        'position': '',
                                        'status': status,
                                        'injury': status_text,
                                        'source': 'espn',
                                        'team_abbr': team_abbr
                                    })
            
            if injuries:
                print(f"   [ESPN_SCRAPER] ✅ Trovati {len(injuries)} injury reports ESPN per {team_abbr}")
            else:
                print(f"   [ESPN_SCRAPER] ⚠️ Nessun injury report ESPN per {team_abbr}")
                
            return injuries
            
        except Exception as e:
            print(f"   [ESPN_SCRAPER] ❌ Errore ESPN scraping: {e}")
            return []

    def _cross_validate_injuries(self, all_injuries):
        """Esegue cross-validation tra le fonti per identificare injuries consistenti."""
        if not all_injuries:
            return []
        
        # Raggruppa injuries per nome giocatore simile
        player_groups = {}
        
        for injury in all_injuries:
            player_name = injury['player'].lower().strip()
            matched_group = None
            
            # Cerca se questo giocatore matcha con un gruppo esistente
            for group_key in player_groups.keys():
                if self._players_match(player_name, group_key):
                    matched_group = group_key
                    break
            
            if matched_group:
                player_groups[matched_group].append(injury)
            else:
                player_groups[player_name] = [injury]
        
        # Valida ogni gruppo
        validated_injuries = []
        
        for player_name, injuries_list in player_groups.items():
            if len(injuries_list) == 1:
                # Un solo report - accetta se da fonte affidabile
                injury = injuries_list[0]
                if injury['source'] in ['cbs_sports', 'espn']:
                    validated_injuries.append(injury)
            else:
                # Multiple reports - prendi il più autorevole o fai merge
                # Priorità: cbs_sports > espn
                best_injury = None
                for injury in injuries_list:
                    if injury['source'] == 'cbs_sports':
                        best_injury = injury
                        break
                    elif injury['source'] == 'espn' and not best_injury:
                        best_injury = injury
                
                if best_injury:
                    validated_injuries.append(best_injury)
        
        return validated_injuries

    def get_team_injuries(self, team_abbr, max_age_hours=6):
        """
        Ottieni injury reports per un team da fonti multiple con sistema di fallback robusto.
        
        GERARCHIA DELLE FONTI SEMPLIFICATE (solo fonti operative):
        1. 📺 CBS SPORTS - Tabelle ben strutturate e affidabili  
        2. 📊 ESPN - Fonte mainstream con dati consistenti
        3. 🔄 CROSS-VALIDATION - Validazione incrociata tra CBS e ESPN
        4. 💾 FALLBACK DATABASE - Solo per situazioni di emergenza
        """
        try:
            print(f"\n🏥 === INJURY REPORT DUAL-SOURCE per {team_abbr} ===")
            
            # Cache check
            cache_key = f"injuries_{team_abbr}"
            cached_result = self._get_cached_data(cache_key, max_age_hours)
            
            if cached_result:
                print(f"   💾 [CACHE] Usando dati cached (età: {cached_result['age_hours']:.1f}h)")
                return cached_result['data']
            
            # Inizializzazione
            all_injuries = []
            sources_tried = []
            sources_successful = []
            
            print(f"   [DUAL-SOURCE] Avvio ricerca injury per {team_abbr}...")
            
            # FONTE 1: CBS SPORTS (primaria)
            print(f"\n   📺 [FONTE 1] CBS SPORTS")
            cbs_injuries = []
            try:
                cbs_injuries = self._fetch_cbs_sports_injuries(team_abbr)
                sources_tried.append('cbs_sports')
                
                if cbs_injuries:
                    all_injuries.extend(cbs_injuries)
                    sources_successful.append('cbs_sports')
                    print(f"   [CBS_SPORTS] ✅ {len(cbs_injuries)} injuries trovati")
                else:
                    print(f"   [CBS_SPORTS] ⚠️ Nessun dato da CBS Sports")
                    
            except Exception as e:
                print(f"   [CBS_SPORTS] ❌ Errore: {e}")
            
            # FONTE 2: ESPN (secondaria)
            print(f"\n   📊 [FONTE 2] ESPN")
            espn_injuries = []
            try:
                espn_injuries = self._fetch_espn_injuries(team_abbr)
                sources_tried.append('espn')
                
                if espn_injuries:
                    all_injuries.extend(espn_injuries)
                    sources_successful.append('espn')
                    print(f"   [ESPN] ✅ {len(espn_injuries)} injuries trovati")
                else:
                    print(f"   [ESPN] ⚠️ Nessun dato da ESPN")
                    
            except Exception as e:
                print(f"   [ESPN] ❌ Errore: {e}")
            
            # CROSS-VALIDATION tra CBS e ESPN
            if len(sources_successful) >= 2:
                print(f"\n   🔄 [CROSS-VALIDATION] Validazione tra {len(sources_successful)} fonti")
                validated_injuries = self._cross_validate_injuries(all_injuries)
                print(f"   [VALIDATION] {len(validated_injuries)} injuries validati")
                all_injuries = validated_injuries
            elif len(sources_successful) == 1:
                print(f"\n   ⚠️ [SINGLE-SOURCE] Solo una fonte disponibile: {sources_successful[0]}")
                
            # FALLBACK DATABASE (solo se entrambe le fonti falliscono)
            if not all_injuries:
                print(f"\n   💾 [FALLBACK] Entrambe le fonti primarie fallite - usando database emergenza")
                try:
                    fallback_injuries = self._get_fallback_injuries(team_abbr)
                    if fallback_injuries:
                        all_injuries.extend(fallback_injuries)
                        sources_successful.append('fallback_db')
                        print(f"   [FALLBACK] ✅ {len(fallback_injuries)} injuries dal database emergenza")
                    else:
                        print(f"   [FALLBACK] ⚠️ Nessun dato nel database emergenza")
                        
                except Exception as e:
                    print(f"   [FALLBACK] ❌ Errore database: {e}")
            
            # Calcolo confidence score semplificato
            confidence_score = self._calculate_confidence_score(sources_successful, len(all_injuries))
            
            # Preparazione risultato finale
            result = {
                'injuries': all_injuries,
                'metadata': {
                    'team_abbr': team_abbr,
                    'sources_tried': sources_tried,
                    'sources_successful': sources_successful,
                    'total_injuries': len(all_injuries),
                    'confidence_score': confidence_score,
                    'quality': 'high' if confidence_score > 0.8 else 'medium' if confidence_score > 0.5 else 'low',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Cache del risultato
            self._cache_data(cache_key, result)
            
            # Summary report
            print(f"\n   📊 === INJURY REPORT SUMMARY per {team_abbr} ===")
            print(f"   Fonti tentate: {len(sources_tried)} ({', '.join(sources_tried)})")
            print(f"   Fonti riuscite: {len(sources_successful)} ({', '.join(sources_successful)})")
            print(f"   Injuries totali: {len(all_injuries)}")
            print(f"   Confidence Score: {confidence_score}")
            print(f"   Qualità: {result['metadata']['quality'].upper()}")
            
            if all_injuries:
                print(f"\n   🏥 TOP 5 INJURIES RILEVATI:")
                for i, injury in enumerate(all_injuries[:5], 1):
                    print(f"     {i}. {injury['player']} ({injury['position']}) - {injury['status'].upper()} - {injury['injury']} [Fonte: {injury['source']}]")
                
                if len(all_injuries) > 5:
                    print(f"     ... e altri {len(all_injuries) - 5} injuries")
            
            return result
            
        except Exception as e:
            print(f"   ❌ [SISTEMA] Errore generale nel sistema injury: {e}")
            return {
                'injuries': [],
                'metadata': {
                    'team_abbr': team_abbr,
                    'error': str(e),
                    'confidence_score': 0.0,
                    'quality': 'failed'
                }
            }

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
            
            # Get team abbreviation for multi-source injury fetching
            team_info = next((t for t in nba_teams.get_teams() if t['id'] == team_id), None)
            if not team_info:
                print(f"⚠️ [INJURY_REPORTER] Team ID {team_id} non trovato")
                return []
                
            team_abbr = team_info['abbreviation']
            
            # Fetch validated injuries from multiple sources
            injury_result = self.get_team_injuries(team_abbr, max_age_hours=6)
            injury_list = injury_result.get('injuries', []) if isinstance(injury_result, dict) else []
            
            # Create mapping con handling migliorato per multiple fonti
            injury_map = {}
            for player_injury in injury_list:
                player_name_lower = player_injury['player'].lower()
                # Prova anche varianti del nome (nome, cognome, nome+cognome)
                name_parts = player_name_lower.split()
                
                injury_map[player_name_lower] = player_injury
                
                # Aggiungi varianti del nome per matching migliore
                if len(name_parts) >= 2:
                    # "John Smith" -> anche "john", "smith"
                    injury_map[name_parts[0]] = player_injury  # First name
                    injury_map[name_parts[-1]] = player_injury  # Last name
                    # "J. Smith" style
                    if len(name_parts[0]) == 1:
                        injury_map[f"{name_parts[0]}. {name_parts[-1]}"] = player_injury
            
            if injury_list:
                sources = set(injury['source'] for injury in injury_list)
                print(f"   ✅ Trovati {len(injury_list)} report infortuni validati per {team_abbr} (fonti: {', '.join(sources)})")
            
            # Get base team roster from nba-api
            roster_data = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                headers=self.headers
            )
            
            # Convert to DataFrame
            roster_df = roster_data.get_data_frames()[0]
            
            # Integrate injury status into the roster with improved matching
            roster_with_injuries = []
            injury_matches_found = 0
            
            for _, row in roster_df.iterrows():
                player_name = row['PLAYER']
                player_name_lower = player_name.lower()
                
                # Default values
                status = 'active'
                injury_details = 'N/A'
                injury_source = 'none'
                injury_confidence = 'none'
                
                # Prova multiple varianti del nome per matching
                player_injury_data = None
                
                # 1. Nome completo esatto
                if player_name_lower in injury_map:
                    player_injury_data = injury_map[player_name_lower]
                
                # 2. Se non trovato, prova solo cognome
                elif not player_injury_data:
                    name_parts = player_name_lower.split()
                    if len(name_parts) >= 2:
                        last_name = name_parts[-1]
                        if last_name in injury_map:
                            player_injury_data = injury_map[last_name]
                
                # 3. Se non trovato, prova solo nome
                elif not player_injury_data:
                    name_parts = player_name_lower.split()
                    if len(name_parts) >= 1:
                        first_name = name_parts[0]
                        if first_name in injury_map:
                            player_injury_data = injury_map[first_name]
                
                # Update status based on validated injury data
                if player_injury_data:
                    injury_matches_found += 1
                    status = player_injury_data.get('status', 'questionable')
                    injury_details = player_injury_data.get('injury', 'N/A')
                    injury_source = player_injury_data.get('source', 'unknown')
                    injury_confidence = player_injury_data.get('confidence', 'medium')
                    
                    print(f"   🔗 [INJURY_MATCH] {player_name} -> {status} ({injury_source}, {injury_confidence})")
                
                # Create player dictionary with all relevant fields
                player = {
                    'id': row['PLAYER_ID'],
                    'PLAYER_ID': row['PLAYER_ID'],  # Aggiungi anche PLAYER_ID per compatibilità
                    'name': player_name,
                    'PLAYER_NAME': player_name,  # Aggiungi anche PLAYER_NAME per compatibilità
                    'jersey': row['NUM'],
                    'position': row['POSITION'],
                    'POSITION': row['POSITION'],  # Aggiungi anche POSITION per compatibilità
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
                
                # INTEGRAZIONE STATISTICHE NBA per giocatori infortunati
                if status != 'active' and self.nba_data_provider:
                    try:
                        print(f"   📊 [NBA_STATS] Recupero statistiche per {player_name}...")
                        
                        # Recupera le statistiche stagionali del giocatore dal NBADataProvider
                        player_stats = self.nba_data_provider.get_player_stats(
                            player_id=row['PLAYER_ID'], 
                            season=season or "2024-25"
                        )
                        
                        if player_stats is not None and not player_stats.empty:
                            # Integra le statistiche reali NBA
                            stats_row = player_stats.iloc[0]  # Prendi la prima riga
                            
                            nba_stats = {
                                'PTS': float(stats_row.get('PTS', 0)),
                                'AST': float(stats_row.get('AST', 0)), 
                                'REB': float(stats_row.get('REB', 0)),
                                'STL': float(stats_row.get('STL', 0)),
                                'BLK': float(stats_row.get('BLK', 0)),
                                'FGM': float(stats_row.get('FGM', 0)),
                                'FGA': float(stats_row.get('FGA', 1)),  # Evita divisione per 0
                                'FTM': float(stats_row.get('FTM', 0)),
                                'FTA': float(stats_row.get('FTA', 1)),  # Evita divisione per 0
                                'TOV': float(stats_row.get('TOV', 0)),
                                'PF': float(stats_row.get('PF', 0)),
                                'MIN': float(stats_row.get('MIN', 20))
                            }
                            
                            # Aggiungi le statistiche al player dictionary
                            player.update(nba_stats)
                            print(f"   ✅ [NBA_STATS] Statistiche NBA integrate per {player_name}: {nba_stats['PTS']:.1f} PTS, {nba_stats['AST']:.1f} AST, {nba_stats['REB']:.1f} REB")
                            
                        else:
                            print(f"   ⚠️ [NBA_STATS] Nessuna statistica trovata per {player_name}")
                            
                    except Exception as e:
                        print(f"   ⚠️ [NBA_STATS_ERROR] Errore recupero statistiche per {player_name}: {e}")

                roster_with_injuries.append(player)
            
            # Report finale con statistiche injury matching
            print(f"✅ [INJURY_REPORTER] Trovati e processati {len(roster_with_injuries)} giocatori per il team {team_abbr} ({team_id})")
            
            if injury_list:
                match_rate = (injury_matches_found / len(injury_list)) * 100 if injury_list else 0
                print(f"   📊 [INJURY_STATS] Injury matches: {injury_matches_found}/{len(injury_list)} ({match_rate:.1f}%)")
                
                if injury_matches_found == 0 and len(injury_list) > 0:
                    print(f"   ⚠️ [INJURY_WARNING] NESSUN MATCH TROVATO - Possibili problemi con i nomi dei giocatori!")
                    print(f"   🔍 [INJURY_DEBUG] Injury list: {[i['player'] for i in injury_list[:3]]}")
                    print(f"   🔍 [INJURY_DEBUG] Roster sample: {[roster_with_injuries[i]['name'] for i in range(min(3, len(roster_with_injuries)))]}")
            else:
                print(f"   ⚠️ [INJURY_WARNING] SISTEMA OPERANDO SENZA INJURY DATA - Verifica connettività web!")
            
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

    def _get_fallback_injuries(self, team_abbr):
        """
        Sistema di fallback con database injuries mock per testing e situazioni di emergenza.
        Fornisce dati verosimili per team NBA comuni.
        """
        try:
            print(f"   [FALLBACK_DB] Attivazione sistema fallback per {team_abbr}")
            
            # Database fallback con injuries mock per team comuni  
            fallback_injury_database = {
                'IND': [  # Indiana Pacers
                    {'player': 'Jarace Walker', 'position': 'F', 'status': 'out', 'injury': 'Ankle injury', 'source': 'fallback_db'},
                    {'player': 'James Johnson', 'position': 'F', 'status': 'out', 'injury': 'Back injury', 'source': 'fallback_db'},
                    {'player': 'Isaiah Jackson', 'position': 'C', 'status': 'out', 'injury': 'Achilles injury', 'source': 'fallback_db'},
                    {'player': 'Quenton Jackson', 'position': 'G', 'status': 'out', 'injury': 'Hip injury', 'source': 'fallback_db'}
                ],
                'OKC': [  # Oklahoma City Thunder  
                    {'player': 'Dillon Jones', 'position': 'F', 'status': 'out', 'injury': 'Hip injury', 'source': 'fallback_db'},
                    {'player': 'Jaylin Williams', 'position': 'F', 'status': 'out', 'injury': 'Hamstring injury', 'source': 'fallback_db'},
                    {'player': 'Jalen Williams', 'position': 'G', 'status': 'questionable', 'injury': 'Knee injury', 'source': 'fallback_db'},
                    {'player': 'Kenrich Williams', 'position': 'F', 'status': 'questionable', 'injury': 'Knee injury', 'source': 'fallback_db'}
                ],
                'LAL': [  # Los Angeles Lakers
                    {'player': 'Christian Wood', 'position': 'F', 'status': 'out', 'injury': 'Knee surgery', 'source': 'fallback_db'},
                    {'player': 'Jalen Hood-Schifino', 'position': 'G', 'status': 'questionable', 'injury': 'Back injury', 'source': 'fallback_db'},
                    {'player': 'Jarred Vanderbilt', 'position': 'F', 'status': 'out', 'injury': 'Foot injury', 'source': 'fallback_db'}
                ],
                'GSW': [  # Golden State Warriors
                    {'player': 'De\'Anthony Melton', 'position': 'G', 'status': 'out', 'injury': 'Back injury', 'source': 'fallback_db'},
                    {'player': 'Jonathan Kuminga', 'position': 'F', 'status': 'questionable', 'injury': 'Ankle injury', 'source': 'fallback_db'}
                ],
                'BOS': [  # Boston Celtics
                    {'player': 'Kristaps Porzingis', 'position': 'C', 'status': 'questionable', 'injury': 'Rest', 'source': 'fallback_db'}
                ],
                'LAC': [  # LA Clippers
                    {'player': 'Kawhi Leonard', 'position': 'F', 'status': 'out', 'injury': 'Knee injury', 'source': 'fallback_db'},
                    {'player': 'P.J. Tucker', 'position': 'F', 'status': 'out', 'injury': 'Not with team', 'source': 'fallback_db'}
                ],
                'PHX': [  # Phoenix Suns
                    {'player': 'Collin Gillespie', 'position': 'G', 'status': 'out', 'injury': 'Ankle injury', 'source': 'fallback_db'}
                ],
                'DEN': [  # Denver Nuggets
                    {'player': 'Vlatko Cancar', 'position': 'F', 'status': 'out', 'injury': 'Knee injury', 'source': 'fallback_db'},
                    {'player': 'DaRon Holmes II', 'position': 'F', 'status': 'out', 'injury': 'Achilles injury', 'source': 'fallback_db'}
                ],
                'MIA': [  # Miami Heat
                    {'player': 'Josh Richardson', 'position': 'G', 'status': 'questionable', 'injury': 'Heel injury', 'source': 'fallback_db'}
                ],
                'MIL': [  # Milwaukee Bucks
                    {'player': 'Khris Middleton', 'position': 'F', 'status': 'questionable', 'injury': 'Ankle injury', 'source': 'fallback_db'}
                ]
            }
            
            team_injuries = fallback_injury_database.get(team_abbr.upper(), [])
            
            if team_injuries:
                print(f"   [FALLBACK_DB] ✅ Trovati {len(team_injuries)} injuries mock per {team_abbr}")
                for injury in team_injuries:
                    injury['team_abbr'] = team_abbr
                return team_injuries
            else:
                print(f"   [FALLBACK_DB] ⚠️ Nessun injury mock per {team_abbr} - roster pulito")
                return []
                
        except Exception as e:
            print(f"   [FALLBACK_DB] ❌ Errore nel sistema fallback: {e}")
            return []

    def _get_cached_data(self, cache_key, max_age_hours):
        """Recupera dati dalla cache se disponibili e non scaduti."""
        if cache_key in self.injury_cache:
            cached_data = self.injury_cache[cache_key]
            timestamp = datetime.fromisoformat(cached_data['timestamp'])
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            
            if age_hours <= max_age_hours:
                cached_data['age_hours'] = age_hours
                return cached_data
                
        return None
    
    def _cache_data(self, cache_key, data):
        """Salva dati nella cache con timestamp."""
        self.injury_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_confidence_score(self, sources_successful, total_injuries):
        """Calcola un confidence score basato su fonti e cross-validation."""
        if total_injuries == 0:
            return 0.0
            
        # Base score sulla quantità di fonti
        sources_score = min(len(sources_successful) / 3.0, 1.0)  # Max con 3+ fonti
        
        # Bonus per fonti autorevoli
        authority_bonus = 0.0
        if 'cbs_sports' in sources_successful:
            authority_bonus += 0.2
        if 'espn' in sources_successful:
            authority_bonus += 0.1
            
        # Penalità per fallback
        fallback_penalty = 0.0
        if 'fallback_db' in sources_successful and len(sources_successful) == 1:
            fallback_penalty = 0.4
            
        # Score finale
        confidence = min(sources_score + authority_bonus - fallback_penalty, 1.0)
        return max(confidence, 0.0)
    
    def _players_match(self, player1, player2, threshold=0.8):
        """
        Verifica se due nomi di giocatori si riferiscono alla stessa persona.
        Usa fuzzy matching per gestire variazioni nei nomi.
        """
        if not player1 or not player2:
            return False
            
        # Normalizza i nomi
        name1 = player1.lower().strip().replace('.', '').replace(',', '')
        name2 = player2.lower().strip().replace('.', '').replace(',', '')
        
        # Match esatto
        if name1 == name2:
            return True
            
        # Split nei componenti del nome
        parts1 = name1.split()
        parts2 = name2.split()
        
        # Se uno dei nomi è contenuto nell'altro
        if name1 in name2 or name2 in name1:
            return True
            
        # Verifica se hanno gli stessi cognomi
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Confronta cognomi (ultima parola)
            if parts1[-1] == parts2[-1]:
                # Se i cognomi sono uguali, verifica le iniziali
                if parts1[0][0] == parts2[0][0]:
                    return True
                    
        # Fuzzy matching semplice per gestire errori di battitura
        # Confronta la lunghezza dei nomi e similarità caratteri
        if len(name1) > 3 and len(name2) > 3:
            common_chars = sum(1 for c1, c2 in zip(name1, name2) if c1 == c2)
            similarity = common_chars / max(len(name1), len(name2))
            if similarity >= threshold:
                return True
                
        return False



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
