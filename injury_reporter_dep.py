# injury_reporter.py (v3.1 - Scraper corretto in base alla nuova struttura della tabella)
import time
import re
import requests
import unicodedata
import pandas as pd
from datetime import datetime
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import commonteamroster
# Assumendo che config.py esista e contenga NBA_API_REQUEST_DELAY
try:
    from config import NBA_API_REQUEST_DELAY
except ImportError:
    NBA_API_REQUEST_DELAY = 0.6 # Valore di fallback

import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nba_injury_scraper.log')
    ]
)
logger = logging.getLogger('nba_injury_scraper')

print("✅ Modulo injury_reporter.py (v3.1 - Scraper Corretto) caricato con successo!")

class InjuryReporter:
    """
    Gestisce il recupero e l'integrazione dei report infortuni
    utilizzando una logica di web scraping robusta per Rotowire.
    """
    
    def __init__(self, nba_data_provider=None):
        self.nba_data_provider = nba_data_provider
        self.injury_cache = {}
        self.cache_expiry_seconds = 1800  # Cache valida per 30 minuti
        self.ROTOWIRE_BASE_URL = "https://www.rotowire.com/basketball/injury-report.php" # URL più specifico per gli infortuni
        self.REQUEST_HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        }

    def _normalize_name(self, name):
        """
        Normalizza i nomi per il matching.
        Gestisce anche i nomi con suffissi come Jr., Sr., III, etc.
        """
        if not name:
            return ""
            
        # Rimuovi accenti e caratteri speciali
        name = ''.join(c for c in unicodedata.normalize('NFD', str(name).strip()) 
                     if unicodedata.category(c) != 'Mn')
        
        # Gestisci i suffissi comuni nei nomi
        suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
        name_parts = []
        
        for part in name.split():
            part = part.lower().strip('.').strip(',').strip()
            if part not in suffixes and not part.isdigit():
                name_parts.append(part)
        
        # Prendi solo i primi due componenti (nome e cognome)
        normalized = ' '.join(name_parts[:2])
        
        # Sostituisci caratteri speciali e rimuovi spazi extra
        normalized = re.sub(r'[^a-z ]', '', normalized)
        return ' '.join(normalized.split())

    def _fetch_injuries_from_rotowire(self, use_cache=True):
        """
        Recupera le formazioni iniziali da Rotowire per determinare gli infortuni.
        Un giocatore non in formazione è considerato infortunato o non disponibile.
        """
        cache_key = "rotowire_starting_lineups"
        if use_cache and cache_key in self.injury_cache:
            if (datetime.now() - self.injury_cache[cache_key]['timestamp']).total_seconds() < self.cache_expiry_seconds:
                logger.info("📦 [INJURY] Formazioni iniziali (Rotowire) recuperate dalla cache.")
                return self.injury_cache[cache_key]['data']

        lineups_url = "https://www.rotowire.com/basketball/nba-lineups.php"
        logger.info(f"⬇️  [LINEUP] Download formazioni da: {lineups_url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(lineups_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Salva la risposta per debug
            with open('rotowire_lineups_debug.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injury_map = {}
            
            # Trova tutti i giocatori in formazione
            players = soup.find_all(class_='lineup__player')
            
            # I giocatori in formazione hanno la classe is-pct-play-100
            active_players = set()
            for player in players:
                try:
                    if 'is-pct-play-100' in player.get('class', []):
                        player_link = player.find('a')
                        if player_link and 'title' in player_link.attrs:
                            full_name = player_link['title']
                            active_players.add(self._normalize_name(full_name))
                except Exception as e:
                    logger.warning(f"⚠️ [LINEUP] Errore nell'elaborazione di un giocatore: {e}")
            
            # Crea la mappa degli infortuni
            # Considera come infortunati tutti i giocatori non in formazione
            # Nota: Questo è un'approssimazione, poiché un giocatore potrebbe essere fuori per riposo o decisione tecnica
            
            # Prima popoliamo la mappa con tutti i giocatori come infortunati
            # Poi rimuoviamo quelli che sono in formazione
            all_players = set()
            
            # Aggiungi tutti i giocatori trovati nella pagina
            for player in players:
                try:
                    player_link = player.find('a')
                    position_elem = player.find(class_='lineup__pos')
                    position = position_elem.get_text(strip=True) if position_elem else ''
                    
                    # Prova a ottenere il nome completo da diversi attributi
                    full_name = None
                    if player_link and 'title' in player_link.attrs:
                        full_name = player_link['title']
                    elif hasattr(player, 'text'):
                        full_name = player.text.strip()
                    
                    if not full_name:
                        continue
                        
                    normalized_name = self._normalize_name(full_name)
                    all_players.add(normalized_name)
                    
                    # Se non è in formazione, lo segniamo come infortunato
                    if normalized_name not in active_players:
                        injury_map[normalized_name] = {
                            'status': 'O',  # Out
                            'original_name': full_name,
                            'normalized_name': normalized_name,  # Aggiungiamo il nome normalizzato
                            'team': 'UNK',  # Sarà aggiornato in seguito
                            'position': position,
                            'full_status': 'Not in starting lineup',
                            'details': ''
                        }
                except Exception as e:
                    logger.warning(f"⚠️ [LINEUP] Errore nell'elaborazione di un giocatore: {e}")
            
            logger.info(f"✅ [LINEUP] Trovati {len(active_players)} giocatori in formazione su {len(all_players)} totali.")
            
            # Aggiungiamo anche i giocatori in formazione con stato attivo
            for player in players:
                try:
                    if 'is-pct-play-100' not in player.get('class', []):
                        continue
                        
                    player_link = player.find('a')
                    position_elem = player.find(class_='lineup__pos')
                    position = position_elem.get_text(strip=True) if position_elem else ''
                    
                    # Prova a ottenere il nome completo da diversi attributi
                    full_name = None
                    if player_link and 'title' in player_link.attrs:
                        full_name = player_link['title']
                    elif hasattr(player, 'text'):
                        full_name = player.text.strip()
                    
                    if not full_name:
                        continue
                        
                    normalized_name = self._normalize_name(full_name)
                    
                    # Se il giocatore è in formazione, lo aggiungiamo con stato attivo
                    injury_map[normalized_name] = {
                        'status': 'A',  # Active
                        'original_name': full_name,
                        'normalized_name': normalized_name,  # Aggiungiamo il nome normalizzato
                        'team': 'UNK',  # Sarà aggiornato in seguito
                        'position': position,
                        'full_status': 'In starting lineup',
                        'details': ''
                    }
                except Exception as e:
                    logger.warning(f"⚠️ [LINEUP] Errore nell'aggiunta di un giocatore attivo: {e}")
            
            self.injury_cache[cache_key] = {'data': injury_map, 'timestamp': datetime.now()}
            return injury_map
            
        except Exception as e:
            logger.error(f"❌ [LINEUP] Errore nel recupero delle formazioni da Rotowire: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _find_best_match(self, player_name, injury_map, threshold=0.8):
        """
        Trova la migliore corrispondenza per un giocatore nella mappa degli infortuni.
        Utilizza la distanza di Levenshtein per trovare corrispondenze approssimative.
        """
        from difflib import get_close_matches
        
        # Se c'è una corrispondenza esatta, usala
        if player_name in injury_map:
            return player_name, injury_map[player_name]
            
        # Altrimenti cerca corrispondenze approssimate
        matches = get_close_matches(player_name, injury_map.keys(), n=1, cutoff=threshold)
        if matches:
            match_name = matches[0]
            return match_name, injury_map[match_name]
            
        return None, None

    def get_team_roster_with_injuries(self, team_id, league_injuries_map, season=None):
        """
        Recupera il roster di una squadra con informazioni sugli infortuni.
        
        Args:
            team_id (int): ID della squadra
            league_injuries_map (dict): Mappa degli infortuni della lega
            season (str, optional): Stagione nel formato 'YYYY-YY'. Default a quella corrente.
            
        Returns:
            list: Lista di dizionari con i dati dei giocatori e il loro status infortuni
        """
        if not season:
            # Usa la stagione corrente se non specificata
            current_year = datetime.now().year
            season = f"{current_year}-{str(current_year + 1)[-2:]}"
        
        try:
            # Recupera il roster della squadra
            roster_data = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster_data.get_data_frames()[0]
            
            roster_with_injuries = []
            
            for _, row in roster_df.iterrows():
                try:
                    player_data = row.to_dict()
                    original_name = player_data['PLAYER']
                    normalized_name = self._normalize_name(original_name)
                    
                    # Cerca corrispondenza esatta o approssimativa negli infortuni
                    injury_info = {}
                    if normalized_name in league_injuries_map:
                        injury_info = league_injuries_map[normalized_name]
                    else:
                        # Prova a trovare una corrispondenza approssimativa
                        match_name, match_info = self._find_best_match(normalized_name, league_injuries_map)
                        if match_name:
                            logger.debug(f"🔍 [MATCH] Trovata corrispondenza: '{original_name}' -> '{match_info['original_name']}'")
                            injury_info = match_info
                    
                    # Determina lo status di rotazione in base all'esperienza
                    exp = player_data.get('EXP')
                    if exp == 'R':
                        rotation_status = 'BENCH'  # I rookie partono tipicamente dalla panchina
                    else:
                        try:
                            exp_years = int(exp) if exp and str(exp).isdigit() else 0
                            rotation_status = 'STARTER' if exp_years >= 3 else 'BENCH'
                        except (ValueError, TypeError):
                            rotation_status = 'BENCH'  # Default a panchina in caso di errore
                    
                    # Prepara le informazioni sul giocatore
                    player_info = {
                        'id': player_data['PLAYER_ID'],
                        'name': original_name,
                        'normalized_name': normalized_name,
                        'jersey': player_data['NUM'],
                        'position': player_data['POSITION'],
                        'height': player_data['HEIGHT'],
                        'weight': player_data['WEIGHT'],
                        'birth_date': player_data['BIRTH_DATE'],
                        'experience': exp,
                        'school': player_data['SCHOOL'],
                        'rotation_status': rotation_status,
                        'injury_status': 'A',  # Default a attivo
                        'injury_details': '',
                        'injury_position': ''
                    }
                    
                    # Se il giocatore è negli infortuni, aggiorna le informazioni
                    if injury_info:
                        player_info.update({
                            'injury_status': injury_info.get('status', 'O'),
                            'injury_details': injury_info.get('full_status', 'Not in starting lineup'),
                            'injury_position': injury_info.get('position', '')
                        })
                    
                    roster_with_injuries.append(player_info)
                    
                except Exception as e:
                    logger.error(f"⚠️ [ROSTER] Errore nell'elaborazione del giocatore {player_data.get('PLAYER', 'sconosciuto')}: {e}")
                    continue
            
            logger.info(f"✅ [ROSTER] Roster per team {team_id} processato con {len(roster_with_injuries)} giocatori.")
            return roster_with_injuries
            
        except Exception as e:
            logger.error(f"❌ [ROSTER] Errore nel recupero del roster per il team {team_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []