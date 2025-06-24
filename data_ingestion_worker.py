# data_ingestion_worker.py

import requests
import os
from dotenv import load_dotenv
from database_manager import save_odds_data, save_consensus_data
from datetime import datetime
import time
from bs4 import BeautifulSoup
import json

load_dotenv()

# Configurazione API
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY')
THERUNDOWN_API_KEY = os.getenv('THERUNDOWN_API_KEY')
API_SPORTS_KEY = os.getenv('API_SPORTS_KEY')
MARKET_TO_MONITOR = os.getenv('MARKET_TO_MONITOR', 'totals')
TARGET_REGIONS = os.getenv('TARGET_REGIONS', 'eu')

def fetch_from_theoddsapi():
    """
    Recupera dati da The Odds API.
    Restituisce una lista di dizionari normalizzati.
    """
    if not THE_ODDS_API_KEY:
        print("⚠️ THE_ODDS_API_KEY non configurata, salto The Odds API")
        return []
    
    url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'apiKey': THE_ODDS_API_KEY,
        'regions': TARGET_REGIONS,
        'markets': MARKET_TO_MONITOR,
        'oddsFormat': 'decimal',
    }
    
    try:
        print("📡 Fetching data from The Odds API...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        remaining_requests = response.headers.get('x-requests-remaining', 'N/A')
        print(f"✅ The Odds API: {remaining_requests} requests remaining")
        
        api_data = response.json()
        normalized_data = []
        
        for game in api_data:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']

            for bookmaker in game['bookmakers']:
                bookie_key = bookmaker['key']
                for market in bookmaker['markets']:
                    if market['key'] == MARKET_TO_MONITOR:
                        line = market['outcomes'][0]['point']
                        odds_over = next((o['price'] for o in market['outcomes'] if o['name'] == 'Over'), None)
                        odds_under = next((o['price'] for o in market['outcomes'] if o['name'] == 'Under'), None)

                        if odds_over and odds_under:
                            normalized_data.append({
                                'game_id': game_id,
                                'sport_key': game['sport_key'],
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker': bookie_key,
                                'market_key': market['key'],
                                'line': line,
                                'odds_over': odds_over,
                                'odds_under': odds_under,
                                'api_source': 'theoddsapi'
                            })
        
        print(f"✅ The Odds API: normalizzati {len(normalized_data)} record")
        return normalized_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore The Odds API: {e}")
        return []

def fetch_from_therundown():
    """
    Recupera dati da TheRundown API.
    Normalizza l'output per renderlo compatibile con la tabella odds_history.
    """
    if not THERUNDOWN_API_KEY:
        print("⚠️ THERUNDOWN_API_KEY non configurata, salto TheRundown API")
        return []
    
    # Nota: Gli endpoint specifici di TheRundown potrebbero variare
    # Questo è un esempio di implementazione che dovrà essere adattato
    url = 'https://therundown-v1.p.rapidapi.com/sports/2/events'
    headers = {
        'X-RapidAPI-Key': THERUNDOWN_API_KEY,
        'X-RapidAPI-Host': 'therundown-v1.p.rapidapi.com'
    }
    
    try:
        print("📡 Fetching data from TheRundown API...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        api_data = response.json()
        normalized_data = []
        
        # Adatta la logica di parsing in base alla struttura effettiva di TheRundown
        if 'events' in api_data:
            for event in api_data['events']:
                # Implementa la logica di parsing specifica per TheRundown
                # Questo esempio è generico e va adattato
                normalized_data.append({
                    'game_id': event.get('event_id', ''),
                    'sport_key': 'basketball_nba',
                    'home_team': event.get('teams_normalized', [{}])[0].get('name', ''),
                    'away_team': event.get('teams_normalized', [{}])[1].get('name', '') if len(event.get('teams_normalized', [])) > 1 else '',
                    'bookmaker': 'therundown',
                    'market_key': 'totals',
                    'line': 0.0,  # Da estrarre dai dati effettivi
                    'odds_over': 0.0,  # Da estrarre dai dati effettivi
                    'odds_under': 0.0,  # Da estrarre dai dati effettivi
                    'api_source': 'therundown'
                })
        
        print(f"✅ TheRundown API: normalizzati {len(normalized_data)} record")
        return normalized_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore TheRundown API: {e}")
        return []

def fetch_from_apisports():
    """
    Recupera dati da API-Sports.
    Implementazione per integrare un'altra fonte di quote.
    """
    if not API_SPORTS_KEY:
        print("⚠️ API_SPORTS_KEY non configurata, salto API-Sports")
        return []
    
    url = 'https://v2.api-sports.io/odds'
    headers = {
        'X-RapidAPI-Key': API_SPORTS_KEY,
        'X-RapidAPI-Host': 'v2.api-sports.io'
    }
    params = {
        'league': '12',  # NBA league ID
        'season': '2024'
    }
    
    try:
        print("📡 Fetching data from API-Sports...")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        api_data = response.json()
        normalized_data = []
        
        # Implementa la logica di parsing per API-Sports
        # Questo è un esempio che va adattato alla struttura effettiva
        
        print(f"✅ API-Sports: normalizzati {len(normalized_data)} record")
        return normalized_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore API-Sports: {e}")
        return []

def fetch_betting_consensus():
    """
    Recupera i dati di Public/Sharp Money.
    Implementa scraping o chiamate API per fonti come VSIN, Sports Insights, Action Network.
    """
    print("📊 Fetching betting consensus data...")
    consensus_data = []
    
    # Esempio di implementazione per scraping dati di consenso
    # Questo dovrebbe essere adattato alle fonti effettive disponibili
    
    try:
        # Esempio: scraping da una fonte ipotetica di dati di consenso
        # In realtà dovresti usare fonti come:
        # - Sports Insights API
        # - Action Network
        # - VSIN (se disponibile)
        
        # Placeholder per dati di esempio
        # In produzione, implementa il vero scraping/API call
        sample_consensus = [
            {
                'game_id': 'sample_game_1',
                'source': 'ActionNetwork',
                'public_tickets_percentage_over': 75.0,
                'public_money_percentage_over': 81.0,
                'sharp_money_indicator': 'Sharp money on Under',
                'reverse_line_movement_detected': 'Potential RLM',
                'line_at_time': 215.5
            }
        ]
        
        consensus_data.extend(sample_consensus)
        print(f"✅ Consensus data: raccolti {len(consensus_data)} record")
        
    except Exception as e:
        print(f"❌ Errore raccolta dati consenso: {e}")
    
    return consensus_data

def deduplicate_odds_data(all_odds_data):
    """
    Rimuove duplicati basati su game_id, bookmaker e timestamp ravvicinati.
    """
    seen = set()
    deduplicated = []
    
    for record in all_odds_data:
        key = (record['game_id'], record['bookmaker'], record['line'])
        if key not in seen:
            seen.add(key)
            deduplicated.append(record)
    
    if len(all_odds_data) != len(deduplicated):
        print(f"🧹 Rimossi {len(all_odds_data) - len(deduplicated)} duplicati")
    
    return deduplicated

def run_ingestion_cycle():
    """
    Orchestra l'intero ciclo di recupero dati.
    Implementa una strategia per gestire i limiti delle API usando multiple fonti.
    """
    print("🚀 --- Starting comprehensive data ingestion cycle ---")
    
    # 1. Recupero Quote da multiple API
    all_odds_data = []
    
    # Prima prova TheRundown (spesso più generoso con i limiti)
    rundown_odds = fetch_from_therundown()
    if rundown_odds:
        all_odds_data.extend(rundown_odds)
    
    # Poi The Odds API come integrazione/fallback
    oddsapi_odds = fetch_from_theoddsapi()
    if oddsapi_odds:
        all_odds_data.extend(oddsapi_odds)
    
    # Infine API-Sports per completare la copertura
    apisports_odds = fetch_from_apisports()
    if apisports_odds:
        all_odds_data.extend(apisports_odds)

    # Deduplicazione e salvataggio quote
    if all_odds_data:
        deduplicated_odds = deduplicate_odds_data(all_odds_data)
        save_odds_data(deduplicated_odds)
    else:
        print("⚠️ Nessun dato quote recuperato da tutte le API")

    # 2. Recupero Dati di Consenso
    consensus_data = fetch_betting_consensus()
    if consensus_data:
        save_consensus_data(consensus_data)
    else:
        print("⚠️ Nessun dato consenso recuperato")
        
    print("✅ --- Data ingestion cycle completed ---")

if __name__ == '__main__':
    # Test del modulo
    run_ingestion_cycle() 