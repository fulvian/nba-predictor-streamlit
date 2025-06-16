"""
Script per testare il recupero delle stagioni disponibili dall'API NBA.
"""

import sys
import os
import http.client
import json
from datetime import datetime, timedelta

# Aggiungi la directory principale al path per permettere l'importazione dei moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carica la chiave API dalle variabili d'ambiente
API_KEY = os.getenv('APISPORTS_API_KEY')
if not API_KEY:
    print("Errore: Impostare la variabile d'ambiente APISPORTS_API_KEY")
    sys.exit(1)

def get_available_seasons():
    """Recupera le stagioni disponibili dall'API."""
    conn = http.client.HTTPSConnection("v1.basketball.api-sports.io")
    headers = {
        'x-rapidapi-host': "v1.basketball.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    
    print("🔍 Richiesta delle stagioni disponibili...")
    conn.request("GET", "/seasons", headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    if res.status != 200:
        print(f"❌ Errore nella richiesta: {res.status} {res.reason}")
        print("Risposta:", data.decode("utf-8"))
        return []
    
    result = json.loads(data)
    
    if 'response' not in result or not result['response']:
        print("❌ Nessun dato di stagione trovato nella risposta")
        return []
    
    seasons = result['response']
    print(f"✅ Trovate {len(seasons)} stagioni:")
    for season in seasons:
        print(f"- {season}")
    
    return seasons

def get_teams(season):
    """Recupera le squadre per una stagione specifica."""
    conn = http.client.HTTPSConnection("v1.basketball.api-sports.io")
    headers = {
        'x-rapidapi-host': "v1.basketball.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    
    print(f"\n🔍 Richiesta delle squadre per la stagione {season}...")
    conn.request("GET", f"/teams?league=12&season={season}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    if res.status != 200:
        print(f"❌ Errore nella richiesta: {res.status} {res.reason}")
        print("Risposta:", data.decode("utf-8"))
        return []
    
    result = json.loads(data)
    
    if 'response' not in result or not result['response']:
        print(f"❌ Nessuna squadra trovata per la stagione {season}")
        return []
    
    teams = result['response']
    print(f"✅ Trovate {len(teams)} squadre per la stagione {season}:")
    for team in teams[:5]:  # Mostra solo le prime 5 squadre per brevità
        print(f"- {team['name']} (ID: {team['id']})")
    if len(teams) > 5:
        print(f"- ... e altre {len(teams) - 5} squadre")
    
    return teams

def get_games(season, days_back=90):
    """
    Recupera le partite per una stagione specifica.
    
    Args:
        season (str): La stagione nel formato 'YYYY-YY' o 'YYYY'
        days_back (int): Numero di giorni precedenti alla data corrente da includere
    """
    conn = http.client.HTTPSConnection("v1.basketball.api-sports.io")
    headers = {
        'x-rapidapi-host': "v1.basketball.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    
    # Calcola la data di inizio (oggi - days_back) e fine (oggi - 1 giorno)
    today = datetime.now().date()
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")  # Ieri
    start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")  # days_back giorni fa
    
    print(f"\n🔍 Richiesta delle partite dal {start_date} al {end_date}...")
    
    # Prova con diversi endpoint e parametri
    endpoints = [
        f"/games?league=12&season={season}&date={start_date}-{end_date}",
        f"/games?league=12&season={season}&timezone=Europe/Rome",
        f"/games?league=12&season={season}",
        f"/games?league=12&season={season}&status=FT"  # Solo partite terminate
    ]
    
    for endpoint in endpoints:
        print(f"  🔄 Provo endpoint: {endpoint.split('?')[0]}...")
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        
        if res.status != 200:
            print(f"  ❌ Errore {res.status}: {res.reason}")
            continue
            
        result = json.loads(data)
        
        if 'response' in result and result['response']:
            games = result['response']
            print(f"  ✅ Trovate {len(games)} partite!")
            return games
    
    # Se nessun endpoint ha funzionato, restituisci lista vuota
    print("  ❌ Nessun dato trovato con nessun endpoint")
    return []

if __name__ == "__main__":
    print("=== Test API NBA Sports ===\n")
    
    # 1. Recupera le stagioni disponibili
    seasons = get_available_seasons()
    
    if not seasons:
        print("\nImpossibile recuperare le stagioni. Verifica la connessione e la chiave API.")
        sys.exit(1)
    
    # Funzione per convertire la stagione in un anno intero per l'ordinamento
    def get_season_year(season):
        try:
            if isinstance(season, int):
                return season
            if isinstance(season, str) and '-' in season:
                return int(season.split('-')[0])
            return int(season)
        except (ValueError, TypeError, AttributeError):
            return 0
    
    # Prova con le stagioni NBA più recenti in ordine di probabilità
    test_seasons = [
        "2024-2025",  # Formato stagione corrente
        "2024",        # Solo anno
        "2023-2024",  # Stagione precedente
        "2023"
    ]
    
    for season in test_seasons:
        print(f"\n=== TEST STAGIONE: {season} ===")
        # 1. Prova a recuperare le squadre
        teams = get_teams(season)
        
        if teams:
            # 2. Se ci sono squadre, prova a recuperare le partite
            games = get_games(season)
            
            if games:
                print(f"\n✅ TROVATI DATI PER LA STAGIONE: {season}")
                print(f"Utilizzare '{season}' come parametro della stagione.")
                break
    else:
        print("\n❌ Impossibile trovare partite per nessuna delle stagioni testate.")
        print("Verifica la connessione e la chiave API, o contatta il supporto di api-sports.io")
