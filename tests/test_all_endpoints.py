import http.client
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione
API_KEY = os.getenv('APISPORTS_API_KEY')
BASE_URL = "v2.nba.api-sports.io"
HEADERS = {
    'x-rapidapi-host': BASE_URL,
    'x-rapidapi-key': API_KEY
}

def make_api_request(endpoint, params=None):
    """Esegue una richiesta all'API e restituisce i dati JSON"""
    conn = http.client.HTTPSConnection(BASE_URL)
    
    # Aggiungi i parametri all'endpoint se presenti
    if params:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items() if v is not None])
        endpoint = f"{endpoint}?{query_string}"
    
    try:
        print(f"\n🔍 Richiesta a: {endpoint}")
        conn.request("GET", endpoint, headers=HEADERS)
        res = conn.getresponse()
        data = res.read().decode('utf-8')
        
        if res.status != 200:
            print(f"❌ Errore {res.status}: {res.reason}")
            print("Risposta:", data)
            return None
            
        return json.loads(data)
    except Exception as e:
        print(f"❌ Errore durante la richiesta: {str(e)}")
        return None
    finally:
        conn.close()

def test_seasons():
    """Test per ottenere le stagioni disponibili"""
    print("\n=== Test Stagioni ===")
    data = make_api_request("/seasons")
    
    if data and 'response' in data:
        print("✅ Stagioni ottenute con successo")
        for season in data['response']:
            print(f"- {season}")
    else:
        print("❌ Impossibile ottenere le stagioni")

def test_leagues():
    """Test per ottenere le leghe disponibili"""
    print("\n=== Test Leghe ===")
    data = make_api_request("/leagues")
    
    if data and 'response' in data:
        print("✅ Leghe ottenute con successo")
        if isinstance(data['response'], list):
            for league in data['response']:
                if isinstance(league, dict):
                    print(f"- {league.get('name', 'N/A')} (Tipo: {league.get('type', 'N/A')})")
                else:
                    print(f"- {league} (formato non previsto)")
        else:
            print("⚠️ Formato risposta non previsto per le leghe")
            print(f"Dettagli: {data['response']}")
    else:
        print("❌ Impossibile ottenere le leghe")

def test_teams():
    """Test per ottenere le squadre"""
    print("\n=== Test Squadre ===")
    data = make_api_request("/teams")
    
    if data and 'response' in data:
        print(f"✅ Trovate {len(data['response'])} squadre")
        for team in data['response'][:5]:  # Mostra solo le prime 5
            print(f"- {team.get('name')} (ID: {team.get('id')})")
    else:
        print("❌ Impossibile ottenere le squadre")

def test_games():
    """Test per ottenere le partite di una data specifica"""
    print("\n=== Test Partite per Data ===")
    # Usa la data di ieri
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    data = make_api_request("/games", {"date": yesterday})
    
    if data and 'response' in data:
        games = data['response']
        print(f"✅ Trovate {len(games)} partite per il {yesterday}")
        
        for game in games[:3]:  # Mostra solo le prime 3 partite
            home = game.get('teams', {}).get('home', {}).get('name', 'N/A')
            away = game.get('teams', {}).get('away', {}).get('name', 'N/A')
            score_home = game.get('scores', {}).get('home', {}).get('total', 'N/A')
            score_away = game.get('scores', {}).get('away', {}).get('total', 'N/A')
            print(f"- {home} {score_home} - {score_away} {away}")
    else:
        print(f"❌ Nessuna partita trovata per il {yesterday}")

def test_standings():
    """Test per ottenere la classifica"""
    print("\n=== Test Classifica ===")
    data = make_api_request("/standings", {"league": "standard", "season": 2023})
    
    if data and 'response' in data:
        print("✅ Classifica ottenuta con successo")
        
        for conference in ['east', 'west']:
            print(f"\n🏆 Conference {conference.upper()}:")
            for team in data['response']:
                if team.get('conference', {}).get('name', '').lower() == conference:
                    print(f"{team.get('position')}. {team.get('team', {}).get('name')} - "
                          f"V: {team.get('win')} P: {team.get('loss')} ({team.get('winPercentage')}%)")
    else:
        print("❌ Impossibile ottenere la classifica")

if __name__ == "__main__":
    print("=== Test Completo Endpoint API NBA v2 ===\n")
    
    # Esegui tutti i test
    test_seasons()
    test_leagues()
    test_teams()
    test_games()
    test_standings()
