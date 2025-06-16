import http.client
import json
import os
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
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"{endpoint}?{query_string}"
    
    try:
        conn.request("GET", endpoint, headers=HEADERS)
        res = conn.getresponse()
        data = res.read().decode('utf-8')
        
        if res.status != 200:
            print(f"❌ Errore {res.status}: {res.reason}")
            print("Risposta:", data)
            return None
            
        return json.loads(data)
    except Exception as e:
        print(f"❌ Errore durante la richiesta a {endpoint}: {str(e)}")
        return None
    finally:
        conn.close()

def test_team_statistics():
    """Test per ottenere le statistiche di una squadra"""
    print("\n=== Test Statistiche Squadra ===")
    
    # Esempio: statistiche dei Boston Celtics (ID: 2) per la stagione 2023-2024
    data = make_api_request(
        "/teams/statistics",
        params={"id": 2, "season": 2023}
    )
    
    if data and 'response' in data and data['response']:
        team_stats = data['response'][0]  # Prendi il primo elemento dell'array
        print("✅ Statistiche squadra ottenute con successo")
        print(f"Squadra: {team_stats.get('team', {}).get('name', 'N/A')}")
        print(f"Partite giocate: {team_stats.get('games', 'N/A')}")
        print(f"Vittorie: {team_stats.get('wins', {}).get('total', 'N/A')}")
        print(f"Sconfitte: {team_stats.get('losses', {}).get('total', 'N/A')}")
        
        # Mostra alcune statistiche avanzate se disponibili
        if 'points' in team_stats:
            print("\n📊 Statistiche medie per partita:")
            print(f"Punti: {team_stats['points']}")
            print(f"Rimbalzi: {team_stats.get('totReb', 'N/A')}")
            print(f"Assist: {team_stats.get('assists', 'N/A')}")
    else:
        print("❌ Impossibile ottenere le statistiche della squadra")

def test_game_statistics():
    """Test per ottenere le statistiche di una partita"""
    print("\n=== Test Statistiche Partita ===")
    
    # Esempio: statistiche di una partita specifica
    data = make_api_request("/games/statistics", params={"id": 10403})
    
    if data and 'response' in data:
        print("✅ Statistiche partita ottenute con successo")
        for team in data['response']:
            print(f"\nSquadra: {team['team']['name']}")
            stats = team['statistics'][0] if team['statistics'] else {}
            print(f"Punti: {stats.get('points', 'N/A')}")
            print(f"Rimbalzi: {stats.get('totReb', 'N/A')}")
            print(f"Assist: {stats.get('assists', 'N/A')}")
    else:
        print("❌ Impossibile ottenere le statistiche della partita")

def test_team_info():
    """Test per ottenere informazioni su una squadra"""
    print("\n=== Test Info Squadra ===")
    
    # Esempio: informazioni sui Los Angeles Lakers (ID: 14)
    data = make_api_request("/teams", params={"id": 14})
    
    if data and 'response' in data and data['response']:
        team = data['response'][0]
        print("✅ Info squadra ottenute con successo")
        print(f"Nome: {team['name']}")
        print(f"Nickname: {team['nickname']}")
        print(f"Città: {team['city']}")
        print(f"Arena: {team.get('arena', {}).get('name', 'N/A')}")
    else:
        print("❌ Impossibile ottenere le informazioni della squadra")

def test_player_info():
    """Test per ottenere informazioni su un giocatore"""
    print("\n=== Test Info Giocatore ===")
    
    # Esempio: informazioni su un giocatore specifico (ID: 265 è un esempio)
    data = make_api_request("/players", params={"id": 265})
    
    if data and 'response' in data and data['response']:
        player = data['response'][0]
        print("✅ Info giocatore ottenute con successo")
        print(f"Nome: {player['firstname']} {player['lastname']}")
        print(f"Posizione: {player.get('position', 'N/A')}")
        print(f"Altezza: {player.get('height', {}).get('meters', 'N/A')}m")
        print(f"Peso: {player.get('weight', {}).get('kilograms', 'N/A')}kg")
    else:
        print("❌ Impossibile ottenere le informazioni del giocatore")

if __name__ == "__main__":
    print("=== Test Endpoint API NBA v2 ===\n")
    
    # Esegui i test
    test_team_info()
    test_team_statistics()
    test_game_statistics()
    test_player_info()
