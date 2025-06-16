import os
import json
import requests
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

def test_leagues_endpoint():
    """Test per l'endpoint delle leghe NBA"""
    url = "https://v2.nba.api-sports.io/leagues"
    
    # Usa la chiave API dalle variabili d'ambiente
    api_key = os.getenv('APISPORTS_API_KEY')
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v2.nba.api-sports.io'
    }
    
    print(f"\n🔍 Invio richiesta a: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Solleva un'eccezione per risposte HTTP non riuscite
        
        print(f"✅ Status Code: {response.status_code}")
        
        # Stampa la risposta formattata
        data = response.json()
        print("\n📄 Risposta JSON formattata:")
        print(json.dumps(data, indent=2))
        
        # Analisi della risposta
        if 'response' in data and data['response']:
            print("\n🏀 Leghe trovate:")
            for league in data['response']:
                if isinstance(league, dict):
                    print(f"- {league.get('name', 'N/A')} (ID: {league.get('id', 'N/A')}, Tipo: {league.get('type', 'N/A')})")
                else:
                    print(f"- {league} (formato non previsto)")
        else:
            print("⚠️ Nessun dato di lega trovato nella risposta")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore durante la richiesta: {e}")
    except json.JSONDecodeError:
        print("❌ Impossibile decodificare la risposta JSON")

if __name__ == "__main__":
    print("=== Test Endpoint Leghe NBA v2 ===")
    test_leagues_endpoint()
