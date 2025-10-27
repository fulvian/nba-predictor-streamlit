# odds_ingestion_worker.py

import requests
from config import THE_ODDS_API_KEY, TARGET_REGIONS, MARKET_TO_MONITOR
from database_manager import save_odds_data

# Costanti per l'API
API_URL = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'

def fetch_nba_odds():
    """
    Chiama The Odds API per ottenere le quote per il mercato 'totals'.
    Questo è il punto in cui implementare la rotazione delle API
    per aggirare i limiti del piano gratuito, come descritto nei documenti.
    """
    params = {
        'apiKey': THE_ODDS_API_KEY,
        'regions': TARGET_REGIONS,
        'markets': MARKET_TO_MONITOR,
        'oddsFormat': 'decimal',
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status() # Lancia un'eccezione per errori HTTP
        api_data = response.json()
        print(f"API Usage: {response.headers.get('x-requests-remaining')} requests remaining.")
        return api_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from The Odds API: {e}")
        return None

def process_and_store_odds():
    """
    Orchestra il processo: recupera i dati, li formatta e li salva.
    """
    raw_odds_data = fetch_nba_odds()
    if not raw_odds_data:
        print("No data fetched. Skipping save process.")
        return

    processed_data = []
    for game in raw_odds_data:
        game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']

        for bookmaker in game['bookmakers']:
            bookie_key = bookmaker['key']
            for market in bookmaker['markets']:
                if market['key'] == MARKET_TO_MONITOR:
                    # Assumiamo che ci sia sempre un over e un under
                    line = market['outcomes'][0]['point'] # La linea (es. 215.5)
                    odds_over = next((o['price'] for o in market['outcomes'] if o['name'] == 'Over'), None)
                    odds_under = next((o['price'] for o in market['outcomes'] if o['name'] == 'Under'), None)

                    if odds_over and odds_under:
                        processed_data.append({
                            'game_id': game_id,
                            'sport_key': game['sport_key'],
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookie_key,
                            'market_key': market['key'],
                            'line': line,
                            'odds_over': odds_over,
                            'odds_under': odds_under,
                        })

    if processed_data:
        save_odds_data(processed_data)
    else:
        print("No relevant odds found in the API response.")

if __name__ == '__main__':
    # Per testare il modulo singolarmente
    process_and_store_odds() 