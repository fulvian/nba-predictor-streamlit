import http.client
import json
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni la chiave API dalle variabili d'ambiente
API_KEY = os.getenv('APISPORTS_API_KEY')
if not API_KEY:
    print("❌ Errore: Imposta la variabile d'ambiente APISPORTS_API_KEY nel file .env")
    exit(1)

def get_game_statistics(game_id):
    """Recupera le statistiche dettagliate di una partita"""
    conn = http.client.HTTPSConnection("v2.nba.api-sports.io")
    
    headers = {
        'x-rapidapi-host': "v2.nba.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    
    print(f"\n🔍 Recupero statistiche per la partita ID: {game_id}...")
    conn.request("GET", f"/games/statistics?id={game_id}", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    if res.status != 200:
        print(f"❌ Errore {res.status}: {res.reason}")
        return None
    
    return json.loads(data)

def test_v2_api():
    """Test della connessione all'API v2"""
    conn = http.client.HTTPSConnection("v2.nba.api-sports.io")
    
    headers = {
        'x-rapidapi-host': "v2.nba.api-sports.io",
        'x-rapidapi-key': API_KEY
    }
    
    # Prova a recuperare le partite recenti
    print("🔍 Prova a recuperare le partite recenti...")
    conn.request("GET", "/games?season=2024", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    if res.status != 200:
        print(f"❌ Errore {res.status}: {res.reason}")
        print("Risposta:", data.decode("utf-8"))
        return
    
    result = json.loads(data)
    
    if 'response' not in result:
        print("❌ Formato della risposta non valido")
        print("Risposta:", result)
        return
    
    games = result.get('response', [])
    print(f"✅ Trovate {len(games)} partite")
    
    # Seleziona le ultime 5 partite finite
    finished_games = [g for g in games if g.get('status', {}).get('short') == '3']
    
    if not finished_games:
        print("❌ Nessuna partita terminata trovata")
        return
    
    # Mostra le ultime 5 partite finite
    for i, game in enumerate(finished_games[:5]):
        game_date = game.get('date', {})
        if isinstance(game_date, dict):
            date_str = game_date.get('date', 'N/A')
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
        else:
            date_str = str(game_date)
        
        game_id = game.get('id')
        
        teams = game.get('teams', {})
        home_team = teams.get('home', {}).get('name', 'N/A') if isinstance(teams, dict) else 'N/A'
        away_team = teams.get('away', {}).get('name', 'N/A') if isinstance(teams, dict) else 'N/A'
        
        scores = game.get('scores', {})
        home_score = scores.get('home', {}).get('total', 'N/A') if isinstance(scores, dict) else 'N/A'
        away_score = scores.get('away', {}).get('total', 'N/A') if isinstance(scores, dict) else 'N/A'
        
        print(f"\n📅 Partita {i+1} - {date_str} (ID: {game_id})")
        print(f"   {home_team} {home_score} - {away_score} {away_team}")
        print(f"   Status: {game.get('status', {}).get('long', 'N/A')}")
        
        # Recupera le statistiche dettagliate per la partita
        if game_id:
            stats = get_game_statistics(game_id)
            if stats and 'response' in stats and stats['response']:
                print("\n   📊 Statistiche della partita:")
                for team_stats in stats['response']:
                    team_name = team_stats.get('team', {}).get('name', 'Sconosciuto')
                    print(f"   Squadra: {team_name}")
                    
                    statistics = team_stats.get('statistics', [])
                    if statistics and len(statistics) > 0:
                        stats_data = statistics[0]
                        print(f"   Punti: {stats_data.get('points', 'N/A')}")
                        print(f"   Rimbalzi: {stats_data.get('totReb', 'N/A')}")
                        print(f"   Assist: {stats_data.get('assists', 'N/A')}")
                        print(f"   Punti da 3: {stats_data.get('tpm', 'N/A')}/{stats_data.get('tpa', 'N/A')} ({stats_data.get('tpp', 'N/A')}%)")
                        print(f"   Tiri liberi: {stats_data.get('ftm', 'N/A')}/{stats_data.get('fta', 'N/A')} ({stats_data.get('ftp', 'N/A')}%)")
                        print(f"   Punti in area: {stats_data.get('pointsInPaint', 'N/A')}")
                        print(f"   Punti da sottobanco: {stats_data.get('benchPoints', 'N/A')}")
                        print("   ---")
    
    if len(finished_games) > 5:
        print(f"\n... e altre {len(finished_games) - 5} partite terminate")

if __name__ == "__main__":
    print("=== Test API NBA v2 ===\n")
    test_v2_api()
