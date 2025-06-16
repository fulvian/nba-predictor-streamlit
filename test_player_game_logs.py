"""
Script di test per verificare il funzionamento del metodo get_player_game_logs.
"""
import sys
import os
import pandas as pd
from datetime import datetime, date

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa il data provider
from data_provider import NBADataProvider

def test_get_player_game_logs():
    """Testa il recupero dei log delle partite di un giocatore."""
    print("🚀 Avvio test per get_player_game_logs...")
    
    # Inizializza il data provider
    data_provider = NBADataProvider()
    
    # ID di un giocatore noto (es. LeBron James)
    player_id = 2544  # ID di LeBron James
    
    # Stagione corrente
    current_season = data_provider._get_season_str_for_nba_api(date.today())
    
    print(f"🔍 Recupero log partite per il giocatore con ID: {player_id} (stagione: {current_season})...")
    
    try:
        # Prova a recuperare i log delle partite
        game_logs = data_provider.get_player_game_logs(player_id, season=current_season, last_n_games=5)
        
        if game_logs is not None and not game_logs.empty:
            print("✅ Test superato con successo!")
            print(f"📊 Trovate {len(game_logs)} partite recenti:")
            
            # Mostra alcune informazioni di base
            print("\nUltime 5 partite:")
            print(game_logs[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']].to_string(index=False))
            
            # Verifica che i dati abbiano il formato atteso
            required_columns = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK']
            missing_columns = [col for col in required_columns if col not in game_logs.columns]
            
            if missing_columns:
                print(f"⚠️ Attenzione: Manca/no le seguenti colonne nei dati: {', '.join(missing_columns)}")
            else:
                print("✅ Tutte le colonne richieste sono presenti nei dati.")
                
            return True
        else:
            print("❌ Test fallito: Nessun dato restituito o DataFrame vuoto.")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_get_player_game_logs()
