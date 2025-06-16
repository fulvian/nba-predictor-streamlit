"""
Script di test per verificare il flusso completo di calcolo del momentum.
"""
import sys
import os
import pandas as pd
from datetime import datetime, date
import time

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa le classi necessarie
from data_provider import NBADataProvider
from player_momentum_predictor import PlayerMomentumPredictor

def test_momentum_calculation():
    """Testa il calcolo del momentum per un giocatore specifico."""
    print("🚀 Avvio test del flusso di calcolo del momentum...")
    
    try:
        # 1. Inizializza il data provider
        print("\n1. Inizializzazione NBADataProvider...")
        data_provider = NBADataProvider()
        
        # 2. Inizializza il predictor di momentum
        print("\n2. Inizializzazione PlayerMomentumPredictor...")
        momentum_predictor = PlayerMomentumPredictor(
            data_dir='data',
            models_dir='models',
            nba_data_provider=data_provider
        )
        
        # 3. ID di un giocatore noto per il test (es. LeBron James)
        test_player_id = 2544  # ID di LeBron James
        
        print(f"\n3. Calcolo del punteggio di momentum per il giocatore ID: {test_player_id}")
        
        # 4. Recupera i log delle partite direttamente per verifica
        print("\n4. Recupero log partite direttamente...")
        game_logs = data_provider.get_player_game_logs(test_player_id, last_n_games=5)
        
        if game_logs is not None and not game_logs.empty:
            print(f"✅ Trovate {len(game_logs)} partite recenti:")
            print(game_logs[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK']].to_string(index=False))
        else:
            print("❌ Nessun log partite trovato. Verificare la connessione o l'ID giocatore.")
            return False
        
        # 5. Calcola il punteggio di momentum
        print("\n5. Calcolo punteggio di momentum...")
        momentum_score = momentum_predictor._get_player_momentum_score(test_player_id)
        
        print(f"\n✅ Punteggio di momentum calcolato: {momentum_score:.2f}")
        
        # 6. Verifica che il punteggio sia nel range atteso (0-100)
        if 0 <= momentum_score <= 100:
            print(f"✅ Il punteggio di momentum è nel range atteso (0-100): {momentum_score:.2f}")
            return True
        else:
            print(f"❌ Il punteggio di momentum non è nel range atteso: {momentum_score:.2f}")
            return False
            
    except Exception as e:
        print(f"\n❌ Errore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_time = time.time()
    
    if test_momentum_calculation():
        print("\n🎉 Test completato con successo!")
    else:
        print("\n❌ Test fallito. Vedere i messaggi di errore sopra per i dettagli.")
    
    end_time = time.time()
    print(f"\n⏱️ Tempo di esecuzione: {end_time - start_time:.2f} secondi")
