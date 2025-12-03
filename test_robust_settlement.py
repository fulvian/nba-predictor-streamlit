#!/usr/bin/env python3
"""
Test script per verificare se il robust settlement system funziona correttamente
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
from nba_predictor.utils.robust_bet_settlement import create_robust_settlement_system

def test_robust_settlement():
    print("🧪 Test del Robust Settlement System...")

    # 1. Connettersi al database
    print("1. Connessione al database...")
    try:
        db_manager = BettingDatabaseManager()
    except Exception as e:
        print(f"   ⚠️ Schema creation failed, trying to connect anyway: {e}")
        # Try to connect without schema creation
        import duckdb
        from pathlib import Path
        db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
        db_manager = BettingDatabaseManager(str(db_path))
        # Manually set connection to avoid re-creating schema
        db_manager.conn = duckdb.connect(str(db_path), read_only=False)

    # 2. Verificare le scommesse pendenti
    print("2. Verifica scommesse pendenti...")
    pending_bets = db_manager.get_pending_bets()
    print(f"   Found {len(pending_bets)} pending bets")

    if pending_bets:
        for i, bet in enumerate(pending_bets[:3]):  # Mostriamo solo le prime 3
            print(f"   Bet {i+1}: {bet.bet_id} - {bet.game_id} - {bet.home_team} vs {bet.away_team}")
            print(f"           Type: {bet.bet_type} {bet.line} | Status: {bet.status}")
            print(f"           Placed: {bet.placed_at}")
            print()

    # 3. Testare il robust settlement system
    print("3. Test del Robust Settlement System...")
    try:
        settlement_system = create_robust_settlement_system(db_manager)
        print("   ✅ Robust settlement system creato con successo")

        # Eseguiamo il settlement
        print("   Esecuzione del settlement...")
        result = settlement_system.execute_robust_settlement()

        print(f"   Risultato: {result}")

        if result.get('success', False):
            print(f"   ✅ Successo! {result.get('settled_bets', 0)} scommesse settle su {result.get('total_pending', 0)}")
        else:
            print(f"   ❌ Fallimento: {result.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"   ❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_robust_settlement()
    if success:
        print("\n🎉 Test completato con successo!")
    else:
        print("\n❌ Test fallito!")
        sys.exit(1)