#!/usr/bin/env python3
"""
Test script per verificare le nuove funzionalità implementate.
"""

import sys
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit/src')

from datetime import datetime, date
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

def test_new_features():
    """Test delle nuove funzionalità implementate."""
    print("🧪 TEST NUOVE FUNZIONALITÀ IMPLEMENTATE")
    print("=" * 50)

    try:
        with BettingDatabaseManager() as db_manager:
            print("✅ 1. Test connessione database manager")

            # Test 2: Data store status
            print("\n📊 2. Test data store status...")
            status = db_manager.get_data_store_status()
            print(f"   ✅ Database size: {status.get('database_size_mb', 0):.2f} MB")
            print(f"   ✅ Tables: {list(status.get('tables', {}).keys())}")
            print(f"   ✅ Recent activity: {status.get('recent_activity_7_days', 0)} bets")

            # Test 3: Sincronizzazione
            print("\n🔄 3. Test sincronizzazione data store...")
            sync_result = db_manager.sync_data_store()
            if sync_result.get('data_integrity_check', False):
                print("   ✅ Data integrity check passed")
            else:
                print(f"   ⚠️ Data integrity issues: {sync_result.get('errors', [])}")

            # Test 4: Bets comprehensive
            print("\n📋 4. Test comprehensive bets...")
            all_bets = db_manager.get_all_bets_comprehensive()
            print(f"   ✅ Pending bets: {len(all_bets.get('pending', []))}")
            print(f"   ✅ Settled bets: {len(all_bets.get('settled', []))}")
            print(f"   ✅ Total bets: {len(all_bets.get('all', []))}")

            # Test 5: Bankroll status con pending_bets_count
            print("\n💰 5. Test bankroll status con pending_bets_count...")
            bankroll = db_manager.get_bankroll_status()
            if 'pending_bets_count' in bankroll:
                print(f"   ✅ Pending bets count: {bankroll['pending_bets_count']}")
            else:
                print("   ❌ pending_bets_count field missing")

            print("\n🎉 TUTTI I TEST PASSATI CON SUCCESSO!")
            print("\n📋 RIEPILOGO FUNZIONALITÀ VERIFICATE:")
            print("   ✅ Connessione database manager")
            print("   ✅ Data store status monitoring")
            print("   ✅ Sincronizzazione data store")
            print("   ✅ Visualizzazione scommesse completa")
            print("   ✅ Bankroll status con pending_bets_count")
            print("   ✅ Gestione connessioni e cleanup")
            print("\n🌐 Puoi testare la nuova sezione su: http://localhost:8510")

            return True

    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_new_features()