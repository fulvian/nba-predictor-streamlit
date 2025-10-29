#!/usr/bin/env python3
"""
Debug script per verificare il comportamento del pulsante "Piazza Scommessa"
"""

import sys
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit/src')

from datetime import datetime
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis
from nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

def debug_bet_placement():
    """Debug completo del processo di scommessa."""
    print("🔍 DEBUG BET PLACEMENT COMPLETO")
    print("=" * 50)

    # 1. Creazione analisi di test
    print("\n📝 Step 1: Creazione BetAnalysis...")
    test_analysis = BetAnalysis(
        bet_type="Over",
        line=225.5,
        odds=1.85,
        edge=3.2,
        probability=0.54,
        implied_probability=0.54,
        true_probability=0.57,
        quality_score=0.75,
        edge_score=0.65,
        confidence_score=0.80,
        risk_score=0.45,
        consistency_score=0.90,
        kelly_fraction=0.02,
        stake=2.0,
        roi=12.5,
        is_value=True,
        risk_level="Medium",
        game_id="DEBUG_GAME_001",
        central_line=225.5,
        timestamp=datetime.now()
    )
    print(f"✅ BetAnalysis creata: {test_analysis.bet_type} {test_analysis.line} @ {test_analysis.odds}")

    # 2. Test salvataggio analisi
    print("\n💾 Step 2: Salvataggio analisi...")
    try:
        with BettingDatabaseManager() as db_manager:
            analysis_id = db_manager.save_bet_analysis(test_analysis)
            print(f"✅ Analisi salvata con ID: {analysis_id}")
    except Exception as e:
        print(f"❌ Errore salvataggio analisi: {e}")
        return False

    # 3. Test piazzamento scommessa
    print("\n💰 Step 3: Piazzamento scommessa...")
    try:
        with BettingDatabaseManager() as db_manager:
            bet_id = db_manager.place_bet(
                analysis=test_analysis,
                selected_stake=3.50,
                notes="Debug test bet"
            )
            print(f"✅ Scommessa piazzata con ID: {bet_id}")

            # Verifica bankroll dopo piazzamento
            bankroll_status = db_manager.get_bankroll_status()
            print(f"💳 Bankroll attuale: €{bankroll_status.get('current_bankroll', 0):.2f}")
            print(f"📊 Pending bets: {bankroll_status.get('pending_bets_count', 0)}")

    except Exception as e:
        print(f"❌ Errore piazzamento scommessa: {e}")
        return False

    # 4. Verifica dati salvati
    print("\n🔍 Step 4: Verifica dati salvati...")
    try:
        with BettingDatabaseManager() as db_manager:
            # Verifica placed_bets
            pending_bets = db_manager.get_pending_bets()
            print(f"📋 Pending bets trovati: {len(pending_bets)}")

            if pending_bets:
                latest_bet = pending_bets[0]
                print(f"✅ Ultima scommessa: {latest_bet.bet_type} {latest_bet.line} @ {latest_bet.odds}")
                print(f"💰 Stake: €{latest_bet.stake}, Status: {latest_bet.status}")

            # Verifica bankroll_history
            conn = db_manager.conn
            history_count = conn.execute("SELECT COUNT(*) FROM bankroll_history").fetchone()[0]
            print(f"📈 Record bankroll_history: {history_count}")

            if history_count > 0:
                latest_history = conn.execute("""
                    SELECT change_type, amount, balance_before, balance_after, notes
                    FROM bankroll_history
                    ORDER BY created_at DESC
                    LIMIT 1
                """).fetchone()
                print(f"✅ Ultimo record: {latest_history[0]} €{latest_history[1]:.2f}")
                print(f"   Before: €{latest_history[2]:.2f} → After: €{latest_history[3]:.2f}")
                print(f"   Note: {latest_history[4]}")

    except Exception as e:
        print(f"❌ Errore verifica dati: {e}")
        return False

    print("\n🎉 DEBUG COMPLETATO CON SUCCESSO!")
    print("✅ Tutti i passaggi funzionano correttamente")
    return True

if __name__ == "__main__":
    debug_bet_placement()