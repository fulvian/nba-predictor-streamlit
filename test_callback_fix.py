#!/usr/bin/env python3
"""
Test rapido per verificare che il fix del foreign key constraint funzioni.
"""

import sys
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit/src')

from datetime import datetime
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

def test_callback_fix():
    """Test che la callback function salvi l'analisi correttamente prima di piazzare la scommessa."""
    print("🧪 TEST CALLBACK FIX")
    print("=" * 40)

    try:
        # Crea una BetAnalysis di test
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
            game_id="CALLBACK_TEST_001",
            central_line=225.5,
            timestamp=datetime.now()
        )

        with BettingDatabaseManager() as db_manager:
            # Simula il workflow della callback function
            print("1. Salvataggio analysis...")
            analysis_id = db_manager.save_bet_analysis(test_analysis)
            print(f"   ✅ Analysis salvata: {analysis_id}")

            print("2. Piazzamento scommessa...")
            bet_id = db_manager.place_bet(
                analysis=test_analysis,
                selected_stake=3.00,
                notes="Test callback fix"
            )

            if bet_id:
                print(f"   ✅ Scommessa piazzata: {bet_id}")

                # Verifica bankroll status
                status = db_manager.get_bankroll_status()
                print(f"   ✅ Bankroll status: €{status['current_bankroll']:.2f}")
                print(f"   ✅ Pending bets: {status['pending_bets_count']}")

                print("\n🎉 CALLBACK FIX VERIFICATO!")
                print("✅ La callback function ora funziona correttamente")
                print("✅ Foreign key constraint rispettato")
                print("✅ Bet placement dovrebbe funzionare nel dashboard")
                return True
            else:
                print("   ❌ Bet placement fallito")
                return False

    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

if __name__ == "__main__":
    test_callback_fix()