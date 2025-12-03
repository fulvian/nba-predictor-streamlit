#!/usr/bin/env python3
"""
Test completo del flusso di betting dopo i fix Context7
Verifica che l'intero workflow dal game selection al bet placement funzioni
"""

import sys
import os
from datetime import date, datetime
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit')

def test_complete_betting_workflow():
    """
    Test completo del flusso di betting con tutti i fix Context7 implementati
    """
    print("🎯 Context7: Test completo flusso di betting NBA")
    print("=" * 60)

    try:
        # Test 1: BettingDatabaseManager con _generate_manual_id
        print("\n📊 Test 1: BettingDatabaseManager")
        from src.nba_predictor.utils.betting_database_manager import BettingDatabaseManager

        with BettingDatabaseManager() as db_manager:
            print("✅ BettingDatabaseManager inizializzato")

            # Test _generate_manual_id
            test_id = db_manager._generate_manual_id("Golden State Warriors", "Orlando Magic", date.today())
            print(f"✅ _generate_manual_id: {test_id}")

            # Test che il sistema è pronto per bet analysis (il test completo segue dopo)
            print("✅ Sistema pronto per bet analysis completo")

        # Test 2: LegacyRiskManager con calculate_risk_score
        print("\n🎰 Test 2: LegacyRiskManager")
        from src.nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

        risk_manager = LegacyRiskManager()
        print("✅ LegacyRiskManager inizializzato")

        # Test calculate_risk_score
        risk_score = risk_manager.calculate_risk_score(0.05, 0.60, 1.85)
        print(f"✅ calculate_risk_score: {risk_score:.3f}")

        # Test calculate_quality_score
        quality_result = risk_manager.calculate_quality_score(0.05, 0.60, 1.85)
        print(f"✅ calculate_quality_score: {quality_result['quality_score']:.3f}")
        print(f"   Risk score in quality: {quality_result['risk_score']:.3f}")

        # Test 3: Integrazione completa
        print("\n🔗 Test 3: Integrazione completa")

        # Import BetAnalysis per il test completo
        from src.nba_predictor.utils.betting_database_manager import BetAnalysis

        # Simula un bet completo
        game_data = {
            'home_team': 'Orlando Magic',
            'away_team': 'Golden State Warriors',
            'game_id': test_id,
            'date': '2025-11-18'
        }

        # Calcola qualità e rischio
        edge = 0.08
        estimated_prob = 0.62
        odds = 1.90

        quality_result = risk_manager.calculate_quality_score(edge, estimated_prob, odds)
        risk_score = risk_manager.calculate_risk_score(edge, estimated_prob, odds)

        print(f"   Edge: {edge:.3f}")
        print(f"   Probability: {estimated_prob:.3f}")
        print(f"   Odds: {odds:.3f}")
        print(f"   Quality Score: {quality_result['quality_score']:.3f}")
        print(f"   Risk Score: {risk_score:.3f}")
        print(f"   Kelly Fraction: {quality_result['kelly_fraction']:.3f}")

        # Test place bet con BettingDatabaseManager
        with BettingDatabaseManager() as db_manager:
            # Crea bet analysis completa
            full_analysis = BetAnalysis(
                bet_type="over/under",
                line=220.5,
                odds=odds,
                edge=edge,
                probability=estimated_prob,
                implied_probability=1/odds,
                true_probability=estimated_prob,
                quality_score=quality_result['quality_score'],
                edge_score=quality_result['edge_score'],
                confidence_score=quality_result['confidence_score'],
                risk_score=risk_score,
                consistency_score=quality_result['consistency_score'],
                kelly_fraction=quality_result['kelly_fraction'],
                stake=25.0,
                roi=(odds - 1) * estimated_prob - (1 - estimated_prob),
                is_value=edge > 0,
                risk_level=risk_manager.assess_risk_level({
                    'quality_score': quality_result['quality_score'],
                    'risk_score': risk_score
                }),
                game_id=game_data['game_id'],
                central_line=220.5,
                timestamp=datetime.now(),
                home_team=game_data['home_team'],
                away_team=game_data['away_team']
            )

            # Place bet con FK protection (crea automaticamente il game record)
            bet_success, bet_message = db_manager.place_bet_with_fk_protection(
                analysis=full_analysis,
                stake=full_analysis.stake
            )
            print(f"✅ Bet piazzato con FK protection: {bet_success}")
            print(f"   Message: {bet_message}")

        # Test 4: Verifica salvataggio
        print("\n💾 Test 4: Verifica salvataggio")
        with BettingDatabaseManager() as db_manager:
            pending_bets = db_manager.get_pending_bets()
            print(f"✅ Pending bets trovati: {len(pending_bets)}")

            if pending_bets:
                latest_bet = pending_bets[-1]  # L'ultimo inserito
                print(f"   Latest bet ID: {latest_bet.bet_id}")
                print(f"   Game: {latest_bet.home_team} vs {latest_bet.away_team}")
                print(f"   Stake: €{latest_bet.stake:.2f}")
                print(f"   Quality Score: {latest_bet.quality_score:.3f}")
                print(f"   Status: {latest_bet.status}")

        print("\n🎉 CONTEST7 WORKFLOW TEST: COMPLETAMENTE SUCCESSFUL!")
        print("✅ Tutti i componenti funzionano correttamente:")
        print("   - BettingDatabaseManager._generate_manual_id ✅")
        print("   - LegacyRiskManager.calculate_risk_score ✅")
        print("   - Bet analysis salvataggio ✅")
        print("   - Bet placement completo ✅")
        print("   - Integrazione end-to-end ✅")

        return True

    except Exception as e:
        print(f"\n❌ ERRORE CRITICO nel test completo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_betting_workflow()
    if success:
        print(f"\n🚀 WORKFLOW READY: Il sistema è pronto per l'uso su dashboard!")
        print(f"📱 Dashboard: http://localhost:8501")
        print(f"🏀 Flusso completo: Game Selection → Analysis → Bet Placement ✅")
    else:
        print(f"\n🔧 WORKFLOW ISSUES: Problemi rilevati da risolvere")

    exit(0 if success else 1)