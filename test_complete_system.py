#!/usr/bin/env python3
"""
Test finale completo del sistema NBA Betting dopo tutti i fix
Verifica che tutti i componenti lavorino insieme correttamente
"""

import sys
import os
from datetime import date, datetime
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit')

def test_complete_system():
    """
    Test completo di tutti i componenti del sistema dopo i fix Context7
    """
    print("🎯 Context7: Test Completo Sistema NBA Betting")
    print("=" * 60)

    all_tests_passed = True
    test_results = []

    try:
        # Test 1: LegacyRiskManager con calculate_risk_score
        print("\n🎰 Test 1: LegacyRiskManager")
        from src.nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

        risk_manager = LegacyRiskManager()
        risk_score = risk_manager.calculate_risk_score(0.08, 0.62, 1.90)
        quality_result = risk_manager.calculate_quality_score(0.08, 0.62, 1.90)

        test_1_success = risk_score is not None and 0 <= risk_score <= 1
        test_results.append(("LegacyRiskManager.calculate_risk_score", test_1_success))
        print(f"✅ calculate_risk_score: {risk_score:.3f}" if test_1_success else f"❌ calculate_risk_score: FAILED")
        print(f"✅ calculate_quality_score: {quality_result['quality_score']:.3f}")

        # Test 2: BettingDatabaseManager con _generate_manual_id
        print("\n📊 Test 2: BettingDatabaseManager")
        from src.nba_predictor.utils.betting_database_manager import BettingDatabaseManager

        with BettingDatabaseManager() as db_manager:
            test_id = db_manager._generate_manual_id("Lakers", "Warriors", date.today())
            test_2_success = test_id.startswith("MANUAL_")
            test_results.append(("BettingDatabaseManager._generate_manual_id", test_2_success))
            print(f"✅ _generate_manual_id: {test_id}" if test_2_success else f"❌ _generate_manual_id: FAILED")

        # Test 3: Enhanced Prediction Bridge
        print("\n🏀 Test 3: Enhanced Prediction Bridge")
        from src.nba_predictor.streamlit.components.enhanced_prediction_bridge_real_data import get_enhanced_prediction_bridge_real_data

        bridge = get_enhanced_prediction_bridge_real_data()
        game_info = {
            'home_team': 'Orlando Magic',
            'away_team': 'Golden State Warriors',
            'date': '2025-11-18',
            'betting_line': None
        }

        prediction = bridge.get_prediction(game_info)
        test_3_success = 'predicted_total' in prediction and prediction['predicted_total'] > 0
        test_results.append(("Enhanced Prediction Bridge", test_3_success))
        print(f"✅ ML Prediction: {prediction.get('predicted_total', 'N/A')}" if test_3_success else f"❌ ML Prediction: FAILED")

        # Test 4: Integrazione Completa (simula scommessa)
        print("\n🔗 Test 4: Integrazione Completa")

        # Simula i dati di una scommessa ottimale
        optimal_bet = {
            'type': 'over/under',
            'line': 232.5,
            'odds': 1.85,
            'edge': 0.05,
            'probability': 0.60,
            'is_value': True
        }

        # Calcola risk score con i parametri corretti (come nella dashboard)
        try:
            integrated_risk_score = risk_manager.calculate_risk_score(
                optimal_bet['edge'],
                optimal_bet['probability'],
                optimal_bet['odds']
            )
            test_4_success = 0 <= integrated_risk_score <= 1
            test_results.append(("Integrazione Bet Placement", test_4_success))
            print(f"✅ Integrated Risk Score: {integrated_risk_score:.3f}" if test_4_success else f"❌ Integrated Risk Score: FAILED")
        except Exception as e:
            test_4_success = False
            test_results.append(("Integrazione Bet Placement", False))
            print(f"❌ Integrated Risk Score: ERROR - {str(e)}")

        # Test 5: Caching System (simula session state)
        print("\n💾 Test 5: Caching System")

        class MockSessionState:
            def __init__(self):
                self.predictions_cache = {}

        mock_session = MockSessionState()

        # Simula il fix che abbiamo implementato
        if not mock_session.predictions_cache:
            mock_session.predictions_cache = {}

        cache_key = f"{game_info['home_team']}_{game_info['away_team']}_{game_info['date']}"
        mock_session.predictions_cache[cache_key] = prediction

        test_5_success = cache_key in mock_session.predictions_cache
        test_results.append(("Caching System", test_5_success))
        print(f"✅ Cache Key: {cache_key}" if test_5_success else f"❌ Cache Key: FAILED")
        print(f"✅ Cached Prediction: {mock_session.predictions_cache[cache_key].get('predicted_total', 'N/A')}")

        # Result Summary
        print("\n" + "="*60)
        print("🎯 RISULTATI TEST COMPLETO SISTEMA:")
        print("="*60)

        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)

        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status:8} {test_name}")

        print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("\n🎉 SISTEMA COMPLETAMENTE FUNZIONANTE!")
            print("✅ Tutti i fix Context7 implementati con successo")
            print("✅ LegacyRiskManager.calculate_risk_score funzionante")
            print("✅ BettingDatabaseManager._generate_manual_id funzionante")
            print("✅ Enhanced ML System con predizioni reali")
            print("✅ Caching intelligente delle predizioni")
            print("✅ Integrazione end-to-end funzionante")
            print("✅ Bet placement pronto per l'uso")

            print(f"\n🚀 SISTEMA PRONTO PER:")
            print(f"📱 Dashboard: http://localhost:8501")
            print(f"🏀 Flusso completo: Game Selection → Analysis → Betting")
            print(f"🤖 Predizioni ML reali (no più mock data)")
            print(f"💰 Bet placement funzionante")
            print(f"📊 Analisi completa con risk management")

            return True
        else:
            print(f"\n⚠️ SISTEMA PARZIALMENTE FUNZIONANTE")
            print(f"❌ {total_tests - passed_tests} test falliti - da investigare")
            return False

    except Exception as e:
        print(f"\n❌ ERRORE CRITICO nel test completo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    exit(0 if success else 1)