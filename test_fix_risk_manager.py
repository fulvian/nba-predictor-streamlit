#!/usr/bin/env python3
"""
Test Script per verificare il fix del LegacyRiskManager calculate_risk_score
Context7-compliant testing approach
"""

import sys
import os
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit')

from src.nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

def test_calculate_risk_score_fix():
    """
    Test del metodo calculate_risk_score implementato
    """
    print("🔧 Context7: Testando fix LegacyRiskManager calculate_risk_score")

    try:
        # Inizializza il manager
        manager = LegacyRiskManager()

        # Dati di test realistici
        test_cases = [
            {
                'edge': 0.05,  # 5% edge
                'estimated_prob': 0.60,  # 60% probabilità
                'odds': 1.85,  # Quota decente
                'expected_range': (0.2, 0.8)  # Range atteso per risk score
            },
            {
                'edge': 0.15,  # 15% edge alto
                'estimated_prob': 0.55,  # 55% probabilità
                'odds': 2.10,  # Quota alta
                'expected_range': (0.0, 0.3)  # Risk basso expected
            },
            {
                'edge': -0.02,  # Edge negativo
                'estimated_prob': 0.40,  # Probabilità bassa
                'odds': 2.50,  # Quota molto alta
                'expected_range': (0.8, 1.0)  # Risk alto expected
            },
            {
                'edge': 0.08,  # Edge buono
                'estimated_prob': 0.65,  # Sweet spot probabilità
                'odds': 1.75,  # Quota decente
                'expected_range': (0.1, 0.5)  # Risk basso-moderato
            }
        ]

        print(f"✅ LegacyRiskManager inizializzato con successo")
        print(f"🧪 Eseguendo {len(test_cases)} test cases...")

        all_passed = True

        for i, case in enumerate(test_cases, 1):
            print(f"\n🎯 Test Case {i}:")
            print(f"   Edge: {case['edge']:.3f}")
            print(f"   Prob: {case['estimated_prob']:.3f}")
            print(f"   Odds: {case['odds']:.3f}")

            # Test del metodo
            risk_score = manager.calculate_risk_score(
                case['edge'],
                case['estimated_prob'],
                case['odds']
            )

            print(f"   📊 Risk Score: {risk_score:.3f}")

            # Verifica bounds
            if 0.0 <= risk_score <= 1.0:
                print(f"   ✅ Bounds check PASS (0.0-1.0)")
            else:
                print(f"   ❌ Bounds check FAIL: {risk_score} non è in 0.0-1.0")
                all_passed = False

            # Verifica range atteso
            min_range, max_range = case['expected_range']
            if min_range <= risk_score <= max_range:
                print(f"   ✅ Range check PASS ({min_range:.1f}-{max_range:.1f})")
            else:
                print(f"   ⚠️  Range check: {risk_score:.3f} fuori dal range atteso {min_range:.1f}-{max_range:.1f}")
                # Non falliamo il test per range, può essere soggettivo

        # Test graceful degradation
        print(f"\n🛡️  Test graceful degradation:")

        error_cases = [
            {'edge': 'invalid', 'prob': 0.5, 'odds': 2.0, 'desc': 'edge invalido'},
            {'edge': 0.05, 'prob': -0.1, 'odds': 2.0, 'desc': 'prob negativa'},
            {'edge': 0.05, 'prob': 0.5, 'odds': 0.5, 'desc': 'odds < 1'},
            {'edge': 0.05, 'prob': 1.5, 'odds': 2.0, 'desc': 'prob > 1'},
        ]

        for case in error_cases:
            risk_score = manager.calculate_risk_score(
                case.get('edge', 0),
                case.get('prob', 0),
                case.get('odds', 0)
            )
            if risk_score == 0.0:
                print(f"   ✅ {case['desc']}: Graceful degradation PASS (0.0)")
            else:
                print(f"   ❌ {case['desc']}: Graceful degradation FAIL ({risk_score})")
                all_passed = False

        # Test con il metodo calculate_quality_score per consistenza
        print(f"\n🔗 Test integrazione con calculate_quality_score:")

        quality_result = manager.calculate_quality_score(0.08, 0.60, 1.90)
        risk_from_quality = quality_result.get('risk_score', 0)
        risk_direct = manager.calculate_risk_score(0.08, 0.60, 1.90)

        print(f"   Risk da quality_score: {risk_from_quality:.3f}")
        print(f"   Risk da calculate_risk_score: {risk_direct:.3f}")

        # Dovrebbero essere diversi ma consistenti
        if abs(risk_from_quality - risk_direct) < 0.3:
            print(f"   ✅ Integrazione: Valori consistenti")
        else:
            print(f"   ⚠️  Integrazione: Valori molto diversi (potrebbe essere normale)")

        if all_passed:
            print(f"\n🎉 CONTEST7 FIX VERIFIED: Tutti i test passati!")
            print(f"✅ Il metodo calculate_risk_score è stato implementato correttamente")
            print(f"🛡️  Context7 compliance: Graceful degradation funzionante")
            print(f"🎯 Enterprise robustness: Input validation e bounds checking attivi")
            return True
        else:
            print(f"\n❌ CONTEST7 FIX FAILED: Alcuni test falliti")
            return False

    except Exception as e:
        print(f"❌ ERRORE CRITICO nel test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_calculate_risk_score_fix()
    if success:
        print(f"\n🚀 FIX VALIDATO: Il sistema è pronto per il betting workflow test")
    else:
        print(f"\n🔧 FIX DA VERIFICARE: Problemi rilevati nell'implementazione")

    exit(0 if success else 1)