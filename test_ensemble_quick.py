#!/usr/bin/env python3
"""
🧪 Quick Test NBA Ensemble Predictor - TensorFlow Verification

Test rapido per verificare che TensorFlow sia installato e l'ensemble predictor sia inizializzato.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "nba_predictor" / "streamlit" / "components"))

def test_tensorflow_availability():
    """Test 1: Verifica che TensorFlow sia disponibile"""
    print("\n🧪 Test 1: Verifica TensorFlow Availability")
    print("=" * 50)

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow importato con successo")
        print(f"   - Versione: {tf.__version__}")
        print(f"   - Keras disponibile: {hasattr(tf, 'keras')}")
        print(f"   - Sequential disponibile: {hasattr(tf.keras.models, 'Sequential')}")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow non disponibile: {e}")
        return False

def test_ensemble_predictor_initialization():
    """Test 2: Verifica inizializzazione del NBA Ensemble Predictor"""
    print("\n🧪 Test 2: Inizializzazione NBA Ensemble Predictor")
    print("=" * 60)

    try:
        from ml_integration_bridge import MLIntegrationBridge

        print("🔄 Creazione ML Integration Bridge...")
        start_time = time.time()

        # Crea bridge con timeout breve
        bridge = MLIntegrationBridge(
            health_check_interval=5,
            max_retries=1,
            cache_ttl_minutes=1
        )

        init_time = (time.time() - start_time) * 1000
        print(f"✅ Bridge inizializzato in {init_time:.2f}ms")

        # Verifica che l'ensemble predictor sia stato inizializzato
        ensemble_predictor = bridge.get_ensemble_predictor()

        if ensemble_predictor is not None:
            print("✅ NBA Ensemble Predictor inizializzato correttamente")
            print(f"   - Tipo: {type(ensemble_predictor).__name__}")
            print(f"   - Metodi disponibili: {hasattr(ensemble_predictor, 'predict')}")

            # Verifica attributi chiave
            attrs_to_check = ['ensemble_method', 'enable_bayesian_optimization', 'is_trained']
            for attr in attrs_to_check:
                if hasattr(ensemble_predictor, attr):
                    value = getattr(ensemble_predictor, attr)
                    print(f"   - {attr}: {value}")

            # Pulizia
            bridge.cleanup()
            return True
        else:
            print("❌ NBA Ensemble Predictor non inizializzato")
            bridge.cleanup()
            return False

    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_neural_availability():
    """Test 3: Verifica disponibilità XGBoost e Neural Network"""
    print("\n🧪 Test 3: Disponibilità Modelli")
    print("=" * 40)

    xgboost_available = False
    neural_available = False

    try:
        import xgboost as xgb
        print("✅ XGBoost disponibile")
        print(f"   - Versione: {xgb.__version__}")
        xgboost_available = True
    except ImportError:
        print("❌ XGBoost non disponibile")

    try:
        import tensorflow as tf
        print("✅ TensorFlow/Keras disponibile per Neural Network")
        neural_available = True
    except ImportError:
        print("❌ TensorFlow/Keras non disponibile per Neural Network")

    print(f"\n📊 Status Modelli:")
    print(f"   - XGBoost: {'✅' if xgboost_available else '❌'}")
    print(f"   - Neural Network: {'✅' if neural_available else '❌'}")

    return xgboost_available and neural_available

def run_quick_test():
    """Esegue test rapido dell'ensemble predictor"""
    print("🧪 QUICK TEST NBA ENSEMBLE PREDICTOR")
    print("=" * 50)
    print("Verifica rapida installazione TensorFlow e inizializzazione Ensemble")
    print("=" * 50)

    tests = [
        ("TensorFlow Availability", test_tensorflow_availability),
        ("Ensemble Predictor Initialization", test_ensemble_predictor_initialization),
        ("XGBoost + Neural Network Availability", test_xgboost_neural_availability)
    ]

    results = []
    total_tests = len(tests)
    passed_tests = 0

    for test_name, test_func in tests:
        print(f"\n🔍 Eseguendo: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                passed_tests += 1
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary finale
    print("\n" + "=" * 50)
    print("🎉 RIEPILOGO QUICK TEST")
    print("=" * 50)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")

    print(f"\n📊 Risultati finali:")
    print(f"   - Tests eseguiti: {total_tests}")
    print(f"   - Tests superati: {passed_tests}")
    print(f"   - Tests falliti: {total_tests - passed_tests}")
    print(f"   - Success rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print(f"\n🎉 TUTTI I TEST SUPERATI!")
        print(f"✅ Task 2.2.1 PRONTO PER IL COMPLETAMENTO")
        print(f"✅ TensorFlow installato e funzionante")
        print(f"✅ NBA Ensemble Predictor inizializzato")
        print(f"✅ Modelli XGBoost + Neural Network disponibili")
        return True
    else:
        print(f"\n⚠️ ALCUNI TEST FALLITI")
        print(f"⚠️ Verificare l'implementazione")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)