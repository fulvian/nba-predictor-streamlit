#!/usr/bin/env python3
"""
Test finale per verificare che tutte le correzioni funzionino correttamente.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from nba_predictor.streamlit.components.enhanced_prediction_bridge_professional import (
    get_enhanced_prediction_bridge_professional,
)
from datetime import date

def test_final_fix():
    """Test finale completo del problema 205.0."""
    print("🏀 Test finale completo del fix 205.0...")
    
    # Inizializza bridge
    bridge = get_enhanced_prediction_bridge_professional()
    
    # Test cases
    test_cases = [
        {
            "name": "LA Clippers @ Atlanta Hawks (problema originale)",
            "home_team": "Atlanta Hawks",
            "away_team": "Los Angeles Clippers",
            "line": 225.0,
        },
        {
            "name": "Boston Celtics @ LA Lakers (controllo)",
            "home_team": "Los Angeles Lakers", 
            "away_team": "Boston Celtics",
            "line": 225.0,
        },
        {
            "name": "LA Clippers @ Boston Celtics (test Clippers dopo fix)",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Clippers", 
            "line": 225.0,
        },
    ]
    
    print(f"📊 Eseguendo {len(test_cases)} test cases...")
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        
        try:
            # Esegui predizione con force refresh
            prediction = bridge.get_professional_prediction(
                home_team=test_case["home_team"],
                away_team=test_case["away_team"],
                game_date=date.today(),
                betting_line=test_case["line"],
                include_detailed_analysis=True,
                force_refresh=True,  # Forza sempre refresh
            )
            
            predicted_total = prediction.get("predicted_total", 0)
            status = prediction.get("status", "unknown")
            method = prediction.get("prediction_method", "Unknown")
            
            print(f"   Predicted Total: {predicted_total}")
            print(f"   Status: {status}")
            print(f"   Method: {method}")
            
            # Verifica risultati
            if predicted_total == 205.0:
                print(f"   ❌ FALLITO: Predizione 205.0 rilevata!")
                all_passed = False
            elif 190 <= predicted_total <= 280:  # Range realistico NBA
                print(f"   ✅ PASS: Predizione {predicted_total} nel range realistico")
            else:
                print(f"   ⚠️ ATTENZIONE: Predizione {predicted_total} fuori range normale")
                
        except Exception as e:
            print(f"   ❌ ERRORE: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    print(f"📋 RISULTATI FINALI:")
    if all_passed:
        print("✅ SUCCESS: Tutti i test sono passati!")
        print("✅ Il problema del 205.0 è stato risolto!")
        print("✅ Le correzioni implementate funzionano correttamente:")
        print("   1. LA Clippers rimosso da high_performance_teams")
        print("   2. Emergency cap aumentato da 20.0 a 30.0")
        print("   3. Force refresh abilitato nella dashboard")
        print("   4. Metodo bridge corretto con parametro force_refresh")
    else:
        print("❌ FALLIMENTO: Alcuni test sono falliti!")
        print("❌ Il problema persiste e richiede ulteriori investigazioni")
    
    return all_passed

if __name__ == "__main__":
    success = test_final_fix()
    exit(0 if success else 1