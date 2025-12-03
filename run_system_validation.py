#!/usr/bin/env python3
"""
Script per eseguire la validazione completa del sistema NBA Predictor
per identificare i problemi critici prima del deployment in produzione.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Aggiungi il path del progetto al Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from nba_predictor.testing.system_validator import create_system_validator
    from nba_predictor.core.data_store import UnifiedDataStore
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    print("Assicurati che il modulo sia nel path corretto")
    sys.exit(1)

def main():
    """Esegui la validazione completa del sistema"""
    print("🚀 Iniziando validazione completa del sistema NBA Predictor...")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Inizializza il data store
        print("📊 Inizializzando UnifiedDataStore...")
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(exist_ok=True)
        
        data_store = UnifiedDataStore(
            base_path=str(data_dir),
            cache_enabled=True
        )
        
        # Inizializza il system validator
        print("🧪 Inizializzando SystemValidator...")
        validator = create_system_validator(data_store)
        
        # Esegui la validazione completa
        print("🔍 Eseguendo validazione completa...")
        report = validator.run_comprehensive_validation()
        
        # Salva il report
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"💾 Salvando report su {report_file}...")
        
        if validator.save_validation_report(report, report_file):
            print(f"✅ Report salvato con successo: {report_file}")
        else:
            print("❌ Errore nel salvare il report")
        
        # Mostra riassunto
        print("\n" + "=" * 60)
        print("📊 RIEPILOGO VALIDAZIONE")
        print("=" * 60)
        
        summary = report.summary
        print(f"📈 Status complessivo: {report.overall_status.upper()}")
        print(f"📋 Test totali: {summary.get('total_tests', 0)}")
        print(f"✅ Test passati: {summary.get('passed_tests', 0)}")
        print(f"❌ Test falliti: {summary.get('failed_tests', 0)}")
        print(f"⏭️ Test saltati: {summary.get('skipped_tests', 0)}")
        print(f"📊 Pass rate: {summary.get('pass_rate', '0%')}")
        
        # Mostra dettagli dei test falliti
        failed_tests = [r for r in report.test_results if r.status == "failed"]
        if failed_tests:
            print(f"\n❌ TEST FALLITI ({len(failed_tests)}):")
            print("-" * 40)
            for i, test in enumerate(failed_tests, 1):
                print(f"{i}. {test.test_name} ({test.test_type})")
                print(f"   Errore: {test.error_message}")
                print(f"   Durata: {test.execution_time_ms:.1f}ms")
                print()
        
        # Mostra metriche di performance
        if report.performance_metrics:
            print(f"⚡ METRICHE PERFORMANCE:")
            print("-" * 30)
            for key, value in report.performance_metrics.items():
                print(f"   {key}: {value}")
            print()
        
        # Mostra target di performance
        if "performance_targets_met" in summary:
            targets = summary["performance_targets_met"]
            print(f"🎯 TARGET DI PERFORMANCE:")
            print("-" * 35)
            for target, met in targets.items():
                status = "✅" if met else "❌"
                print(f"   {status} {target}")
            print()
        
        # Mostra criteri di user acceptance
        if "user_acceptance_criteria_met" in summary:
            criteria = summary["user_acceptance_criteria_met"]
            print(f"👥 CRITERI USER ACCEPTANCE:")
            print("-" * 40)
            for criterion, met in criteria.items():
                status = "✅" if met else "❌"
                print(f"   {status} {criterion}")
            print()
        
        # Raccomandazioni finali
        if report.overall_status in ["failed", "passed_with_major_issues"]:
            print("🚨 AZIONI CRITICHE RICHIESTE:")
            print("-" * 40)
            
            if summary.get('failed_tests', 0) > 0:
                print("1. Risolvere i test falliti prima del deployment")
            
            if not summary.get("performance_targets_met", {}).get("prediction_time_target", False):
                print("2. Ottimizzare i tempi di predizione (< 20ms)")
            
            if not summary.get("user_acceptance_criteria_met", {}).get("system_usability", False):
                print("3. Migliorare l'usabilità del sistema")
            
            print("4. Eseguire nuovamente la validazione dopo le correzioni")
        
        elif report.overall_status == "passed_with_minor_issues":
            print("⚠️ AZIONI CONSIGLIATE:")
            print("-" * 30)
            print("1. Monitorare i problemi minori identificati")
            print("2. Considerare fix per i test di performance")
        
        else:
            print("🎉 SISTEMA PRONTO PER LA PRODUZIONE!")
            print("-" * 40)
            print("✅ Tutti i test passati")
            print("✅ Target di performance raggiunti")
            print("✅ Criteri user acceptance soddisfatti")
        
        return report.overall_status in ["passed", "passed_with_minor_issues"]
        
    except Exception as e:
        print(f"❌ Errore critico durante la validazione: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
