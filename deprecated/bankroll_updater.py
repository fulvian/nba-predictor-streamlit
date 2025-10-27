#!/usr/bin/env python3
"""
Script per l'aggiornamento automatico del bankroll.
Controlla le scommesse pendenti e aggiorna il bankroll automaticamente
quando le partite sono completate.

Usage:
    python bankroll_updater.py              # Controllo singolo
    python bankroll_updater.py --daemon     # Esecuzione continua ogni ora
"""

import argparse
import time
import schedule
from datetime import datetime
from main import NBACompleteSystem
from data_provider import NBADataProvider

def update_bankroll():
    """Esegue l'aggiornamento del bankroll per le scommesse completate."""
    try:
        print(f"\n{'='*60}")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Controllo Automatico Bankroll")
        print(f"{'='*60}")
        
        data_provider = NBADataProvider()
        system = NBACompleteSystem(data_provider, auto_mode=True)
        
        # Controlla e aggiorna scommesse pendenti
        system.check_and_update_pending_bets()
        
        print(f"✅ Controllo completato alle {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Aggiornatore automatico bankroll NBA')
    parser.add_argument('--daemon', action='store_true', 
                       help='Esegui in modalità daemon (controllo ogni ora)')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Intervallo in minuti per modalità daemon (default: 60)')
    
    args = parser.parse_args()
    
    if args.daemon:
        print("🤖 Avvio modalità daemon - Controllo automatico ogni {} minuti".format(args.interval))
        print("   Premi Ctrl+C per interrompere")
        
        # Esegui subito un controllo
        update_bankroll()
        
        # Programma controlli periodici
        schedule.every(args.interval).minutes.do(update_bankroll)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Controlla ogni minuto se ci sono task scheduled
        except KeyboardInterrupt:
            print("\n⏹️ Modalità daemon interrotta dall'utente")
    else:
        # Esecuzione singola
        update_bankroll()

if __name__ == "__main__":
    main() 