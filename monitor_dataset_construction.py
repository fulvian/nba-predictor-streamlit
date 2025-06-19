#!/usr/bin/env python3
"""
📊 DATASET CONSTRUCTION MONITOR
Monitora il progresso della costruzione del dataset completo
"""

import os
import time
import pandas as pd
from datetime import datetime

def monitor_dataset_construction():
    """Monitora il progresso della costruzione del dataset"""
    
    data_dir = "data"
    target_file = os.path.join(data_dir, "nba_complete_dataset.csv")
    
    print("🔍 === MONITOR COSTRUZIONE DATASET ===")
    print(f"📁 Monitorando: {target_file}")
    print(f"⏰ Avviato alle: {datetime.now().strftime('%H:%M:%S')}")
    print("\n" + "="*50)
    
    while True:
        try:
            if os.path.exists(target_file):
                # File creato - leggi e analizza
                df = pd.read_csv(target_file)
                
                print(f"\n✅ DATASET TROVATO! ({datetime.now().strftime('%H:%M:%S')})")
                print(f"📊 PARTITE TOTALI: {len(df):,}")
                
                if 'SEASON' in df.columns:
                    season_counts = df['SEASON'].value_counts().sort_index()
                    print(f"📅 DISTRIBUZIONE PER STAGIONE:")
                    for season, count in season_counts.items():
                        print(f"   {season}: {count:,} partite")
                
                if 'target_mu' in df.columns:
                    print(f"🎯 PUNTEGGI - Media: {df['target_mu'].mean():.1f}, "
                          f"Range: {df['target_mu'].min()}-{df['target_mu'].max()}")
                
                print(f"\n🎉 COSTRUZIONE COMPLETATA!")
                print(f"📁 File disponibile: {target_file}")
                break
            else:
                print(f"⏳ In attesa... ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(30)  # Controlla ogni 30 secondi
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoraggio interrotto dall'utente")
            break
        except Exception as e:
            print(f"⚠️ Errore durante monitoraggio: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_dataset_construction() 