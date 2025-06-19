#!/usr/bin/env python3
"""
Script specifico per recuperare dati playoff storici (2019-20, 2020-21, 2021-22, 2022-23)
e integrarli nel dataset di momentum esistente.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from data_provider import NBADataProvider
from momentum_dataset_builder import MomentumDatasetBuilder
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class PlayoffHistoricalDataBuilder:
    """
    Costruisce dati playoff storici per integrare il dataset esistente.
    """
    
    def __init__(self):
        self.data_provider = NBADataProvider()
        self.momentum_builder = MomentumDatasetBuilder(self.data_provider)
        self.output_dir = os.path.join(os.path.dirname(__file__), 'data', 'momentum_v2')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # File paths
        self.main_dataset_path = os.path.join(self.output_dir, 'momentum_training_dataset.csv')
        self.historical_dataset_path = os.path.join(self.output_dir, 'momentum_historical_playoffs.csv')
        self.checkpoint_path = os.path.join(self.output_dir, 'historical_playoffs_checkpoint.csv')
        
        # Stagioni storiche da recuperare (solo playoff)
        self.historical_seasons = ['2019-20', '2020-21', '2021-22', '2022-23']
    
    def build_historical_playoff_data(self):
        """
        Recupera i dati playoff storici e li salva in un file separato.
        """
        print("🏆 RECUPERO DATI PLAYOFF STORICI")
        print("=" * 60)
        print(f"📋 Stagioni target: {self.historical_seasons}")
        print(f"🎯 Solo playoff (nessuna Regular Season)")
        
        # Controlla checkpoint esistente
        processed_games_df = pd.DataFrame()
        processed_game_ids = set()
        
        if os.path.exists(self.checkpoint_path):
            print(f"🔄 Checkpoint trovato: {self.checkpoint_path}")
            try:
                processed_games_df = pd.read_csv(self.checkpoint_path)
                if not processed_games_df.empty:
                    processed_games_df = processed_games_df.drop_duplicates(subset=['game_id'], keep='last')
                    processed_games_df['game_id'] = processed_games_df['game_id'].astype(str).str.lstrip('0')
                    processed_game_ids = set(processed_games_df['game_id'])
                    print(f"   ✅ Caricati {len(processed_game_ids)} partite playoff storiche già processate")
                else:
                    print(f"   📁 File checkpoint vuoto")
            except Exception as e:
                print(f"   ⚠️ Errore caricamento checkpoint: {e}")
                processed_games_df = pd.DataFrame()
                processed_game_ids = set()
        else:
            print("   📁 Nessun checkpoint - iniziamo da zero")
        
        # Recupera partite playoff per ogni stagione storica
        all_playoff_games = []
        for season in self.historical_seasons:
            print(f"\n🔍 Recupero playoff {season}...")
            try:
                # SOLO playoff - specificamente richiesto
                playoff_log = self.data_provider.get_season_game_log(season, season_type='Playoffs')
                
                if playoff_log is not None and not playoff_log.empty:
                    print(f"   ✅ Trovate {len(playoff_log)} partite playoff per {season}")
                    all_playoff_games.append(playoff_log)
                else:
                    print(f"   ⚠️ Nessuna partita playoff trovata per {season}")
                    
                # Rate limiting per evitare problemi API
                time.sleep(1.0)
                    
            except Exception as e:
                print(f"   ❌ Errore recupero playoff {season}: {e}")
                continue
        
        if not all_playoff_games:
            print("❌ Nessuna partita playoff storica recuperata!")
            return
        
        # Combina tutte le partite playoff
        combined_playoff_log = pd.concat(all_playoff_games, ignore_index=True)
        combined_playoff_log = combined_playoff_log.sort_values(by=['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
        
        total_playoff_games = len(combined_playoff_log)
        remaining_games = total_playoff_games - len(processed_game_ids)
        
        print(f"\n📊 RIEPILOGO PLAYOFF STORICI:")
        print(f"   📋 Partite playoff totali trovate: {total_playoff_games}")
        print(f"   ✅ Già processate: {len(processed_game_ids)}")
        print(f"   🔄 Da processare: {remaining_games}")
        
        if remaining_games == 0:
            print("🎉 Tutti i playoff storici già processati!")
            return
        
        # Pre-carica roster per ottimizzazione
        print(f"\n🚀 Pre-caricamento roster per ottimizzazione...")
        unique_team_ids = set()
        for _, game in combined_playoff_log.iterrows():
            unique_team_ids.add(game['HOME_TEAM_ID'])
            unique_team_ids.add(game['AWAY_TEAM_ID'])
        
        print(f"   🏀 Squadre uniche da caricare: {len(unique_team_ids)}")
        
        for i, team_id in enumerate(unique_team_ids, 1):
            try:
                roster = self.data_provider.get_team_roster(team_id)
                if roster is not None:
                    print(f"   ✅ [{i}/{len(unique_team_ids)}] Roster caricato team {team_id}")
                else:
                    print(f"   ⚠️ [{i}/{len(unique_team_ids)}] Roster vuoto team {team_id}")
                time.sleep(0.2)  # Rate limiting leggero
            except Exception as e:
                print(f"   ❌ [{i}/{len(unique_team_ids)}] Errore team {team_id}: {e}")
        
        # Processa partite playoff storiche
        print(f"\n🔄 ELABORAZIONE PARTITE PLAYOFF STORICHE")
        print("=" * 50)
        
        new_games_processed = 0
        batch_size = 3  # Salva ogni 3 partite
        
        try:
            for index, game in combined_playoff_log.iterrows():
                game_id_str = str(game['GAME_ID']).lstrip('0')
                current_progress = index + 1
                
                # Salta se già processata
                if game_id_str in processed_game_ids:
                    print(f"   ⏭️ [{current_progress}/{total_playoff_games}] Già processata: {game['MATCHUP']} {game['GAME_DATE']}")
                    continue
                
                print(f"   🔄 [{current_progress}/{total_playoff_games}] Processando: {game['MATCHUP']} {game['GAME_DATE']}")
                
                try:
                    # Trova il log della stagione corretta
                    game_season = game['SEASON_YEAR']
                    current_season_log = next((log for log in all_playoff_games 
                                             if not log.empty and log['SEASON_YEAR'].iloc[0] == game_season), None)
                    
                    if current_season_log is None:
                        print(f"      ⚠️ Log stagione {game_season} non trovato. Skip.")
                        continue
                    
                    # Estrai features usando il momentum builder esistente
                    game_features = self.momentum_builder._extract_features_for_game(game, current_season_log)
                    
                    if game_features:
                        # Aggiungi season year per identificazione
                        game_features['season'] = game_season
                        
                        # Aggiungi al dataset
                        new_row = pd.DataFrame([game_features])
                        processed_games_df = pd.concat([processed_games_df, new_row], ignore_index=True)
                        processed_game_ids.add(game_id_str)
                        new_games_processed += 1
                        
                        print(f"      ✅ Features estratte con successo")
                        
                        # Salva checkpoint ogni batch_size
                        if new_games_processed % batch_size == 0:
                            self._save_checkpoint(processed_games_df)
                            print(f"      💾 Checkpoint: {len(processed_games_df)} partite totali")
                    else:
                        print(f"      ❌ Errore estrazione features")
                        
                except Exception as e:
                    print(f"      ❌ Errore processamento: {e}")
                    continue
                
                # Rate limiting per evitare sovraccarico API
                time.sleep(0.3)
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Interruzione utente. Salvataggio...")
        except Exception as e:
            print(f"\n❌ Errore generale: {e}. Salvataggio...")
        finally:
            # Salvataggio finale garantito
            if new_games_processed > 0:
                self._save_checkpoint(processed_games_df)
                print(f"💾 Salvataggio finale: {len(processed_games_df)} partite playoff storiche")
                
                # Salva anche dataset finale
                final_path = self.historical_dataset_path
                processed_games_df.to_csv(final_path, index=False)
                print(f"🎉 Dataset playoff storici salvato: {final_path}")
                
                return processed_games_df
            else:
                print("ℹ️ Nessuna nuova partita processata in questa sessione")
                return processed_games_df if not processed_games_df.empty else None
    
    def _save_checkpoint(self, df: pd.DataFrame):
        """Salva checkpoint del progresso."""
        try:
            df.to_csv(self.checkpoint_path, index=False)
        except Exception as e:
            print(f"❌ Errore salvataggio checkpoint: {e}")
    
    def merge_with_main_dataset(self):
        """
        Unisce i dati playoff storici con il dataset principale esistente.
        """
        print(f"\n🔗 UNIONE CON DATASET PRINCIPALE")
        print("=" * 40)
        
        # Carica dataset principale
        if not os.path.exists(self.main_dataset_path):
            print(f"❌ Dataset principale non trovato: {self.main_dataset_path}")
            return
        
        main_df = pd.read_csv(self.main_dataset_path)
        print(f"📊 Dataset principale: {len(main_df)} partite")
        
        # Carica dataset playoff storici
        if not os.path.exists(self.historical_dataset_path):
            print(f"❌ Dataset playoff storici non trovato: {self.historical_dataset_path}")
            return
        
        historical_df = pd.read_csv(self.historical_dataset_path)
        print(f"🏆 Dataset playoff storici: {len(historical_df)} partite")
        
        # Verifica compatibilità colonne
        main_cols = set(main_df.columns)
        hist_cols = set(historical_df.columns)
        
        # Trova colonne comuni
        common_cols = main_cols.intersection(hist_cols)
        missing_in_main = hist_cols - main_cols
        missing_in_hist = main_cols - hist_cols
        
        print(f"📋 Colonne comuni: {len(common_cols)}")
        if missing_in_main:
            print(f"⚠️ Colonne mancanti nel dataset principale: {missing_in_main}")
        if missing_in_hist:
            print(f"⚠️ Colonne mancanti nei playoff storici: {missing_in_hist}")
        
        # Prepara dataframe per unione
        if missing_in_hist:
            # Aggiungi colonne mancanti con NaN/valori default
            for col in missing_in_hist:
                if col == 'season':
                    # Inferisci la stagione dalla data se possibile
                    historical_df['season'] = historical_df.get('season', 'unknown')
                else:
                    historical_df[col] = np.nan
        
        if missing_in_main:
            # Aggiungi colonne mancanti al dataset principale
            for col in missing_in_main:
                if col == 'season':
                    # Inferisci stagioni per il dataset principale (2023-24, 2024-25)
                    main_df['season'] = main_df.apply(self._infer_season_from_date, axis=1)
                else:
                    main_df[col] = np.nan
        
        # Unisci i dataset
        print(f"🔗 Unione dei dataset...")
        combined_df = pd.concat([main_df, historical_df], ignore_index=True)
        
        # Rimuovi duplicati basati su game_id
        initial_size = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['game_id'], keep='last')
        final_size = len(combined_df)
        
        print(f"📊 Risultato unione:")
        print(f"   - Partite prima deduplicazione: {initial_size}")
        print(f"   - Partite dopo deduplicazione: {final_size}")
        print(f"   - Duplicati rimossi: {initial_size - final_size}")
        
        # Salva dataset combinato finale
        backup_path = self.main_dataset_path.replace('.csv', '_backup.csv')
        main_df.to_csv(backup_path, index=False)
        print(f"💾 Backup dataset originale: {backup_path}")
        
        combined_df.to_csv(self.main_dataset_path, index=False)
        print(f"🎉 Dataset combinato salvato: {self.main_dataset_path}")
        
        # Statistiche finali per tipo
        if 'season' in combined_df.columns:
            season_stats = combined_df['season'].value_counts().sort_index()
            print(f"\n📈 Distribuzione per stagione:")
            for season, count in season_stats.items():
                print(f"   {season}: {count} partite")
        
        return combined_df
    
    def _infer_season_from_date(self, row):
        """Inferisce la stagione dalla data della partita."""
        try:
            if 'game_date' in row and pd.notna(row['game_date']):
                date_str = str(row['game_date'])
                if '2023' in date_str:
                    return '2023-24'
                elif '2024' in date_str:
                    return '2024-25'
            return 'unknown'
        except:
            return 'unknown'
    
    def run_complete_historical_update(self):
        """
        Esegue l'intero processo di aggiornamento con dati playoff storici.
        """
        print("🚀 AVVIO AGGIORNAMENTO DATI PLAYOFF STORICI")
        print("=" * 60)
        
        # Step 1: Costruisci dati playoff storici
        historical_df = self.build_historical_playoff_data()
        
        if historical_df is None or historical_df.empty:
            print("❌ Nessun dato playoff storico recuperato. Processo interrotto.")
            return
        
        # Step 2: Unisci con dataset principale
        combined_df = self.merge_with_main_dataset()
        
        if combined_df is not None:
            print(f"\n🎉 PROCESSO COMPLETATO!")
            print(f"   📊 Dataset finale: {len(combined_df)} partite")
            print(f"   🏆 Include playoff storici: {self.historical_seasons}")
            print(f"   💾 Salvato in: {self.main_dataset_path}")
            
            return combined_df
        else:
            print("❌ Errore durante l'unione dei dataset")
            return None

def main():
    """Funzione principale per l'esecuzione dello script."""
    try:
        builder = PlayoffHistoricalDataBuilder()
        result = builder.run_complete_historical_update()
        
        if result is not None:
            print(f"\n✅ Successo! Dataset aggiornato con {len(result)} partite totali.")
            print(f"   Ora puoi rilanciare il training del modello playoff!")
        else:
            print(f"\n❌ Processo fallito.")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Processo interrotto dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore generale: {e}")

if __name__ == "__main__":
    main() 