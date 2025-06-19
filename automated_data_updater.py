#!/usr/bin/env python3
"""
🔄 AUTOMATED NBA DATA UPDATER
Sistema automatico per aggiornamento incrementale dei dati NBA
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, date, timedelta
import time
import logging
from typing import Dict, List, Optional

from data_provider import NBADataProvider

class AutomatedDataUpdater:
    """Sistema automatico per aggiornamento dati NBA"""
    
    def __init__(self):
        self.data_provider = NBADataProvider()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.main_dataset_path = os.path.join(self.data_dir, 'nba_fixed_training_dataset.csv')
        self.cache_path = os.path.join(self.data_dir, 'update_cache.json')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/update_log.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_daily_update(self):
        """
        Esegue aggiornamento giornaliero:
        1. Controlla nuove partite completate
        2. Raccoglie dati mancanti
        3. Aggiorna dataset principale
        4. Riaddestra modelli se necessario
        """
        
        self.logger.info("🔄 === STARTING DAILY DATA UPDATE ===")
        
        try:
            # 1. LOAD CURRENT DATASET
            current_df = self._load_current_dataset()
            self.logger.info(f"📊 Current dataset: {len(current_df)} games")
            
            # 2. FIND NEW GAMES TO PROCESS
            new_games = self._find_new_completed_games(current_df)
            
            if not new_games:
                self.logger.info("✅ No new games to process")
                return {'status': 'up_to_date', 'new_games': 0}
            
            self.logger.info(f"🆕 Found {len(new_games)} new completed games")
            
            # 3. PROCESS NEW GAMES
            new_data_rows = self._process_new_games(new_games)
            
            if not new_data_rows:
                self.logger.warning("❌ No valid data extracted from new games")
                return {'status': 'error', 'message': 'No valid data extracted'}
            
            # 4. UPDATE MAIN DATASET
            updated_df = self._update_main_dataset(current_df, new_data_rows)
            
            # 5. SAVE UPDATED DATASET
            self._save_updated_dataset(updated_df)
            
            # 6. UPDATE CACHE
            self._update_cache({
                'last_update': datetime.now().isoformat(),
                'games_added': len(new_data_rows),
                'total_games': len(updated_df)
            })
            
            self.logger.info(f"✅ Dataset updated: +{len(new_data_rows)} games (Total: {len(updated_df)})")
            
            # 7. RETRAIN MODELS IF SIGNIFICANT UPDATE
            if len(new_data_rows) >= 50:  # Soglia per retraining
                self.logger.info("🔄 Triggering model retraining due to significant data update")
                self._trigger_model_retraining()
            
            return {
                'status': 'success',
                'new_games': len(new_data_rows),
                'total_games': len(updated_df),
                'retrained': len(new_data_rows) >= 50
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error during data update: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _load_current_dataset(self) -> pd.DataFrame:
        """Carica il dataset corrente"""
        
        if not os.path.exists(self.main_dataset_path):
            self.logger.warning("📊 Main dataset not found, creating empty one")
            return pd.DataFrame()
        
        df = pd.read_csv(self.main_dataset_path)
        self.logger.info(f"📊 Loaded dataset: {len(df)} games, columns: {len(df.columns)}")
        return df
    
    def _find_new_completed_games(self, current_df: pd.DataFrame) -> List[Dict]:
        """
        Trova nuove partite completate da aggiungere al dataset
        """
        
        # Determina range di date da controllare
        if current_df.empty:
            # Se dataset vuoto, parti da inizio stagione corrente
            start_date = self._get_season_start_date()
        else:
            # Altrimenti, parti dall'ultima data nel dataset
            if 'GAME_DATE_EST' in current_df.columns:
                last_date_str = current_df['GAME_DATE_EST'].max()
                start_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            else:
                start_date = date.today() - timedelta(days=7)  # Fallback: ultima settimana
        
        end_date = date.today() - timedelta(days=1)  # Fino a ieri
        
        self.logger.info(f"🔍 Searching for completed games from {start_date} to {end_date}")
        
        # Lista per raccogliere tutte le partite trovate
        all_completed_games = []
        
        # Scansiona ogni giorno nel range
        current_date = start_date
        while current_date <= end_date:
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                self.logger.info(f"   📅 Checking games for {date_str}")
                
                # Usa data_provider per ottenere partite del giorno
                games = self.data_provider.get_scheduled_games(
                    days_ahead=1,
                    specific_date=date_str
                )
                
                # Filtra solo partite completate
                for game in games:
                    if self._is_game_completed(game, current_date):
                        # Controlla se già presente nel dataset
                        if not self._is_game_in_dataset(game, current_df):
                            all_completed_games.append(game)
                            self.logger.info(f"      ✅ New completed game: {game['away_team']} @ {game['home_team']}")
                        else:
                            self.logger.info(f"      ⏭️ Game already in dataset: {game['away_team']} @ {game['home_team']}")
                
                # Piccola pausa per evitare rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"   ❌ Error checking {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"🆕 Found {len(all_completed_games)} new completed games")
        return all_completed_games
    
    def _is_game_completed(self, game: Dict, game_date: date) -> bool:
        """
        Verifica se una partita è completata
        - Se è di ieri o prima → Dovrebbe essere completata
        - Se è di oggi → Controllo più specifico
        """
        
        if game_date < date.today():
            # Partite passate dovrebbero essere completate
            return True
        elif game_date == date.today():
            # Partite di oggi: controllo status se disponibile
            status = game.get('status', '').lower()
            return 'final' in status or 'completed' in status
        else:
            # Partite future
            return False
    
    def _is_game_in_dataset(self, game: Dict, df: pd.DataFrame) -> bool:
        """Controlla se una partita è già presente nel dataset"""
        
        if df.empty:
            return False
        
        # Cerca per data e squadre
        game_date = game['date']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Controlla esistenza nel dataset
        if 'GAME_DATE_EST' in df.columns and 'HOME_TEAM_NAME' in df.columns:
            mask = (
                (df['GAME_DATE_EST'] == game_date) &
                (df['HOME_TEAM_NAME'] == home_team) &
                (df['AWAY_TEAM_NAME'] == away_team)
            )
            return mask.any()
        
        return False
    
    def _process_new_games(self, new_games: List[Dict]) -> List[Dict]:
        """
        Processa le nuove partite per estrarre dati necessari
        """
        
        self.logger.info(f"⚙️ Processing {len(new_games)} new games...")
        
        processed_data = []
        
        for i, game in enumerate(new_games):
            try:
                self.logger.info(f"   🏀 [{i+1}/{len(new_games)}] {game['away_team']} @ {game['home_team']}")
                
                # 1. OTTIENI STATISTICHE SQUADRE
                team_stats = self.data_provider.get_team_stats_for_game(
                    game['home_team'], 
                    game['away_team']
                )
                
                if not team_stats:
                    self.logger.warning(f"      ❌ Could not get team stats")
                    continue
                
                # 2. OTTIENI RISULTATO PARTITA (se disponibile)
                game_result = self._get_game_final_score(game)
                
                if not game_result:
                    self.logger.warning(f"      ❌ Could not get final score")
                    continue
                
                # 3. COSTRUISCI ROW DATI
                data_row = self._build_data_row(game, team_stats, game_result)
                
                if data_row:
                    processed_data.append(data_row)
                    self.logger.info(f"      ✅ Data extracted successfully")
                else:
                    self.logger.warning(f"      ❌ Failed to build data row")
                
                # Pausa per rate limiting
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"      ❌ Error processing game: {e}")
                continue
        
        self.logger.info(f"✅ Successfully processed {len(processed_data)}/{len(new_games)} games")
        return processed_data
    
    def _get_game_final_score(self, game: Dict) -> Optional[Dict]:
        """
        Ottiene il punteggio finale di una partita
        """
        
        try:
            # Prova a usare NBA API per ottenere boxscore
            from nba_api.stats.endpoints import boxscoretraditionalv2
            
            time.sleep(0.6)  # Rate limiting
            
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game.get('game_id', ''),
                headers=self.data_provider.headers
            )
            
            team_stats = boxscore.team_stats.get_data_frame()
            
            if team_stats.empty:
                return None
            
            # Estrai punteggi
            home_score = None
            away_score = None
            
            for _, row in team_stats.iterrows():
                team_name = row.get('TEAM_NAME', '')
                points = row.get('PTS', 0)
                
                if team_name in game['home_team'] or game['home_team'] in team_name:
                    home_score = points
                elif team_name in game['away_team'] or game['away_team'] in team_name:
                    away_score = points
            
            if home_score is not None and away_score is not None:
                total_score = home_score + away_score
                return {
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_score': total_score
                }
            
        except Exception as e:
            self.logger.error(f"      Error getting final score: {e}")
        
        return None
    
    def _build_data_row(self, game: Dict, team_stats: Dict, game_result: Dict) -> Optional[Dict]:
        """
        Costruisce una riga di dati compatibile con il dataset principale
        """
        
        try:
            home_stats = team_stats['home']
            away_stats = team_stats['away']
            
            # Estrai stagione dalla data
            game_date = datetime.strptime(game['date'], '%Y-%m-%d').date()
            season = self._get_season_from_date(game_date)
            
            # Costruisci riga dati seguendo formato del dataset esistente
            data_row = {
                # Identificativi
                'GAME_DATE_EST': game['date'],
                'SEASON': season,
                'HOME_TEAM_NAME': game['home_team'],
                'AWAY_TEAM_NAME': game['away_team'],
                
                # Target variables (quello che vogliamo predire)
                'target_mu': game_result['total_score'],  # Punteggio totale come target
                'target_sigma': self._estimate_sigma_from_game(game_result),  # Stima sigma
                
                # Home team features
                'HOME_W_PCT': home_stats.get('win_percentage', 0.5),
                'HOME_PACE': home_stats.get('pace', 100.0),
                'HOME_OFF_RATING': home_stats.get('offensive_rating', 110.0),
                'HOME_DEF_RATING': home_stats.get('defensive_rating', 110.0),
                'HOME_NET_RATING': home_stats.get('offensive_rating', 110.0) - home_stats.get('defensive_rating', 110.0),
                'HOME_EFG_PCT': home_stats.get('efg_pct', 0.5),
                'HOME_FT_RATE': home_stats.get('ft_rate', 0.2),
                'HOME_TOV_PCT': home_stats.get('tov_pct', 0.14),
                'HOME_OREB_PCT': home_stats.get('oreb_pct', 0.25),
                
                # Away team features
                'AWAY_W_PCT': away_stats.get('win_percentage', 0.5),
                'AWAY_PACE': away_stats.get('pace', 100.0),
                'AWAY_OFF_RATING': away_stats.get('offensive_rating', 110.0),
                'AWAY_DEF_RATING': away_stats.get('defensive_rating', 110.0),
                'AWAY_NET_RATING': away_stats.get('offensive_rating', 110.0) - away_stats.get('defensive_rating', 110.0),
                'AWAY_EFG_PCT': away_stats.get('efg_pct', 0.5),
                'AWAY_FT_RATE': away_stats.get('ft_rate', 0.2),
                'AWAY_TOV_PCT': away_stats.get('tov_pct', 0.14),
                'AWAY_OREB_PCT': away_stats.get('oreb_pct', 0.25),
                
                # Derived features
                'PACE_DIFFERENTIAL': home_stats.get('pace', 100.0) - away_stats.get('pace', 100.0),
                'OFFENSIVE_MISMATCH': home_stats.get('offensive_rating', 110.0) - away_stats.get('defensive_rating', 110.0),
                'DEFENSIVE_MISMATCH': away_stats.get('offensive_rating', 110.0) - home_stats.get('defensive_rating', 110.0),
                
                # Metadati
                'DATA_SOURCE': 'automated_update',
                'UPDATE_TIMESTAMP': datetime.now().isoformat()
            }
            
            return data_row
            
        except Exception as e:
            self.logger.error(f"Error building data row: {e}")
            return None
    
    def _estimate_sigma_from_game(self, game_result: Dict) -> float:
        """
        Stima la sigma (incertezza) per una partita basandosi sul risultato
        """
        
        total_score = game_result['total_score']
        
        # Stima sigma basata su range tipico NBA (più alto = più incertezza)
        if total_score < 200:
            return 16.0  # Partite basse = più incertezza
        elif total_score > 250:
            return 18.0  # Partite alte = più incertezza  
        else:
            return 14.0  # Range normale = incertezza standard
    
    def _get_season_from_date(self, game_date: date) -> int:
        """Determina la stagione NBA dalla data"""
        
        year = game_date.year
        month = game_date.month
        
        # Stagione NBA: Ottobre Anno X - Giugno Anno X+1
        if month >= 10:  # Da Ottobre in poi
            return year
        else:  # Da Gennaio a Giugno
            return year - 1
    
    def _get_season_start_date(self) -> date:
        """Ottiene data di inizio della stagione corrente"""
        
        today = date.today()
        year = today.year
        
        if today.month >= 10:  # Stagione corrente
            return date(year, 10, 1)
        else:  # Stagione precedente
            return date(year - 1, 10, 1)
    
    def _update_main_dataset(self, current_df: pd.DataFrame, new_rows: List[Dict]) -> pd.DataFrame:
        """Aggiorna il dataset principale con nuovi dati"""
        
        # Converti nuovi dati in DataFrame
        new_df = pd.DataFrame(new_rows)
        
        if current_df.empty:
            return new_df
        
        # CORREZIONE: Allinea le colonne prima del merge
        # Assicurati che entrambi i dataframe abbiano le stesse colonne
        current_columns = set(current_df.columns)
        new_columns = set(new_df.columns)
        
        # Aggiungi colonne mancanti con valori default
        for col in new_columns - current_columns:
            current_df[col] = None
        
        for col in current_columns - new_columns:
            new_df[col] = None
        
        # Riordina colonne per consistenza
        all_columns = sorted(list(current_columns | new_columns))
        current_df = current_df.reindex(columns=all_columns)
        new_df = new_df.reindex(columns=all_columns)
        
        self.logger.info(f"   🔧 Merging: current_df({len(current_df)} rows) + new_df({len(new_df)} rows)")
        
        # Unisci dataset
        updated_df = pd.concat([current_df, new_df], ignore_index=True)
        
        self.logger.info(f"   ✅ After concat: {len(updated_df)} rows")
        
        # Rimuovi eventuali duplicati
        if 'GAME_DATE_EST' in updated_df.columns and 'HOME_TEAM_NAME' in updated_df.columns:
            before_dedup = len(updated_df)
            updated_df = updated_df.drop_duplicates(
                subset=['GAME_DATE_EST', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME'],
                keep='last'
            )
            after_dedup = len(updated_df)
            self.logger.info(f"   🧹 Deduplication: {before_dedup} → {after_dedup} rows")
        
        # Ordina per data se disponibile
        if 'GAME_DATE_EST' in updated_df.columns:
            updated_df = updated_df.sort_values('GAME_DATE_EST').reset_index(drop=True)
        
        self.logger.info(f"   📊 Final dataset: {len(updated_df)} rows")
        return updated_df
    
    def _save_updated_dataset(self, df: pd.DataFrame):
        """Salva dataset aggiornato"""
        
        # Backup del dataset precedente
        if os.path.exists(self.main_dataset_path):
            backup_path = self.main_dataset_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.rename(self.main_dataset_path, backup_path)
            self.logger.info(f"📁 Backup saved: {backup_path}")
        
        # Salva nuovo dataset
        df.to_csv(self.main_dataset_path, index=False)
        self.logger.info(f"💾 Updated dataset saved: {self.main_dataset_path}")
    
    def _update_cache(self, cache_data: Dict):
        """Aggiorna cache con informazioni ultimo update"""
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _trigger_model_retraining(self):
        """Avvia processo di riaddestramento modelli"""
        
        try:
            # Esegui script di retraining ottimizzato
            import subprocess
            
            self.logger.info("🔄 Starting model retraining...")
            
            result = subprocess.run([
                'python', 'optimized_model_trainer.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Model retraining completed successfully")
            else:
                self.logger.error(f"❌ Model retraining failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Error triggering retraining: {e}")

def setup_automated_updates():
    """Setup per aggiornamenti automatici giornalieri"""
    
    print("🔄 Setting up automated NBA data updates...")
    
    # Crea script scheduler
    scheduler_script = '''#!/bin/bash
# NBA Data Update Scheduler
cd "$(dirname "$0")"
python automated_data_updater.py
'''
    
    with open('run_daily_update.sh', 'w') as f:
        f.write(scheduler_script)
    
    os.chmod('run_daily_update.sh', 0o755)
    
    print("✅ Setup completed!")
    print("📋 To schedule daily updates, add this to your crontab:")
    print("   0 9 * * * /path/to/your/project/run_daily_update.sh")

def main():
    """Esegue aggiornamento manuale"""
    
    updater = AutomatedDataUpdater()
    result = updater.run_daily_update()
    
    print(f"\n📊 Update Result: {result}")

if __name__ == "__main__":
    main() 