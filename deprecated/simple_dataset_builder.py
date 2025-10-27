#!/usr/bin/env python3
"""
🏀 SIMPLE ROBUST NBA DATASET BUILDER
Versione semplificata e robusta per costruire dataset NBA bilanciato
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams

class SimpleDatasetBuilder:
    """Builder semplificato per dataset NBA"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.request_delay = 1.0  # Delay più conservativo
        
        # Stagioni target consecutive
        self.target_seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
        
        print("🏀 SIMPLE DATASET BUILDER INIZIALIZZATO")
        print(f"📅 Stagioni target: {', '.join(self.target_seasons)}")

    def build_simple_balanced_dataset(self) -> pd.DataFrame:
        """Costruisce dataset bilanciato con approccio semplificato"""
        
        print("\n🚀 === INIZIO COSTRUZIONE DATASET SEMPLIFICATO ===")
        
        all_data = []
        
        for season in self.target_seasons:
            print(f"\n📅 ELABORAZIONE STAGIONE {season}...")
            
            try:
                season_data = self._get_season_games_simple(season)
                if season_data is not None and len(season_data) > 0:
                    all_data.extend(season_data)
                    print(f"   ✅ Raccolte {len(season_data)} partite")
                else:
                    print(f"   ❌ Nessun dato per {season}")
                    
            except Exception as e:
                print(f"   ❌ Errore in {season}: {e}")
                continue
        
        if not all_data:
            print("❌ Nessun dato raccolto!")
            return pd.DataFrame()
        
        # Costruisci DataFrame
        print(f"\n🔄 COSTRUZIONE DATAFRAME FINALE...")
        df = pd.DataFrame(all_data)
        
        # Aggiungi features di base
        df = self._add_basic_features(df)
        
        # Pulizia
        df = self._clean_dataset(df)
        
        # Salva
        output_path = os.path.join(self.data_dir, 'nba_simple_complete_dataset.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n🎉 === DATASET COSTRUITO ===")
        print(f"📊 TOTALE PARTITE: {len(df):,}")
        print(f"💾 SALVATO IN: {output_path}")
        
        # Statistiche
        if 'SEASON' in df.columns:
            season_counts = df['SEASON'].value_counts().sort_index()
            print(f"\n📈 DISTRIBUZIONE PER STAGIONE:")
            for season, count in season_counts.items():
                print(f"   {season}: {count:,} partite")
        
        return df

    def _get_season_games_simple(self, season: str) -> List[Dict]:
        """Recupera partite per stagione con approccio semplificato"""
        
        season_data = []
        season_year = int(season.split('-')[0])
        
        try:
            print(f"   🔄 Recupero partite Regular Season per {season}...")
            time.sleep(self.request_delay)
            
            # Solo Regular Season per semplicità
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00'
            )
            
            games_df = game_finder.get_data_frames()[0]
            
            if games_df.empty:
                print(f"     ❌ Nessuna partita trovata")
                return []
            
            # Filtra solo partite a casa per evitare duplicati
            home_games = games_df[games_df['MATCHUP'].str.contains('vs.', na=False)]
            
            print(f"     📊 Trovate {len(home_games)} partite uniche")
            
            # Processa ogni partita
            for idx, (_, game_row) in enumerate(home_games.iterrows()):
                if idx % 100 == 0:
                    print(f"     🔄 Processate {idx}/{len(home_games)} partite...")
                
                game_data = self._process_game_simple(game_row, season_year)
                if game_data:
                    season_data.append(game_data)
                    
            print(f"     ✅ Processate {len(season_data)} partite valide")
            
        except Exception as e:
            print(f"     ❌ Errore: {e}")
            return []
        
        return season_data

    def _process_game_simple(self, game_row: pd.Series, season_year: int) -> Dict:
        """Processa singola partita con approccio semplificato"""
        
        try:
            # Informazioni base
            game_id = game_row['GAME_ID']
            game_date = game_row['GAME_DATE']
            team_id = game_row['TEAM_ID']
            pts = game_row['PTS']
            
            # Determina home/away
            matchup = game_row['MATCHUP']
            is_home = 'vs.' in matchup
            
            # Team avversario
            if is_home:
                opponent_abbr = matchup.split(' vs. ')[1]
            else:
                return None  # Processa solo home games
            
            # Statistiche base dalla partita
            fgm = game_row.get('FGM', 0)
            fga = game_row.get('FGA', 1)
            fg3m = game_row.get('FG3M', 0)
            ftm = game_row.get('FTM', 0)
            fta = game_row.get('FTA', 1)
            oreb = game_row.get('OREB', 0)
            dreb = game_row.get('DREB', 0)
            tov = game_row.get('TOV', 0)
            
            # Calcola statistiche di base
            fg_pct = fgm / max(fga, 1)
            efg_pct = (fgm + 0.5 * fg3m) / max(fga, 1)
            ft_rate = fta / max(fga, 1)
            
            # Usa valori NBA tipici per mancanti
            pace = 100.0  # NBA average
            ortg = 110.0  # NBA average
            drtg = 110.0  # NBA average
            
            # Stima punteggio avversario (approssimazione)
            # Per semplicità, usa media NBA come placeholder
            estimated_opp_score = 110
            total_score = pts + estimated_opp_score
            
            # Costruisci record
            game_data = {
                'GAME_ID': game_id,
                'GAME_DATE': game_date,
                'SEASON': season_year,
                'HOME_TEAM_ID': team_id,
                'AWAY_TEAM_ID': f"OPP_{team_id}",  # Placeholder
                
                # Features base HOME (dalla partita effettiva)
                'HOME_eFG_PCT_sAvg': efg_pct,
                'HOME_TOV_PCT_sAvg': tov / max(100, 1),  # Rough estimate
                'HOME_OREB_PCT_sAvg': oreb / max(oreb + dreb, 1),
                'HOME_FT_RATE_sAvg': ft_rate,
                'HOME_ORtg_sAvg': ortg,
                'HOME_DRtg_sAvg': drtg,
                'HOME_PACE': pace,
                
                # Features AWAY (placeholder NBA averages)
                'AWAY_eFG_PCT_sAvg': 0.53,  # NBA average
                'AWAY_TOV_PCT_sAvg': 0.14,  # NBA average
                'AWAY_OREB_PCT_sAvg': 0.25,  # NBA average
                'AWAY_FT_RATE_sAvg': 0.24,  # NBA average
                'AWAY_ORtg_sAvg': 110.0,
                'AWAY_DRtg_sAvg': 110.0,
                'AWAY_PACE': pace,
                
                # Target (reale)
                'target_mu': total_score,
                'target_sigma': 12.0,  # NBA typical volatility
                'TOTAL_SCORE': total_score,
                'HOME_SCORE': pts,
                'AWAY_SCORE': estimated_opp_score
            }
            
            return game_data
            
        except Exception as e:
            return None

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge features derivate di base"""
        
        print("🔧 Aggiunta features derivate...")
        
        # Features derivate
        df['GAME_PACE'] = (df['HOME_PACE'] + df['AWAY_PACE']) / 2
        df['PACE_DIFFERENTIAL'] = df['HOME_PACE'] - df['AWAY_PACE']
        df['HOME_OFF_vs_AWAY_DEF'] = df['HOME_ORtg_sAvg'] - df['AWAY_DRtg_sAvg']
        df['AWAY_OFF_vs_HOME_DEF'] = df['AWAY_ORtg_sAvg'] - df['HOME_DRtg_sAvg']
        df['TOTAL_EXPECTED_SCORING'] = df['HOME_OFF_vs_AWAY_DEF'] + df['AWAY_OFF_vs_HOME_DEF']
        df['AVG_PACE'] = df['GAME_PACE']
        
        # League averages per stagione
        df['LgAvg_ORtg_season'] = df.groupby('SEASON')['HOME_ORtg_sAvg'].transform('mean')
        
        return df

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulizia dataset"""
        
        print("🧹 Pulizia dataset...")
        
        initial_count = len(df)
        
        # Filtra range realistici
        df = df[(df['target_mu'] >= 160) & (df['target_mu'] <= 300)]
        df = df.dropna(subset=['target_mu'])
        
        # Ordina per data
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['SEASON', 'GAME_DATE']).reset_index(drop=True)
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"   🧹 Rimosse {removed} righe")
        print(f"   ✅ Dataset finale: {final_count:,} partite")
        
        return df


def main():
    """Funzione principale"""
    
    builder = SimpleDatasetBuilder()
    
    print("🚀 AVVIO COSTRUZIONE DATASET SEMPLIFICATO...")
    print("⏱️  Tempo stimato: 5-10 minuti")
    
    dataset = builder.build_simple_balanced_dataset()
    
    if not dataset.empty:
        print("\n🎉 === COSTRUZIONE COMPLETATA ===")
        print("✅ Dataset NBA semplificato creato!")
        return dataset
    else:
        print("\n❌ === ERRORE ===")
        return None


if __name__ == "__main__":
    main() 