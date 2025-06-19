#!/usr/bin/env python3
"""
🔨 ENHANCED TRAINING DATA BUILDER
Ricostruisce il dataset con target MU/SIGMA validi basati sui risultati reali
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from datetime import datetime
import joblib

class EnhancedTrainingDataBuilder:
    """Costruisce dataset robusto per training del modello probabilistico"""
    
    def __init__(self):
        self.raw_data_path = 'data/nba_data_with_mu_sigma_for_ml.csv'
        self.output_path = 'data/nba_clean_training_dataset.csv'
        
    def build_enhanced_dataset(self):
        """Costruisce dataset pulito con target validi"""
        
        print("🔨 === ENHANCED DATASET BUILDER ===")
        
        # 1. CARICA DATI RAW
        try:
            print("\n📊 Loading raw dataset...")
            df = pd.read_csv(self.raw_data_path)
            print(f"   ✅ Raw data loaded: {len(df)} samples")
            
            # Filtra solo righe con TOTAL_SCORE valido (ground truth disponibile)
            valid_games = df.dropna(subset=['TOTAL_SCORE'])
            print(f"   🎯 Games with actual scores: {len(valid_games)}")
            
            if len(valid_games) == 0:
                print("   ❌ No games with valid total scores found!")
                return None
                
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
        
        # 2. GENERA TARGET REALISTICI DA PUNTEGGI REALI
        print("\n🎯 Generating realistic targets from actual scores...")
        
        # Mu = TOTAL_SCORE (punteggio reale della partita)
        valid_games['target_mu'] = valid_games['TOTAL_SCORE'].astype(float)
        
        # Sigma = Calcolato come deviazione standard rolling del total score
        # per squadre simili o periodo temporale
        valid_games['target_sigma'] = self._calculate_realistic_sigma(valid_games)
        
        print(f"   ✅ Target Mu range: {valid_games['target_mu'].min():.1f} - {valid_games['target_mu'].max():.1f}")
        print(f"   ✅ Target Sigma range: {valid_games['target_sigma'].min():.1f} - {valid_games['target_sigma'].max():.1f}")
        
        # 3. SELEZIONA E PULISCE FEATURES
        print("\n⚙️ Selecting and cleaning features...")
        
        # Features principali usate dal modello attuale
        feature_candidates = [
            # Offensive/Defensive Ratings
            'HOME_ORtg', 'HOME_DRtg', 'AWAY_ORtg', 'AWAY_DRtg',
            'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg',
            
            # Pace
            'HOME_PACE', 'AWAY_PACE', 'GAME_PACE',
            
            # Four Factors
            'HOME_eFG_PCT', 'HOME_TOV_PCT', 'HOME_OREB_PCT', 'HOME_FT_RATE',
            'AWAY_eFG_PCT', 'AWAY_TOV_PCT', 'AWAY_OREB_PCT', 'AWAY_FT_RATE',
            'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
            'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
            
            # Last 5 games trends
            'HOME_ORtg_L5Avg', 'HOME_DRtg_L5Avg', 'AWAY_ORtg_L5Avg', 'AWAY_DRtg_L5Avg',
            
            # Additional context
            'SEASON', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'
        ]
        
        # Trova features disponibili
        available_features = []
        for feature in feature_candidates:
            if feature in valid_games.columns:
                available_features.append(feature)
            else:
                # Cerca alternative simili
                similar = [col for col in valid_games.columns if 
                          any(part in col for part in feature.split('_'))]
                if similar:
                    print(f"   📝 {feature} -> using {similar[0]}")
                    available_features.append(similar[0])
        
        print(f"   ✅ Selected {len(available_features)} features")
        
        # 4. PULISCI E STANDARDIZZA
        print("\n🧹 Cleaning and standardizing...")
        
        # Crea dataset pulito
        clean_data = valid_games[available_features + ['target_mu', 'target_sigma', 'GAME_DATE_EST']].copy()
        
        # Rimuovi outliers estremi
        clean_data = self._remove_outliers(clean_data)
        
        # Fill missing values con valori realistici NBA
        clean_data = self._fill_missing_values(clean_data)
        
        # Ordina per data
        clean_data['GAME_DATE'] = pd.to_datetime(clean_data['GAME_DATE_EST'])
        clean_data = clean_data.sort_values('GAME_DATE')
        
        print(f"   ✅ Clean dataset: {len(clean_data)} samples")
        
        # 5. AGGIUNGI ENGINEERED FEATURES
        print("\n🔧 Adding engineered features...")
        
        # Differenziali squadre
        if 'HOME_ORtg_sAvg' in clean_data.columns and 'AWAY_DRtg_sAvg' in clean_data.columns:
            clean_data['HOME_OFF_vs_AWAY_DEF'] = clean_data['HOME_ORtg_sAvg'] - clean_data['AWAY_DRtg_sAvg']
            clean_data['AWAY_OFF_vs_HOME_DEF'] = clean_data['AWAY_ORtg_sAvg'] - clean_data['HOME_DRtg_sAvg']
        
        # Pace differential
        if 'HOME_PACE' in clean_data.columns and 'AWAY_PACE' in clean_data.columns:
            clean_data['PACE_DIFFERENTIAL'] = abs(clean_data['HOME_PACE'] - clean_data['AWAY_PACE'])
        
        # Season trend (inizio vs fine stagione)
        if 'GAME_DATE' in clean_data.columns:
            clean_data['DAYS_FROM_SEASON_START'] = (clean_data['GAME_DATE'] - clean_data['GAME_DATE'].min()).dt.days
        
        print(f"   ✅ Added engineered features")
        
        # 6. SALVA DATASET
        try:
            # Rimuovi colonne temporanee
            final_columns = [col for col in clean_data.columns if col not in ['GAME_DATE_EST', 'GAME_DATE']]
            final_dataset = clean_data[final_columns]
            
            final_dataset.to_csv(self.output_path, index=False)
            print(f"\n💾 Dataset saved to: {self.output_path}")
            print(f"   📊 Final shape: {final_dataset.shape}")
            print(f"   🎯 Target Mu stats: mean={final_dataset['target_mu'].mean():.1f}, std={final_dataset['target_mu'].std():.1f}")
            print(f"   📏 Target Sigma stats: mean={final_dataset['target_sigma'].mean():.1f}, std={final_dataset['target_sigma'].std():.1f}")
            
            return final_dataset
            
        except Exception as e:
            print(f"   ❌ Error saving dataset: {e}")
            return None
    
    def _calculate_realistic_sigma(self, df):
        """Calcola sigma realistico basato su variabilità storica"""
        
        # Metodo 1: Sigma basato su rolling std dei total scores
        df_sorted = df.sort_values('GAME_DATE_EST')
        rolling_std = df_sorted['TOTAL_SCORE'].rolling(window=50, min_periods=10).std()
        
        # Metodo 2: Sigma basato su differenze team-specific
        team_variance = []
        for _, row in df.iterrows():
            # Variance basata su pace e rating differentials
            pace_factor = abs(row.get('HOME_PACE', 100) - 100) / 10  # Higher pace = more variance
            rating_uncertainty = abs(row.get('HOME_ORtg_sAvg', 110) - row.get('AWAY_DRtg_sAvg', 110)) / 20
            
            estimated_sigma = 8.0 + pace_factor + rating_uncertainty  # Base sigma NBA ~ 8-12
            team_variance.append(estimated_sigma)
        
        # Combina entrambi i metodi
        combined_sigma = []
        for i, (rolling_val, team_val) in enumerate(zip(rolling_std, team_variance)):
            if pd.isna(rolling_val):
                combined_sigma.append(team_val)
            else:
                # Media pesata: 70% rolling, 30% team-specific
                combined_sigma.append(0.7 * rolling_val + 0.3 * team_val)
        
        # Clamp a range realistico NBA
        return np.clip(combined_sigma, 6.0, 18.0)
    
    def _remove_outliers(self, df):
        """Rimuove outliers estremi dai dati"""
        
        initial_len = len(df)
        
        # Rimuovi punteggi impossibili
        df = df[(df['target_mu'] >= 150) & (df['target_mu'] <= 300)]
        
        # Rimuovi sigma impossibili
        df = df[(df['target_sigma'] >= 4) & (df['target_sigma'] <= 25)]
        
        # Rimuovi outliers nelle features principali usando IQR
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature not in ['target_mu', 'target_sigma']:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 3*IQR invece di 1.5 per essere meno aggressivi
                upper_bound = Q3 + 3 * IQR
                df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        
        print(f"   🗑️ Removed {initial_len - len(df)} outliers")
        return df
    
    def _fill_missing_values(self, df):
        """Riempie valori mancanti con defaults NBA realistici"""
        
        # Defaults basati su statistiche NBA reali
        nba_defaults = {
            'ORtg': 110.0, 'DRtg': 110.0, 'PACE': 100.0,
            'eFG_PCT': 0.52, 'TOV_PCT': 12.5, 'OREB_PCT': 25.0, 'FT_RATE': 0.25,
            'PTS': 110.0, 'AST': 25.0, 'REB': 45.0
        }
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64'] and df[column].isna().any():
                # Trova default appropriato
                default_value = 0.0
                for keyword, value in nba_defaults.items():
                    if keyword in column.upper():
                        default_value = value
                        break
                
                df[column] = df[column].fillna(default_value)
        
        return df

def main():
    """Esegue il rebuild del dataset"""
    builder = EnhancedTrainingDataBuilder()
    dataset = builder.build_enhanced_dataset()
    
    if dataset is not None:
        print("\n🎉 SUCCESS! Enhanced dataset ready for training.")
        print("\nNext steps:")
        print("1. 📊 Review the dataset quality")
        print("2. 🤖 Retrain the probabilistic models")
        print("3. 📈 Validate on recent games")
    else:
        print("\n❌ Failed to build enhanced dataset.")

if __name__ == "__main__":
    main() 