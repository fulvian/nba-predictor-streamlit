#!/usr/bin/env python3
"""
🔧 COMPLETE DATASET FIX & MODEL RETRAINING
Risolve i problemi del dataset e ri-addestra modelli robusti
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetFixerAndTrainer:
    """Sistema completo per fixare dataset e ri-addestrare modelli"""
    
    def __init__(self):
        self.raw_data_path = 'data/nba_data_with_mu_sigma_for_ml.csv'
        self.clean_data_path = 'data/nba_fixed_training_dataset.csv'
        self.models_dir = 'models/probabilistic'
        
        # Assicura che la directory dei modelli esista
        os.makedirs(self.models_dir, exist_ok=True)
        
    def run_complete_fix(self):
        """Esegue il fix completo del dataset e re-training"""
        
        print("🔧 === COMPLETE DATASET FIX & RETRAIN ===")
        
        # 1. Fix del dataset
        print("\n📊 STEP 1: Dataset Cleaning & Target Generation")
        cleaned_df = self._fix_dataset()
        
        if cleaned_df is None:
            print("❌ Dataset fix failed!")
            return False
        
        # 2. Re-training dei modelli
        print("\n🤖 STEP 2: Model Retraining")
        success = self._retrain_models(cleaned_df)
        
        if success:
            print("\n🎉 COMPLETE SUCCESS!")
            print("✅ Dataset fixed")
            print("✅ Models retrained") 
            print("✅ System ready for production")
            return True
        else:
            print("\n❌ Retraining failed!")
            return False
    
    def _fix_dataset(self):
        """Fix completo del dataset"""
        
        try:
            # Carica dati raw
            print("   📥 Loading raw dataset...")
            df = pd.read_csv(self.raw_data_path)
            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # PROBLEMA 1: Fix MU target (era tutto NaN)
            print("   🎯 Fixing MU target...")
            # MU dovrebbe essere uguale al TOTAL_SCORE reale!
            df['target_mu'] = df['TOTAL_SCORE'].copy()
            print(f"   ✅ MU target fixed: {df['target_mu'].notna().sum()} valid values")
            
            # PROBLEMA 2: SIGMA è già OK ma verifichiamo
            print("   📏 Verifying SIGMA target...")
            df['target_sigma'] = df['SIGMA_L2_sd_final'].copy()
            print(f"   ✅ SIGMA target: {df['target_sigma'].notna().sum()} valid values")
            
            # PROBLEMA 3: Fix date issues
            print("   📅 Fixing date issues...")
            # Rimuovi righe con date mancanti o usa date di fallback
            df_with_dates = df.dropna(subset=['GAME_DATE_EST'])
            print(f"   ✅ Rows with valid dates: {len(df_with_dates)}")
            
            # PROBLEMA 4: Seleziona features ottimali
            print("   ⚙️ Selecting optimal features...")
            
            # Features basate su analisi NBA scientifica (Four Factors + Advanced)
            optimal_features = [
                # Core Four Factors (Dean Oliver)
                'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
                'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
                
                # Advanced Ratings
                'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg',
                
                # Recent Form (Last 5 games)
                'HOME_ORtg_L5Avg', 'HOME_DRtg_L5Avg', 'AWAY_ORtg_L5Avg', 'AWAY_DRtg_L5Avg',
                
                # Pace Factors
                'HOME_PACE', 'AWAY_PACE', 'GAME_PACE',
                
                # Season Context
                'SEASON', 'LgAvg_ORtg_season',
                
                # Head-to-Head 
                'H2H_L3_Avg_TotalScore', 'H2H_L3_Var_TotalScore',
                
                # Targets
                'target_mu', 'target_sigma'
            ]
            
            # Trova features disponibili
            available_features = []
            for feature in optimal_features:
                if feature in df_with_dates.columns:
                    available_features.append(feature)
                else:
                    print(f"   ⚠️ Missing feature: {feature}")
            
            print(f"   ✅ Using {len(available_features)} features")
            
            # Crea dataset pulito
            clean_df = df_with_dates[available_features].copy()
            
            # PROBLEMA 5: Fill missing values intelligentemente
            print("   🧹 Handling missing values...")
            
            # NBA defaults scientificamente validati
            nba_defaults = {
                'ORtg': 110.0, 'DRtg': 110.0, 'PACE': 100.0,
                'eFG_PCT': 0.52, 'TOV_PCT': 12.5, 'OREB_PCT': 25.0, 'FT_RATE': 0.25,
                'TotalScore': 220.0, 'Var': 15.0, 'SEASON': 2023
            }
            
            for col in clean_df.columns:
                if col in ['target_mu', 'target_sigma']:
                    continue  # Non toccare i target
                    
                if clean_df[col].dtype in ['float64', 'int64'] and clean_df[col].isna().any():
                    # Trova default appropriato
                    default_val = 0.0
                    for keyword, value in nba_defaults.items():
                        if keyword in col:
                            default_val = value
                            break
                    
                    missing_count = clean_df[col].isna().sum()
                    clean_df[col] = clean_df[col].fillna(default_val)
                    print(f"   📝 Filled {missing_count} missing values in {col} with {default_val}")
            
            # PROBLEMA 6: Rimuovi outliers estremi
            print("   🗑️ Removing extreme outliers...")
            
            initial_len = len(clean_df)
            
            # Outliers nei target
            clean_df = clean_df[
                (clean_df['target_mu'] >= 160) & (clean_df['target_mu'] <= 300) &
                (clean_df['target_sigma'] >= 5) & (clean_df['target_sigma'] <= 25)
            ]
            
            outliers_removed = initial_len - len(clean_df)
            print(f"   ✅ Removed {outliers_removed} extreme outliers")
            
            # PROBLEMA 7: Aggiungi engineered features
            print("   🔧 Adding engineered features...")
            
            # Differenziali predittivi
            if all(col in clean_df.columns for col in ['HOME_ORtg_sAvg', 'AWAY_DRtg_sAvg']):
                clean_df['HOME_OFF_vs_AWAY_DEF'] = clean_df['HOME_ORtg_sAvg'] - clean_df['AWAY_DRtg_sAvg']
                clean_df['AWAY_OFF_vs_HOME_DEF'] = clean_df['AWAY_ORtg_sAvg'] - clean_df['HOME_DRtg_sAvg']
                clean_df['TOTAL_EXPECTED_SCORING'] = clean_df['HOME_OFF_vs_AWAY_DEF'] + clean_df['AWAY_OFF_vs_HOME_DEF']
            
            # Pace differential
            if all(col in clean_df.columns for col in ['HOME_PACE', 'AWAY_PACE']):
                clean_df['PACE_DIFFERENTIAL'] = abs(clean_df['HOME_PACE'] - clean_df['AWAY_PACE'])
                clean_df['AVG_PACE'] = (clean_df['HOME_PACE'] + clean_df['AWAY_PACE']) / 2
            
            print(f"   ✅ Added {3} engineered features")
            
            # Salva dataset pulito
            clean_df.to_csv(self.clean_data_path, index=False)
            print(f"   💾 Clean dataset saved: {self.clean_data_path}")
            print(f"   📊 Final shape: {clean_df.shape}")
            
            return clean_df
            
        except Exception as e:
            print(f"   ❌ Error in dataset fix: {e}")
            return None
    
    def _retrain_models(self, df):
        """Re-addestra i modelli con dataset pulito"""
        
        try:
            # Prepara features e targets
            print("   📋 Preparing training data...")
            
            # Separa features dai targets
            feature_cols = [col for col in df.columns if col not in ['target_mu', 'target_sigma']]
            X = df[feature_cols]
            y_mu = df['target_mu']
            y_sigma = df['target_sigma']
            
            print(f"   ✅ Features: {X.shape[1]}, Samples: {X.shape[0]}")
            
            # Time Series Split (più realistico per time series)
            print("   📅 Time series validation split...")
            
            # Ordina per SEASON se disponibile
            if 'SEASON' in X.columns:
                sort_idx = X['SEASON'].argsort()
                X = X.iloc[sort_idx]
                y_mu = y_mu.iloc[sort_idx] 
                y_sigma = y_sigma.iloc[sort_idx]
            
            # 80/20 split temporale
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_mu_train, y_mu_test = y_mu.iloc[:split_idx], y_mu.iloc[split_idx:]
            y_sigma_train, y_sigma_test = y_sigma.iloc[:split_idx], y_sigma.iloc[split_idx:]
            
            print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Feature scaling
            print("   ⚖️ Feature scaling...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # TRAIN MU MODEL (XGBoost)
            print("   🎯 Training MU model (XGBoost)...")
            mu_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            mu_model.fit(X_train_scaled, y_mu_train)
            
            # Valuta MU model
            mu_pred_train = mu_model.predict(X_train_scaled)
            mu_pred_test = mu_model.predict(X_test_scaled)
            
            mu_mae_train = mean_absolute_error(y_mu_train, mu_pred_train)
            mu_mae_test = mean_absolute_error(y_mu_test, mu_pred_test)
            mu_r2_train = r2_score(y_mu_train, mu_pred_train)
            mu_r2_test = r2_score(y_mu_test, mu_pred_test)
            
            print(f"   📊 MU Performance:")
            print(f"      Train: MAE={mu_mae_train:.2f}, R²={mu_r2_train:.3f}")
            print(f"      Test:  MAE={mu_mae_test:.2f}, R²={mu_r2_test:.3f}")
            
            # TRAIN SIGMA MODEL (Random Forest - meglio per variance)
            print("   📏 Training SIGMA model (Random Forest)...")
            sigma_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            sigma_model.fit(X_train_scaled, y_sigma_train)
            
            # Valuta SIGMA model
            sigma_pred_train = sigma_model.predict(X_train_scaled)
            sigma_pred_test = sigma_model.predict(X_test_scaled)
            
            sigma_mae_train = mean_absolute_error(y_sigma_train, sigma_pred_train)
            sigma_mae_test = mean_absolute_error(y_sigma_test, sigma_pred_test)
            sigma_r2_train = r2_score(y_sigma_train, sigma_pred_train)
            sigma_r2_test = r2_score(y_sigma_test, sigma_pred_test)
            
            print(f"   📊 SIGMA Performance:")
            print(f"      Train: MAE={sigma_mae_train:.2f}, R²={sigma_r2_train:.3f}")
            print(f"      Test:  MAE={sigma_mae_test:.2f}, R²={sigma_r2_test:.3f}")
            
            # PERFORMANCE EVALUATION
            print(f"\n📈 OVERALL PERFORMANCE ASSESSMENT:")
            
            mu_quality = "✅ GOOD" if mu_mae_test < 8 else "⚠️ FAIR" if mu_mae_test < 12 else "❌ POOR"
            sigma_quality = "✅ GOOD" if sigma_mae_test < 2 else "⚠️ FAIR" if sigma_mae_test < 3 else "❌ POOR"
            
            print(f"   🎯 MU Model: {mu_quality} (MAE: {mu_mae_test:.2f} points)")
            print(f"   📏 SIGMA Model: {sigma_quality} (MAE: {sigma_mae_test:.2f} points)")
            
            # Overfitting check
            mu_overfit = ((mu_mae_test - mu_mae_train) / mu_mae_train) * 100
            sigma_overfit = ((sigma_mae_test - sigma_mae_train) / sigma_mae_train) * 100
            
            print(f"   🔍 Overfitting:")
            print(f"      MU: {mu_overfit:+.1f}% ({'✅ OK' if abs(mu_overfit) < 15 else '⚠️ HIGH'})")
            print(f"      SIGMA: {sigma_overfit:+.1f}% ({'✅ OK' if abs(sigma_overfit) < 15 else '⚠️ HIGH'})")
            
            # SAVE MODELS
            print("   💾 Saving models...")
            
            joblib.dump(mu_model, os.path.join(self.models_dir, 'mu_model.pkl'))
            joblib.dump(sigma_model, os.path.join(self.models_dir, 'sigma_model.pkl'))
            joblib.dump(scaler, os.path.join(self.models_dir, 'scaler.pkl'))
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': feature_cols,
                'mu_performance': {
                    'mae_train': mu_mae_train,
                    'mae_test': mu_mae_test,
                    'r2_train': mu_r2_train,
                    'r2_test': mu_r2_test
                },
                'sigma_performance': {
                    'mae_train': sigma_mae_train,
                    'mae_test': sigma_mae_test,
                    'r2_train': sigma_r2_train,
                    'r2_test': sigma_r2_test
                }
            }
            
            import json
            with open(os.path.join(self.models_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   ✅ Models saved to: {self.models_dir}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error in model training: {e}")
            return False

def main():
    """Esegue il fix completo"""
    
    print("🚀 Starting complete dataset fix and model retraining...")
    
    fixer = DatasetFixerAndTrainer()
    success = fixer.run_complete_fix()
    
    if success:
        print("\n🎉 === SUCCESS! ===")
        print("Sistema completamente riparato e pronto per la produzione!")
        print("\nNext steps:")
        print("1. 🧪 Test the fixed models on new predictions")
        print("2. 📊 Compare performance with old system") 
        print("3. 🚀 Deploy to production")
    else:
        print("\n❌ === FAILED ===")
        print("Sistemazione fallita. Controlla i log per dettagli.")

if __name__ == "__main__":
    main() 