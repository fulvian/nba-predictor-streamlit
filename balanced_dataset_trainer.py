#!/usr/bin/env python3
"""
🎯 BALANCED DATASET TRAINER
Trainer ottimizzato per dataset NBA bilanciato con 5 stagioni consecutive
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

class BalancedDatasetTrainer:
    """Trainer per dataset NBA bilanciato con validazione temporale"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.models_dir = os.path.join(self.base_dir, 'models', 'probabilistic')
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Features scientificamente validate (Four Factors + Advanced)
        self.selected_features = [
            # Four Factors (Dean Oliver)
            'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
            'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
            
            # Advanced Ratings
            'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg',
            
            # Pace & Matchups
            'HOME_PACE', 'AWAY_PACE', 'GAME_PACE', 'PACE_DIFFERENTIAL',
            'HOME_OFF_vs_AWAY_DEF', 'AWAY_OFF_vs_HOME_DEF', 'TOTAL_EXPECTED_SCORING',
            
            # Context
            'LgAvg_ORtg_season', 'AVG_PACE'
        ]
        
        print("🎯 BALANCED DATASET TRAINER INIZIALIZZATO")
        print(f"📊 Features selezionate: {len(self.selected_features)}")

    def load_and_prepare_data(self, dataset_path: str = None):
        """Carica e prepara il dataset bilanciato"""
        
        if not dataset_path:
            dataset_path = os.path.join(self.data_dir, 'nba_complete_dataset.csv')
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset non trovato: {dataset_path}")
            return pd.DataFrame(), False
        
        print(f"📂 Caricamento dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Verifica bilanciamento
        print(f"📊 Dataset originale: {len(df):,} partite")
        
        if 'SEASON' in df.columns:
            season_counts = df['SEASON'].value_counts().sort_index()
            print(f"📅 DISTRIBUZIONE PER STAGIONE:")
            for season, count in season_counts.items():
                print(f"   {season}: {count:,} partite")
            
            cv = season_counts.std() / season_counts.mean()
            print(f"📈 Coefficiente di Variazione: {cv:.3f}")
            is_balanced = cv < 0.5
            
            print(f"⚖️ Dataset {'BILANCIATO' if is_balanced else 'SBILANCIATO'}")
        else:
            is_balanced = False
        
        # Pulizia e preparazione
        df = self._clean_and_prepare(df)
        
        return df, is_balanced

    def _clean_and_prepare(self, df):
        """Pulizia e preparazione del dataset"""
        
        print("🧹 Pulizia e preparazione dataset...")
        
        initial_count = len(df)
        
        # Rimuovi valori mancanti
        df = df.dropna(subset=['target_mu', 'target_sigma'])
        df = df.dropna(subset=self.selected_features, how='any')
        
        # Filtra range realistici
        df = df[(df['target_mu'] >= 150) & (df['target_mu'] <= 320)]
        df = df[(df['target_sigma'] >= 5) & (df['target_sigma'] <= 25)]
        
        # Ordina cronologicamente
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['SEASON', 'GAME_DATE']).reset_index(drop=True)
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"   🧹 Rimosse {removed} righe ({removed/initial_count*100:.1f}%)")
        print(f"   ✅ Dataset pulito: {final_count:,} partite")
        
        return df

    def train_models(self, df):
        """Training con validazione temporale"""
        
        print("\n🚀 === TRAINING CON VALIDAZIONE TEMPORALE ===")
        
        X = df[self.selected_features].copy()
        y_mu = df['target_mu'].copy()
        y_sigma = df['target_sigma'].copy()
        
        print(f"📊 Shape dati: X{X.shape}, y_mu{y_mu.shape}, y_sigma{y_sigma.shape}")
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Modelli da testare
        models_to_test = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        best_models = {}
        
        for target_name, y_target in [('mu', y_mu), ('sigma', y_sigma)]:
            print(f"\n🎯 === TRAINING TARGET: {target_name.upper()} ===")
            
            best_score = float('inf')
            best_model_info = None
            
            for model_name, model in models_to_test.items():
                print(f"\n🔄 Testing {model_name} per {target_name}...")
                
                # Scalatura
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y_target, 
                    cv=tscv, 
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                mae_scores = -cv_scores
                mean_mae = mae_scores.mean()
                std_mae = mae_scores.std()
                
                print(f"   📊 {model_name}: MAE = {mean_mae:.2f} ± {std_mae:.2f}")
                
                if mean_mae < best_score:
                    best_score = mean_mae
                    best_model_info = {
                        'name': model_name,
                        'model': model,
                        'scaler': scaler,
                        'mae': mean_mae
                    }
            
            print(f"\n🏆 MIGLIOR MODELLO per {target_name}: {best_model_info['name']}")
            print(f"   📊 MAE: {best_model_info['mae']:.2f}")
            
            best_models[target_name] = best_model_info
        
        return self._final_training_and_save(X, y_mu, y_sigma, best_models)

    def _final_training_and_save(self, X, y_mu, y_sigma, best_models):
        """Training finale e salvataggio"""
        
        print("\n💾 === TRAINING FINALE E SALVATAGGIO ===")
        
        # Split finale
        X_train, X_test, y_mu_train, y_mu_test, y_sigma_train, y_sigma_test = train_test_split(
            X, y_mu, y_sigma, test_size=0.2, random_state=42, shuffle=False
        )
        
        final_results = {}
        
        for target_name in ['mu', 'sigma']:
            print(f"\n🎯 Training finale {target_name.upper()}...")
            
            model_info = best_models[target_name]
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Prepara dati
            y_train = y_mu_train if target_name == 'mu' else y_sigma_train
            y_test = y_mu_test if target_name == 'mu' else y_sigma_test
            
            # Scala features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Training finale
            model.fit(X_train_scaled, y_train)
            
            # Predizioni
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Metriche
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Overfitting (protezione da divisione per zero)
            overfitting = ((train_mae - test_mae) / max(train_mae, 1e-6)) * 100
            
            print(f"   📊 {model_info['name']} {target_name.upper()}:")
            print(f"      Train MAE: {train_mae:.2f}")
            print(f"      Test MAE:  {test_mae:.2f}")
            print(f"      Train R²:  {train_r2:.3f}")
            print(f"      Test R²:   {test_r2:.3f}")
            print(f"      Overfitting: {overfitting:.1f}%")
            
            # Salva modello
            model_filename = f"{target_name}_model.pkl"
            scaler_filename = f"{target_name}_scaler.pkl"
            
            joblib.dump(model, os.path.join(self.models_dir, model_filename))
            joblib.dump(scaler, os.path.join(self.models_dir, scaler_filename))
            
            print(f"   💾 Salvato: {model_filename}")
            
            final_results[target_name] = {
                'model_name': model_info['name'],
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'overfitting_percent': overfitting
            }
        
        # Salva metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'features_used': self.selected_features,
            'dataset_size': len(X),
            'test_size': len(X_test),
            'results': final_results
        }
        
        metadata_path = os.path.join(self.models_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n✅ TRAINING COMPLETATO!")
        print(f"📁 Modelli salvati in: {self.models_dir}")
        
        return final_results


def main():
    """Funzione principale"""
    
    trainer = BalancedDatasetTrainer()
    
    print("🎯 === TRAINING SU DATASET BILANCIATO ===")
    
    # Carica dataset
    df, is_balanced = trainer.load_and_prepare_data()
    
    if df.empty:
        print("❌ Impossibile caricare il dataset")
        return None
    
    if not is_balanced:
        print("⚠️ ATTENZIONE: Dataset non perfettamente bilanciato")
    
    # Training
    results = trainer.train_models(df)
    
    print("\n🎉 === TRAINING COMPLETATO ===")
    print("✅ Modelli ottimizzati per dataset bilanciato!")
    
    return results


if __name__ == "__main__":
    main() 