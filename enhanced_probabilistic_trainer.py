#!/usr/bin/env python3
"""
🚀 ENHANCED PROBABILISTIC MODEL TRAINER
Sistema di training avanzato con validazione robusta e hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedProbabilisticTrainer:
    """Trainer avanzato per modelli probabilistici NBA"""
    
    def __init__(self, data_path='data/nba_clean_training_dataset.csv'):
        self.data_path = data_path
        self.models_dir = 'models/probabilistic_enhanced'
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.mu_model = None
        self.sigma_model = None
        self.scaler = None
        
        # Hyperparameters grids
        self.xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        self.rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    def load_and_prepare_data(self):
        """Carica e prepara i dati per il training"""
        
        print("📊 === DATA PREPARATION ===")
        
        try:
            # Carica dataset
            df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {df.shape}")
            
            # Separa features e targets
            target_cols = ['target_mu', 'target_sigma']
            feature_cols = [col for col in df.columns if col not in target_cols]
            
            X = df[feature_cols]
            y_mu = df['target_mu']
            y_sigma = df['target_sigma']
            
            # Rimuovi features non numeriche o problematiche
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_features]
            
            print(f"📈 Features: {len(numeric_features)} numeric features")
            print(f"🎯 Targets: μ={len(y_mu)}, σ={len(y_sigma)}")
            
            # Check per missing values
            if X_numeric.isnull().sum().sum() > 0:
                print("⚠️ Missing values detected, filling with median...")
                X_numeric = X_numeric.fillna(X_numeric.median())
            
            return X_numeric, y_mu, y_sigma
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None
    
    def temporal_train_test_split(self, X, y_mu, y_sigma, test_size=0.2):
        """Split temporale dei dati per validazione realistica"""
        
        print(f"\n📅 === TEMPORAL TRAIN/TEST SPLIT ===")
        
        # Split temporale: ultimi 20% per test
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_mu_train = y_mu.iloc[:split_idx]
        y_mu_test = y_mu.iloc[split_idx:]
        y_sigma_train = y_sigma.iloc[:split_idx]
        y_sigma_test = y_sigma.iloc[split_idx:]
        
        print(f"🏋️ Train set: {len(X_train)} samples")
        print(f"🧪 Test set: {len(X_test)} samples")
        print(f"📊 Train μ: {y_mu_train.mean():.1f} ± {y_mu_train.std():.1f}")
        print(f"📊 Test μ: {y_mu_test.mean():.1f} ± {y_mu_test.std():.1f}")
        
        return X_train, X_test, y_mu_train, y_mu_test, y_sigma_train, y_sigma_test
    
    def hyperparameter_optimization(self, X_train, y_train, model_type='xgboost', target='mu'):
        """Ottimizzazione hyperparameters con TimeSeriesSplit"""
        
        print(f"\n🔧 === HYPERPARAMETER OPTIMIZATION ({model_type.upper()} - {target.upper()}) ===")
        
        # Configurazione Cross-Validation temporale
        tscv = TimeSeriesSplit(n_splits=3)
        
        if model_type == 'xgboost':
            base_model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            param_grid = self.xgb_param_grid
        else:  # RandomForest
            base_model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
            param_grid = self.rf_param_grid
        
        # Usa RandomizedSearchCV per efficiency
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,  # Numero di combinazioni da testare
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print(f"🎯 Running {search.n_iter} random searches with {tscv.n_splits}-fold TimeSeriesCV...")
        search.fit(X_train, y_train)
        
        print(f"✅ Best score: {-search.best_score_:.3f} MAE")
        print(f"🏆 Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_, -search.best_score_
    
    def train_and_validate(self, optimize_hyperparams=True):
        """Training completo con validazione robusta"""
        
        print("🚀 === ENHANCED PROBABILISTIC MODEL TRAINING ===")
        
        # 1. Carica dati
        X, y_mu, y_sigma = self.load_and_prepare_data()
        if X is None:
            return False
        
        # 2. Split temporale
        X_train, X_test, y_mu_train, y_mu_test, y_sigma_train, y_sigma_test = \
            self.temporal_train_test_split(X, y_mu, y_sigma)
        
        # 3. Scaling
        print(f"\n⚖️ === FEATURE SCALING ===")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"✅ Features scaled using StandardScaler")
        
        # 4. Training MU MODEL
        print(f"\n🎯 === MU MODEL TRAINING ===")
        
        if optimize_hyperparams:
            # Ottimizzazione hyperparameters
            self.mu_model, mu_best_params, mu_cv_score = \
                self.hyperparameter_optimization(X_train_scaled, y_mu_train, 'xgboost', 'mu')
        else:
            # Modello con parametri default buoni
            self.mu_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
            self.mu_model.fit(X_train_scaled, y_mu_train)
        
        # Valutazione Mu Model
        mu_pred_train = self.mu_model.predict(X_train_scaled)
        mu_pred_test = self.mu_model.predict(X_test_scaled)
        
        mu_mae_train = mean_absolute_error(y_mu_train, mu_pred_train)
        mu_mae_test = mean_absolute_error(y_mu_test, mu_pred_test)
        mu_r2_train = r2_score(y_mu_train, mu_pred_train)
        mu_r2_test = r2_score(y_mu_test, mu_pred_test)
        
        print(f"📊 MU RESULTS:")
        print(f"   🏋️ Train: MAE={mu_mae_train:.2f}, R²={mu_r2_train:.3f}")
        print(f"   🧪 Test:  MAE={mu_mae_test:.2f}, R²={mu_r2_test:.3f}")
        
        # 5. Training SIGMA MODEL
        print(f"\n📏 === SIGMA MODEL TRAINING ===")
        
        if optimize_hyperparams:
            # Ottimizzazione hyperparameters per Sigma
            self.sigma_model, sigma_best_params, sigma_cv_score = \
                self.hyperparameter_optimization(X_train_scaled, y_sigma_train, 'xgboost', 'sigma')
        else:
            # Modello default per sigma (spesso meno complesso)
            self.sigma_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.sigma_model.fit(X_train_scaled, y_sigma_train)
        
        # Valutazione Sigma Model
        sigma_pred_train = self.sigma_model.predict(X_train_scaled)
        sigma_pred_test = self.sigma_model.predict(X_test_scaled)
        
        sigma_mae_train = mean_absolute_error(y_sigma_train, sigma_pred_train)
        sigma_mae_test = mean_absolute_error(y_sigma_test, sigma_pred_test)
        sigma_r2_train = r2_score(y_sigma_train, sigma_pred_train)
        sigma_r2_test = r2_score(y_sigma_test, sigma_pred_test)
        
        print(f"📊 SIGMA RESULTS:")
        print(f"   🏋️ Train: MAE={sigma_mae_train:.2f}, R²={sigma_r2_train:.3f}")
        print(f"   🧪 Test:  MAE={sigma_mae_test:.2f}, R²={sigma_r2_test:.3f}")
        
        # 6. Overfitting Analysis
        print(f"\n🔍 === OVERFITTING ANALYSIS ===")
        mu_overfit = (mu_mae_test - mu_mae_train) / mu_mae_train * 100
        sigma_overfit = (sigma_mae_test - sigma_mae_train) / sigma_mae_train * 100
        
        print(f"🎯 Mu overfitting: {mu_overfit:+.1f}% ({'⚠️ HIGH' if abs(mu_overfit) > 15 else '✅ OK'})")
        print(f"📏 Sigma overfitting: {sigma_overfit:+.1f}% ({'⚠️ HIGH' if abs(sigma_overfit) > 15 else '✅ OK'})")
        
        # 7. Feature Importance
        self._analyze_feature_importance(X_train.columns)
        
        # 8. Save Models
        print(f"\n💾 === SAVING MODELS ===")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            mu_path = os.path.join(self.models_dir, f'mu_model_{timestamp}.pkl')
            sigma_path = os.path.join(self.models_dir, f'sigma_model_{timestamp}.pkl')
            scaler_path = os.path.join(self.models_dir, f'scaler_{timestamp}.pkl')
            
            joblib.dump(self.mu_model, mu_path)
            joblib.dump(self.sigma_model, sigma_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Salva anche nelle directory principali per compatibilità
            joblib.dump(self.mu_model, 'models/probabilistic/mu_model.pkl')
            joblib.dump(self.sigma_model, 'models/probabilistic/sigma_model.pkl')
            joblib.dump(self.scaler, 'models/probabilistic/scaler.pkl')
            
            print(f"✅ Models saved:")
            print(f"   📁 {mu_path}")
            print(f"   📁 {sigma_path}")
            print(f"   📁 {scaler_path}")
            
            # Salva metadata
            metadata = {
                'timestamp': timestamp,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(X_train.columns),
                'mu_performance': {
                    'train_mae': mu_mae_train,
                    'test_mae': mu_mae_test,
                    'train_r2': mu_r2_train,
                    'test_r2': mu_r2_test,
                    'overfitting_pct': mu_overfit
                },
                'sigma_performance': {
                    'train_mae': sigma_mae_train,
                    'test_mae': sigma_mae_test,
                    'train_r2': sigma_r2_train,
                    'test_r2': sigma_r2_test,
                    'overfitting_pct': sigma_overfit
                }
            }
            
            metadata_path = os.path.join(self.models_dir, f'training_metadata_{timestamp}.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Metadata saved: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
        
        # 9. Final Summary
        print(f"\n🎉 === TRAINING COMPLETED ===")
        print(f"✅ μ Model: MAE {mu_mae_test:.1f} points ({'🟢 Excellent' if mu_mae_test < 8 else '🟡 Good' if mu_mae_test < 12 else '🔴 Needs work'})")
        print(f"✅ σ Model: MAE {sigma_mae_test:.1f} points ({'🟢 Excellent' if sigma_mae_test < 3 else '🟡 Good' if sigma_mae_test < 5 else '🔴 Needs work'})")
        
        return True
    
    def _analyze_feature_importance(self, feature_names):
        """Analizza l'importanza delle features"""
        
        print(f"\n🔍 === FEATURE IMPORTANCE ===")
        
        try:
            # Mu model importance
            mu_importance = self.mu_model.feature_importances_
            mu_top_features = sorted(zip(feature_names, mu_importance), 
                                   key=lambda x: x[1], reverse=True)[:10]
            
            print(f"🎯 TOP 10 MU FEATURES:")
            for i, (feature, importance) in enumerate(mu_top_features, 1):
                print(f"   {i:2d}. {feature:<25} {importance:.3f}")
            
            # Sigma model importance
            sigma_importance = self.sigma_model.feature_importances_
            sigma_top_features = sorted(zip(feature_names, sigma_importance), 
                                      key=lambda x: x[1], reverse=True)[:10]
            
            print(f"\n📏 TOP 10 SIGMA FEATURES:")
            for i, (feature, importance) in enumerate(sigma_top_features, 1):
                print(f"   {i:2d}. {feature:<25} {importance:.3f}")
                
        except Exception as e:
            print(f"⚠️ Could not analyze feature importance: {e}")

def main():
    """Esegue il training enhanced"""
    
    # Verifica se esiste il dataset pulito
    if not os.path.exists('data/nba_clean_training_dataset.csv'):
        print("❌ Clean dataset not found!")
        print("   Please run: python enhanced_training_data_builder.py")
        return
    
    trainer = EnhancedProbabilisticTrainer()
    
    print("🚀 Starting Enhanced Probabilistic Model Training...")
    print("=" * 60)
    
    success = trainer.train_and_validate(optimize_hyperparams=True)
    
    if success:
        print("\n🎉 SUCCESS! Enhanced models trained and saved.")
        print("\nNext steps:")
        print("1. 🧪 Test the models on recent games")
        print("2. 📊 Compare with old model performance")
        print("3. 🚀 Deploy to production system")
    else:
        print("\n❌ Training failed. Check logs above.")

if __name__ == "__main__":
    main() 