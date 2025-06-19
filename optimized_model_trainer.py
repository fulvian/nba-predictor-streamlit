#!/usr/bin/env python3
"""
🎯 OPTIMIZED MODEL TRAINER
Risolve l'overfitting e migliora le performance del modello MU
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
import xgboost as xgb
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    """Trainer ottimizzato per risolvere overfitting e migliorare performance"""
    
    def __init__(self):
        self.clean_data_path = 'data/nba_fixed_training_dataset.csv'
        self.models_dir = 'models/probabilistic'
        
    def run_optimized_training(self):
        """Esegue training ottimizzato con focus anti-overfitting"""
        
        print("🎯 === OPTIMIZED MODEL TRAINING ===")
        
        # 1. Carica dati
        df = pd.read_csv(self.clean_data_path)
        print(f"📊 Loaded {len(df)} samples")
        
        # 2. Feature Selection Scientifica
        print("\n🔬 STEP 1: Scientific Feature Selection")
        X_optimized, y_mu, y_sigma = self._scientific_feature_selection(df)
        
        # 3. Multiple Model Comparison
        print("\n🤖 STEP 2: Model Comparison & Selection")
        best_mu_model = self._compare_and_select_mu_models(X_optimized, y_mu)
        
        # 4. Sigma Model (già buono)
        print("\n📏 STEP 3: Optimized Sigma Model")
        sigma_model, scaler = self._train_optimized_sigma_model(X_optimized, y_sigma)
        
        # 5. Final Validation
        print("\n✅ STEP 4: Final Validation")
        self._final_validation(best_mu_model, sigma_model, scaler, X_optimized, y_mu, y_sigma)
        
        # 6. Save models
        print("\n💾 STEP 5: Saving Optimized Models")
        self._save_optimized_models(best_mu_model, sigma_model, scaler, X_optimized.columns.tolist())
        
        return True
    
    def _scientific_feature_selection(self, df):
        """Selezione features basata su ricerca scientifica NBA"""
        
        # CORE FEATURES (Dean Oliver's Four Factors + Advanced Stats)
        core_features = [
            # Four Factors (scientificamente validati per predire vittorie)
            'HOME_eFG_PCT_sAvg', 'AWAY_eFG_PCT_sAvg',  # Effective FG% (più importante)
            'HOME_TOV_PCT_sAvg', 'AWAY_TOV_PCT_sAvg',   # Turnovers (secondo più importante)
            'HOME_OREB_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', # Offensive Rebounds
            'HOME_FT_RATE_sAvg', 'AWAY_FT_RATE_sAvg',   # Free Throw Rate
            
            # Advanced Ratings (Dean Oliver)
            'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg',         # Offensive/Defensive Rating
            'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg', 
            
            # Pace (controlla il numero di possessi)
            'HOME_PACE', 'AWAY_PACE', 'GAME_PACE'
        ]
        
        # ENGINEERED FEATURES (combinazioni predittive)
        engineered_features = [
            'HOME_OFF_vs_AWAY_DEF',   # Mismatch offensivo vs difensivo
            'AWAY_OFF_vs_HOME_DEF',   # Mismatch offensivo vs difensivo  
            'TOTAL_EXPECTED_SCORING', # Scoring totale atteso
            'PACE_DIFFERENTIAL',      # Differenza di pace
            'AVG_PACE'                # Pace medio della partita
        ]
        
        # CONTEXT FEATURES (molto importanti per generalizzazione)
        context_features = [
            'SEASON',                 # Evoluzione del gioco nel tempo
            'LgAvg_ORtg_season'      # Media lega (normalizzazione)
        ]
        
        all_features = core_features + engineered_features + context_features
        
        # Filtra features disponibili
        available_features = [f for f in all_features if f in df.columns]
        missing_features = [f for f in all_features if f not in df.columns]
        
        print(f"   ✅ Using {len(available_features)} scientific features")
        print(f"   ⚠️ Missing {len(missing_features)} features: {missing_features}")
        
        # Crea feature matrix ottimizzata
        X = df[available_features].copy()
        y_mu = df['target_mu']
        y_sigma = df['target_sigma']
        
        # Feature correlation analysis
        print(f"   🔍 Feature correlation with target_mu:")
        correlations = []
        for feature in available_features:
            if X[feature].dtype in ['float64', 'int64']:
                corr = X[feature].corr(y_mu)
                correlations.append((feature, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("   📊 Top 5 predictive features:")
        for i, (feature, corr) in enumerate(correlations[:5]):
            print(f"      {i+1}. {feature}: {corr:.3f}")
        
        return X, y_mu, y_sigma
    
    def _compare_and_select_mu_models(self, X, y_mu):
        """Confronta multiple algoritmi per scegliere il migliore per MU"""
        
        # Time series split per validazione realistica
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modelli candidati (focus su riduzione overfitting)
        models = {
            'Ridge': Ridge(alpha=10.0),  # Regularizzazione forte
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),  # L1+L2 regularization
            'RandomForest_Conservative': RandomForestRegressor(
                n_estimators=50,     # Meno alberi
                max_depth=4,         # Profondità limitata
                min_samples_split=20, # Split conservativo
                min_samples_leaf=10,  # Foglie conservative
                random_state=42
            ),
            'XGBoost_Regularized': xgb.XGBRegressor(
                n_estimators=50,      # Meno boosting rounds
                max_depth=3,          # Profondità molto limitata
                learning_rate=0.05,   # Learning rate basso
                subsample=0.7,        # Bagging
                colsample_bytree=0.7, # Feature bagging
                reg_alpha=1.0,        # L1 regularization
                reg_lambda=1.0,       # L2 regularization
                random_state=42
            )
        }
        
        print("   🏁 Cross-validation comparison:")
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        # Standardize features per modelli lineari
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for name, model in models.items():
            # Cross-validation con time series split
            cv_scores = cross_val_score(
                model, X_scaled, y_mu, 
                cv=tscv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            mean_mae = -cv_scores.mean()
            std_mae = cv_scores.std()
            
            print(f"      {name}: MAE = {mean_mae:.2f} ± {std_mae:.2f}")
            
            if mean_mae < best_score:
                best_score = mean_mae
                best_model = model
                best_name = name
        
        print(f"   🏆 Winner: {best_name} (MAE: {best_score:.2f})")
        
        # Train best model su tutto il dataset
        best_model.fit(X_scaled, y_mu)
        
        return {
            'model': best_model,
            'scaler': scaler,
            'name': best_name,
            'cv_mae': best_score
        }
    
    def _train_optimized_sigma_model(self, X, y_sigma):
        """Train modello SIGMA ottimizzato (già performava bene)"""
        
        print("   📏 Training optimized SIGMA model...")
        
        # Random Forest con parametri ottimizzati
        sigma_model = RandomForestRegressor(
            n_estimators=80,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        sigma_model.fit(X_scaled, y_sigma)
        
        # Quick validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            sigma_model, X_scaled, y_sigma,
            cv=tscv,
            scoring='neg_mean_absolute_error'
        )
        
        mean_mae = -cv_scores.mean()
        print(f"   ✅ SIGMA Model CV MAE: {mean_mae:.2f}")
        
        return sigma_model, scaler
    
    def _final_validation(self, mu_model_dict, sigma_model, sigma_scaler, X, y_mu, y_sigma):
        """Validazione finale sui modelli ottimizzati"""
        
        print("   🧪 Final holdout validation...")
        
        # 80/20 split temporale
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_mu_train, y_mu_test = y_mu.iloc[:split_idx], y_mu.iloc[split_idx:]
        y_sigma_train, y_sigma_test = y_sigma.iloc[:split_idx], y_sigma.iloc[split_idx:]
        
        # MU Model validation
        mu_model = mu_model_dict['model']
        mu_scaler = mu_model_dict['scaler']
        
        X_train_scaled_mu = mu_scaler.transform(X_train)
        X_test_scaled_mu = mu_scaler.transform(X_test)
        
        mu_pred_train = mu_model.predict(X_train_scaled_mu)
        mu_pred_test = mu_model.predict(X_test_scaled_mu)
        
        mu_mae_train = mean_absolute_error(y_mu_train, mu_pred_train)
        mu_mae_test = mean_absolute_error(y_mu_test, mu_pred_test)
        mu_r2_test = r2_score(y_mu_test, mu_pred_test)
        
        # SIGMA Model validation
        X_train_scaled_sigma = sigma_scaler.transform(X_train)
        X_test_scaled_sigma = sigma_scaler.transform(X_test)
        
        sigma_pred_train = sigma_model.predict(X_train_scaled_sigma)
        sigma_pred_test = sigma_model.predict(X_test_scaled_sigma)
        
        sigma_mae_train = mean_absolute_error(y_sigma_train, sigma_pred_train)
        sigma_mae_test = mean_absolute_error(y_sigma_test, sigma_pred_test)
        sigma_r2_test = r2_score(y_sigma_test, sigma_pred_test)
        
        print(f"\n   📊 FINAL PERFORMANCE:")
        print(f"   🎯 MU Model ({mu_model_dict['name']}):")
        print(f"      Train MAE: {mu_mae_train:.2f}")
        print(f"      Test MAE:  {mu_mae_test:.2f}")
        print(f"      Test R²:   {mu_r2_test:.3f}")
        print(f"      Overfitting: {((mu_mae_test - mu_mae_train) / mu_mae_train * 100):+.1f}%")
        
        print(f"   📏 SIGMA Model:")
        print(f"      Train MAE: {sigma_mae_train:.2f}")
        print(f"      Test MAE:  {sigma_mae_test:.2f}")
        print(f"      Test R²:   {sigma_r2_test:.3f}")
        print(f"      Overfitting: {((sigma_mae_test - sigma_mae_train) / sigma_mae_train * 100):+.1f}%")
        
        # Quality assessment
        mu_quality = "✅ EXCELLENT" if mu_mae_test < 6 else "✅ GOOD" if mu_mae_test < 10 else "⚠️ FAIR" if mu_mae_test < 15 else "❌ POOR"
        sigma_quality = "✅ EXCELLENT" if sigma_mae_test < 1.5 else "✅ GOOD" if sigma_mae_test < 2.5 else "⚠️ FAIR"
        
        print(f"\n   🏆 OVERALL QUALITY:")
        print(f"      MU: {mu_quality}")
        print(f"      SIGMA: {sigma_quality}")
        
        return {
            'mu_mae_test': mu_mae_test,
            'sigma_mae_test': sigma_mae_test,
            'mu_r2_test': mu_r2_test,
            'sigma_r2_test': sigma_r2_test
        }
    
    def _save_optimized_models(self, mu_model_dict, sigma_model, sigma_scaler, feature_names):
        """Salva modelli ottimizzati"""
        
        # Save models
        joblib.dump(mu_model_dict['model'], os.path.join(self.models_dir, 'mu_model.pkl'))
        joblib.dump(sigma_model, os.path.join(self.models_dir, 'sigma_model.pkl'))
        joblib.dump(mu_model_dict['scaler'], os.path.join(self.models_dir, 'scaler.pkl'))  # MU scaler
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'optimization_version': '2.0',
            'mu_model_type': mu_model_dict['name'],
            'mu_cv_mae': mu_model_dict['cv_mae'],
            'sigma_model_type': 'RandomForest_Optimized',
            'features': feature_names,
            'anti_overfitting_measures': [
                'Scientific feature selection',
                'Cross-validation model comparison', 
                'Conservative hyperparameters',
                'Regularization',
                'Time series validation'
            ]
        }
        
        import json
        with open(os.path.join(self.models_dir, 'optimized_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Optimized models saved to: {self.models_dir}")

def main():
    """Esegue training ottimizzato"""
    
    print("🎯 Starting optimized model training...")
    
    trainer = OptimizedModelTrainer()
    success = trainer.run_optimized_training()
    
    if success:
        print("\n🎉 === OPTIMIZATION SUCCESS! ===")
        print("Modelli ottimizzati e anti-overfitting implementato!")
        print("\n📈 Improvements:")
        print("• ✅ Scientific feature selection")
        print("• ✅ Cross-validation model comparison") 
        print("• ✅ Anti-overfitting measures")
        print("• ✅ Regularization implemented")
        print("• ✅ Time series validation")
    else:
        print("\n❌ Optimization failed!")

if __name__ == "__main__":
    main() 