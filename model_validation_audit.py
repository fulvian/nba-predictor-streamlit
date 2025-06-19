#!/usr/bin/env python3
"""
🔍 MODEL VALIDATION AUDIT SCRIPT
Analizza le performance del modello probabilistico esistente
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def audit_model_performance():
    """Esegue audit completo delle performance del modello"""
    
    print("🔍 === NBA PREDICTOR MODEL AUDIT ===")
    
    # 1. CARICA DATASET TRAINING
    try:
        print("\n📊 Loading dataset...")
        df = pd.read_csv('data/nba_data_with_mu_sigma_for_ml.csv')
        print(f"   ✅ Dataset loaded: {len(df)} samples, {len(df.columns)} features")
        
        # Analisi dataset
        print(f"\n📈 DATASET ANALYSIS:")
        print(f"   📅 Date range: {df['GAME_DATE_EST'].min()} to {df['GAME_DATE_EST'].max()}")
        print(f"   🏀 Seasons: {df['SEASON'].unique()}")
        print(f"   📊 Target stats:")
        print(f"      MU_L1_Media_punti_stimati_finale: {df['MU_L1_Media_punti_stimati_finale'].mean():.1f} ± {df['MU_L1_Media_punti_stimati_finale'].std():.1f}")
        print(f"      SIGMA_L2_sd_final: {df['SIGMA_L2_sd_final'].mean():.1f} ± {df['SIGMA_L2_sd_final'].std():.1f}")
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return
    
    # 2. CARICA MODELLI
    try:
        print("\n🤖 Loading models...")
        mu_model = joblib.load('models/probabilistic/mu_model.pkl')
        sigma_model = joblib.load('models/probabilistic/sigma_model.pkl')
        scaler = joblib.load('models/probabilistic/scaler.pkl')
        print(f"   ✅ Models loaded:")
        print(f"      Mu Model: {type(mu_model).__name__}")
        print(f"      Sigma Model: {type(sigma_model).__name__}")
        print(f"      Scaler: {type(scaler).__name__}")
        
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return
    
    # 3. PREPARA FEATURES (replico la logica del probabilistic_model.py)
    try:
        print("\n⚙️ Preparing features...")
        
        # Features originali (basate sul codice probabilistic_model.py)
        feature_columns = [
            'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'HOME_PACE_season',
            'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg', 'AWAY_PACE_season', 
            'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
            'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
            'HOME_ORtg_L5Avg', 'HOME_DRtg_L5Avg',
            'AWAY_ORtg_L5Avg', 'AWAY_DRtg_L5Avg',
            'GAME_PACE'
        ]
        
        # Trova colonne disponibili nel dataset
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                available_features.append(col)
            else:
                # Cerca alternative
                alternatives = [c for c in df.columns if col.replace('_season', '').replace('_sAvg', '') in c]
                if alternatives:
                    print(f"   ⚠️ {col} not found, using {alternatives[0]}")
                    available_features.append(alternatives[0])
                else:
                    print(f"   ❌ {col} not found in dataset")
        
        print(f"   ✅ Using {len(available_features)} features")
        
        # Estrai features e targets
        X = df[available_features].fillna(method='ffill').fillna(110.0)  # Default NBA values
        y_mu = df['MU_L1_Media_punti_stimati_finale'].fillna(220.0)
        y_sigma = df['SIGMA_L2_sd_final'].fillna(12.0)
        
        print(f"   📊 Feature matrix: {X.shape}")
        print(f"   🎯 Targets: μ={len(y_mu)}, σ={len(y_sigma)}")
        
    except Exception as e:
        print(f"   ❌ Error preparing features: {e}")
        return
    
    # 4. SPLIT TEMPORALE (più realistico per time series)
    try:
        print("\n📅 Temporal train/test split...")
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'])
        df_sorted = df.sort_values('GAME_DATE')
        
        # 80/20 split temporale
        split_idx = int(len(df_sorted) * 0.8)
        
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_mu_train, y_mu_test = y_mu.loc[train_idx], y_mu.loc[test_idx]
        y_sigma_train, y_sigma_test = y_sigma.loc[train_idx], y_sigma.loc[test_idx]
        
        print(f"   ✅ Train: {len(X_train)} samples")
        print(f"   ✅ Test: {len(X_test)} samples")
        print(f"   📅 Train period: {df_sorted.loc[train_idx, 'GAME_DATE'].min()} to {df_sorted.loc[train_idx, 'GAME_DATE'].max()}")
        print(f"   📅 Test period: {df_sorted.loc[test_idx, 'GAME_DATE'].min()} to {df_sorted.loc[test_idx, 'GAME_DATE'].max()}")
        
    except Exception as e:
        print(f"   ❌ Error in train/test split: {e}")
        return
    
    # 5. SCALE FEATURES
    try:
        print("\n⚖️ Scaling features...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"   ✅ Features scaled using existing scaler")
        
    except Exception as e:
        print(f"   ❌ Error scaling features: {e}")
        return
    
    # 6. EVALUATE MODELS
    try:
        print("\n📊 === MODEL EVALUATION ===")
        
        # Predizioni Mu
        print("\n🎯 MU MODEL PERFORMANCE:")
        mu_pred_train = mu_model.predict(X_train_scaled)
        mu_pred_test = mu_model.predict(X_test_scaled)
        
        mu_mae_train = mean_absolute_error(y_mu_train, mu_pred_train)
        mu_mae_test = mean_absolute_error(y_mu_test, mu_pred_test)
        mu_r2_train = r2_score(y_mu_train, mu_pred_train)
        mu_r2_test = r2_score(y_mu_test, mu_pred_test)
        mu_rmse_train = np.sqrt(mean_squared_error(y_mu_train, mu_pred_train))
        mu_rmse_test = np.sqrt(mean_squared_error(y_mu_test, mu_pred_test))
        
        print(f"   🏋️ TRAIN: MAE={mu_mae_train:.2f}, RMSE={mu_rmse_train:.2f}, R²={mu_r2_train:.3f}")
        print(f"   🧪 TEST:  MAE={mu_mae_test:.2f}, RMSE={mu_rmse_test:.2f}, R²={mu_r2_test:.3f}")
        
        # Predizioni Sigma
        print("\n📏 SIGMA MODEL PERFORMANCE:")
        sigma_pred_train = sigma_model.predict(X_train_scaled)
        sigma_pred_test = sigma_model.predict(X_test_scaled)
        
        sigma_mae_train = mean_absolute_error(y_sigma_train, sigma_pred_train)
        sigma_mae_test = mean_absolute_error(y_sigma_test, sigma_pred_test)
        sigma_r2_train = r2_score(y_sigma_train, sigma_pred_train)
        sigma_r2_test = r2_score(y_sigma_test, sigma_pred_test)
        sigma_rmse_train = np.sqrt(mean_squared_error(y_sigma_train, sigma_pred_train))
        sigma_rmse_test = np.sqrt(mean_squared_error(y_sigma_test, sigma_pred_test))
        
        print(f"   🏋️ TRAIN: MAE={sigma_mae_train:.2f}, RMSE={sigma_rmse_train:.2f}, R²={sigma_r2_train:.3f}")
        print(f"   🧪 TEST:  MAE={sigma_mae_test:.2f}, RMSE={sigma_rmse_test:.2f}, R²={sigma_r2_test:.3f}")
        
        # Overfitting check
        print(f"\n🔍 OVERFITTING ANALYSIS:")
        mu_overfit = (mu_mae_test - mu_mae_train) / mu_mae_train * 100
        sigma_overfit = (sigma_mae_test - sigma_mae_train) / sigma_mae_train * 100
        
        print(f"   🎯 Mu overfitting: {mu_overfit:+.1f}% ({'⚠️ HIGH' if abs(mu_overfit) > 15 else '✅ OK'})")
        print(f"   📏 Sigma overfitting: {sigma_overfit:+.1f}% ({'⚠️ HIGH' if abs(sigma_overfit) > 15 else '✅ OK'})")
        
    except Exception as e:
        print(f"   ❌ Error in model evaluation: {e}")
        return
    
    # 7. RESIDUAL ANALYSIS
    try:
        print(f"\n🔬 RESIDUAL ANALYSIS:")
        
        mu_residuals = y_mu_test - mu_pred_test
        sigma_residuals = y_sigma_test - sigma_pred_test
        
        # Statistiche residui
        print(f"   🎯 Mu residuals: mean={mu_residuals.mean():.2f}, std={mu_residuals.std():.2f}")
        print(f"   📏 Sigma residuals: mean={sigma_residuals.mean():.2f}, std={sigma_residuals.std():.2f}")
        
        # Test normalità
        mu_shapiro = stats.shapiro(mu_residuals.sample(min(5000, len(mu_residuals))))
        sigma_shapiro = stats.shapiro(sigma_residuals.sample(min(5000, len(sigma_residuals))))
        
        print(f"   📊 Mu normality test: p={mu_shapiro.pvalue:.4f} ({'✅ Normal' if mu_shapiro.pvalue > 0.05 else '⚠️ Non-normal'})")
        print(f"   📊 Sigma normality test: p={sigma_shapiro.pvalue:.4f} ({'✅ Normal' if sigma_shapiro.pvalue > 0.05 else '⚠️ Non-normal'})")
        
    except Exception as e:
        print(f"   ❌ Error in residual analysis: {e}")
    
    # 8. SUMMARY & RECOMMENDATIONS
    print(f"\n🎯 === AUDIT SUMMARY ===")
    print(f"✅ Model Type: XGBoost (good choice)")
    print(f"✅ Dataset Size: {len(df)} samples (adequate)")
    print(f"⚠️ Test MAE Mu: {mu_mae_test:.1f} points ({'✅ Good' if mu_mae_test < 8 else '⚠️ Needs improvement' if mu_mae_test < 12 else '❌ Poor'})")
    print(f"⚠️ Test MAE Sigma: {sigma_mae_test:.1f} points ({'✅ Good' if sigma_mae_test < 3 else '⚠️ Needs improvement' if sigma_mae_test < 5 else '❌ Poor'})")
    
    recommendations = []
    if mu_mae_test > 10:
        recommendations.append("🔧 Improve Mu model: try feature engineering")
    if sigma_mae_test > 4:
        recommendations.append("🔧 Improve Sigma model: may need different approach")
    if abs(mu_overfit) > 15:
        recommendations.append("🔧 Reduce overfitting: regularization, early stopping")
    if len(available_features) < len(feature_columns):
        recommendations.append("🔧 Feature alignment: some features missing from dataset")
    
    if recommendations:
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print(f"\n🎉 Model performance is acceptable!")

if __name__ == "__main__":
    audit_model_performance() 