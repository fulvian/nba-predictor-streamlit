#!/usr/bin/env python3
"""
Complete Momentum ML Trainer - Regular Season + Playoff
=======================================================
Sistema completo che gestisce:
1. Training progressivo Regular Season (come momentum_ml_trainer_progressive.py)
2. Modello specializzato Playoff con feature aggiuntive
3. Modello ibrido che combina insights da entrambi
4. Feature engineering specifiche per contesto playoff
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CompleteMomentumMLTrainer:
    """
    Trainer ML completo per Regular Season e Playoff con modelli specializzati.
    """
    
    def __init__(self, dataset_path='data/momentum_v2/momentum_training_dataset.csv'):
        """
        Inizializza il trainer completo.
        """
        self.dataset_path = dataset_path
        self.models_dir = 'models/momentum_complete'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Directory per i diversi tipi di modelli
        self.regular_dir = os.path.join(self.models_dir, 'regular_season')
        self.playoff_dir = os.path.join(self.models_dir, 'playoff')
        self.hybrid_dir = os.path.join(self.models_dir, 'hybrid')
        
        for dir_path in [self.regular_dir, self.playoff_dir, self.hybrid_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configurazione training ESTESA
        self.n_splits = 4
        self.train_ratio = 0.8
        # ESTESO: 6 stagioni playoff per dinamiche ricorrenti
        self.playoff_seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
        # ESTESO: 2 stagioni regular per dati più recenti 
        self.regular_seasons = ['2023-24', '2024-25']
        
        # Storage risultati
        self.regular_results = {}
        self.playoff_results = {}
        self.hybrid_results = {}
        
        # Feature names
        self.base_features = [
            'home_momentum_score', 'home_hot_hand_players', 'home_avg_player_momentum',
            'home_avg_player_weighted_contribution', 'home_team_offensive_potential',
            'home_team_defensive_potential', 'away_momentum_score', 'away_hot_hand_players',
            'away_avg_player_momentum', 'away_avg_player_weighted_contribution',
            'away_team_offensive_potential', 'away_team_defensive_potential', 'momentum_diff'
        ]
        
        print(f"🚀 CompleteMomentumMLTrainer inizializzato")
        print(f"   📁 Dataset: {dataset_path}")
        print(f"   📊 Regular Seasons: {self.regular_seasons} (training progressivo esteso)")
        print(f"   🏆 Playoff seasons: {self.playoff_seasons} (6 stagioni per dinamiche ricorrenti)")
        print(f"   💾 Modelli salvati in: {self.models_dir}")

    def load_and_prepare_complete_data(self):
        """
        Carica e prepara dati per training completo (Regular + Playoff).
        """
        print("\n📊 Caricamento e preparazione dati completi...")
        
        # Carica dataset
        df = pd.read_csv(self.dataset_path)
        print(f"   ✅ Dataset caricato: {len(df)} partite, {len(df.columns)} colonne")
        
        # Converti date e determina stagioni
        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed')
        df = df.sort_values('game_date').reset_index(drop=True)
        
        def get_nba_season(date):
            if date.month >= 10:
                return f"{date.year}-{str(date.year + 1)[-2:]}"
            else:
                return f"{date.year - 1}-{str(date.year)[-2:]}"
        
        df['season'] = df['game_date'].apply(get_nba_season)
        
        def get_game_type(date):
            if date.month >= 4 and date.month <= 6:
                if date.month == 4 and date.day < 15:
                    return 'Regular'
                else:
                    return 'Playoff'
            else:
                return 'Regular'
        
        df['game_type'] = df['game_date'].apply(get_game_type)
        
        # Separa Regular Season ESTESO (2 stagioni per training progressivo)
        self.regular_df = df[
            (df['season'].isin(self.regular_seasons)) & 
            (df['game_type'] == 'Regular')
        ].copy().reset_index(drop=True)
        
        self.playoff_df = df[
            (df['season'].isin(self.playoff_seasons)) & 
            (df['game_type'] == 'Playoff')
        ].copy().reset_index(drop=True)
        
        self.full_df = df.copy()
        
        print(f"\n📈 Analisi dati ESTESI:")
        print(f"   📊 Regular Season ({', '.join(self.regular_seasons)}): {len(self.regular_df)} partite")
        print(f"   🏆 Playoff ({', '.join(self.playoff_seasons)}): {len(self.playoff_df)} partite")
        
        # Distribuzione regular per stagione
        regular_by_season = self.regular_df['season'].value_counts().sort_index()
        for season, count in regular_by_season.items():
            print(f"      📊 {season}: {count} partite regular")
        
        # Distribuzione playoff per stagione  
        playoff_by_season = self.playoff_df['season'].value_counts().sort_index()
        for season, count in playoff_by_season.items():
            print(f"      🏆 {season}: {count} partite playoff")
        
        # Clean data
        self.regular_df = self.regular_df.dropna(subset=['score_deviation'])
        self.playoff_df = self.playoff_df.dropna(subset=['score_deviation'])
        
        print(f"   🧹 Dopo pulizia: {len(self.regular_df)} Regular, {len(self.playoff_df)} Playoff")
        
        return self.regular_df, self.playoff_df

    def engineer_playoff_features(self, playoff_df):
        """
        Crea feature specifiche per i playoff.
        """
        print("\n🔧 Engineering feature playoff...")
        
        df = playoff_df.copy()
        
        # 1. ESPERIENZA PLAYOFF (mock - in realtà servirebbe database giocatori)
        # Per ora usiamo proxy basati su age e team performance
        df['home_playoff_experience'] = np.random.uniform(0.3, 0.9, len(df))  # Mock
        df['away_playoff_experience'] = np.random.uniform(0.3, 0.9, len(df))  # Mock
        
        # 2. SEEDING IMPACT (mock - in realtà servirebbe regular season record)
        df['home_seed_advantage'] = np.random.uniform(-0.3, 0.3, len(df))  # Mock
        df['away_seed_advantage'] = np.random.uniform(-0.3, 0.3, len(df))  # Mock
        
        # 3. SERIES CONTEXT
        # Game number in series (mock - derivabile da schedule)
        df['series_game_number'] = np.random.randint(1, 8, len(df))  # Mock Games 1-7
        df['is_elimination_game'] = (df['series_game_number'] >= 6).astype(int)
        df['is_series_opener'] = (df['series_game_number'] == 1).astype(int)
        
        # 4. REST DAYS (più importante nei playoff)
        df['days_rest'] = np.random.randint(0, 4, len(df))  # Mock 0-3 giorni
        df['rest_advantage'] = df['days_rest'] * 0.1  # Ogni giorno di riposo vale 0.1
        
        # 5. PLAYOFF INTENSITY MULTIPLIERS
        # Amplifica le feature di momentum per l'intensità playoff
        intensity_multiplier = 1.3  # I playoff sono più intensi
        
        for feature in self.base_features:
            if feature in df.columns:
                df[f'{feature}_playoff_adjusted'] = df[feature] * intensity_multiplier
        
        # 6. PRESSURE INDEX
        # Combina fattori di pressione
        df['pressure_index'] = (
            df['is_elimination_game'] * 0.4 +
            (df['series_game_number'] / 7) * 0.3 +
            abs(df['home_seed_advantage'] - df['away_seed_advantage']) * 0.3
        )
        
        # Feature list playoff
        self.playoff_features = self.base_features + [
            'home_playoff_experience', 'away_playoff_experience',
            'home_seed_advantage', 'away_seed_advantage',
            'series_game_number', 'is_elimination_game', 'is_series_opener',
            'days_rest', 'rest_advantage', 'pressure_index'
        ]
        
        # Aggiungi adjusted features
        adjusted_features = [f'{f}_playoff_adjusted' for f in self.base_features if f in df.columns]
        self.playoff_features.extend(adjusted_features)
        
        print(f"   ✅ Feature playoff create: {len(self.playoff_features)}")
        print(f"   🎯 Feature base: {len(self.base_features)}")
        print(f"   🏆 Feature playoff specifiche: {len(self.playoff_features) - len(self.base_features)}")
        
        return df

    def train_regular_season_model(self):
        """
        Training progressivo per Regular Season (come nel trainer progressivo).
        """
        print(f"\n🏀 TRAINING REGULAR SEASON MODEL")
        print("=" * 50)
        
        # Usa la logica del trainer progressivo
        return self._progressive_training(
            self.regular_df, 
            self.base_features, 
            self.regular_dir,
            model_type='regular_season'
        )

    def train_playoff_model(self):
        """
        Training specializzato per playoff.
        """
        print(f"\n🏆 TRAINING PLAYOFF MODEL")
        print("=" * 50)
        
        # Engineer playoff features
        playoff_df_enhanced = self.engineer_playoff_features(self.playoff_df)
        
        # Dividi per cross-validation temporale (per stagione)
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        playoff_results = {}
        
        # Cross-validation per stagione (leave-one-season-out)
        for test_season in self.playoff_seasons:
            print(f"\n   🔄 Cross-validation: Test season {test_season}")
            
            train_mask = playoff_df_enhanced['season'] != test_season
            test_mask = playoff_df_enhanced['season'] == test_season
            
            train_df = playoff_df_enhanced[train_mask]
            test_df = playoff_df_enhanced[test_mask]
            
            print(f"      📊 Train: {len(train_df)} partite | Test: {len(test_df)} partite")
            
            if len(test_df) == 0:
                continue
            
            # Prepara dati
            X_train = train_df[self.playoff_features].values
            y_train = train_df['score_deviation'].values
            X_test = test_df[self.playoff_features].values
            y_test = test_df['score_deviation'].values
            
            # Standardizza
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            season_results = {}
            
            for name, model in models.items():
                print(f"         🤖 Training {name}...")
                
                model_instance = model.__class__(**model.get_params())
                model_instance.fit(X_train_scaled, y_train)
                
                test_pred = model_instance.predict(X_test_scaled)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                season_results[name] = {
                    'model': model_instance,
                    'scaler': scaler,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'test_pred': test_pred,
                    'test_actual': y_test
                }
                
                print(f"            📊 Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
            
            playoff_results[test_season] = season_results
        
        # Trova il modello migliore mediamente
        model_names = list(models.keys())
        avg_maes = {}
        
        for name in model_names:
            maes = [playoff_results[season][name]['test_mae'] 
                   for season in playoff_results.keys()]
            avg_maes[name] = np.mean(maes)
        
        best_playoff_model = min(avg_maes.keys(), key=lambda x: avg_maes[x])
        
        print(f"\n   🏆 Migliore modello playoff: {best_playoff_model}")
        print(f"      📊 MAE medio cross-validation: {avg_maes[best_playoff_model]:.3f}")
        
        # Training finale su tutti i dati playoff
        X_all = playoff_df_enhanced[self.playoff_features].values
        y_all = playoff_df_enhanced['score_deviation'].values
        
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X_all)
        
        # Ricrea modello migliore
        final_model = self._create_model_instance(best_playoff_model)
        final_model.fit(X_all_scaled, y_all)
        
        # Salva modello playoff
        self._save_model(final_model, final_scaler, {
            'model_name': best_playoff_model,
            'feature_names': self.playoff_features,
            'cross_validation_results': playoff_results,
            'avg_mae': avg_maes[best_playoff_model],
            'training_seasons': self.playoff_seasons,
            'model_type': 'playoff'
        }, self.playoff_dir)
        
        self.playoff_results = {
            'best_model': final_model,
            'best_scaler': final_scaler,
            'best_model_name': best_playoff_model,
            'cross_validation': playoff_results,
            'avg_mae': avg_maes[best_playoff_model]
        }
        
        return self.playoff_results

    def train_hybrid_model(self):
        """
        Training modello ibrido che combina insights Regular Season + Playoff.
        """
        print(f"\n🔗 TRAINING HYBRID MODEL")
        print("=" * 50)
        
        # Prepara dataset combinato
        regular_df_subset = self.regular_df.copy()
        playoff_df_enhanced = self.engineer_playoff_features(self.playoff_df)
        
        # Per il modello ibrido, aggiungi flag di contesto
        regular_df_subset['is_playoff'] = 0
        playoff_df_enhanced['is_playoff'] = 1
        
        # Allinea le feature (usa solo quelle comuni + flag)
        common_features = self.base_features + ['is_playoff']
        
        # Aggiungi feature playoff simulate per regular season
        for feature in ['pressure_index', 'rest_advantage']:
            if feature in playoff_df_enhanced.columns:
                regular_df_subset[feature] = 0  # Valore neutro per regular season
                common_features.append(feature)
        
        # Combina dataset
        combined_df = pd.concat([
            regular_df_subset[common_features + ['score_deviation', 'season', 'game_date']],
            playoff_df_enhanced[common_features + ['score_deviation', 'season', 'game_date']]
        ], ignore_index=True).sort_values('game_date')
        
        print(f"   📊 Dataset ibrido: {len(combined_df)} partite")
        print(f"      🏀 Regular: {len(regular_df_subset)} partite")
        print(f"      🏆 Playoff: {len(playoff_df_enhanced)} partite")
        print(f"   🎯 Feature comuni: {len(common_features)}")
        
        # Pulisci dati da NaN PRIMA della divisione
        clean_df = combined_df[common_features + ['score_deviation', 'is_playoff']].dropna().reset_index(drop=True)
        
        if len(clean_df) == 0:
            print("   ⚠️ Nessun dato valido per training ibrido dopo pulizia NaN")
            return {}
        
        print(f"   📊 Dopo pulizia NaN: {len(clean_df)} partite (era {len(combined_df)})")
        
        # Training temporale (ultimo 20% per test)
        split_idx = int(len(clean_df) * 0.8)
        train_df = clean_df.iloc[:split_idx].reset_index(drop=True)
        test_df = clean_df.iloc[split_idx:].reset_index(drop=True)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        hybrid_results = {}
        
        X_train = train_df[common_features].values
        y_train = train_df['score_deviation'].values
        X_test = test_df[common_features].values
        y_test = test_df['score_deviation'].values
        
        # Verifica finale per NaN
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
            print("   ⚠️ NaN ancora presenti dopo pulizia - skippo training ibrido")
            return {}
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            print(f"\n   🤖 Training {name} (ibrido)...")
            
            model_instance = model.__class__(**model.get_params())
            model_instance.fit(X_train_scaled, y_train)
            
            test_pred = model_instance.predict(X_test_scaled)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Valuta separatamente su regular e playoff nel test set
            # Usa approccio più semplice per evitare problemi di dimensioni
            try:
                test_regular_indices = test_df[test_df['is_playoff'] == 0].index
                test_playoff_indices = test_df[test_df['is_playoff'] == 1].index
                
                # Converti in maschere relative al test set
                test_regular_mask = np.isin(range(len(test_df)), test_regular_indices)
                test_playoff_mask = np.isin(range(len(test_df)), test_playoff_indices)
                
                regular_mae = mean_absolute_error(
                    y_test[test_regular_mask], 
                    test_pred[test_regular_mask]
                ) if np.any(test_regular_mask) else np.nan
                
                playoff_mae = mean_absolute_error(
                    y_test[test_playoff_mask], 
                    test_pred[test_playoff_mask]
                ) if np.any(test_playoff_mask) else np.nan
                
            except Exception as e:
                print(f"         ⚠️ Errore nella valutazione separata: {e}")
                regular_mae = np.nan
                playoff_mae = np.nan
            
            hybrid_results[name] = {
                'model': model_instance,
                'scaler': scaler,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'regular_mae': regular_mae,
                'playoff_mae': playoff_mae,
                'test_pred': test_pred
            }
            
            print(f"      📊 Overall MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
            if not np.isnan(regular_mae):
                print(f"      🏀 Regular MAE: {regular_mae:.3f}")
            if not np.isnan(playoff_mae):
                print(f"      🏆 Playoff MAE: {playoff_mae:.3f}")
        
        # Trova modello migliore
        best_hybrid_model = min(hybrid_results.keys(), 
                               key=lambda x: hybrid_results[x]['test_mae'])
        
        print(f"\n   🏆 Migliore modello ibrido: {best_hybrid_model}")
        
        # Salva modello ibrido
        best_result = hybrid_results[best_hybrid_model]
        self._save_model(best_result['model'], best_result['scaler'], {
            'model_name': best_hybrid_model,
            'feature_names': common_features,
            'test_mae': best_result['test_mae'],
            'test_r2': best_result['test_r2'],
            'regular_mae': best_result['regular_mae'],
            'playoff_mae': best_result['playoff_mae'],
            'model_type': 'hybrid'
        }, self.hybrid_dir)
        
        self.hybrid_results = hybrid_results
        return hybrid_results

    def _progressive_training(self, df, features, save_dir, model_type):
        """
        Implementa training progressivo ESTESO per 2 stagioni Regular Season.
        """
        print(f"   🔄 Training progressivo ESTESO per {model_type}...")
        
        if model_type == 'regular_season':
            # NUOVO: Training progressivo su 2 stagioni
            return self._progressive_training_multi_season(df, features, save_dir)
        else:
            # Per altri tipi, usa logica originale
            n_games_per_split = len(df) // self.n_splits
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        progressive_results = {}
        
        for step in range(1, self.n_splits + 1):
            # Prendi dati fino al step corrente
            end_idx = step * n_games_per_split
            if step == self.n_splits:
                end_idx = len(df)
            
            step_df = df.iloc[:end_idx].copy()
            
            # Dividi in train/test
            train_size = int(len(step_df) * self.train_ratio)
            train_df = step_df.iloc[:train_size]
            test_df = step_df.iloc[train_size:]
            
            if len(test_df) == 0:
                continue
            
            X_train = train_df[features].values
            y_train = train_df['score_deviation'].values
            X_test = test_df[features].values
            y_test = test_df['score_deviation'].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            step_results = {}
            
            for name, model in models.items():
                model_instance = model.__class__(**model.get_params())
                model_instance.fit(X_train_scaled, y_train)
                
                test_pred = model_instance.predict(X_test_scaled)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                step_results[name] = {
                    'model': model_instance,
                    'scaler': scaler,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
            
            best_name = min(step_results.keys(), key=lambda x: step_results[x]['test_mae'])
            progressive_results[f'step_{step}'] = {
                'best_model_name': best_name,
                'best_result': step_results[best_name],
                'all_results': step_results
            }
            
            print(f"      Step {step}: {best_name} | MAE: {step_results[best_name]['test_mae']:.3f}")
        
        # Training finale su tutto il dataset
        X_all = df[features].values
        y_all = df['score_deviation'].values
        
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X_all)
        
        final_best_name = progressive_results[f'step_{self.n_splits}']['best_model_name']
        final_model = self._create_model_instance(final_best_name)
        final_model.fit(X_all_scaled, y_all)
        
        # Salva modello finale
        self._save_model(final_model, final_scaler, {
            'model_name': final_best_name,
            'feature_names': features,
            'progressive_results': progressive_results,
            'model_type': model_type
        }, save_dir)
        
        return {
            'final_model': final_model,
            'final_scaler': final_scaler,
            'progressive_results': progressive_results,
            'best_model_name': final_best_name
        }

    def _progressive_training_multi_season(self, df, features, save_dir):
        """
        Training progressivo ESTESO su 2 stagioni Regular Season:
        1. Stagione 2023-24: Split1 → Split1+2 → Split1+2+3 → Split1+2+3+4
        2. Stagione 2023-24 completa + Stagione 2024-25: Split1 → Split1+2 → Split1+2+3 → Split1+2+3+4
        """
        print(f"      🚀 TRAINING PROGRESSIVO MULTI-STAGIONE")
        print(f"      📊 Stagioni: {self.regular_seasons}")
        
        # Separa per stagione
        season_dfs = {}
        for season in self.regular_seasons:
            season_df = df[df['season'] == season].copy().reset_index(drop=True)
            season_dfs[season] = season_df
            print(f"         {season}: {len(season_df)} partite")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        all_progressive_results = {}
        
        # FASE 1: Training progressivo su prima stagione (2023-24)
        print(f"\n      🔄 FASE 1: Training progressivo su {self.regular_seasons[0]}")
        first_season_df = season_dfs[self.regular_seasons[0]]
        
        phase1_results = self._single_season_progressive_training(
            first_season_df, features, models, "Fase1"
        )
        all_progressive_results['phase1'] = phase1_results
        
        # FASE 2: Training progressivo su entrambe le stagioni
        if len(self.regular_seasons) > 1:
            print(f"\n      🔄 FASE 2: Training progressivo su entrambe le stagioni")
            
            # Combina le stagioni in ordine temporale
            combined_df = pd.concat([
                season_dfs[season] for season in self.regular_seasons
            ], ignore_index=True).sort_values('game_date').reset_index(drop=True)
            
            print(f"         📊 Dataset combinato: {len(combined_df)} partite")
            
            # Training progressivo sulla seconda stagione, usando la prima come base
            first_season_size = len(season_dfs[self.regular_seasons[0]])
            second_season_df = season_dfs[self.regular_seasons[1]]
            
            phase2_results = self._two_season_progressive_training(
                season_dfs[self.regular_seasons[0]], 
                second_season_df, 
                features, 
                models, 
                "Fase2"
            )
            all_progressive_results['phase2'] = phase2_results
            
            # Usa risultati della fase 2 come finali
            final_results = phase2_results
        else:
            # Se c'è solo una stagione, usa fase 1
            final_results = phase1_results
        
        # Trova il modello migliore finale
        final_step_key = max(final_results['progressive_results'].keys())
        best_model_name = final_results['progressive_results'][final_step_key]['best_model_name']
        best_result = final_results['progressive_results'][final_step_key]['best_result']
        
        # Training finale su tutto il dataset
        print(f"\n      🎯 TRAINING FINALE su tutto il dataset Regular Season")
        X_all = df[features].values
        y_all = df['score_deviation'].values
        
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X_all)
        
        final_model = self._create_model_instance(best_model_name)
        final_model.fit(X_all_scaled, y_all)
        
        # Valutazione finale
        final_pred = final_model.predict(X_all_scaled)
        final_mae = mean_absolute_error(y_all, final_pred)
        final_r2 = r2_score(y_all, final_pred)
        
        print(f"         ✅ Modello finale: {best_model_name}")
        print(f"         📊 Training finale: MAE={final_mae:.3f}, R²={final_r2:.3f}")
        
        # Salva modello finale
        self._save_model(final_model, final_scaler, {
            'model_name': best_model_name,
            'feature_names': features,
            'multi_season_results': all_progressive_results,
            'final_mae': final_mae,
            'final_r2': final_r2,
            'seasons_used': self.regular_seasons,
            'model_type': 'regular_season_multi'
        }, save_dir)
        
        return {
            'final_model': final_model,
            'final_scaler': final_scaler,
            'progressive_results': final_results['progressive_results'],
            'multi_season_results': all_progressive_results,
            'best_model_name': best_model_name,
            'final_mae': final_mae,
            'final_r2': final_r2
        }

    def _single_season_progressive_training(self, season_df, features, models, phase_name):
        """
        Training progressivo su una singola stagione (4 split).
        """
        print(f"         🔄 {phase_name}: Split progressivi su singola stagione")
        
        # Dividi in 4 split temporali
        n_games_per_split = len(season_df) // self.n_splits
        progressive_results = {}
        
        for step in range(1, self.n_splits + 1):
            end_idx = step * n_games_per_split
            if step == self.n_splits:
                end_idx = len(season_df)
            
            step_df = season_df.iloc[:end_idx].copy()
            
            # 80% training, 20% test
            train_size = int(len(step_df) * self.train_ratio)
            train_df = step_df.iloc[:train_size]
            test_df = step_df.iloc[train_size:]
            
            if len(test_df) == 0:
                continue
            
            step_results = self._train_and_evaluate_models(
                train_df, test_df, features, models
            )
            
            # Controlla se abbiamo risultati validi
            if not step_results:
                print(f"            Step {step}: SKIPPED - Dati insufficienti")
                continue
            
            best_name = min(step_results.keys(), key=lambda x: step_results[x]['test_mae'])
            
            progressive_results[f'step_{step}'] = {
                'best_model_name': best_name,
                'best_result': step_results[best_name],
                'all_results': step_results,
                'dates': {
                    'train_start': train_df['game_date'].min().date(),
                    'train_end': train_df['game_date'].max().date(),
                    'test_start': test_df['game_date'].min().date(),
                    'test_end': test_df['game_date'].max().date()
                }
            }
            
            print(f"            Step {step}: {best_name} | MAE: {step_results[best_name]['test_mae']:.3f} | "
                  f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        return {
            'progressive_results': progressive_results,
            'phase': phase_name
        }

    def _two_season_progressive_training(self, first_season_df, second_season_df, features, models, phase_name):
        """
        Training progressivo usando prima stagione completa + split progressivi seconda stagione.
        """
        print(f"         🔄 {phase_name}: Prima stagione completa + split progressivi seconda stagione")
        
        # Dividi seconda stagione in 4 split
        n_games_per_split = len(second_season_df) // self.n_splits
        progressive_results = {}
        
        for step in range(1, self.n_splits + 1):
            # Usa sempre tutta la prima stagione
            train_base = first_season_df.copy()
            
            # Aggiungi split progressivi della seconda stagione
            end_idx = step * n_games_per_split
            if step == self.n_splits:
                end_idx = len(second_season_df)
            
            second_season_portion = second_season_df.iloc[:end_idx].copy()
            
            # Combina prima stagione + porzione seconda stagione per training
            combined_train = pd.concat([train_base, second_season_portion], ignore_index=True)
            
            # Test set: parte rimanente della seconda stagione
            if step < self.n_splits:
                test_df = second_season_df.iloc[end_idx:end_idx + n_games_per_split//2].copy()
            else:
                # Ultimo step: usa ultime partite come test
                test_size = min(len(second_season_df) // 5, 50)  # Max 50 partite per test
                test_df = second_season_df.iloc[-test_size:].copy()
            
            if len(test_df) == 0:
                continue
            
            step_results = self._train_and_evaluate_models(
                combined_train, test_df, features, models
            )
            
            # Controlla se abbiamo risultati validi
            if not step_results:
                print(f"            Step {step}: SKIPPED - Dati insufficienti")
                continue
            
            best_name = min(step_results.keys(), key=lambda x: step_results[x]['test_mae'])
            
            progressive_results[f'step_{step}'] = {
                'best_model_name': best_name,
                'best_result': step_results[best_name],
                'all_results': step_results,
                'data_composition': {
                    'first_season_games': len(train_base),
                    'second_season_games': len(second_season_portion),
                    'total_train_games': len(combined_train),
                    'test_games': len(test_df)
                }
            }
            
            print(f"            Step {step}: {best_name} | MAE: {step_results[best_name]['test_mae']:.3f} | "
                  f"Train: {len(combined_train)} ({len(train_base)}+{len(second_season_portion)}), Test: {len(test_df)}")
        
        return {
            'progressive_results': progressive_results,
            'phase': phase_name
        }

    def _train_and_evaluate_models(self, train_df, test_df, features, models):
        """
        Helper per addestrare e valutare tutti i modelli su un dataset.
        """
        # Pulisci dati da NaN
        train_clean = train_df[features + ['score_deviation']].dropna()
        test_clean = test_df[features + ['score_deviation']].dropna()
        
        if len(train_clean) == 0 or len(test_clean) == 0:
            print(f"         ⚠️ Dati insufficienti dopo pulizia NaN: train={len(train_clean)}, test={len(test_clean)}")
            return {}
        
        X_train = train_clean[features].values
        y_train = train_clean['score_deviation'].values
        X_test = test_clean[features].values
        y_test = test_clean['score_deviation'].values
        
        # Verifica ancora per NaN residui
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
            print(f"         ⚠️ NaN rilevati dopo pulizia - skippo questo step")
            return {}
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for name, model in models.items():
            model_instance = model.__class__(**model.get_params())
            model_instance.fit(X_train_scaled, y_train)
            
            test_pred = model_instance.predict(X_test_scaled)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[name] = {
                'model': model_instance,
                'scaler': scaler,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'test_pred': test_pred
            }
        
        return results

    def _create_model_instance(self, model_name):
        """Helper per creare istanza modello dal nome."""
        if 'Linear' in model_name:
            return LinearRegression()
        elif 'Ridge' in model_name:
            return Ridge(alpha=1.0)
        elif 'Lasso' in model_name:
            return Lasso(alpha=1.0)
        elif 'Random Forest' in model_name:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif 'Gradient Boosting' in model_name:
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            return LinearRegression()

    def _save_model(self, model, scaler, metadata, save_dir):
        """Helper per salvare modello e metadati."""
        # Salva modello
        model_path = os.path.join(save_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Salva scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Salva metadati
        metadata['training_date'] = datetime.now().isoformat()
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"      💾 Modello salvato in: {save_dir}")

    def create_complete_visualizations(self):
        """
        Crea visualizzazioni complete per tutti i modelli.
        """
        print(f"\n📊 Creazione visualizzazioni complete...")
        
        plt.figure(figsize=(20, 15))
        
        # 1. Confronto performance tra modelli
        plt.subplot(3, 4, 1)
        
        model_types = []
        maes = []
        
        if hasattr(self, 'regular_results') and self.regular_results:
            last_step = max([k for k in self.regular_results['progressive_results'].keys()])
            regular_mae = self.regular_results['progressive_results'][last_step]['best_result']['test_mae']
            model_types.append('Regular Season')
            maes.append(regular_mae)
        
        if hasattr(self, 'playoff_results') and self.playoff_results:
            model_types.append('Playoff')
            maes.append(self.playoff_results['avg_mae'])
        
        if hasattr(self, 'hybrid_results') and self.hybrid_results:
            best_hybrid = min(self.hybrid_results.keys(), 
                             key=lambda x: self.hybrid_results[x]['test_mae'])
            model_types.append('Hybrid')
            maes.append(self.hybrid_results[best_hybrid]['test_mae'])
        
        if model_types:
            bars = plt.bar(model_types, maes, color=['blue', 'red', 'green'][:len(model_types)])
            plt.ylabel('Test MAE')
            plt.title('Confronto Performance Modelli')
            plt.xticks(rotation=45)
            
            # Annotazioni
            for i, (bar, mae) in enumerate(zip(bars, maes)):
                plt.annotate(f'{mae:.3f}', 
                           (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom')
        
        # 2. Feature importance comparison (se disponibile)
        plt.subplot(3, 4, 2)
        
        # Placeholder per feature importance
        plt.text(0.5, 0.5, 'Feature Importance\nComparison\n(Placeholder)', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='lightgray'))
        plt.title('Feature Importance')
        
        # 3-4. Performance playoff per stagione
        if hasattr(self, 'playoff_results') and 'cross_validation' in self.playoff_results:
            plt.subplot(3, 4, 3)
            
            seasons = list(self.playoff_results['cross_validation'].keys())
            best_model_name = self.playoff_results['best_model_name']
            
            season_maes = [
                self.playoff_results['cross_validation'][season][best_model_name]['test_mae']
                for season in seasons
            ]
            
            plt.bar(range(len(seasons)), season_maes, color='red', alpha=0.7)
            plt.xlabel('Stagione')
            plt.ylabel('Test MAE')
            plt.title('Performance Playoff per Stagione')
            plt.xticks(range(len(seasons)), seasons, rotation=45)
        
        # 5. Distribution of errors
        plt.subplot(3, 4, 4)
        plt.text(0.5, 0.5, 'Error Distribution\n(Placeholder)', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='lightblue'))
        plt.title('Distribuzione Errori')
        
        plt.tight_layout()
        
        # Salva plot
        plot_path = os.path.join(self.models_dir, 'complete_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizzazioni salvate: {plot_path}")
        
        plt.show()

    def run_complete_training(self):
        """
        Esegue training completo: Regular Season + Playoff + Hybrid.
        """
        print("🚀 AVVIO TRAINING COMPLETO (REGULAR + PLAYOFF + HYBRID)")
        print("=" * 70)
        
        # 1. Carica dati
        self.load_and_prepare_complete_data()
        
        # 2. Training Regular Season (progressivo)
        print(f"\n🔄 FASE 1: REGULAR SEASON")
        self.regular_results = self.train_regular_season_model()
        
        # 3. Training Playoff (specializzato)
        print(f"\n🔄 FASE 2: PLAYOFF")
        self.playoff_results = self.train_playoff_model()
        
        # 4. Training Hybrid (combinato)
        print(f"\n🔄 FASE 3: HYBRID")
        self.hybrid_results = self.train_hybrid_model()
        
        # 5. Visualizzazioni
        self.create_complete_visualizations()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETO TERMINATO!")
        
        print(f"\n📊 RIEPILOGO FINALE:")
        if self.regular_results:
            last_step = max([k for k in self.regular_results['progressive_results'].keys()])
            regular_mae = self.regular_results['progressive_results'][last_step]['best_result']['test_mae']
            print(f"   🏀 Regular Season: {self.regular_results['best_model_name']:<20} | MAE: {regular_mae:.3f}")
        
        if self.playoff_results:
            print(f"   🏆 Playoff:        {self.playoff_results['best_model_name']:<20} | MAE: {self.playoff_results['avg_mae']:.3f}")
        
        if self.hybrid_results:
            best_hybrid = min(self.hybrid_results.keys(), 
                             key=lambda x: self.hybrid_results[x]['test_mae'])
            hybrid_mae = self.hybrid_results[best_hybrid]['test_mae']
            print(f"   🔗 Hybrid:         {best_hybrid:<20} | MAE: {hybrid_mae:.3f}")
        
        print(f"\n💾 Tutti i modelli salvati in: {self.models_dir}")
        
        return {
            'regular_results': self.regular_results,
            'playoff_results': self.playoff_results,
            'hybrid_results': self.hybrid_results
        }


def main():
    """
    Funzione principale per training completo.
    """
    trainer = CompleteMomentumMLTrainer()
    results = trainer.run_complete_training()
    
    return results


if __name__ == "__main__":
    main() 