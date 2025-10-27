#!/usr/bin/env python3
"""
Momentum ML Trainer con Walk-Forward Validation Progressivo
===========================================================
Implementa il sistema di training progressivo simile al modello probabilistico:
1. Divide Regular Season 2023-24 in 4 split temporali
2. Per ogni split: usa 80% training, 20% test
3. Training progressivo: Split1 → Split1+2 → Split1+2+3 → Split1+2+3+4
4. Validazione su ogni step prima di procedere
5. Training finale su tutto il dataset
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

class ProgressiveMomentumMLTrainer:
    """
    Trainer ML con walk-forward validation progressivo per evitare overfitting.
    """
    
    def __init__(self, dataset_path='data/momentum_v2/momentum_training_dataset.csv'):
        """
        Inizializza il trainer progressivo.
        """
        self.dataset_path = dataset_path
        self.models_dir = 'models/momentum_progressive'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configurazione progressive training
        self.base_season = '2023-24'  # Stagione base per splits
        self.n_splits = 4            # Numero di split temporali
        self.train_ratio = 0.8       # 80% training, 20% test per split
        
        # Storage per risultati progressivi
        self.splits_data = {}
        self.progressive_results = {}
        self.validation_history = []
        self.best_models_history = []
        
        # Risultati finali
        self.final_model = None
        self.final_scaler = None
        self.feature_names = None
        
        print(f"🚀 ProgressiveMomentumMLTrainer inizializzato")
        print(f"   📁 Dataset: {dataset_path}")
        print(f"   📚 Stagione base: {self.base_season}")
        print(f"   🔄 Numero splits: {self.n_splits}")
        print(f"   📊 Train/Test ratio: {self.train_ratio:.0%}/{1-self.train_ratio:.0%}")
        print(f"   💾 Modelli salvati in: {self.models_dir}")

    def load_and_prepare_progressive_data(self):
        """
        Carica e prepara i dati per training progressivo.
        """
        print("\n📊 Caricamento e preparazione dati progressivi...")
        
        # Carica dataset
        df = pd.read_csv(self.dataset_path)
        print(f"   ✅ Dataset caricato: {len(df)} partite, {len(df.columns)} colonne")
        
        # Converti date
        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed')
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Determina stagioni NBA
        def get_nba_season(date):
            if date.month >= 10:
                return f"{date.year}-{str(date.year + 1)[-2:]}"
            else:
                return f"{date.year - 1}-{str(date.year)[-2:]}"
        
        df['season'] = df['game_date'].apply(get_nba_season)
        
        # Determina tipo partita
        def get_game_type(date):
            if date.month >= 4 and date.month <= 6:
                if date.month == 4 and date.day < 15:
                    return 'Regular'
                else:
                    return 'Playoff'
            else:
                return 'Regular'
        
        df['game_type'] = df['game_date'].apply(get_game_type)
        
        # Filtra solo Regular Season della stagione base
        season_mask = df['season'] == self.base_season
        regular_mask = df['game_type'] == 'Regular'
        
        self.base_df = df[season_mask & regular_mask].copy().reset_index(drop=True)
        self.full_df = df.copy()  # Mantieni tutto per training finale
        
        print(f"\n📈 Analisi dati progressivi:")
        print(f"   🗓️ Stagione base {self.base_season}: {len(self.base_df)} partite Regular Season")
        print(f"   🗓️ Dataset completo: {len(self.full_df)} partite totali")
        print(f"   📅 Date base: da {self.base_df['game_date'].min().date()} a {self.base_df['game_date'].max().date()}")
        
        # Rimuovi righe con valori mancanti nel target
        initial_len = len(self.base_df)
        self.base_df = self.base_df.dropna(subset=['score_deviation'])
        print(f"   🧹 Rimosse {initial_len - len(self.base_df)} righe con target mancante")
        
        # Definisci feature
        feature_columns = [
            'home_momentum_score', 'home_hot_hand_players', 'home_avg_player_momentum',
            'home_avg_player_weighted_contribution', 'home_team_offensive_potential',
            'home_team_defensive_potential', 'away_momentum_score', 'away_hot_hand_players',
            'away_avg_player_momentum', 'away_avg_player_weighted_contribution',
            'away_team_offensive_potential', 'away_team_defensive_potential', 'momentum_diff'
        ]
        
        # Verifica feature
        missing_features = [f for f in feature_columns if f not in self.base_df.columns]
        if missing_features:
            print(f"   ⚠️ Feature mancanti: {missing_features}")
            feature_columns = [f for f in feature_columns if f in self.base_df.columns]
        
        print(f"   🎯 Feature utilizzate: {len(feature_columns)}")
        self.feature_names = feature_columns
        
        return self.base_df

    def create_temporal_splits(self):
        """
        Crea 4 split temporali della stagione Regular 2023-24.
        """
        print(f"\n🔄 Creazione {self.n_splits} split temporali...")
        
        df = self.base_df.copy()
        n_games_per_split = len(df) // self.n_splits
        
        splits = {}
        
        for i in range(self.n_splits):
            start_idx = i * n_games_per_split
            if i == self.n_splits - 1:  # Ultimo split prende tutto il resto
                end_idx = len(df)
            else:
                end_idx = (i + 1) * n_games_per_split
            
            split_df = df.iloc[start_idx:end_idx].copy()
            
            # Dividi ogni split in 80% training, 20% test
            split_train_size = int(len(split_df) * self.train_ratio)
            
            train_df = split_df.iloc[:split_train_size].copy()
            test_df = split_df.iloc[split_train_size:].copy()
            
            splits[f'split_{i+1}'] = {
                'full': split_df,
                'train': train_df,
                'test': test_df,
                'date_range': (split_df['game_date'].min().date(), split_df['game_date'].max().date()),
                'train_date_range': (train_df['game_date'].min().date(), train_df['game_date'].max().date()),
                'test_date_range': (test_df['game_date'].min().date(), test_df['game_date'].max().date())
            }
            
            print(f"   📊 Split {i+1}: {len(split_df)} partite totali")
            print(f"      🟢 Train: {len(train_df)} partite ({train_df['game_date'].min().date()} - {train_df['game_date'].max().date()})")
            print(f"      🔴 Test:  {len(test_df)} partite ({test_df['game_date'].min().date()} - {test_df['game_date'].max().date()})")
        
        self.splits_data = splits
        return splits

    def progressive_training(self):
        """
        Esegue training progressivo: Split1 → Split1+2 → Split1+2+3 → Split1+2+3+4
        """
        print("\n🚀 AVVIO TRAINING PROGRESSIVO")
        print("=" * 60)
        
        # Modelli da testare
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        progressive_results = {}
        
        for step in range(1, self.n_splits + 1):
            print(f"\n🔄 STEP {step}/{self.n_splits}: Training su Split 1-{step}")
            
            # Combina dati di training progressivamente
            train_dfs = []
            test_dfs = []
            
            for i in range(1, step + 1):
                split_key = f'split_{i}'
                train_dfs.append(self.splits_data[split_key]['train'])
                test_dfs.append(self.splits_data[split_key]['test'])
            
            # Combina in un unico dataset
            combined_train = pd.concat(train_dfs, ignore_index=True)
            combined_test = pd.concat(test_dfs, ignore_index=True)
            
            print(f"   📊 Dati combinati: {len(combined_train)} training, {len(combined_test)} test")
            print(f"   📅 Range training: {combined_train['game_date'].min().date()} - {combined_train['game_date'].max().date()}")
            print(f"   📅 Range test: {combined_test['game_date'].min().date()} - {combined_test['game_date'].max().date()}")
            
            # Prepara X e y
            X_train = combined_train[self.feature_names].values
            y_train = combined_train['score_deviation'].values
            X_test = combined_test[self.feature_names].values
            y_test = combined_test['score_deviation'].values
            
            # Standardizza
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Addestra e valuta ogni modello
            step_results = {}
            
            for name, model in models.items():
                print(f"\n      🤖 Training {name}...")
                
                # Addestra
                model_instance = model.__class__(**model.get_params())
                model_instance.fit(X_train_scaled, y_train)
                
                # Predizioni
                train_pred = model_instance.predict(X_train_scaled)
                test_pred = model_instance.predict(X_test_scaled)
                
                # Metriche
                train_mae = mean_absolute_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                step_results[name] = {
                    'model': model_instance,
                    'scaler': scaler,
                    'train_mae': train_mae,
                    'train_r2': train_r2,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'train_pred': train_pred,
                    'test_pred': test_pred,
                    'step': step,
                    'training_splits': list(range(1, step + 1))
                }
                
                print(f"         📊 Train MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
                print(f"         📊 Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
            
            # Trova migliore modello per questo step
            best_name = min(step_results.keys(), key=lambda x: step_results[x]['test_mae'])
            best_result = step_results[best_name]
            
            print(f"\n   🏆 Migliore Step {step}: {best_name}")
            print(f"      📊 Test MAE: {best_result['test_mae']:.3f}")
            print(f"      📊 Test R²: {best_result['test_r2']:.3f}")
            
            # Salva risultati step
            progressive_results[f'step_{step}'] = {
                'all_models': step_results,
                'best_model_name': best_name,
                'best_model': best_result,
                'validation_mae': best_result['test_mae'],
                'validation_r2': best_result['test_r2']
            }
            
            # Salva modello migliore di questo step
            self.save_step_model(step, best_name, best_result)
            
            # Validazione: controlla se il modello sta migliorando
            if step > 1:
                prev_mae = progressive_results[f'step_{step-1}']['validation_mae']
                current_mae = best_result['test_mae']
                improvement = prev_mae - current_mae
                
                print(f"   📈 Miglioramento vs Step {step-1}: {improvement:.3f} MAE")
                if improvement < 0:
                    print(f"   ⚠️ Performance peggiorata di {abs(improvement):.3f}")
                else:
                    print(f"   ✅ Performance migliorata di {improvement:.3f}")
        
        self.progressive_results = progressive_results
        
        # Seleziona modello finale (ultimo step)
        final_step = f'step_{self.n_splits}'
        self.final_model = progressive_results[final_step]['best_model']['model']
        self.final_scaler = progressive_results[final_step]['best_model']['scaler']
        
        print(f"\n🎯 TRAINING PROGRESSIVO COMPLETATO")
        print(f"   🏆 Modello finale: {progressive_results[final_step]['best_model_name']}")
        print(f"   📊 MAE finale: {progressive_results[final_step]['validation_mae']:.3f}")
        
        return progressive_results

    def save_step_model(self, step, model_name, model_data):
        """
        Salva il modello migliore per ogni step.
        """
        step_dir = os.path.join(self.models_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)
        
        # Salva modello
        model_path = os.path.join(step_dir, f'best_model_{model_name.replace(" ", "_").lower()}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data['model'], f)
        
        # Salva scaler
        scaler_path = os.path.join(step_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(model_data['scaler'], f)
        
        # Salva metadati
        metadata = {
            'step': step,
            'model_name': model_name,
            'training_splits': model_data['training_splits'],
            'train_mae': model_data['train_mae'],
            'train_r2': model_data['train_r2'],
            'test_mae': model_data['test_mae'],
            'test_r2': model_data['test_r2'],
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(step_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      💾 Step {step} salvato in: {step_dir}")

    def final_training_on_all_data(self):
        """
        Training finale su tutto il dataset (inclusi playoff e altre stagioni).
        """
        print(f"\n🚀 TRAINING FINALE SU TUTTO IL DATASET")
        print("=" * 50)
        
        # Prepara dataset completo
        full_df = self.full_df.copy()
        full_df = full_df.dropna(subset=['score_deviation'])
        
        # Dividi in training (2023-24) e test (2024-25)
        train_mask = full_df['season'] == '2023-24'
        test_mask = full_df['season'] == '2024-25'
        
        train_df = full_df[train_mask]
        test_df = full_df[test_mask]
        
        print(f"   📊 Training completo: {len(train_df)} partite (stagione 2023-24)")
        print(f"   📊 Test finale: {len(test_df)} partite (stagione 2024-25)")
        
        # Prepara dati
        X_train = train_df[self.feature_names].values
        y_train = train_df['score_deviation'].values
        X_test = test_df[self.feature_names].values
        y_test = test_df['score_deviation'].values
        
        # Standardizza
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train)
        X_test_scaled = final_scaler.transform(X_test)
        
        # Usa il tipo di modello migliore dal training progressivo
        best_model_name = self.progressive_results[f'step_{self.n_splits}']['best_model_name']
        
        # Ricrea modello dello stesso tipo
        if 'Linear' in best_model_name:
            final_model = LinearRegression()
        elif 'Ridge' in best_model_name:
            final_model = Ridge(alpha=1.0)
        elif 'Lasso' in best_model_name:
            final_model = Lasso(alpha=1.0)
        elif 'Random Forest' in best_model_name:
            final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif 'Gradient Boosting' in best_model_name:
            final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            final_model = LinearRegression()  # Fallback
        
        # Addestra modello finale
        print(f"   🤖 Training finale: {best_model_name}")
        final_model.fit(X_train_scaled, y_train)
        
        # Valutazione finale
        train_pred = final_model.predict(X_train_scaled)
        test_pred = final_model.predict(X_test_scaled)
        
        final_train_mae = mean_absolute_error(y_train, train_pred)
        final_train_r2 = r2_score(y_train, train_pred)
        final_test_mae = mean_absolute_error(y_test, test_pred)
        final_test_r2 = r2_score(y_test, test_pred)
        
        print(f"   📊 Training finale: MAE={final_train_mae:.3f}, R²={final_train_r2:.3f}")
        print(f"   📊 Test finale: MAE={final_test_mae:.3f}, R²={final_test_r2:.3f}")
        
        # Salva modello finale
        self.save_final_model(final_model, final_scaler, {
            'model_name': best_model_name,
            'train_mae': final_train_mae,
            'train_r2': final_train_r2,
            'test_mae': final_test_mae,
            'test_r2': final_test_r2,
            'train_size': len(train_df),
            'test_size': len(test_df)
        })
        
        return {
            'model': final_model,
            'scaler': final_scaler,
            'train_mae': final_train_mae,
            'test_mae': final_test_mae,
            'train_r2': final_train_r2,
            'test_r2': final_test_r2
        }

    def save_final_model(self, model, scaler, metadata):
        """
        Salva il modello finale addestrato su tutto il dataset.
        """
        print(f"\n💾 Salvataggio modello finale...")
        
        # Salva modello finale
        model_path = os.path.join(self.models_dir, 'final_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Modello finale salvato: {model_path}")
        
        # Salva scaler finale
        scaler_path = os.path.join(self.models_dir, 'final_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   ✅ Scaler finale salvato: {scaler_path}")
        
        # Metadati completi
        complete_metadata = {
            'training_type': 'progressive_walk_forward',
            'n_splits': self.n_splits,
            'base_season': self.base_season,
            'train_ratio_per_split': self.train_ratio,
            'feature_names': self.feature_names,
            'final_model': metadata,
            'progressive_history': {},
            'training_date': datetime.now().isoformat()
        }
        
        # Aggiungi storia progressiva
        for step_key, step_data in self.progressive_results.items():
            complete_metadata['progressive_history'][step_key] = {
                'best_model_name': step_data['best_model_name'],
                'validation_mae': step_data['validation_mae'],
                'validation_r2': step_data['validation_r2']
            }
        
        metadata_path = os.path.join(self.models_dir, 'complete_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(complete_metadata, f, indent=2)
        print(f"   ✅ Metadati completi salvati: {metadata_path}")

    def create_progressive_visualizations(self):
        """
        Crea visualizzazioni del training progressivo.
        """
        print(f"\n📊 Creazione visualizzazioni progressive...")
        
        # Setup plot
        plt.figure(figsize=(20, 15))
        
        # 1. Evoluzione MAE progressiva
        plt.subplot(3, 4, 1)
        steps = list(range(1, self.n_splits + 1))
        maes = [self.progressive_results[f'step_{s}']['validation_mae'] for s in steps]
        
        plt.plot(steps, maes, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Training Step')
        plt.ylabel('Validation MAE')
        plt.title('Evoluzione MAE Progressiva')
        plt.grid(True, alpha=0.3)
        for i, mae in enumerate(maes):
            plt.annotate(f'{mae:.3f}', (steps[i], mae), textcoords="offset points", xytext=(0,10), ha='center')
        
        # 2. Evoluzione R² progressiva
        plt.subplot(3, 4, 2)
        r2s = [self.progressive_results[f'step_{s}']['validation_r2'] for s in steps]
        
        plt.plot(steps, r2s, 's-', linewidth=2, markersize=8, color='red')
        plt.xlabel('Training Step')
        plt.ylabel('Validation R²')
        plt.title('Evoluzione R² Progressiva')
        plt.grid(True, alpha=0.3)
        for i, r2 in enumerate(r2s):
            plt.annotate(f'{r2:.3f}', (steps[i], r2), textcoords="offset points", xytext=(0,10), ha='center')
        
        # 3. Confronto modelli per step finale
        plt.subplot(3, 4, 3)
        final_step_results = self.progressive_results[f'step_{self.n_splits}']['all_models']
        model_names = list(final_step_results.keys())
        model_maes = [final_step_results[name]['test_mae'] for name in model_names]
        
        bars = plt.bar(range(len(model_names)), model_maes, color='skyblue')
        plt.xlabel('Modelli')
        plt.ylabel('Test MAE')
        plt.title(f'Confronto Modelli - Step {self.n_splits}')
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=0)
        
        # Evidenzia il migliore
        best_idx = model_names.index(self.progressive_results[f'step_{self.n_splits}']['best_model_name'])
        bars[best_idx].set_color('gold')
        
        # 4. Miglioramenti step-by-step
        plt.subplot(3, 4, 4)
        if len(steps) > 1:
            improvements = []
            for i in range(1, len(steps)):
                prev_mae = maes[i-1]
                curr_mae = maes[i]
                improvement = prev_mae - curr_mae
                improvements.append(improvement)
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            plt.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            plt.xlabel('Step Transition')
            plt.ylabel('MAE Improvement')
            plt.title('Miglioramenti Step-by-Step')
            plt.xticks(range(len(improvements)), [f'{i+1}→{i+2}' for i in range(len(improvements))])
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 5-8. Predizioni vs Reali per ogni step
        for step in range(1, min(5, self.n_splits + 1)):  # Massimo 4 plot
            plt.subplot(3, 4, 4 + step)
            
            step_data = self.progressive_results[f'step_{step}']['best_model']
            y_test = step_data['test_pred']  # Nota: invertito per consistenza
            y_pred = step_data['test_pred']  
            
            # Correggi: usa dati reali dal test set del step
            # Ricostruisci i dati di test per questo step
            test_dfs = []
            for i in range(1, step + 1):
                test_dfs.append(self.splits_data[f'split_{i}']['test'])
            combined_test = pd.concat(test_dfs, ignore_index=True)
            y_actual = combined_test['score_deviation'].values
            
            plt.scatter(y_actual, step_data['test_pred'], alpha=0.6, s=30)
            min_val = min(y_actual.min(), step_data['test_pred'].min())
            max_val = max(y_actual.max(), step_data['test_pred'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('Score Deviation Reale')
            plt.ylabel('Score Deviation Predetta')
            plt.title(f'Step {step}: {step_data["test_mae"]:.3f} MAE')
            
        # 9. Distribuzione dimensioni training set
        plt.subplot(3, 4, 9)
        train_sizes = []
        test_sizes = []
        for step in steps:
            step_data = self.progressive_results[f'step_{step}']['all_models']
            # Calcola dimensioni dai splits
            train_size = sum(len(self.splits_data[f'split_{i}']['train']) for i in range(1, step + 1))
            test_size = sum(len(self.splits_data[f'split_{i}']['test']) for i in range(1, step + 1))
            train_sizes.append(train_size)
            test_sizes.append(test_size)
        
        x = np.arange(len(steps))
        width = 0.35
        
        plt.bar(x - width/2, train_sizes, width, label='Training', alpha=0.8)
        plt.bar(x + width/2, test_sizes, width, label='Test', alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Numero Partite')
        plt.title('Dimensioni Dataset per Step')
        plt.xticks(x, [f'Step {s}' for s in steps])
        plt.legend()
        
        # 10. Feature importance finale (se disponibile)
        final_model = self.progressive_results[f'step_{self.n_splits}']['best_model']['model']
        if hasattr(final_model, 'feature_importances_'):
            plt.subplot(3, 4, 10)
            importance = final_model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]
            
            plt.bar(range(len(indices)), importance[indices])
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title('Top 10 Feature Importance\n(Modello Finale)')
            plt.xticks(range(len(indices)), 
                      [self.feature_names[i].replace('_', '\n') for i in indices], 
                      rotation=45, ha='right')
        
        # 11. Timeline dei dati
        plt.subplot(3, 4, 11)
        colors = ['blue', 'green', 'orange', 'red']
        for i, split_key in enumerate(self.splits_data.keys()):
            split_data = self.splits_data[split_key]
            start_date = split_data['date_range'][0]
            end_date = split_data['date_range'][1]
            
            # Converti date in numeri per il plot
            import matplotlib.dates as mdates
            start_num = mdates.date2num(start_date)
            end_num = mdates.date2num(end_date)
            
            plt.barh(i, end_num - start_num, left=start_num, 
                    color=colors[i % len(colors)], alpha=0.7, 
                    label=f'Split {i+1}')
        
        plt.xlabel('Date')
        plt.ylabel('Split')
        plt.title('Timeline Split Temporali')
        plt.legend()
        
        # 12. Distribuzione errori finale
        plt.subplot(3, 4, 12)
        final_best = self.progressive_results[f'step_{self.n_splits}']['best_model']
        
        # Ricostruisci errori finali
        test_dfs = []
        for i in range(1, self.n_splits + 1):
            test_dfs.append(self.splits_data[f'split_{i}']['test'])
        combined_test = pd.concat(test_dfs, ignore_index=True)
        errors = np.abs(combined_test['score_deviation'].values - final_best['test_pred'])
        
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=errors.mean(), color='r', linestyle='--', 
                   label=f'Media: {errors.mean():.2f}')
        plt.axvline(x=errors.median(), color='g', linestyle='--', 
                   label=f'Mediana: {errors.median():.2f}')
        plt.xlabel('Errore Assoluto')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione Errori Finali')
        plt.legend()
        
        plt.tight_layout()
        
        # Salva plot
        plot_path = os.path.join(self.models_dir, 'progressive_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizzazioni salvate: {plot_path}")
        
        plt.show()

    def run_complete_progressive_training(self):
        """
        Esegue l'intero processo di training progressivo.
        """
        print("🚀 AVVIO TRAINING PROGRESSIVO COMPLETO")
        print("=" * 70)
        
        # 1. Carica e prepara dati
        self.load_and_prepare_progressive_data()
        
        # 2. Crea split temporali
        self.create_temporal_splits()
        
        # 3. Training progressivo
        self.progressive_training()
        
        # 4. Training finale su tutto il dataset
        final_results = self.final_training_on_all_data()
        
        # 5. Crea visualizzazioni
        self.create_progressive_visualizations()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING PROGRESSIVO COMPLETATO!")
        print(f"\n📊 RIEPILOGO PROGRESSIVO:")
        
        # Mostra evoluzione
        for step in range(1, self.n_splits + 1):
            step_data = self.progressive_results[f'step_{step}']
            print(f"   Step {step}: {step_data['best_model_name']:<20} | "
                  f"MAE: {step_data['validation_mae']:.3f} | "
                  f"R²: {step_data['validation_r2']:.3f}")
        
        print(f"\n🎯 MODELLO FINALE:")
        print(f"   📊 Test MAE: {final_results['test_mae']:.3f}")
        print(f"   📊 Test R²: {final_results['test_r2']:.3f}")
        print(f"   💾 Salvato in: {self.models_dir}")
        
        return {
            'progressive_results': self.progressive_results,
            'final_results': final_results,
            'splits_data': self.splits_data
        }


def main():
    """
    Funzione principale per training progressivo.
    """
    trainer = ProgressiveMomentumMLTrainer()
    results = trainer.run_complete_progressive_training()
    
    return results


if __name__ == "__main__":
    main() 