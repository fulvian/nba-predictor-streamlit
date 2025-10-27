#!/usr/bin/env python3
"""
Momentum ML Trainer
===================
Addestra modelli di machine learning per predire score_deviation usando feature di momentum.
Il dataset contiene 2624 partite con feature di momentum pre-calcolate.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MomentumMLTrainer:
    """
    Trainer per modelli ML che predicono score_deviation usando feature di momentum.
    """
    
    def __init__(self, dataset_path='data/momentum_v2/momentum_training_dataset.csv'):
        """
        Inizializza il trainer.
        
        Args:
            dataset_path: Percorso al dataset momentum
        """
        self.dataset_path = dataset_path
        self.models_dir = 'models/momentum_ml'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Risultati training
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        
        print(f"🚀 MomentumMLTrainer inizializzato")
        print(f"   📁 Dataset: {dataset_path}")
        print(f"   💾 Modelli salvati in: {self.models_dir}")

    def load_and_prepare_data(self):
        """
        Carica e prepara i dati per il training.
        """
        print("\n📊 Caricamento e preparazione dati...")
        
        # Carica dataset
        df = pd.read_csv(self.dataset_path)
        print(f"   ✅ Dataset caricato: {len(df)} partite, {len(df.columns)} colonne")
        
        # Rimuovi righe con valori mancanti nel target
        initial_len = len(df)
        df = df.dropna(subset=['score_deviation'])
        print(f"   🧹 Rimosse {initial_len - len(df)} righe con target mancante")
        
        # Definisci feature e target
        feature_columns = [
            'home_momentum_score', 'home_hot_hand_players', 'home_avg_player_momentum',
            'home_avg_player_weighted_contribution', 'home_team_offensive_potential',
            'home_team_defensive_potential', 'away_momentum_score', 'away_hot_hand_players',
            'away_avg_player_momentum', 'away_avg_player_weighted_contribution',
            'away_team_offensive_potential', 'away_team_defensive_potential', 'momentum_diff'
        ]
        
        # Verifica che tutte le feature esistano
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            print(f"   ⚠️ Feature mancanti: {missing_features}")
            feature_columns = [f for f in feature_columns if f in df.columns]
        
        print(f"   🎯 Feature utilizzate: {len(feature_columns)}")
        for i, feature in enumerate(feature_columns, 1):
            print(f"      {i:2d}. {feature}")
        
        # Prepara X e y
        X = df[feature_columns].copy()
        y = df['score_deviation'].copy()
        
        # Rimuovi righe con valori mancanti nelle feature
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        print(f"   📈 Dataset finale: {len(X)} partite")
        print(f"   🎯 Target statistics:")
        print(f"      Media: {y.mean():.2f}")
        print(f"      Std: {y.std():.2f}")
        print(f"      Min: {y.min():.2f}")
        print(f"      Max: {y.max():.2f}")
        
        self.X = X
        self.y = y
        self.feature_names = feature_columns
        
        return X, y

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divide i dati in training e test set.
        """
        print(f"\n🔄 Divisione dati (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"   📊 Training set: {len(X_train)} partite")
        print(f"   📊 Test set: {len(X_test)} partite")
        
        # Standardizza le feature
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self):
        """
        Addestra diversi modelli ML e li confronta.
        """
        print("\n🤖 Training modelli ML...")
        
        # Definisci modelli da testare
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n   🔄 Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Training completo
            model.fit(self.X_train, self.y_train)
            
            # Predizioni
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            # Metriche
            train_mae = mean_absolute_error(self.y_train, train_pred)
            train_mse = mean_squared_error(self.y_train, train_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            results[name] = {
                'model': model,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'train_pred': train_pred,
                'test_pred': test_pred
            }
            
            print(f"      📊 CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
            print(f"      📊 Test MAE: {test_mae:.3f}")
            print(f"      📊 Test R²: {test_r2:.3f}")
        
        self.results = results
        
        # Trova il modello migliore (basato su test MAE)
        best_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\n🏆 Modello migliore: {best_name}")
        print(f"   📊 Test MAE: {results[best_name]['test_mae']:.3f}")
        print(f"   📊 Test R²: {results[best_name]['test_r2']:.3f}")
        
        return results

    def optimize_best_model(self):
        """
        Ottimizza gli iperparametri del modello migliore.
        """
        print(f"\n⚙️ Ottimizzazione iperparametri per {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        elif 'Ridge' in self.best_model_name:
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            base_model = Ridge()
            
        elif 'Lasso' in self.best_model_name:
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            base_model = Lasso()
            
        else:
            print("   ⚠️ Modello non supportato per ottimizzazione")
            return self.best_model
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        optimized_model = grid_search.best_estimator_
        
        # Valuta modello ottimizzato
        test_pred_opt = optimized_model.predict(self.X_test)
        test_mae_opt = mean_absolute_error(self.y_test, test_pred_opt)
        test_r2_opt = r2_score(self.y_test, test_pred_opt)
        
        print(f"   🎯 Migliori parametri: {grid_search.best_params_}")
        print(f"   📊 Test MAE ottimizzato: {test_mae_opt:.3f}")
        print(f"   📊 Test R² ottimizzato: {test_r2_opt:.3f}")
        
        # Aggiorna se il modello ottimizzato è migliore
        original_mae = self.results[self.best_model_name]['test_mae']
        if test_mae_opt < original_mae:
            print(f"   ✅ Miglioramento: {original_mae:.3f} → {test_mae_opt:.3f}")
            self.best_model = optimized_model
            self.results[self.best_model_name]['optimized_model'] = optimized_model
            self.results[self.best_model_name]['optimized_test_mae'] = test_mae_opt
            self.results[self.best_model_name]['optimized_test_r2'] = test_r2_opt
        else:
            print(f"   ℹ️ Nessun miglioramento significativo")
        
        return optimized_model

    def save_models(self):
        """
        Salva il modello migliore e il scaler.
        """
        print(f"\n💾 Salvataggio modelli...")
        
        # Salva modello migliore
        model_path = os.path.join(self.models_dir, 'best_momentum_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"   ✅ Modello salvato: {model_path}")
        
        # Salva scaler
        scaler_path = os.path.join(self.models_dir, 'momentum_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ Scaler salvato: {scaler_path}")
        
        # Salva metadati
        metadata = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(self.X),
            'test_mae': self.results[self.best_model_name]['test_mae'],
            'test_r2': self.results[self.best_model_name]['test_r2'],
            'target_stats': {
                'mean': float(self.y.mean()),
                'std': float(self.y.std()),
                'min': float(self.y.min()),
                'max': float(self.y.max())
            }
        }
        
        metadata_path = os.path.join(self.models_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadati salvati: {metadata_path}")

    def create_visualizations(self):
        """
        Crea visualizzazioni dei risultati.
        """
        print(f"\n📊 Creazione visualizzazioni...")
        
        # Confronto modelli
        plt.figure(figsize=(15, 10))
        
        # 1. Confronto MAE
        plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        test_maes = [self.results[name]['test_mae'] for name in model_names]
        
        bars = plt.bar(range(len(model_names)), test_maes, color='skyblue')
        plt.xlabel('Modelli')
        plt.ylabel('Test MAE')
        plt.title('Confronto Test MAE')
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=0)
        
        # Evidenzia il migliore
        best_idx = model_names.index(self.best_model_name)
        bars[best_idx].set_color('gold')
        
        # 2. Confronto R²
        plt.subplot(2, 3, 2)
        test_r2s = [self.results[name]['test_r2'] for name in model_names]
        
        bars = plt.bar(range(len(model_names)), test_r2s, color='lightcoral')
        plt.xlabel('Modelli')
        plt.ylabel('Test R²')
        plt.title('Confronto Test R²')
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=0)
        
        # Evidenzia il migliore
        bars[best_idx].set_color('gold')
        
        # 3. Predizioni vs Reali (modello migliore)
        plt.subplot(2, 3, 3)
        test_pred = self.results[self.best_model_name]['test_pred']
        plt.scatter(self.y_test, test_pred, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Score Deviation Reale')
        plt.ylabel('Score Deviation Predetta')
        plt.title(f'Predizioni vs Reali\n{self.best_model_name}')
        
        # 4. Residui
        plt.subplot(2, 3, 4)
        residuals = self.y_test - test_pred
        plt.scatter(test_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predizioni')
        plt.ylabel('Residui')
        plt.title('Analisi Residui')
        
        # 5. Feature Importance (se disponibile)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 3, 5)
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]  # Top 10
            
            plt.bar(range(len(indices)), importance[indices])
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.xticks(range(len(indices)), 
                      [self.feature_names[i].replace('_', '\n') for i in indices], 
                      rotation=45, ha='right')
        
        # 6. Distribuzione errori
        plt.subplot(2, 3, 6)
        errors = np.abs(self.y_test - test_pred)
        plt.hist(errors, bins=30, alpha=0.7, color='orange')
        plt.xlabel('Errore Assoluto')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione Errori Assoluti')
        plt.axvline(x=errors.mean(), color='r', linestyle='--', label=f'Media: {errors.mean():.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        # Salva plot
        plot_path = os.path.join(self.models_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizzazioni salvate: {plot_path}")
        
        plt.show()

    def run_complete_training(self):
        """
        Esegue il training completo.
        """
        print("🚀 AVVIO TRAINING COMPLETO MOMENTUM ML")
        print("=" * 50)
        
        # 1. Carica dati
        self.load_and_prepare_data()
        
        # 2. Dividi dati
        self.split_data()
        
        # 3. Addestra modelli
        self.train_models()
        
        # 4. Ottimizza modello migliore
        self.optimize_best_model()
        
        # 5. Salva modelli
        self.save_models()
        
        # 6. Crea visualizzazioni
        self.create_visualizations()
        
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETATO CON SUCCESSO!")
        print(f"🏆 Modello migliore: {self.best_model_name}")
        print(f"📊 Test MAE: {self.results[self.best_model_name]['test_mae']:.3f}")
        print(f"📊 Test R²: {self.results[self.best_model_name]['test_r2']:.3f}")
        print(f"💾 Modelli salvati in: {self.models_dir}")


def main():
    """
    Funzione principale per eseguire il training.
    """
    # Crea trainer
    trainer = MomentumMLTrainer()
    
    # Esegui training completo
    trainer.run_complete_training()


if __name__ == "__main__":
    main()