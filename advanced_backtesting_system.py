#!/usr/bin/env python3
"""
📊 ADVANCED BACKTESTING SYSTEM
Sistema di validazione avanzato con split stagionali multipli
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import json

class AdvancedBacktestingSystem:
    """Sistema di backtesting avanzato per modelli NBA"""
    
    def __init__(self):
        self.results = {}
        
    def run_comprehensive_backtest(self, X, y_mu, y_sigma, seasons):
        """
        Esegue backtesting completo con multiple strategie di split
        
        Args:
            X: Feature matrix
            y_mu: Target MU (punteggi totali)
            y_sigma: Target SIGMA (deviazione standard)
            seasons: Array delle stagioni (es. [2020, 2021, 2022, 2023, 2024])
        """
        
        print("📊 === COMPREHENSIVE BACKTESTING SYSTEM ===")
        
        # 1. SEASON-BY-SEASON BACKTESTING
        print("\n🏀 STEP 1: Season-by-Season Validation")
        season_results = self._season_by_season_validation(X, y_mu, y_sigma, seasons)
        
        # 2. ROLLING WINDOW BACKTESTING  
        print("\n🔄 STEP 2: Rolling Window Validation")
        rolling_results = self._rolling_window_validation(X, y_mu, y_sigma, seasons)
        
        # 3. EXPANDING WINDOW BACKTESTING
        print("\n📈 STEP 3: Expanding Window Validation")
        expanding_results = self._expanding_window_validation(X, y_mu, y_sigma, seasons)
        
        # 4. CROSS-SEASON VALIDATION
        print("\n🔀 STEP 4: Cross-Season Validation")
        cross_season_results = self._cross_season_validation(X, y_mu, y_sigma, seasons)
        
        # 5. PERFORMANCE SUMMARY
        print("\n📋 STEP 5: Performance Summary")
        self._generate_performance_summary({
            'season_by_season': season_results,
            'rolling_window': rolling_results,
            'expanding_window': expanding_results,
            'cross_season': cross_season_results
        })
        
        return self.results
    
    def _season_by_season_validation(self, X, y_mu, y_sigma, seasons):
        """
        Validazione stagione per stagione
        Train: Stagioni precedenti, Test: Stagione corrente
        """
        
        results = []
        
        for i, test_season in enumerate(seasons[1:], 1):  # Inizia dalla seconda stagione
            # Train: Tutte le stagioni precedenti
            train_mask = seasons[:i]
            train_indices = X.index[X['SEASON'].isin(train_mask)]
            
            # Test: Stagione corrente
            test_indices = X.index[X['SEASON'] == test_season]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            print(f"   🏀 Testing Season {test_season}")
            print(f"      Train: {train_mask} ({len(train_indices)} games)")
            print(f"      Test: {test_season} ({len(test_indices)} games)")
            
            # Split data
            X_train = X.loc[train_indices].drop('SEASON', axis=1)
            X_test = X.loc[test_indices].drop('SEASON', axis=1) 
            y_mu_train = y_mu.loc[train_indices]
            y_mu_test = y_mu.loc[test_indices]
            y_sigma_train = y_sigma.loc[train_indices]
            y_sigma_test = y_sigma.loc[test_indices]
            
            # Train and evaluate
            season_result = self._train_and_evaluate(
                X_train, X_test, y_mu_train, y_mu_test, 
                y_sigma_train, y_sigma_test, f"Season_{test_season}"
            )
            
            season_result['test_season'] = test_season
            season_result['train_seasons'] = train_mask.tolist()
            results.append(season_result)
        
        return results
    
    def _rolling_window_validation(self, X, y_mu, y_sigma, seasons, window_size=2):
        """
        Validazione con finestra mobile
        Train: Ultime N stagioni, Test: Stagione successiva
        """
        
        results = []
        
        for i in range(window_size, len(seasons)):
            # Train: Finestra mobile delle ultime window_size stagioni
            train_seasons = seasons[i-window_size:i]
            test_season = seasons[i]
            
            train_indices = X.index[X['SEASON'].isin(train_seasons)]
            test_indices = X.index[X['SEASON'] == test_season]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            print(f"   🔄 Rolling Window: Train {train_seasons} → Test {test_season}")
            
            # Split data
            X_train = X.loc[train_indices].drop('SEASON', axis=1)
            X_test = X.loc[test_indices].drop('SEASON', axis=1)
            y_mu_train = y_mu.loc[train_indices]
            y_mu_test = y_mu.loc[test_indices]
            y_sigma_train = y_sigma.loc[train_indices]
            y_sigma_test = y_sigma.loc[test_indices]
            
            # Train and evaluate
            rolling_result = self._train_and_evaluate(
                X_train, X_test, y_mu_train, y_mu_test,
                y_sigma_train, y_sigma_test, f"Rolling_{test_season}"
            )
            
            rolling_result['test_season'] = test_season
            rolling_result['train_seasons'] = train_seasons.tolist()
            rolling_result['window_size'] = window_size
            results.append(rolling_result)
        
        return results
    
    def _expanding_window_validation(self, X, y_mu, y_sigma, seasons):
        """
        Validazione con finestra espandente
        Train: Dall'inizio fino alla stagione N-1, Test: Stagione N
        """
        
        results = []
        
        for i, test_season in enumerate(seasons[2:], 2):  # Inizia dalla terza stagione
            # Train: Dall'inizio fino alla stagione precedente
            train_seasons = seasons[:i]
            
            train_indices = X.index[X['SEASON'].isin(train_seasons)]
            test_indices = X.index[X['SEASON'] == test_season]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            print(f"   📈 Expanding: Train {train_seasons} → Test {test_season}")
            
            # Split data
            X_train = X.loc[train_indices].drop('SEASON', axis=1)
            X_test = X.loc[test_indices].drop('SEASON', axis=1)
            y_mu_train = y_mu.loc[train_indices]
            y_mu_test = y_mu.loc[test_indices]
            y_sigma_train = y_sigma.loc[train_indices]
            y_sigma_test = y_sigma.loc[test_indices]
            
            # Train and evaluate
            expanding_result = self._train_and_evaluate(
                X_train, X_test, y_mu_train, y_mu_test,
                y_sigma_train, y_sigma_test, f"Expanding_{test_season}"
            )
            
            expanding_result['test_season'] = test_season
            expanding_result['train_seasons'] = train_seasons.tolist()
            results.append(expanding_result)
        
        return results
    
    def _cross_season_validation(self, X, y_mu, y_sigma, seasons):
        """
        Validazione incrociata tra stagioni
        Train: Stagioni non consecutive, Test: Stagione intermedia
        """
        
        results = []
        
        # Test ogni stagione usando le altre come training
        for test_season in seasons[1:-1]:  # Escludi prima e ultima
            train_seasons = [s for s in seasons if s != test_season]
            
            train_indices = X.index[X['SEASON'].isin(train_seasons)]
            test_indices = X.index[X['SEASON'] == test_season]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            print(f"   🔀 Cross-Season: Test {test_season}, Train {train_seasons}")
            
            # Split data
            X_train = X.loc[train_indices].drop('SEASON', axis=1)
            X_test = X.loc[test_indices].drop('SEASON', axis=1)
            y_mu_train = y_mu.loc[train_indices]
            y_mu_test = y_mu.loc[test_indices]
            y_sigma_train = y_sigma.loc[train_indices]
            y_sigma_test = y_sigma.loc[test_indices]
            
            # Train and evaluate
            cross_result = self._train_and_evaluate(
                X_train, X_test, y_mu_train, y_mu_test,
                y_sigma_train, y_sigma_test, f"Cross_{test_season}"
            )
            
            cross_result['test_season'] = test_season
            cross_result['train_seasons'] = train_seasons
            results.append(cross_result)
        
        return results
    
    def _train_and_evaluate(self, X_train, X_test, y_mu_train, y_mu_test, 
                           y_sigma_train, y_sigma_test, experiment_name):
        """Addestra e valuta modelli per un singolo split"""
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train MU model (Ridge per stabilità)
        mu_model = Ridge(alpha=10.0)
        mu_model.fit(X_train_scaled, y_mu_train)
        
        # Train SIGMA model (RandomForest)
        sigma_model = RandomForestRegressor(
            n_estimators=50, max_depth=6, random_state=42
        )
        sigma_model.fit(X_train_scaled, y_sigma_train)
        
        # Predictions
        mu_pred = mu_model.predict(X_test_scaled)
        sigma_pred = sigma_model.predict(X_test_scaled)
        
        # Metrics
        mu_mae = mean_absolute_error(y_mu_test, mu_pred)
        mu_r2 = r2_score(y_mu_test, mu_pred)
        sigma_mae = mean_absolute_error(y_sigma_test, sigma_pred)
        sigma_r2 = r2_score(y_sigma_test, sigma_pred)
        
        return {
            'experiment': experiment_name,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mu_mae': mu_mae,
            'mu_r2': mu_r2,
            'sigma_mae': sigma_mae,
            'sigma_r2': sigma_r2,
            'mu_predictions': mu_pred.tolist()[:10],  # Prime 10 per debug
            'mu_actuals': y_mu_test.values.tolist()[:10]
        }
    
    def _generate_performance_summary(self, all_results):
        """Genera summary delle performance attraverso tutti i metodi"""
        
        print("\n📋 === BACKTESTING PERFORMANCE SUMMARY ===")
        
        for method_name, method_results in all_results.items():
            if not method_results:
                continue
            
            print(f"\n🔍 {method_name.upper()} RESULTS:")
            
            # Calcola statistiche aggregate
            mu_maes = [r['mu_mae'] for r in method_results]
            sigma_maes = [r['sigma_mae'] for r in method_results]
            mu_r2s = [r['mu_r2'] for r in method_results]
            
            print(f"   🎯 MU Performance:")
            print(f"      Mean MAE: {np.mean(mu_maes):.2f} ± {np.std(mu_maes):.2f}")
            print(f"      Best MAE: {np.min(mu_maes):.2f}")
            print(f"      Worst MAE: {np.max(mu_maes):.2f}")
            print(f"      Mean R²: {np.mean(mu_r2s):.3f}")
            
            print(f"   📏 SIGMA Performance:")
            print(f"      Mean MAE: {np.mean(sigma_maes):.2f} ± {np.std(sigma_maes):.2f}")
            
            # Trova best/worst performer
            best_idx = np.argmin(mu_maes)
            worst_idx = np.argmax(mu_maes)
            
            print(f"   🏆 Best: {method_results[best_idx]['experiment']} (MAE: {mu_maes[best_idx]:.2f})")
            print(f"   📉 Worst: {method_results[worst_idx]['experiment']} (MAE: {mu_maes[worst_idx]:.2f})")
        
        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                method: {
                    'mean_mu_mae': np.mean([r['mu_mae'] for r in results]),
                    'std_mu_mae': np.std([r['mu_mae'] for r in results]),
                    'mean_sigma_mae': np.mean([r['sigma_mae'] for r in results]),
                    'mean_mu_r2': np.mean([r['mu_r2'] for r in results]),
                    'experiments_count': len(results)
                } for method, results in all_results.items() if results
            },
            'detailed_results': all_results
        }
        
        # Save to file
        with open('data/backtesting_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: data/backtesting_results.json")

def run_advanced_backtesting():
    """Esegue il sistema di backtesting avanzato"""
    
    print("🚀 Loading dataset for advanced backtesting...")
    
    # Load data
    df = pd.read_csv('data/nba_fixed_training_dataset.csv')
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['target_mu', 'target_sigma']]
    X = df[feature_cols]
    y_mu = df['target_mu']
    y_sigma = df['target_sigma']
    seasons = sorted(X['SEASON'].unique())
    
    print(f"📊 Dataset: {len(df)} samples across {len(seasons)} seasons")
    print(f"🏀 Seasons: {seasons}")
    
    # Run backtesting
    backtester = AdvancedBacktestingSystem()
    results = backtester.run_comprehensive_backtest(X, y_mu, y_sigma, seasons)
    
    print("\n🎉 Advanced backtesting completed!")
    return results

if __name__ == "__main__":
    run_advanced_backtesting() 