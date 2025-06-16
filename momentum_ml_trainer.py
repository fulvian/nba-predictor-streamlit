# momentum_ml_trainer.py
"""
Sistema opzionale per addestrare modelli ML per l'integrazione momentum.
Implementa metodologie validate dalla ricerca scientifica.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class MomentumMLTrainer:
    """
    Trainer per modelli ML che integrano momentum nei totali NBA.
    Basato su ricerca scientifica per XGBoost + SHAP interpretability.
    """
    
    def __init__(self, data_provider, models_dir='models'):
        self.data_provider = data_provider
        self.models_dir = os.path.join(models_dir, 'momentum_ml')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configurazioni validata dalla ricerca
        self.xgb_params = {
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        self.rf_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }

    def collect_training_data(self, seasons=['2021-22', '2022-23', '2023-24'], min_games_per_season=50):
        """
        Raccoglie dati di training dalle stagioni specificate.
        Include momentum features scientificamente validate.
        """
        print(f"🔍 Raccogliendo dati di training per stagioni: {seasons}")
        
        all_training_data = []
        
        for season in seasons:
            print(f"   Processando stagione {season}...")
            
            # Recupera partite della stagione
            season_games = self._get_season_games(season)
            if len(season_games) < min_games_per_season:
                print(f"   ⚠️ Troppo poche partite per {season}: {len(season_games)}")
                continue
            
            # Per ogni partita, calcola features pre-game
            for game_idx, game in enumerate(season_games):
                if game_idx % 50 == 0:
                    print(f"   Processando partita {game_idx}/{len(season_games)}")
                
                features = self._extract_game_features(game, season_games[:game_idx])
                if features:
                    all_training_data.append(features)
        
        print(f"✅ Raccolte {len(all_training_data)} samples di training")
        return pd.DataFrame(all_training_data)

    def _extract_game_features(self, game, historical_games):
        """
        Estrae features scientificamente validate per una singola partita.
        """
        try:
            # Info base partita
            home_team = game['home_team']
            away_team = game['away_team']
            actual_total = game['home_score'] + game['away_score']
            
            # Team stats base (ultime 10 partite)
            home_recent_stats = self._calculate_team_recent_stats(home_team, historical_games, 10)
            away_recent_stats = self._calculate_team_recent_stats(away_team, historical_games, 10)
            
            # Momentum features avanzate
            home_momentum_features = self._extract_momentum_features_for_team(home_team, historical_games)
            away_momentum_features = self._extract_momentum_features_for_team(away_team, historical_games)
            
            # Combina tutte le features
            features = {
                # Target
                'actual_total': actual_total,
                
                # Team stats base
                'home_ortg_l10': home_recent_stats.get('ortg', 110),
                'home_drtg_l10': home_recent_stats.get('drtg', 110),
                'home_pace_l10': home_recent_stats.get('pace', 100),
                'away_ortg_l10': away_recent_stats.get('ortg', 110),
                'away_drtg_l10': away_recent_stats.get('drtg', 110),
                'away_pace_l10': away_recent_stats.get('pace', 100),
                
                # Momentum features (validate dalla ricerca)
                'home_ts_pct_trend': home_momentum_features.get('ts_trend', 0),
                'away_ts_pct_trend': away_momentum_features.get('ts_trend', 0),
                'home_usage_trend': home_momentum_features.get('usage_trend', 0),
                'away_usage_trend': away_momentum_features.get('usage_trend', 0),
                'home_consistency': home_momentum_features.get('consistency', 0.5),
                'away_consistency': away_momentum_features.get('consistency', 0.5),
                'home_hot_hand_strength': home_momentum_features.get('hot_hand_strength', 0),
                'away_hot_hand_strength': away_momentum_features.get('hot_hand_strength', 0),
                
                # Interaction features
                'combined_pace': (home_recent_stats.get('pace', 100) + away_recent_stats.get('pace', 100)) / 2,
                'offensive_advantage': home_recent_stats.get('ortg', 110) - away_recent_stats.get('drtg', 110),
                'defensive_advantage': away_recent_stats.get('ortg', 110) - home_recent_stats.get('drtg', 110),
                'momentum_differential': home_momentum_features.get('total_momentum', 50) - away_momentum_features.get('total_momentum', 50),
                
                # Meta features
                'both_teams_hot': 1 if (home_momentum_features.get('hot_hand_strength', 0) > 0.6 and 
                                      away_momentum_features.get('hot_hand_strength', 0) > 0.6) else 0,
                'high_variance_game': 1 if (home_momentum_features.get('consistency', 0.5) < 0.4 or
                                          away_momentum_features.get('consistency', 0.5) < 0.4) else 0,
                
                # Context
                'is_home_favored': 1 if home_recent_stats.get('ortg', 110) > away_recent_stats.get('ortg', 110) else 0,
                'season': game.get('season', '2023-24'),
                'game_date': game.get('date', '2024-01-01')
            }
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Errore estrazione features: {e}")
            return None

    def train_momentum_integration_model(self, training_data, test_size=0.2):
        """
        Addestra modello per integrazione momentum usando metodologie validate.
        """
        print("🚀 Addestrando modello integrazione momentum...")
        
        # Prepara dati
        feature_cols = [col for col in training_data.columns if col not in ['actual_total', 'season', 'game_date']]
        X = training_data[feature_cols]
        y = training_data['actual_total']
        
        # Split temporale (critico per time series)
        split_idx = int(len(training_data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   Training size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Addestra modelli
        models = {}
        
        # XGBoost (preferito dalla ricerca)
        if XGBOOST_AVAILABLE:
            print("   Addestrando XGBoost...")
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.fit(X_train, y_train)
            
            # Valutazione
            train_pred = xgb_model.predict(X_train)
            test_pred = xgb_model.predict(X_test)
            
            models['xgboost'] = {
                'model': xgb_model,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred)
            }
            
            print(f"   XGBoost - Train RMSE: {models['xgboost']['train_rmse']:.2f}, Test RMSE: {models['xgboost']['test_rmse']:.2f}")
        
        # Random Forest (backup robusto)
        print("   Addestrando Random Forest...")
        rf_model = RandomForestRegressor(**self.rf_params)
        rf_model.fit(X_train, y_train)
        
        train_pred_rf = rf_model.predict(X_train)
        test_pred_rf = rf_model.predict(X_test)
        
        models['random_forest'] = {
            'model': rf_model,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_rf)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred_rf)),
            'train_mae': mean_absolute_error(y_train, train_pred_rf),
            'test_mae': mean_absolute_error(y_test, test_pred_rf)
        }
        
        print(f"   Random Forest - Train RMSE: {models['random_forest']['train_rmse']:.2f}, Test RMSE: {models['random_forest']['test_rmse']:.2f}")
        
        # Selezione modello migliore
        best_model_name = min(models.keys(), key=lambda k: models[k]['test_rmse'])
        best_model = models[best_model_name]['model']
        
        print(f"✅ Miglior modello: {best_model_name}")
        
        # SHAP analysis se disponibile
        if SHAP_AVAILABLE:
            self._analyze_feature_importance(best_model, X_test, feature_cols)
        
        # Salva modello
        model_path = os.path.join(self.models_dir, 'momentum_integration_model.pkl')
        joblib.dump(best_model, model_path)
        
        # Salva metadati
        metadata = {
            'model_type': best_model_name,
            'feature_columns': feature_cols,
            'performance': models[best_model_name],
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.models_dir, 'momentum_model_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modello salvato in: {model_path}")
        return best_model, models[best_model_name]

    def _analyze_feature_importance(self, model, X_test, feature_cols):
        """Analisi SHAP per interpretabilità."""
        try:
            print("   🔍 Analisi SHAP feature importance...")
            
            # SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test.head(100))  # Sample per performance
            
            # Feature importance
            feature_importance = np.abs(shap_values.values).mean(0)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("   Top 10 feature più importanti:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            print(f"   ⚠️ Errore analisi SHAP: {e}")

    def _get_season_games(self, season):
        """Simula recupero partite stagione."""
        # Questo dovrebbe essere implementato usando il tuo data_provider
        # Per ora restituiamo una struttura di esempio
        return [
            {
                'date': '2024-01-01',
                'season': season,
                'home_team': 'Lakers',
                'away_team': 'Warriors',
                'home_score': 120,
                'away_score': 115
            }
            # ... altre partite
        ]

    def _calculate_team_recent_stats(self, team, historical_games, n_games):
        """Calcola stats recenti team."""
        # Implementazione semplificata
        return {
            'ortg': 112.0,
            'drtg': 108.0,
            'pace': 100.0
        }

    def _extract_momentum_features_for_team(self, team, historical_games):
        """Estrae momentum features per team."""
        # Implementazione semplificata
        return {
            'ts_trend': 0.02,
            'usage_trend': 0.01,
            'consistency': 0.7,
            'hot_hand_strength': 0.5,
            'total_momentum': 55.0
        }

# Script di esempio per l'utilizzo
def main():
    """Esempio di utilizzo del trainer."""
    from data_provider import NBADataProvider  # Il tuo data provider
    
    data_provider = NBADataProvider()
    trainer = MomentumMLTrainer(data_provider)
    
    print("Raccogliendo dati di training...")
    training_data = trainer.collect_training_data(['2021-22', '2022-23'])
    
    if len(training_data) > 100:  # Minimo dataset size
        print("Addestrando modello...")
        model, performance = trainer.train_momentum_integration_model(training_data)
        print(f"Modello addestrato con RMSE: {performance['test_rmse']:.2f}")
    else:
        print("Dataset insufficiente per training")

if __name__ == "__main__":
    main()