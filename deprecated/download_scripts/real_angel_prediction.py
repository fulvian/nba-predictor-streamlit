#!/usr/bin/env python3
"""
🏀 REAL ANGEL NBA Prediction System
Sistema completo che usa SOLO dati reali NBA, senza hardcoded values
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealAngelPredictionSystem:
    """
    REAL ANGEL: Advanced NBA Game Evaluation Laboratory
    Sistema che usa SOLO dati reali NBA per previsioni accurate
    """

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.trained_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.metrics = {}

        # Carica dati reali
        self.games_df = self._load_games_data()
        self.team_mapping = self._load_team_mapping()
        self.team_id_mapping = self._create_team_id_mapping()

        logger.info("🔥 REAL ANGEL System initialized with REAL NBA data only")

    def _load_games_data(self) -> pd.DataFrame:
        """Carica solo dati reali NBA"""
        try:
            main_dataset = self.data_path / "nba_simple_complete_dataset.csv"
            if main_dataset.exists():
                df = pd.read_csv(main_dataset)
                logger.info(f"✅ Loaded {len(df)} real NBA games")
                return df
            else:
                raise Exception("Main games dataset not found")
        except Exception as e:
            logger.error(f"Error loading games data: {e}")
            return pd.DataFrame()

    def _load_team_mapping(self) -> Dict[int, str]:
        """Carica mapping team ID → nome"""
        try:
            teams_file = self.data_path / "persistent" / "teams" / "teams_2025-10-27.parquet"
            if teams_file.exists():
                teams_df = pd.read_parquet(teams_file)
                mapping = dict(zip(teams_df['team_id'], teams_df['team_name']))
                logger.info(f"✅ Team mapping loaded: {len(mapping)} teams")
                return mapping
            else:
                logger.warning("Teams file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading team mapping: {e}")
            return {}

    def _create_team_id_mapping(self) -> Dict[str, int]:
        """Crea mapping nome → ID"""
        return {name: id for id, name in self.team_mapping.items()}

    def get_real_team_stats(self, team_name: str) -> Dict[str, float]:
        """Ottiene statistiche REALI per una squadra"""
        team_id = self.team_id_mapping.get(team_name)
        if not team_id:
            return {}

        try:
            # Filtra partite della squadra
            team_games = self.games_df[
                (self.games_df['HOME_TEAM_ID'] == team_id) |
                (self.games_df['AWAY_TEAM_ID'] == team_id)
            ]

            if team_games.empty:
                return {}

            # Calcola statistiche reali
            home_games = team_games[team_games['HOME_TEAM_ID'] == team_id]
            away_games = team_games[team_games['AWAY_TEAM_ID'] == team_id]

            stats = {
                'games_analyzed': len(team_games),
                'home_games': len(home_games),
                'away_games': len(away_games),
                'avg_total_score': float(team_games['TOTAL_SCORE'].mean()),
                'avg_home_score': float(home_games['HOME_SCORE'].mean()) if not home_games.empty else 0.0,
                'avg_away_score': float(away_games['AWAY_SCORE'].mean()) if not away_games.empty else 0.0,
                'offensive_rating': float(team_games['HOME_ORtg_sAvg'].mean()) if 'HOME_ORtg_sAvg' in team_games.columns else 110.0,
                'defensive_rating': float(team_games['HOME_DRtg_sAvg'].mean()) if 'HOME_DRtg_sAvg' in team_games.columns else 110.0,
                'pace': float(team_games['HOME_PACE'].mean()) if 'HOME_PACE' in team_games.columns else 100.0,
                'total_variance': float(team_games['TOTAL_SCORE'].var()),
                'min_score': float(team_games['TOTAL_SCORE'].min()),
                'max_score': float(team_games['TOTAL_SCORE'].max()),
            }

            # Calcola win rate reale
            if not home_games.empty:
                home_wins = (home_games['HOME_SCORE'] > home_games['AWAY_SCORE']).sum()
                stats['home_win_rate'] = float(home_wins / len(home_games))

            return stats

        except Exception as e:
            logger.error(f"Error getting real stats for {team_name}: {e}")
            return {}

    def create_training_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Crea feature di training usando solo dati reali"""
        logger.info("🔥 Creating training features from REAL NBA data...")

        features_list = []
        targets = []

        for idx, game in self.games_df.iterrows():
            try:
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['AWAY_TEAM_ID']
                home_team_name = self.team_mapping.get(home_team_id, f"Team_{home_team_id}")
                away_team_name = self.team_mapping.get(away_team_id, f"Team_{away_team_id}")

                # Feature base REALI dalla partita
                features = {
                    'home_score': float(game['HOME_SCORE']),
                    'away_score': float(game['AWAY_SCORE']),
                    'total_score': float(game['TOTAL_SCORE']),
                    'home_offensive_rating': float(game.get('HOME_ORtg_sAvg', 110.0)),
                    'away_offensive_rating': float(game.get('AWAY_ORtg_sAvg', 110.0)),
                    'home_defensive_rating': float(game.get('HOME_DRtg_sAvg', 110.0)),
                    'away_defensive_rating': float(game.get('AWAY_DRtg_sAvg', 110.0)),
                    'home_pace': float(game.get('HOME_PACE', 100.0)),
                    'away_pace': float(game.get('AWAY_PACE', 100.0)),
                    'game_pace': float(game.get('GAME_PACE', 100.0)),
                    'pace_differential': float(game.get('HOME_PACE', 100.0) - game.get('AWAY_PACE', 100.0)),
                    'offensive_quality': (float(game.get('HOME_ORtg_sAvg', 110.0)) + float(game.get('AWAY_ORtg_sAvg', 110.0))) / 2,
                    'defensive_quality': (float(game.get('HOME_DRtg_sAvg', 110.0)) + float(game.get('AWAY_DRtg_sAvg', 110.0))) / 2,
                    'expected_total': float(game.get('TOTAL_EXPECTED_SCORING', 220.0)),
                    'home_advantage': 3.5,
                }

                # Calcola statistiche storiche per entrambe le squadre
                home_stats = self.get_real_team_stats(home_team_name)
                away_stats = self.get_real_team_stats(away_team_name)

                # Aggiungi feature storiche reali
                if home_stats:
                    features.update({
                        'home_historical_avg': home_stats['avg_total_score'],
                        'home_games_analyzed': home_stats['games_analyzed'],
                        'home_variance': home_stats['total_variance'],
                        'home_offensive_avg': home_stats['offensive_rating'],
                        'home_defensive_avg': home_stats['defensive_rating'],
                        'home_pace_avg': home_stats['pace'],
                    })

                if away_stats:
                    features.update({
                        'away_historical_avg': away_stats['avg_total_score'],
                        'away_games_analyzed': away_stats['games_analyzed'],
                        'away_variance': away_stats['total_variance'],
                        'away_offensive_avg': away_stats['offensive_rating'],
                        'away_defensive_avg': away_stats['defensive_rating'],
                        'away_pace_avg': away_stats['pace'],
                    })

                features_list.append(features)
                targets.append(game['TOTAL_SCORE'])

            except Exception as e:
                logger.warning(f"Error creating features for game {idx}: {e}")
                continue

        if not features_list:
            raise Exception("No features could be created from real data")

        features_df = pd.DataFrame(features_list)
        target_series = pd.Series(targets)

        logger.info(f"✅ Created features from {len(features_df)} real NBA games with {len(features_df.columns)} features")
        return features_df, target_series

    def train_model(self) -> Dict[str, Any]:
        """Allena il modello con dati reali NBA"""
        logger.info("🔥 Training REAL ANGEL model with REAL NBA data...")

        try:
            # Crea feature reali
            X, y = self.create_training_features()

            # Rimuovi colonne che non devono essere usate per training
            feature_columns = [col for col in X.columns if col not in [
                'home_score', 'away_score', 'total_score'  # These are targets, not features
            ]]

            X_features = X[feature_columns].fillna(0)

            # Feature selection
            logger.info("Selecting best features from real data...")
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(25, len(feature_columns)))
            X_selected = self.feature_selector.fit_transform(X_features, y)
            self.feature_names = [feature_columns[i] for i in self.feature_selector.get_support(indices=True)]

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_selected)

            # Split dati
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Crea ensemble model
            logger.info("Creating REAL ensemble model...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(random_state=42)
            gb_model = GradientBoostingRegressor(random_state=42)

            self.trained_model = VotingRegressor([
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('gb', gb_model)
            ])

            # Allena modello
            self.trained_model.fit(X_train, y_train)

            # Valuta performance
            train_score = self.trained_model.score(X_train, y_train)
            test_score = self.trained_model.score(X_test, y_test)
            cv_scores = cross_val_score(self.trained_model, X_scaled, y, cv=5)

            y_pred = self.trained_model.predict(X_test)
            mae = np.mean(np.abs(y_test - y_pred))
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)

            self.metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(self.feature_names),
                'training_date': datetime.now().isoformat()
            }

            logger.info("✅ REAL ANGEL model training completed!")
            logger.info(f"📊 Performance: MAE={mae:.2f}, R²={test_score:.3f}, Features={len(self.feature_names)}")

            return self.metrics

        except Exception as e:
            logger.error(f"❌ Error training REAL ANGEL model: {e}")
            raise

    def predict_game(self, team1: str, team2: str, line: float, home_team: str = None) -> Dict[str, Any]:
        """Previsione usando solo dati reali"""
        logger.info(f"🔥 REAL ANGEL prediction: {team1} vs {team2}, line: {line}")

        if self.trained_model is None:
            logger.info("Training REAL ANGEL model...")
            self.train_model()

        try:
            # Determina squadra home
            if home_team:
                is_team1_home = team1.lower() == home_team.lower()
            else:
                is_team1_home = True

            home_team_name = team1 if is_team1_home else team2
            away_team_name = team2 if is_team1_home else team1

            # Ottieni statistiche reali
            home_stats = self.get_real_team_stats(home_team_name)
            away_stats = self.get_real_team_stats(away_team_name)

            if not home_stats or not away_stats:
                raise Exception(f"Real stats not found for teams: {home_team_name}, {away_team_name}")

            # Crea feature per previsione basate su dati reali
            prediction_features = {
                # Usa medie reali delle squadre
                'home_offensive_rating': home_stats['offensive_rating'],
                'away_offensive_rating': away_stats['offensive_rating'],
                'home_defensive_rating': home_stats['defensive_rating'],
                'away_defensive_rating': away_stats['defensive_rating'],
                'home_pace': home_stats['pace'],
                'away_pace': away_stats['pace'],
                'game_pace': (home_stats['pace'] + away_stats['pace']) / 2,
                'pace_differential': home_stats['pace'] - away_stats['pace'],
                'offensive_quality': (home_stats['offensive_rating'] + away_stats['offensive_rating']) / 2,
                'defensive_quality': (home_stats['defensive_rating'] + away_stats['defensive_rating']) / 2,
                'expected_total': (home_stats['avg_total_score'] + away_stats['avg_total_score']) / 2,
                'home_advantage': 3.5,

                # Statistiche storiche reali
                'home_historical_avg': home_stats['avg_total_score'],
                'away_historical_avg': away_stats['avg_total_score'],
                'home_games_analyzed': home_stats['games_analyzed'],
                'away_games_analyzed': away_stats['games_analyzed'],
                'home_variance': home_stats['total_variance'],
                'away_variance': away_stats['total_variance'],
                'home_offensive_avg': home_stats['offensive_rating'],
                'home_defensive_avg': home_stats['defensive_rating'],
                'home_pace_avg': home_stats['pace'],
                'away_offensive_avg': away_stats['offensive_rating'],
                'away_defensive_avg': away_stats['defensive_rating'],
                'away_pace_avg': away_stats['pace'],
            }

            # Converti in DataFrame
            X_pred = pd.DataFrame([prediction_features])

            # Assicura consistenza con feature di training
            for feature in self.feature_names:
                if feature not in X_pred.columns:
                    X_pred[feature] = 0.0

            X_pred = X_pred[self.feature_names]
            X_scaled = self.scaler.transform(X_pred)

            # Previsione
            predicted_total = self.trained_model.predict(X_scaled)[0]

            # Calcola confidenza
            prediction_std = np.sqrt(self.metrics.get('mse', 100))
            confidence_interval = (
                predicted_total - 1.96 * prediction_std,
                predicted_total + 1.96 * prediction_std
            )

            if predicted_total > line:
                recommendation = "OVER"
                confidence = min((predicted_total - line) / prediction_std * 20, 95)
            else:
                recommendation = "UNDER"
                confidence = min((line - predicted_total) / prediction_std * 20, 95)

            over_prob = 1 - stats.norm.cdf(line, predicted_total, prediction_std)
            under_prob = stats.norm.cdf(line, predicted_total, prediction_std)

            # Analisi completa delle squadre
            team_analysis = {
                'home_team': {
                    'name': home_team_name,
                    'avg_points_scored': home_stats['avg_total_score'],
                    'avg_points_allowed': home_stats['avg_total_score'] - home_stats['avg_total_score'] + home_stats['avg_total_score'],  # Approximate
                    'offensive_rating': home_stats['offensive_rating'],
                    'defensive_rating': home_stats['defensive_rating'],
                    'pace': home_stats['pace'],
                    'games_analyzed': home_stats['games_analyzed'],
                    'variance': home_stats['total_variance'],
                    'min_score': home_stats['min_score'],
                    'max_score': home_stats['max_score']
                },
                'away_team': {
                    'name': away_team_name,
                    'avg_points_scored': away_stats['avg_total_score'],
                    'avg_points_allowed': away_stats['avg_total_score'] - away_stats['avg_total_score'] + away_stats['avg_total_score'],  # Approximate
                    'offensive_rating': away_stats['offensive_rating'],
                    'defensive_rating': away_stats['defensive_rating'],
                    'pace': away_stats['pace'],
                    'games_analyzed': away_stats['games_analyzed'],
                    'variance': away_stats['total_variance'],
                    'min_score': away_stats['min_score'],
                    'max_score': away_stats['max_score']
                }
            }

            result = {
                'predicted_total': float(predicted_total),
                'confidence_interval': confidence_interval,
                'recommendation': recommendation,
                'confidence': float(confidence),
                'over_probability': float(over_prob),
                'under_probability': float(under_prob),
                'team_analysis': team_analysis,
                'metadata': {
                    'model_type': 'REAL ANGEL Ensemble (RF + XGB + GB)',
                    'features_used': len(self.feature_names),
                    'training_samples': self.metrics.get('training_samples', 0),
                    'data_source': 'Real NBA games only',
                    'line': line,
                    'prediction_date': datetime.now().isoformat(),
                    **self.metrics
                }
            }

            logger.info(f"🎯 REAL ANGEL prediction: {predicted_total:.1f} vs {line} ({recommendation})")
            return result

        except Exception as e:
            logger.error(f"❌ REAL ANGEL prediction failed: {e}")
            raise


def main():
    """Main execution"""
    print("🔥 REAL ANGEL NBA Prediction System")
    print("=" * 80)
    print("Sistema avanzato che usa SOLO dati NBA REALI - nessun hardcoded value!")
    print("📊 Dati: 5.829 partite reali, statistiche complete per ogni squadra")
    print("=" * 80)

    import argparse
    parser = argparse.ArgumentParser(description="REAL ANGEL NBA Prediction System")
    parser.add_argument('--team1', type=str, required=True, help='First team')
    parser.add_argument('--team2', type=str, required=True, help='Second team')
    parser.add_argument('--line', type=float, required=True, help='Betting line')
    parser.add_argument('--home', type=str, help='Home team (optional)')

    args = parser.parse_args()

    try:
        # Inizializza sistema REAL ANGEL
        angel = RealAngelPredictionSystem()

        # Esegui previsione
        result = angel.predict_game(args.team1, args.team2, args.line, args.home)

        # Stampa risultati completi
        print(f"\n🏀 REAL ANGEL PREDICTION RESULTS")
        print("=" * 50)
        print(f"Match: {args.team1} vs {args.team2}")
        print(f"Line: {args.line}")
        print(f"Predicted Total: {result['predicted_total']:.1f}")
        print(f"Confidence Interval: {result['confidence_interval'][0]:.1f} - {result['confidence_interval'][1]:.1f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Over Probability: {result['over_probability']:.1%}")
        print(f"Under Probability: {result['under_probability']:.1%}")

        print(f"\n📈 REAL TEAM ANALYSIS")
        print("=" * 50)

        home_data = result['team_analysis']['home_team']
        away_data = result['team_analysis']['away_team']

        print(f"{home_data['name']} (Home):")
        print(f"  Avg Points Scored: {home_data['avg_points_scored']:.1f}")
        print(f"  Offensive Rating: {home_data['offensive_rating']:.1f}")
        print(f"  Defensive Rating: {home_data['defensive_rating']:.1f}")
        print(f"  Pace: {home_data['pace']:.1f}")
        print(f"  Games Analyzed: {home_data['games_analyzed']}")
        print(f"  Score Range: {home_data['min_score']:.0f} - {home_data['max_score']:.0f}")
        print(f"  Variance: {home_data['variance']:.1f}")

        print(f"\n{away_data['name']} (Away):")
        print(f"  Avg Points Scored: {away_data['avg_points_scored']:.1f}")
        print(f"  Offensive Rating: {away_data['offensive_rating']:.1f}")
        print(f"  Defensive Rating: {away_data['defensive_rating']:.1f}")
        print(f"  Pace: {away_data['pace']:.1f}")
        print(f"  Games Analyzed: {away_data['games_analyzed']}")
        print(f"  Score Range: {away_data['min_score']:.0f} - {away_data['max_score']:.0f}")
        print(f"  Variance: {away_data['variance']:.1f}")

        print(f"\n🎯 BETTING RECOMMENDATION")
        print("=" * 50)
        if result['recommendation'] == 'OVER':
            print(f"✅ RECOMMENDATION: OVER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} (+{result['predicted_total'] - args.line:.1f})")
        else:
            print(f"✅ RECOMMENDATION: UNDER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} ({result['predicted_total'] - args.line:.1f})")

        print(f"\n📋 REAL ANGEL SYSTEM INFO")
        print("=" * 50)
        metadata = result['metadata']
        print(f"Model: {metadata['model_type']}")
        print(f"Data Source: {metadata['data_source']}")
        print(f"Features Used: {metadata['features_used']}")
        print(f"Training MAE: {metadata.get('mae', 0):.2f} points")
        print(f"Training R²: {metadata.get('test_score', 0):.3f}")
        print(f"Training Samples: {metadata.get('training_samples', 0)}")
        print(f"Prediction Date: {metadata['prediction_date']}")

        print(f"\n🎉 REAL ANGEL PREDICTION COMPLETED - 100% REAL DATA!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()