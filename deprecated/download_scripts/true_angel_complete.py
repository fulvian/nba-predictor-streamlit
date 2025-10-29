#!/usr/bin/env python3
"""
🏀 TRUE ANGEL NBA Prediction System - COMPLETE DATA INTEGRATION
Sistema COMPLETO che integra TUTTI i dati disponibili:
- 5.829 partite reali
- 1.926 statistiche giocatori
- 450 momentum giocatori
- 72 injury reports
- 1.056 roster records
- 4.007 partite complete per H2H
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

class TrueAngelCompleteSystem:
    """
    TRUE ANGEL: Complete NBA Game Evaluation Laboratory
    Sistema COMPLETO che integra TUTTI i dati disponibili
    """

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.trained_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.metrics = {}

        # Carica TUTTI i dati disponibili
        logger.info("🔥 Loading ALL data sources for TRUE ANGEL system...")
        self.games_df = self._load_games_data()
        self.team_mapping = self._load_team_mapping()
        self.team_id_mapping = self._create_team_id_mapping()
        self.player_stats = self._load_player_stats()
        self.player_momentum = self._load_player_momentum()
        self.injuries = self._load_injuries()
        self.rosters = self._load_rosters()
        self.complete_games = self._load_complete_games()

        logger.info("✅ TRUE ANGEL System initialized with ALL data sources")

    def _load_games_data(self) -> pd.DataFrame:
        """Carica dataset principale partite"""
        try:
            main_dataset = self.data_path / "nba_simple_complete_dataset.csv"
            if main_dataset.exists():
                df = pd.read_csv(main_dataset)
                logger.info(f"✅ Base games: {len(df)} games")
                return df
        except Exception as e:
            logger.error(f"Error loading games: {e}")
        return pd.DataFrame()

    def _load_team_mapping(self) -> Dict[int, str]:
        """Carica mapping team ID → nome"""
        try:
            teams_file = self.data_path / "persistent" / "teams" / "teams_2025-10-27.parquet"
            if teams_file.exists():
                teams_df = pd.read_parquet(teams_file)
                mapping = dict(zip(teams_df['team_id'], teams_df['team_name']))
                logger.info(f"✅ Teams: {len(mapping)} teams")
                return mapping
        except Exception as e:
            logger.error(f"Error loading teams: {e}")
        return {}

    def _create_team_id_mapping(self) -> Dict[str, int]:
        """Crea mapping nome → ID"""
        return {name: id for id, name in self.team_mapping.items()}

    def _load_player_stats(self) -> pd.DataFrame:
        """Carica statistiche giocatori"""
        try:
            stats_file = self.data_path / "persistent" / "players" / "player_stats_2025-10-27.parquet"
            if stats_file.exists():
                df = pd.read_parquet(stats_file)
                logger.info(f"✅ Player stats: {len(df)} records")
                return df
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
        return pd.DataFrame()

    def _load_player_momentum(self) -> pd.DataFrame:
        """Carica momentum giocatori"""
        try:
            momentum_file = self.data_path / "persistent" / "players" / "player_momentum_2025-10-27.parquet"
            if momentum_file.exists():
                df = pd.read_parquet(momentum_file)
                logger.info(f"✅ Player momentum: {len(df)} records")
                return df
        except Exception as e:
            logger.error(f"Error loading player momentum: {e}")
        return pd.DataFrame()

    def _load_injuries(self) -> pd.DataFrame:
        """Carica injury reports"""
        try:
            injury_file = self.data_path / "persistent" / "injuries" / "injury_reports_2025-10-27.parquet"
            if injury_file.exists():
                df = pd.read_parquet(injury_file)
                logger.info(f"✅ Injuries: {len(df)} records")
                return df
        except Exception as e:
            logger.error(f"Error loading injuries: {e}")
        return pd.DataFrame()

    def _load_rosters(self) -> pd.DataFrame:
        """Carica roster information"""
        try:
            roster_file = self.data_path / "persistent" / "rosters" / "rosters_2025-10-27.parquet"
            if roster_file.exists():
                df = pd.read_parquet(roster_file)
                logger.info(f"✅ Rosters: {len(df)} records")
                return df
        except Exception as e:
            logger.error(f"Error loading rosters: {e}")
        return pd.DataFrame()

    def _load_complete_games(self) -> pd.DataFrame:
        """Carica partite complete per H2H"""
        try:
            complete_file = self.data_path / "persistent" / "games" / "game_results_2024-25_Regular_Season.parquet"
            if complete_file.exists():
                df = pd.read_parquet(complete_file)
                logger.info(f"✅ Complete games: {len(df)} games")
                return df
        except Exception as e:
            logger.error(f"Error loading complete games: {e}")
        return pd.DataFrame()

    def get_complete_team_features(self, team_name: str) -> Dict[str, float]:
        """Ottieni feature COMPLETE per una squadra usando TUTTI i dati"""
        team_id = self.team_id_mapping.get(team_name)
        if not team_id:
            return {}

        features = {}

        try:
            # 1. Statistiche base dal dataset principale
            team_games = self.games_df[
                (self.games_df['HOME_TEAM_ID'] == team_id) |
                (self.games_df['AWAY_TEAM_ID'] == team_id)
            ]

            if not team_games.empty:
                home_games = team_games[team_games['HOME_TEAM_ID'] == team_id]
                features.update({
                    'base_avg_total': float(team_games['TOTAL_SCORE'].mean()),
                    'base_home_avg': float(home_games['HOME_SCORE'].mean()) if not home_games.empty else 0.0,
                    'base_offensive_rating': float(team_games['HOME_ORtg_sAvg'].mean()) if 'HOME_ORtg_sAvg' in team_games.columns else 110.0,
                    'base_defensive_rating': float(team_games['HOME_DRtg_sAvg'].mean()) if 'HOME_DRtg_sAvg' in team_games.columns else 110.0,
                    'base_pace': float(team_games['HOME_PACE'].mean()) if 'HOME_PACE' in team_games.columns else 100.0,
                    'base_variance': float(team_games['TOTAL_SCORE'].var()),
                    'base_games_count': len(team_games)
                })

            # 2. Statistiche giocatori attuali
            if not self.player_stats.empty and 'team_name' in self.player_stats.columns:
                team_players = self.player_stats[self.player_stats['team_name'] == team_name]
                if not team_players.empty:
                    features.update({
                        'player_avg_points': float(team_players['points'].mean()) if 'points' in team_players.columns else 0.0,
                        'player_avg_assists': float(team_players['assists'].mean()) if 'assists' in team_players.columns else 0.0,
                        'player_avg_rebounds': float(team_players['rebounds'].mean()) if 'rebounds' in team_players.columns else 0.0,
                        'player_avg_minutes': float(team_players['minutes'].mean()) if 'minutes' in team_players.columns else 0.0,
                        'player_count': len(team_players),
                        'player_total_production': float(team_players['points'].fillna(0).sum() + team_players['assists'].fillna(0).sum() + team_players['rebounds'].fillna(0).sum()) if all(col in team_players.columns for col in ['points', 'assists', 'rebounds']) else 0.0
                    })

                    # Top 5 players by points
                    if 'points' in team_players.columns:
                        top_players = team_players.nlargest(5, 'points')
                        features.update({
                            'top5_avg_points': float(top_players['points'].mean()),
                            'top5_total_points': float(top_players['points'].sum()),
                            'star_power': float(top_players['points'].sum())
                        })

            # 3. Momentum giocatori
            if not self.player_momentum.empty and 'team_name' in self.player_momentum.columns:
                team_momentum = self.player_momentum[self.player_momentum['team_name'] == team_name]
                if not team_momentum.empty:
                    features.update({
                        'momentum_avg_points': float(team_momentum['points'].mean()) if 'points' in team_momentum.columns else 0.0,
                        'momentum_avg_assists': float(team_momentum['assists'].mean()) if 'assists' in team_momentum.columns else 0.0,
                        'momentum_avg_rebounds': float(team_momentum['rebounds'].mean()) if 'rebounds' in team_momentum.columns else 0.0,
                        'momentum_players_count': len(team_momentum),
                        'momentum_total_production': float(team_momentum['points'].fillna(0).sum() + team_momentum['assists'].fillna(0).sum() + team_momentum['rebounds'].fillna(0).sum()) if all(col in team_momentum.columns for col in ['points', 'assists', 'rebounds']) else 0.0
                    })

                    # Recent form (last 5 games equivalent)
                    if 'points' in team_momentum.columns:
                        momentum_production = team_momentum['points'].fillna(0) + team_momentum['assists'].fillna(0) + team_momentum['rebounds'].fillna(0)
                        features.update({
                            'recent_form_avg': float(momentum_production.mean()),
                            'recent_form_consistency': float(1 / (momentum_production.std() + 1))  # Lower std = higher consistency
                        })

            # 4. Injury impact
            if not self.injuries.empty:
                if 'team' in self.injuries.columns:
                    team_injuries = self.injuries[self.injuries['team'] == team_name]
                elif 'team_name' in self.injuries.columns:
                    team_injuries = self.injuries[self.injuries['team_name'] == team_name]
                else:
                    team_injuries = pd.DataFrame()

                if not team_injuries.empty:
                    total_injured = len(team_injuries)
                    key_injured = 0

                    if 'status' in team_injuries.columns:
                        key_injured = len(team_injuries[team_injuries['status'].isin(['Out', 'Doubtful', 'Questionable'])])

                    features.update({
                        'injury_total_count': total_injured,
                        'injury_key_players': key_injured,
                        'injury_impact_score': float(key_injured * 10 + total_injured * 3),  # Weight key injuries more
                        'injury_severity': key_injured / max(total_injured, 1)
                    })

            # 5. Roster stability
            if not self.rosters.empty and 'team_name' in self.rosters.columns:
                team_roster = self.rosters[self.rosters['team_name'] == team_name]
                if not team_roster.empty:
                    features.update({
                        'roster_size': len(team_roster),
                        'roster_stability': min(len(team_roster) / 15.0, 1.0),  # 15 is ideal roster size
                        'roster_experience': float(team_roster['experience'].mean()) if 'experience' in team_roster.columns else 5.0
                    })

            # 6. Head-to-Head recent performance
            if not self.complete_games.empty:
                # Find all games involving this team
                team_h2h = self.complete_games[
                    (self.complete_games['home_team'] == team_name) |
                    (self.complete_games['away_team'] == team_name)
                ].copy()

                if not team_h2h.empty:
                    # Calculate scoring trends
                    team_h2h['team_score'] = np.where(
                        team_h2h['home_team'] == team_name,
                        team_h2h['home_score'],
                        team_h2h['away_score']
                    )

                    team_h2h['opponent_score'] = np.where(
                        team_h2h['home_team'] == team_name,
                        team_h2h['away_score'],
                        team_h2h['home_score']
                    )

                    team_h2h['total_score'] = team_h2h['team_score'] + team_h2h['opponent_score']

                    # Recent performance (last 10 games)
                    recent_h2h = team_h2h.nlargest(10, 'game_date') if 'game_date' in team_h2h.columns else team_h2h.tail(10)

                    features.update({
                        'h2h_games_total': len(team_h2h),
                        'h2h_avg_team_score': float(team_h2h['team_score'].mean()),
                        'h2h_avg_total_score': float(team_h2h['total_score'].mean()),
                        'h2h_recent_avg': float(recent_h2h['total_score'].mean()),
                        'h2h_scoring_trend': float(recent_h2h['total_score'].iloc[-5:].mean() - recent_h2h['total_score'].iloc[:-5].mean()) if len(recent_h2h) >= 10 else 0.0,
                        'h2h_consistency': float(1 / (team_h2h['total_score'].std() + 1)),
                        'h2h_win_rate': float((team_h2h['team_score'] > team_h2h['opponent_score']).mean())
                    })

        except Exception as e:
            logger.warning(f"Error getting complete features for {team_name}: {e}")

        return features

    def create_comprehensive_training_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Crea feature di training COMPLETE usando TUTTI i dati"""
        logger.info("🔥 Creating COMPREHENSIVE features from ALL data sources...")

        features_list = []
        targets = []

        for idx, game in self.games_df.iterrows():
            try:
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['AWAY_TEAM_ID']
                home_team_name = self.team_mapping.get(home_team_id, f"Team_{home_team_id}")
                away_team_name = self.team_mapping.get(away_team_id, f"Team_{away_team_id}")

                # Ottieni feature COMPLETE per entrambe le squadre
                home_features = self.get_complete_team_features(home_team_name)
                away_features = self.get_complete_team_features(away_team_name)

                # Combina tutte le feature
                combined_features = {
                    # Game-specific features
                    'home_score': float(game['HOME_SCORE']),
                    'away_score': float(game['AWAY_SCORE']),
                    'total_score': float(game['TOTAL_SCORE']),
                    'game_pace': float(game.get('GAME_PACE', 100.0)),
                    'season': float(game.get('SEASON', 2024)),

                    # Home advantage
                    'home_advantage': 3.5,
                }

                # Add home team features (prefix with 'home_')
                for feature, value in home_features.items():
                    combined_features[f'home_{feature}'] = value

                # Add away team features (prefix with 'away_')
                for feature, value in away_features.items():
                    combined_features[f'away_{feature}'] = value

                # Calculate differential features
                if 'base_avg_total' in home_features and 'base_avg_total' in away_features:
                    combined_features['diff_avg_total'] = home_features['base_avg_total'] - away_features['base_avg_total']

                if 'player_avg_points' in home_features and 'player_avg_points' in away_features:
                    combined_features['diff_player_points'] = home_features['player_avg_points'] - away_features['player_avg_points']

                if 'injury_impact_score' in home_features and 'injury_impact_score' in away_features:
                    combined_features['diff_injury_impact'] = away_features['injury_impact_score'] - home_features['injury_impact_score']

                if 'momentum_avg_points' in home_features and 'momentum_avg_points' in away_features:
                    combined_features['diff_momentum'] = home_features['momentum_avg_points'] - away_features['momentum_avg_points']

                features_list.append(combined_features)
                targets.append(game['TOTAL_SCORE'])

            except Exception as e:
                logger.warning(f"Error creating comprehensive features for game {idx}: {e}")
                continue

        if not features_list:
            raise Exception("No comprehensive features could be created")

        features_df = pd.DataFrame(features_list)
        target_series = pd.Series(targets)

        # Fill NaN values with 0 for ML
        features_df = features_df.fillna(0)

        logger.info(f"✅ Created COMPREHENSIVE features: {len(features_df)} samples with {len(features_df.columns)} features")
        return features_df, target_series

    def train_complete_model(self) -> Dict[str, Any]:
        """Allena il modello COMPLETO con tutti i dati"""
        logger.info("🔥 Training TRUE ANGEL model with ALL data sources...")

        try:
            # Crea feature complete
            X, y = self.create_comprehensive_training_features()

            # Rimuovi target columns e non-features
            exclude_columns = ['home_score', 'away_score', 'total_score', 'season']
            feature_columns = [col for col in X.columns if col not in exclude_columns]

            X_features = X[feature_columns]

            logger.info(f"Using {len(feature_columns)} features for training")

            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(40, len(feature_columns)))
            X_selected = self.feature_selector.fit_transform(X_features, y)
            self.feature_names = [feature_columns[i] for i in self.feature_selector.get_support(indices=True)]

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_selected)

            # Split dati
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Crea ensemble model potenziato
            logger.info("Creating COMPLETE ensemble model...")
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
            xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=200, max_depth=8, learning_rate=0.05)
            gb_model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05)

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
                'data_sources_count': 6,  # games, players, momentum, injuries, rosters, H2H
                'training_date': datetime.now().isoformat()
            }

            logger.info("✅ TRUE ANGEL COMPLETE model training finished!")
            logger.info(f"📊 Performance: MAE={mae:.2f}, R²={test_score:.3f}, Features={len(self.feature_names)}")

            return self.metrics

        except Exception as e:
            logger.error(f"❌ Error training complete model: {e}")
            raise

    def predict_complete_game(self, team1: str, team2: str, line: float, home_team: str = None) -> Dict[str, Any]:
        """Previsione COMPLETA usando tutti i dati"""
        logger.info(f"🔥 TRUE ANGEL COMPLETE prediction: {team1} vs {team2}, line: {line}")

        if self.trained_model is None:
            logger.info("Training TRUE ANGEL COMPLETE model...")
            self.train_complete_model()

        try:
            # Determina squadra home
            if home_team:
                is_team1_home = team1.lower() == home_team.lower()
            else:
                is_team1_home = True

            home_team_name = team1 if is_team1_home else team2
            away_team_name = team2 if is_team1_home else team1

            # Ottieni feature COMPLETE per entrambe le squadre
            home_features = self.get_complete_team_features(home_team_name)
            away_features = self.get_complete_team_features(away_team_name)

            if not home_features or not away_features:
                raise Exception(f"Complete features not found for teams: {home_team_name}, {away_team_name}")

            # Costruisci feature per previsione
            prediction_features = {
                'home_advantage': 3.5,
                'game_pace': (home_features.get('base_pace', 100) + away_features.get('base_pace', 100)) / 2,
                'season': 2025.0,
            }

            # Add home team features
            for feature, value in home_features.items():
                prediction_features[f'home_{feature}'] = value

            # Add away team features
            for feature, value in away_features.items():
                prediction_features[f'away_{feature}'] = value

            # Calculate differentials
            if 'base_avg_total' in home_features and 'base_avg_total' in away_features:
                prediction_features['diff_avg_total'] = home_features['base_avg_total'] - away_features['base_avg_total']

            if 'player_avg_points' in home_features and 'player_avg_points' in away_features:
                prediction_features['diff_player_points'] = home_features['player_avg_points'] - away_features['player_avg_points']

            if 'injury_impact_score' in home_features and 'injury_impact_score' in away_features:
                prediction_features['diff_injury_impact'] = away_features['injury_impact_score'] - home_features['injury_impact_score']

            if 'momentum_avg_points' in home_features and 'momentum_avg_points' in away_features:
                prediction_features['diff_momentum'] = home_features['momentum_avg_points'] - away_features['momentum_avg_points']

            # Converti in DataFrame e assicura consistenza
            X_pred = pd.DataFrame([prediction_features])
            X_pred = X_pred.fillna(0)

            # Aggiungi feature mancanti con default 0
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
                    **home_features
                },
                'away_team': {
                    'name': away_team_name,
                    **away_features
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
                    'model_type': 'TRUE ANGEL COMPLETE (All Data Sources)',
                    'features_used': len(self.feature_names),
                    'data_sources': ['games', 'player_stats', 'player_momentum', 'injuries', 'rosters', 'complete_games'],
                    'training_samples': self.metrics.get('training_samples', 0),
                    'data_sources_count': self.metrics.get('data_sources_count', 0),
                    'line': line,
                    'prediction_date': datetime.now().isoformat(),
                    **self.metrics
                }
            }

            logger.info(f"🎯 TRUE ANGEL COMPLETE prediction: {predicted_total:.1f} vs {line} ({recommendation})")
            return result

        except Exception as e:
            logger.error(f"❌ TRUE ANGEL COMPLETE prediction failed: {e}")
            raise


def main():
    """Main execution"""
    print("🔥🔥🔥 TRUE ANGEL NBA Prediction System - COMPLETE DATA INTEGRATION 🔥🔥🔥")
    print("=" * 90)
    print("🏀 Sistema COMPLETO che integra TUTTI i dati disponibili:")
    print("   📊 5.829 partite reali")
    print("   👥 1.926 statistiche giocatori")
    print("   🎯 450 momentum giocatori")
    print("   🏥 72 injury reports")
    print("   📋 1.056 roster records")
    print("   ⚔️ 4.007 partite complete per H2H")
    print("=" * 90)

    import argparse
    parser = argparse.ArgumentParser(description="TRUE ANGEL COMPLETE NBA Prediction System")
    parser.add_argument('--team1', type=str, required=True, help='First team')
    parser.add_argument('--team2', type=str, required=True, help='Second team')
    parser.add_argument('--line', type=float, required=True, help='Betting line')
    parser.add_argument('--home', type=str, help='Home team (optional)')

    args = parser.parse_args()

    try:
        # Inizializza sistema TRUE ANGEL COMPLETO
        angel = TrueAngelCompleteSystem()

        # Esegui previsione completa
        result = angel.predict_complete_game(args.team1, args.team2, args.line, args.home)

        # Stampa risultati completi
        print(f"\n🏀 TRUE ANGEL COMPLETE PREDICTION RESULTS")
        print("=" * 60)
        print(f"Match: {args.team1} vs {args.team2}")
        print(f"Line: {args.line}")
        print(f"Predicted Total: {result['predicted_total']:.1f}")
        print(f"Confidence Interval: {result['confidence_interval'][0]:.1f} - {result['confidence_interval'][1]:.1f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Over Probability: {result['over_probability']:.1%}")
        print(f"Under Probability: {result['under_probability']:.1%}")

        print(f"\n📈 COMPLETE TEAM ANALYSIS")
        print("=" * 60)

        home_data = result['team_analysis']['home_team']
        away_data = result['team_analysis']['away_team']

        print(f"{home_data['name']} (Home):")
        print(f"  🏀 Base Avg Total: {home_data.get('base_avg_total', 0):.1f}")
        print(f"  👥 Player Avg Points: {home_data.get('player_avg_points', 0):.1f}")
        print(f"  🎯 Momentum Avg Points: {home_data.get('momentum_avg_points', 0):.1f}")
        print(f"  🏥 Injury Impact Score: {home_data.get('injury_impact_score', 0):.1f}")
        print(f"  📋 Roster Size: {home_data.get('roster_size', 0)}")
        print(f"  ⚔️ H2H Avg Total: {home_data.get('h2h_avg_total_score', 0):.1f}")
        print(f"  📊 Games Analyzed: {home_data.get('base_games_count', 0)}")

        print(f"\n{away_data['name']} (Away):")
        print(f"  🏀 Base Avg Total: {away_data.get('base_avg_total', 0):.1f}")
        print(f"  👥 Player Avg Points: {away_data.get('player_avg_points', 0):.1f}")
        print(f"  🎯 Momentum Avg Points: {away_data.get('momentum_avg_points', 0):.1f}")
        print(f"  🏥 Injury Impact Score: {away_data.get('injury_impact_score', 0):.1f}")
        print(f"  📋 Roster Size: {away_data.get('roster_size', 0)}")
        print(f"  ⚔️ H2H Avg Total: {away_data.get('h2h_avg_total_score', 0):.1f}")
        print(f"  📊 Games Analyzed: {away_data.get('base_games_count', 0)}")

        print(f"\n🎯 BETTING RECOMMENDATION")
        print("=" * 60)
        if result['recommendation'] == 'OVER':
            print(f"✅ RECOMMENDATION: OVER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} (+{result['predicted_total'] - args.line:.1f})")
        else:
            print(f"✅ RECOMMENDATION: UNDER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} ({result['predicted_total'] - args.line:.1f})")

        print(f"\n📋 TRUE ANGEL COMPLETE SYSTEM INFO")
        print("=" * 60)
        metadata = result['metadata']
        print(f"Model: {metadata['model_type']}")
        print(f"Data Sources: {', '.join(metadata['data_sources'])}")
        print(f"Features Used: {metadata['features_used']}")
        print(f"Training MAE: {metadata.get('mae', 0):.2f} points")
        print(f"Training R²: {metadata.get('test_score', 0):.3f}")
        print(f"Training Samples: {metadata.get('training_samples', 0)}")
        print(f"Data Sources Count: {metadata.get('data_sources_count', 0)}")
        print(f"Prediction Date: {metadata['prediction_date']}")

        print(f"\n🎉🎉🎉 TRUE ANGEL COMPLETE PREDICTION - ALL DATA INTEGRATED! 🎉🎉🎉")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()