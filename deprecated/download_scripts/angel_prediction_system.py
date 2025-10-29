#!/usr/bin/env python3
"""
🏀 ANGEL NBA Prediction System - COMPLETE DATA INTEGRATION
Sistema completo che usa TUTTI i dati disponibili senza semplificazioni
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import data sources - removed UnifiedDataStore dependency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AngelPredictionSystem:
    """
    ANGEL: Advanced NBA Game Evaluation Laboratory
    Sistema completo che integra TUTTI i dati disponibili per previsioni accurate
    """

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.trained_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.metrics = {}

        # Carica mapping team ID → nome
        self.team_mapping = self._load_team_mapping()

        logger.info("🔥 ANGEL NBA Prediction System initialized with COMPLETE data integration")

    def _load_team_mapping(self) -> Dict[int, str]:
        """Carica mapping team ID → nome dai dati reali"""
        try:
            teams_file = self.data_path / "persistent" / "teams" / "teams_2025-10-27.parquet"
            if teams_file.exists():
                teams_df = pd.read_parquet(teams_file)
                mapping = dict(zip(teams_df['team_id'], teams_df['team_name']))
                logger.info(f"✅ Team mapping loaded: {len(mapping)} teams")
                return mapping
            else:
                logger.warning("Teams file not found, using basic mapping")
                return {}
        except Exception as e:
            logger.error(f"Error loading team mapping: {e}")
            return {}

    def load_all_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Carica TUTTI i dati disponibili per l'analisi completa"""
        logger.info("🔥 Loading ALL data sources for ANGEL system...")

        data_sources = {}

        try:
            # 1. Dataset principale partite NBA
            main_dataset = self.data_path / "nba_simple_complete_dataset.csv"
            if main_dataset.exists():
                data_sources['games'] = pd.read_csv(main_dataset)
                logger.info(f"✅ Main games loaded: {len(data_sources['games'])} games")

            # 2. Partite complete con dettagli
            complete_games = self.data_path / "persistent" / "games" / "game_results_2024-25_Regular_Season.parquet"
            if complete_games.exists():
                data_sources['complete_games'] = pd.read_parquet(complete_games)
                logger.info(f"✅ Complete games loaded: {len(data_sources['complete_games'])} games")

            # 3. Statistiche giocatori
            player_stats = self.data_path / "persistent" / "players" / "player_stats_2025-10-27.parquet"
            if player_stats.exists():
                data_sources['player_stats'] = pd.read_parquet(player_stats)
                logger.info(f"✅ Player stats loaded: {len(data_sources['player_stats'])} records")

            # 4. Momentum giocatori
            player_momentum = self.data_path / "persistent" / "players" / "player_momentum_2025-10-27.parquet"
            if player_momentum.exists():
                data_sources['player_momentum'] = pd.read_parquet(player_momentum)
                logger.info(f"✅ Player momentum loaded: {len(data_sources['player_momentum'])} records")

            # 5. Roster
            roster_file = self.data_path / "persistent" / "rosters" / "rosters_2025-10-27.parquet"
            if roster_file.exists():
                data_sources['rosters'] = pd.read_parquet(roster_file)
                logger.info(f"✅ Rosters loaded: {len(data_sources['rosters'])} records")

            # 6. Injuries
            injury_file = self.data_path / "persistent" / "injuries" / "injury_reports_2025-10-27.parquet"
            if injury_file.exists():
                data_sources['injuries'] = pd.read_parquet(injury_file)
                logger.info(f"✅ Injuries loaded: {len(data_sources['injuries'])} records")

            # 7. Team information
            teams_file = self.data_path / "persistent" / "teams" / "teams_2025-10-27.parquet"
            if teams_file.exists():
                data_sources['teams'] = pd.read_parquet(teams_file)
                logger.info(f"✅ Teams loaded: {len(data_sources['teams'])} teams")

            logger.info("🎯 ALL DATA SOURCES SUCCESSFULLY LOADED FOR ANGEL SYSTEM!")
            return data_sources

        except Exception as e:
            logger.error(f"❌ Error loading data sources: {e}")
            return {}

    def create_comprehensive_features(self, games_df: pd.DataFrame, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Crea feature complete usando TUTTI i dati disponibili
        This is the CORE of ANGEL system
        """
        logger.info("🔥 Creating comprehensive features with ALL data sources...")

        features_list = []

        for idx, game in games_df.iterrows():
            try:
                # Convert team IDs to names
                home_team_id = game.get('HOME_TEAM_ID', 0)
                away_team_id = game.get('AWAY_TEAM_ID', 0)
                home_team_name = self.team_mapping.get(home_team_id, f"Team_{home_team_id}")
                away_team_name = self.team_mapping.get(away_team_id, f"Team_{away_team_id}")

                # Base features from game data
                features = {
                    'game_id': game.get('GAME_ID', idx),
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_team_name': home_team_name,
                    'away_team_name': away_team_name,
                    'season': game.get('SEASON', 2024),
                    'game_date': game.get('GAME_DATE', '2024-01-01'),

                    # Base score features
                    'home_score': float(game.get('HOME_SCORE', 0)),
                    'away_score': float(game.get('AWAY_SCORE', 0)),
                    'total_score': float(game.get('TOTAL_SCORE', 0)),
                    'opponent_score': float(game.get('OPPONENT_SCORE', 0)),

                    # Team performance metrics
                    'home_offensive_rating': float(game.get('HOME_ORtg_sAvg', 110.0)),
                    'away_offensive_rating': float(game.get('AWAY_ORtg_sAvg', 110.0)),
                    'home_defensive_rating': float(game.get('HOME_DRtg_sAvg', 110.0)),
                    'away_defensive_rating': float(game.get('AWAY_DRtg_sAvg', 110.0)),
                    'home_pace': float(game.get('HOME_PACE', 100.0)),
                    'away_pace': float(game.get('AWAY_PACE', 100.0)),

                    # Advanced differential features
                    'offensive_rating_diff': float(game.get('HOME_ORtg_sAvg', 110.0)) - float(game.get('AWAY_ORtg_sAvg', 110.0)),
                    'defensive_rating_diff': float(game.get('HOME_DRtg_sAvg', 110.0)) - float(game.get('AWAY_DRtg_sAvg', 110.0)),
                    'pace_diff': float(game.get('HOME_PACE', 100.0)) - float(game.get('AWAY_PACE', 100.0)),
                    'efficiency_diff': (float(game.get('HOME_ORtg_sAvg', 110.0)) - float(game.get('HOME_DRtg_sAvg', 110.0))) - (float(game.get('AWAY_ORtg_sAvg', 110.0)) - float(game.get('AWAY_DRtg_sAvg', 110.0))),

                    # Game context features
                    'home_advantage': 3.5,
                    'offensive_quality': (float(game.get('HOME_ORtg_sAvg', 110.0)) + float(game.get('AWAY_ORtg_sAvg', 110.0))) / 2,
                    'defensive_quality': (float(game.get('HOME_DRtg_sAvg', 110.0)) + float(game.get('AWAY_DRtg_sAvg', 110.0))) / 2,
                    'game_pace': (float(game.get('HOME_PACE', 100.0)) + float(game.get('AWAY_PACE', 100.0))) / 2,
                    'expected_total': float(game.get('HOME_SCORE', 0)) + float(game.get('AWAY_SCORE', 0)),
                }

                # Add injury impact features
                injury_features = self._calculate_injury_features(home_team_name, away_team_name, data_sources.get('injuries'))
                features.update(injury_features)

                # Add roster stability features
                roster_features = self._calculate_roster_features(home_team_name, away_team_name, data_sources.get('rosters'))
                features.update(roster_features)

                # Add player momentum features
                momentum_features = self._calculate_momentum_features(home_team_name, away_team_name, data_sources.get('player_momentum'))
                features.update(momentum_features)

                # Add head-to-head features
                h2h_features = self._calculate_h2h_features(home_team_name, away_team_name, data_sources.get('complete_games'))
                features.update(h2h_features)

                # Add player performance features
                player_features = self._calculate_player_features(home_team_name, away_team_name, data_sources.get('player_stats'))
                features.update(player_features)

                features_list.append(features)

            except Exception as e:
                logger.warning(f"Error creating features for game {idx}: {e}")
                continue

        if not features_list:
            raise Exception("No features could be created")

        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Created comprehensive features: {len(features_df)} samples with {len(features_df.columns)} features")

        return features_df

    def _calculate_injury_features(self, home_team: str, away_team: str, injuries_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calcola feature impatto infortuni"""
        features = {
            'home_injured_players': 0.0,
            'away_injured_players': 0.0,
            'home_key_players_injured': 0.0,
            'away_key_players_injured': 0.0,
            'injury_impact_differential': 0.0
        }

        if injuries_df is None or injuries_df.empty:
            return features

        try:
            # Conta infortunati per squadra
            home_injuries = injuries_df[injuries_df['team'] == home_team] if 'team' in injuries_df.columns else pd.DataFrame()
            away_injuries = injuries_df[injuries_df['team'] == away_team] if 'team' in injuries_df.columns else pd.DataFrame()

            features['home_injured_players'] = float(len(home_injuries))
            features['away_injured_players'] = float(len(away_injuries))

            # Identifica giocatori chiave infortunati (base su status)
            if 'status' in injuries_df.columns:
                home_key_injured = home_injuries[home_injuries['status'].isin(['Out', 'Doubtful'])]
                away_key_injured = away_injuries[away_injuries['status'].isin(['Out', 'Doubtful'])]
                features['home_key_players_injured'] = float(len(home_key_injured))
                features['away_key_players_injured'] = float(len(away_key_injured))

            features['injury_impact_differential'] = features['away_key_players_injured'] - features['home_key_players_injured']

        except Exception as e:
            logger.warning(f"Error calculating injury features: {e}")

        return features

    def _calculate_roster_features(self, home_team: str, away_team: str, rosters_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calcola feature stabilità roster"""
        features = {
            'home_roster_size': 15.0,
            'away_roster_size': 15.0,
            'home_roster_stability': 1.0,
            'away_roster_stability': 1.0,
            'roster_stability_diff': 0.0
        }

        if rosters_df is None or rosters_df.empty:
            return features

        try:
            # Filtra per squadra se possibile
            if 'team_name' in rosters_df.columns:
                home_roster = rosters_df[rosters_df['team_name'] == home_team]
                away_roster = rosters_df[rosters_df['team_name'] == away_team]

                features['home_roster_size'] = float(len(home_roster)) if not home_roster.empty else 15.0
                features['away_roster_size'] = float(len(away_roster)) if not away_roster.empty else 15.0

                # La stabilità è base su dimensione ideale (15 giocatori)
                features['home_roster_stability'] = min(features['home_roster_size'] / 15.0, 1.0)
                features['away_roster_stability'] = min(features['away_roster_size'] / 15.0, 1.0)
                features['roster_stability_diff'] = features['home_roster_stability'] - features['away_roster_stability']

        except Exception as e:
            logger.warning(f"Error calculating roster features: {e}")

        return features

    def _calculate_momentum_features(self, home_team: str, away_team: str, momentum_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calcola feature momentum giocatori"""
        features = {
            'home_player_momentum': 0.0,
            'away_player_momentum': 0.0,
            'home_star_power': 0.0,
            'away_star_power': 0.0,
            'momentum_differential': 0.0
        }

        if momentum_df is None or momentum_df.empty:
            return features

        try:
            # Filtra per squadra
            home_momentum = momentum_df[momentum_df['team_name'] == home_team] if 'team_name' in momentum_df.columns else pd.DataFrame()
            away_momentum = momentum_df[momentum_df['team_name'] == away_team] if 'team_name' in momentum_df.columns else pd.DataFrame()

            # Calcola momentum base su punti, assist, rimbalzi
            if not home_momentum.empty and 'points' in home_momentum.columns:
                home_prod = (home_momentum['points'].fillna(0) +
                           home_momentum['assists'].fillna(0) +
                           home_momentum['rebounds'].fillna(0))
                features['home_player_momentum'] = float(home_prod.mean())

                # Star power = top 3 producers
                if len(home_prod) >= 3:
                    features['home_star_power'] = float(home_prod.nlargest(3).sum())
                else:
                    features['home_star_power'] = float(home_prod.sum())

            if not away_momentum.empty and 'points' in away_momentum.columns:
                away_prod = (away_momentum['points'].fillna(0) +
                           away_momentum['assists'].fillna(0) +
                           away_momentum['rebounds'].fillna(0))
                features['away_player_momentum'] = float(away_prod.mean())

                if len(away_prod) >= 3:
                    features['away_star_power'] = float(away_prod.nlargest(3).sum())
                else:
                    features['away_star_power'] = float(away_prod.sum())

            features['momentum_differential'] = features['home_player_momentum'] - features['away_player_momentum']

        except Exception as e:
            logger.warning(f"Error calculating momentum features: {e}")

        return features

    def _calculate_h2h_features(self, home_team: str, away_team: str, complete_games_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calcola feature head-to-head"""
        features = {
            'h2h_games_count': 0.0,
            'home_h2h_win_rate': 0.5,
            'avg_h2h_total': 220.0,
            'h2h_total_variance': 200.0,
            'h2h_trend': 0.0
        }

        if complete_games_df is None or complete_games_df.empty:
            return features

        try:
            # Filtra scontri diretti
            h2h_mask = (
                ((complete_games_df['home_team'] == home_team) & (complete_games_df['away_team'] == away_team)) |
                ((complete_games_df['home_team'] == away_team) & (complete_games_df['away_team'] == home_team))
            )
            h2h_games = complete_games_df[h2h_mask].copy()

            if h2h_games.empty:
                return features

            features['h2h_games_count'] = float(len(h2h_games))

            # Calcola win rate squadra home
            if 'home_score' in h2h_games.columns and 'away_score' in h2h_games.columns:
                h2h_games['total_score'] = h2h_games['home_score'] + h2h_games['away_score']
                features['avg_h2h_total'] = float(h2h_games['total_score'].mean())
                features['h2h_total_variance'] = float(h2h_games['total_score'].var())

                # Win rate calcolata su squadra home
                h2h_games['home_won'] = np.where(
                    ((h2h_games['home_team'] == home_team) & (h2h_games['home_score'] > h2h_games['away_score'])) |
                    ((h2h_games['away_team'] == home_team) & (h2h_games['away_score'] > h2h_games['home_score'])),
                    1, 0
                )
                features['home_h2h_win_rate'] = float(h2h_games['home_won'].mean())

                # Trend ultimi 5 scontri
                if len(h2h_games) >= 5:
                    h2h_games_sorted = h2h_games.sort_values('game_date', ascending=False).head(5)
                    features['h2h_trend'] = float(h2h_games_sorted['home_won'].mean() - 0.5)

        except Exception as e:
            logger.warning(f"Error calculating H2H features: {e}")

        return features

    def _calculate_player_features(self, home_team: str, away_team: str, player_stats_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calcola feature statistiche giocatori"""
        features = {
            'home_team_avg_points': 110.0,
            'away_team_avg_points': 110.0,
            'home_team_efficiency': 1.0,
            'away_team_efficiency': 1.0,
            'player_quality_diff': 0.0
        }

        if player_stats_df is None or player_stats_df.empty:
            return features

        try:
            # Filtra statistiche per squadra
            home_stats = player_stats_df[player_stats_df['team_name'] == home_team] if 'team_name' in player_stats_df.columns else pd.DataFrame()
            away_stats = player_stats_df[player_stats_df['team_name'] == away_team] if 'team_name' in player_stats_df.columns else pd.DataFrame()

            if not home_stats.empty and 'points' in home_stats.columns:
                features['home_team_avg_points'] = float(home_stats['points'].mean())
                if 'minutes' in home_stats.columns:
                    features['home_team_efficiency'] = float(home_stats['points'].sum() / max(home_stats['minutes'].sum(), 1))

            if not away_stats.empty and 'points' in away_stats.columns:
                features['away_team_avg_points'] = float(away_stats['points'].mean())
                if 'minutes' in away_stats.columns:
                    features['away_team_efficiency'] = float(away_stats['points'].sum() / max(away_stats['minutes'].sum(), 1))

            features['player_quality_diff'] = features['home_team_efficiency'] - features['away_team_efficiency']

        except Exception as e:
            logger.warning(f"Error calculating player features: {e}")

        return features

    def train_model(self) -> Dict[str, Any]:
        """Allena il modello ANGEL con tutte le feature complete"""
        logger.info("🔥 Training ANGEL model with comprehensive features...")

        try:
            # Carica dati
            data_sources = self.load_all_data_sources()
            games_df = data_sources.get('games')

            if games_df is None or games_df.empty:
                raise Exception("No games data available")

            # Crea feature complete
            features_df = self.create_comprehensive_features(games_df, data_sources)

            # Prepara target (total score)
            y = features_df['total_score']

            # Rimuovi colonne non feature
            feature_columns = [col for col in features_df.columns if col not in [
                'game_id', 'home_team_id', 'away_team_id', 'home_team_name',
                'away_team_name', 'season', 'game_date', 'total_score'
            ]]

            X = features_df[feature_columns].fillna(0)

            # Feature selection
            logger.info("Selecting best features...")
            self.feature_selector = SelectKBest(score_func=f_regression, k=30)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.feature_names = [feature_columns[i] for i in self.feature_selector.get_support(indices=True)]

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_selected)

            # Split dati
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Crea ensemble model
            logger.info("Creating ensemble model...")
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

            # Predizioni test set
            y_pred = self.trained_model.predict(X_test)
            mae = np.mean(np.abs(y_test - y_pred))
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)

            # Salva metriche
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

            logger.info("✅ ANGEL model training completed!")
            logger.info(f"📊 Performance: MAE={mae:.2f}, R²={test_score:.3f}, Features={len(self.feature_names)}")

            return self.metrics

        except Exception as e:
            logger.error(f"❌ Error training ANGEL model: {e}")
            raise

    def predict_game(self, team1: str, team2: str, line: float, home_team: str = None) -> Dict[str, Any]:
        """Previsione completa usando sistema ANGEL"""
        logger.info(f"🔥 ANGEL prediction: {team1} vs {team2}, line: {line}")

        if self.trained_model is None:
            logger.info("Training ANGEL model...")
            self.train_model()

        try:
            # Determina squadra home
            if home_team:
                is_team1_home = team1.lower() == home_team.lower()
            else:
                is_team1_home = True  # Default

            home_team_name = team1 if is_team1_home else team2
            away_team_name = team2 if is_team1_home else team1

            # Carica dati attuali
            data_sources = self.load_all_data_sources()

            # Crea mock game con valori reali basati su statistiche storiche
            games_df = data_sources.get('games')
            mock_game = self._create_mock_game(home_team_name, away_team_name, games_df)

            # Crea feature complete per previsione
            features_df = self.create_comprehensive_features(pd.DataFrame([mock_game]), data_sources)

            # Prepara feature usando esattamente quelle del training
            feature_columns = [col for col in features_df.columns if col not in [
                'game_id', 'home_team_id', 'away_team_id', 'home_team_name',
                'away_team_name', 'season', 'game_date', 'total_score'
            ]]

            X = features_df[feature_columns].fillna(0)

            # Assicura che le feature siano quelle del training
            if self.feature_names:
                # Aggiungi feature mancanti con valori di default
                for feature in self.feature_names:
                    if feature not in X.columns:
                        X[feature] = 0.0

                # Riordina colonne secondo training
                X = X[self.feature_names]

            # Scale
            X_scaled = self.scaler.transform(X)

            # Previsione
            predicted_total = self.trained_model.predict(X_scaled)[0]

            # Calcola confidenza e probabilità
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

            # Analisi completa
            analysis = self._create_complete_analysis(home_team_name, away_team_name, data_sources)

            result = {
                'predicted_total': float(predicted_total),
                'confidence_interval': confidence_interval,
                'recommendation': recommendation,
                'confidence': float(confidence),
                'over_probability': float(over_prob),
                'under_probability': float(under_prob),
                'team_analysis': analysis,
                'feature_importance': self._get_feature_importance(),
                'metadata': {
                    'model_type': 'ANGEL Ensemble (RF + XGB + GB)',
                    'features_used': len(self.feature_names),
                    'data_sources': list(data_sources.keys()),
                    'line': line,
                    'prediction_date': datetime.now().isoformat(),
                    **self.metrics
                }
            }

            logger.info(f"🎯 ANGEL prediction: {predicted_total:.1f} vs {line} ({recommendation})")
            return result

        except Exception as e:
            logger.error(f"❌ ANGEL prediction failed: {e}")
            raise

    def _create_mock_game(self, home_team: str, away_team: str, games_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Crea mock game con valori reali basati su statistiche storiche"""

        # Valori default base NBA
        defaults = {
            'GAME_ID': 999999,
            'HOME_TEAM_NAME': home_team,
            'AWAY_TEAM_NAME': away_team,
            'HOME_TEAM_ID': 0,
            'AWAY_TEAM_ID': 0,
            'SEASON': 2025,
            'GAME_DATE': datetime.now().strftime('%Y-%m-%d'),
            'HOME_SCORE': 110,
            'AWAY_SCORE': 105,
            'TOTAL_SCORE': 215,
            'OPPONENT_SCORE': 105,
            'HOME_ORtg_sAvg': 112.0,
            'AWAY_ORtg_sAvg': 110.0,
            'HOME_DRtg_sAvg': 108.0,
            'AWAY_DRtg_sAvg': 110.0,
            'HOME_PACE': 100.0,
            'AWAY_PACE': 98.0
        }

        if games_df is None or games_df.empty:
            return defaults

        try:
            # Cerca statistiche reali per le squadre
            home_games = games_df[
                (games_df['HOME_TEAM_NAME'] == home_team) |
                (games_df['AWAY_TEAM_NAME'] == home_team)
            ]

            away_games = games_df[
                (games_df['HOME_TEAM_NAME'] == away_team) |
                (games_df['AWAY_TEAM_NAME'] == away_team)
            ]

            if not home_games.empty:
                defaults.update({
                    'HOME_ORtg_sAvg': float(home_games['HOME_ORtg_sAvg'].mean()) if 'HOME_ORtg_sAvg' in home_games.columns else 112.0,
                    'HOME_DRtg_sAvg': float(home_games['HOME_DRtg_sAvg'].mean()) if 'HOME_DRtg_sAvg' in home_games.columns else 108.0,
                    'HOME_PACE': float(home_games['HOME_PACE'].mean()) if 'HOME_PACE' in home_games.columns else 100.0,
                    'HOME_SCORE': float(home_games['HOME_SCORE'].mean()) if 'HOME_SCORE' in home_games.columns else 110,
                    'TOTAL_SCORE': float(home_games['TOTAL_SCORE'].mean()) if 'TOTAL_SCORE' in home_games.columns else 215
                })

            if not away_games.empty:
                defaults.update({
                    'AWAY_ORtg_sAvg': float(away_games['AWAY_ORtg_sAvg'].mean()) if 'AWAY_ORtg_sAvg' in away_games.columns else 110.0,
                    'AWAY_DRtg_sAvg': float(away_games['AWAY_DRtg_sAvg'].mean()) if 'AWAY_DRtg_sAvg' in away_games.columns else 110.0,
                    'AWAY_PACE': float(away_games['AWAY_PACE'].mean()) if 'AWAY_PACE' in away_games.columns else 98.0,
                    'AWAY_SCORE': float(away_games['AWAY_SCORE'].mean()) if 'AWAY_SCORE' in away_games.columns else 105
                })

            # Aggiorna total score basato su media delle squadre
            defaults['TOTAL_SCORE'] = defaults['HOME_SCORE'] + defaults['AWAY_SCORE']
            defaults['OPPONENT_SCORE'] = defaults['AWAY_SCORE']

        except Exception as e:
            logger.warning(f"Error creating mock game: {e}")

        return defaults

    def _create_complete_analysis(self, home_team: str, away_team: str, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Crea analisi completa delle squadre"""

        analysis = {
            'home_team': {'name': home_team},
            'away_team': {'name': away_team}
        }

        # Aggiungi statistiche base da dataset
        games_df = data_sources.get('games')
        if games_df is not None:
            home_games = games_df[
                (games_df['HOME_TEAM_NAME'] == home_team) |
                (games_df['AWAY_TEAM_NAME'] == home_team)
            ]

            away_games = games_df[
                (games_df['HOME_TEAM_NAME'] == away_team) |
                (games_df['AWAY_TEAM_NAME'] == away_team)
            ]

            if not home_games.empty:
                analysis['home_team'].update({
                    'avg_points_scored': float(home_games['TOTAL_SCORE'].mean()) if 'TOTAL_SCORE' in home_games.columns else 0.0,
                    'avg_points_allowed': float(home_games['OPPONENT_SCORE'].mean()) if 'OPPONENT_SCORE' in home_games.columns else 0.0,
                    'offensive_rating': float(home_games['HOME_ORtg_sAvg'].mean()) if 'HOME_ORtg_sAvg' in home_games.columns else 110.0,
                    'defensive_rating': float(home_games['HOME_DRtg_sAvg'].mean()) if 'HOME_DRtg_sAvg' in home_games.columns else 110.0,
                    'pace': float(home_games['HOME_PACE'].mean()) if 'HOME_PACE' in home_games.columns else 100.0,
                    'games_analyzed': len(home_games)
                })

            if not away_games.empty:
                analysis['away_team'].update({
                    'avg_points_scored': float(away_games['TOTAL_SCORE'].mean()) if 'TOTAL_SCORE' in away_games.columns else 0.0,
                    'avg_points_allowed': float(away_games['OPPONENT_SCORE'].mean()) if 'OPPONENT_SCORE' in away_games.columns else 0.0,
                    'offensive_rating': float(away_games['AWAY_ORtg_sAvg'].mean()) if 'AWAY_ORtg_sAvg' in away_games.columns else 110.0,
                    'defensive_rating': float(away_games['AWAY_DRtg_sAvg'].mean()) if 'AWAY_DRtg_sAvg' in away_games.columns else 110.0,
                    'pace': float(away_games['AWAY_PACE'].mean()) if 'AWAY_PACE' in away_games.columns else 100.0,
                    'games_analyzed': len(away_games)
                })

        return analysis

    def _get_feature_importance(self) -> Dict[str, float]:
        """Ottiene importanza delle feature dal modello"""
        if self.trained_model is None or not hasattr(self.trained_model, 'estimators_'):
            return {}

        try:
            # Media importanza dai modelli ensemble
            importances = []
            for estimator in self.trained_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)

            if importances and self.feature_names:
                avg_importance = np.mean(importances, axis=0)
                return dict(zip(self.feature_names, avg_importance))
        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}")

        return {}


def main():
    """Main execution"""
    print("🔥 ANGEL NBA Prediction System - COMPLETE DATA INTEGRATION")
    print("=" * 80)
    print("Sistema avanzato che usa TUTTI i dati disponibili per previsioni accurate")
    print("📊 Dati integrati: games, players, injuries, rosters, momentum, H2H")
    print("=" * 80)

    import argparse
    parser = argparse.ArgumentParser(description="ANGEL NBA Prediction System")
    parser.add_argument('--team1', type=str, required=True, help='First team')
    parser.add_argument('--team2', type=str, required=True, help='Second team')
    parser.add_argument('--line', type=float, required=True, help='Betting line')
    parser.add_argument('--home', type=str, help='Home team (optional)')

    args = parser.parse_args()

    try:
        # Inizializza sistema ANGEL
        angel = AngelPredictionSystem()

        # Esegui previsione completa
        result = angel.predict_game(args.team1, args.team2, args.line, args.home)

        # Stampa risultati completi
        print(f"\n🏀 ANGEL PREDICTION RESULTS")
        print("=" * 50)
        print(f"Match: {args.team1} vs {args.team2}")
        print(f"Line: {args.line}")
        print(f"Predicted Total: {result['predicted_total']:.1f}")
        print(f"Confidence Interval: {result['confidence_interval'][0]:.1f} - {result['confidence_interval'][1]:.1f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Over Probability: {result['over_probability']:.1%}")
        print(f"Under Probability: {result['under_probability']:.1%}")

        print(f"\n📈 TEAM ANALYSIS")
        print("=" * 50)

        home_data = result['team_analysis']['home_team']
        away_data = result['team_analysis']['away_team']

        print(f"{home_data['name']}:")
        print(f"  Avg Points Scored: {home_data.get('avg_points_scored', 0):.1f}")
        print(f"  Avg Points Allowed: {home_data.get('avg_points_allowed', 0):.1f}")
        print(f"  Offensive Rating: {home_data.get('offensive_rating', 0):.1f}")
        print(f"  Defensive Rating: {home_data.get('defensive_rating', 0):.1f}")
        print(f"  Pace: {home_data.get('pace', 0):.1f}")
        print(f"  Games Analyzed: {home_data.get('games_analyzed', 0)}")

        print(f"\n{away_data['name']}:")
        print(f"  Avg Points Scored: {away_data.get('avg_points_scored', 0):.1f}")
        print(f"  Avg Points Allowed: {away_data.get('avg_points_allowed', 0):.1f}")
        print(f"  Offensive Rating: {away_data.get('offensive_rating', 0):.1f}")
        print(f"  Defensive Rating: {away_data.get('defensive_rating', 0):.1f}")
        print(f"  Pace: {away_data.get('pace', 0):.1f}")
        print(f"  Games Analyzed: {away_data.get('games_analyzed', 0)}")

        if result['feature_importance']:
            print(f"\n🔍 KEY FACTORS (Top 5)")
            print("=" * 50)
            top_features = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                print(f"  • {feature}: {importance:.3f}")

        print(f"\n🎯 BETTING RECOMMENDATION")
        print("=" * 50)
        if result['recommendation'] == 'OVER':
            print(f"✅ RECOMMENDATION: OVER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} (+{result['predicted_total'] - args.line:.1f})")
        else:
            print(f"✅ RECOMMENDATION: UNDER {args.line}")
            print(f"💰 Predicted total: {result['predicted_total']:.1f} ({result['predicted_total'] - args.line:.1f})")

        print(f"\n📋 ANGEL SYSTEM INFO")
        print("=" * 50)
        metadata = result['metadata']
        print(f"Model: {metadata['model_type']}")
        print(f"Features Used: {metadata['features_used']}")
        print(f"Data Sources: {', '.join(metadata['data_sources'])}")
        print(f"Training MAE: {metadata.get('mae', 0):.2f} points")
        print(f"Training R²: {metadata.get('test_score', 0):.3f}")
        print(f"Training Samples: {metadata.get('training_samples', 0)}")
        print(f"Prediction Date: {metadata['prediction_date']}")

        print(f"\n🎉 ANGEL PREDICTION COMPLETED WITH FULL DATA INTEGRATION!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Import missing stats module
    from scipy import stats
    main()