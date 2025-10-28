#!/usr/bin/env python3
"""
🏀 NBA Prediction Pipeline - Context7 Compliant
Modern prediction system using real NBA data with ensemble methods.

This module implements:
- Context7-compliant data processing pipeline
- Ensemble ML models (VotingRegressor + RandomForest + XGBoost)
- Real NBA data integration with UnifiedDataStore
- Feature engineering for Over/Under predictions
- Performance monitoring and validation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

# Context7-compliant ML imports
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb

# Context7-compliant internal imports
from nba_predictor.core.data_store import UnifiedDataStore
from nba_predictor.integration.real_data_adapter import RealNBADataAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionFeatures:
    """Feature set for NBA game prediction."""
    team1_id: int
    team2_id: int
    team1_avg_points: float
    team2_avg_points: float
    team1_avg_allowed: float
    team2_avg_allowed: float
    team1_momentum: float
    team2_momentum: float
    team1_pace: float
    team2_pace: float
    home_advantage: float
    days_rest_team1: int
    days_rest_team2: int
    historical_avg_total: float
    season_stage: str  # early, mid, late, playoffs
    back_to_back_team1: bool
    back_to_back_team2: bool


@dataclass
class PredictionResult:
    """Context7-compliant prediction result."""
    predicted_total: float
    confidence_interval: Tuple[float, float]
    recommendation: str  # OVER, UNDER, NEUTRAL
    confidence: float
    over_probability: float
    under_probability: float
    model_weights: Dict[str, float]
    feature_importance: Dict[str, float]
    team_analysis: Dict[str, Any]
    metadata: Dict[str, Any]


class PredictionPipelineError(Exception):
    """Custom exception for prediction pipeline operations."""
    pass


class NBAPredictionPipeline:
    """
    Context7-compliant NBA prediction pipeline using ensemble methods.

    This class implements a complete ML pipeline for NBA Over/Under predictions
    using real NBA data and advanced ensemble techniques.

    Attributes:
        data_adapter: Real NBA data adapter
        ensemble_model: VotingRegressor ensemble
        feature_scaler: Feature preprocessing scaler
        feature_selector: Feature selection module
        is_trained: Whether the model is trained
        metrics: Performance metrics
    """

    def __init__(
        self,
        data_path: str = "data",
        model_path: str = "models",
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize the NBA prediction pipeline.

        Args:
            data_path: Path to real NBA data
            model_path: Path to save/load trained models
            cache_enabled: Enable feature caching for performance

        Example:
            >>> pipeline = NBAPredictionPipeline()
            >>> result = pipeline.predict_over_under(
            ...     team1="Los Angeles Lakers",
            ...     team2="Boston Celtics",
            ...     line=225.5
            ... )
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.cache_enabled = cache_enabled

        # Initialize data adapter
        self.data_adapter = RealNBADataAdapter(str(self.data_path))

        # Initialize ML components (Context7 best practices)
        self.ensemble_model = None
        self.feature_scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = SelectKBest(f_regression, k=15)
        self.feature_columns = []

        # Model state
        self.is_trained = False
        self.metrics = {}
        self.feature_cache = {} if cache_enabled else None

        # Initialize ensemble model with Context7 best practices
        self._initialize_ensemble_model()

        logger.info("NBAPredictionPipeline initialized with Context7 compliance")

    def _initialize_ensemble_model(self) -> None:
        """
        Initialize ensemble model with Context7 best practices.

        Uses VotingRegressor with multiple diverse models for improved accuracy.
        """
        # Context7-compliant model configurations
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )),
            ('gbr', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ))
        ]

        # Create VotingRegressor with weighted voting
        self.ensemble_model = VotingRegressor(
            estimators=models,
            weights=[2, 3, 1]  # XGBoost gets higher weight based on typical performance
        )

        logger.info("Ensemble model initialized with RandomForest, XGBoost, GradientBoosting")

    def _load_and_prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load real NBA data and prepare features for training.

        Returns:
            Tuple of (features_df, target_series)

        Raises:
            PredictionPipelineError: If data loading fails
        """
        try:
            logger.info("Loading real NBA data for training...")

            # Load complete games dataset
            games_df = pd.read_csv(self.data_path / "nba_simple_complete_dataset.csv")
            logger.info(f"Loaded {len(games_df)} real NBA games")

            # Prepare features - each row is already a complete game
            features_list = []
            targets = []

            # Process each game as a training sample
            for _, game in games_df.iterrows():
                # Create features from the complete game data
                features = self._create_game_features_from_row(game)
                if features:
                    features_list.append(features)
                    # Target is actual total score
                    targets.append(game['TOTAL_SCORE'])

            if not features_list:
                raise PredictionPipelineError("No valid features could be created from data")

            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            target_series = pd.Series(targets)

            logger.info(f"Created {len(features_df)} training samples with {len(features_df.columns)} features")

            return features_df, target_series

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise PredictionPipelineError(f"Failed to load training data: {e}")

    def _create_game_features_from_row(self, game_row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Create features from a single game row (already contains home/away data).

        Args:
            game_row: Single game row with all home/away statistics

        Returns:
            Feature dictionary or None if invalid
        """
        try:
            # Extract statistics from the game row
            home_score = game_row.get('HOME_SCORE', 0)
            away_score = game_row.get('AWAY_SCORE', 0)
            home_rtg = game_row.get('HOME_ORtg_sAvg', 110.0)
            away_rtg = game_row.get('AWAY_ORtg_sAvg', 110.0)
            home_drtg = game_row.get('HOME_DRtg_sAvg', 110.0)
            away_drtg = game_row.get('AWAY_DRtg_sAvg', 110.0)
            home_pace = game_row.get('HOME_PACE', 100.0)
            away_pace = game_row.get('AWAY_PACE', 100.0)

            # Efficiency differentials
            home_eff_diff = home_rtg - home_drtg
            away_eff_diff = away_rtg - away_drtg

            # Create comprehensive feature set
            features = {
                'home_score': home_score,
                'away_score': away_score,
                'home_offensive_rating': home_rtg,
                'away_offensive_rating': away_rtg,
                'home_defensive_rating': home_drtg,
                'away_defensive_rating': away_drtg,
                'home_pace': home_pace,
                'away_pace': away_pace,
                'home_eff_diff': home_eff_diff,
                'away_eff_diff': away_eff_diff,
                'home_advantage': 3.5,  # Standard NBA home advantage
                'pace_differential': home_pace - away_pace,
                'offensive_quality': (home_rtg + away_rtg) / 2,
                'defensive_quality': (home_drtg + away_drtg) / 2,
                'expected_total': home_score + away_score,
                'game_pace': game_row.get('GAME_PACE', 100.0),
                'pace_differential_actual': game_row.get('PACE_DIFFERENTIAL', 0.0),
                'home_off_vs_away_def': game_row.get('HOME_OFF_vs_AWAY_DEF', 0.0),
                'away_off_vs_home_def': game_row.get('AWAY_OFF_vs_HOME_DEF', 0.0),
                'total_expected_scoring': game_row.get('TOTAL_EXPECTED_SCORING', home_score + away_score)
            }

            return features

        except Exception as e:
            logger.warning(f"Error creating features from game row: {e}")
            return None

    def _create_game_features(self, home_team: pd.Series, away_team: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Create features for a single game.

        Args:
            home_team: Home team statistics
            away_team: Away team statistics

        Returns:
            Feature dictionary or None if invalid
        """
        try:
            # Basic team statistics
            home_points = home_team.get('HOME_SCORE', 0)
            away_points = away_team.get('AWAY_SCORE', 0)
            home_allowed = away_team.get('AWAY_SCORE', 0)
            away_allowed = home_team.get('HOME_SCORE', 0)

            # Advanced metrics
            home_rtg = home_team.get('HOME_ORtg_sAvg', 110.0)
            away_rtg = away_team.get('AWAY_ORtg_sAvg', 110.0)
            home_drtg = home_team.get('HOME_DRtg_sAvg', 110.0)
            away_drtg = away_team.get('AWAY_DRtg_sAvg', 110.0)

            # Pace metrics
            home_pace = home_team.get('HOME_PACE', 100.0)
            away_pace = away_team.get('AWAY_PACE', 100.0)

            # Efficiency differentials
            home_eff_diff = home_rtg - home_drtg
            away_eff_diff = away_rtg - away_drtg

            # Create comprehensive feature set
            features = {
                'home_team_avg_points': home_points,
                'away_team_avg_points': away_points,
                'home_team_avg_allowed': home_allowed,
                'away_team_avg_allowed': away_allowed,
                'home_offensive_rating': home_rtg,
                'away_offensive_rating': away_rtg,
                'home_defensive_rating': home_drtg,
                'away_defensive_rating': away_drtg,
                'home_pace': home_pace,
                'away_pace': away_pace,
                'home_eff_diff': home_eff_diff,
                'away_eff_diff': away_eff_diff,
                'home_advantage': 3.5,  # Standard NBA home advantage
                'pace_differential': home_pace - away_pace,
                'offensive_quality': (home_rtg + away_rtg) / 2,
                'defensive_quality': (home_drtg + away_drtg) / 2,
                'expected_total': home_points + away_points
            }

            return features

        except Exception as e:
            logger.warning(f"Error creating features for game: {e}")
            return None

    def train_model(self, test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the ensemble prediction model using real NBA data.

        Args:
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds

        Returns:
            Training metrics dictionary

        Raises:
            PredictionPipelineError: If training fails
        """
        try:
            logger.info("Starting model training with real NBA data...")

            # Load and prepare data
            X, y = self._load_and_prepare_training_data()

            if len(X) < 100:
                raise PredictionPipelineError(f"Insufficient training data: {len(X)} samples")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Store feature columns
            self.feature_columns = X.columns.tolist()

            # Create Context7-compliant pipeline
            model_pipeline = Pipeline([
                ('scaler', self.feature_scaler),
                ('selector', self.feature_selector),
                ('ensemble', self.ensemble_model)
            ])

            # Train model
            logger.info("Training ensemble model...")
            model_pipeline.fit(X_train, y_train)

            # Evaluate model
            y_pred = model_pipeline.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model_pipeline, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

            # Store trained model and metrics
            self.trained_model = model_pipeline
            self.is_trained = True

            self.metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'cv_mae_mean': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(model_pipeline.named_steps['selector'].get_support(indices=True)),
                'training_date': datetime.now().isoformat()
            }

            # Save model
            self._save_model()

            logger.info(f"Model training completed. MAE: {mae:.2f}, R²: {r2:.3f}")

            return self.metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise PredictionPipelineError(f"Training failed: {e}")

    def predict_over_under(
        self,
        team1: str,
        team2: str,
        line: float,
        home_team: Optional[str] = None
    ) -> PredictionResult:
        """
        Make Over/Under prediction for NBA game using real data.

        Args:
            team1: First team name
            team2: Second team name
            line: Betting line (total points)
            home_team: Which team is playing at home (if None, team2 is home)

        Returns:
            PredictionResult with comprehensive analysis

        Raises:
            PredictionPipelineError: If prediction fails
        """
        try:
            if not self.is_trained:
                # Auto-train if not trained
                logger.info("Model not trained - auto-training...")
                self.train_model()

            # Determine home/away
            if home_team is None:
                home_team = team2

            is_team1_home = (team1 == home_team)

            logger.info(f"Making prediction: {team1} vs {team2}, line: {line}")

            # Get team statistics from real data
            team1_stats = self._get_team_stats(team1)
            team2_stats = self._get_team_stats(team2)

            if not team1_stats or not team2_stats:
                raise PredictionPipelineError(f"Could not find statistics for teams: {team1}, {team2}")

            # Create features using team statistics
            features_dict = self._create_prediction_features(team1_stats, team2_stats, is_team1_home)
            if not features_dict:
                raise PredictionPipelineError("Failed to create prediction features")

            # Convert to DataFrame
            features_df = pd.DataFrame([features_dict])

            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0

            # Reorder columns to match training data
            features_df = features_df[self.feature_columns]

            # Make prediction
            predicted_total = self.trained_model.predict(features_df)[0]

            # Calculate confidence intervals (using prediction variance estimate)
            prediction_std = np.sqrt(self.metrics['mse'])
            confidence_interval = (
                predicted_total - 1.96 * prediction_std,
                predicted_total + 1.96 * prediction_std
            )

            # Determine recommendation and probabilities
            if predicted_total > line:
                recommendation = "OVER"
                confidence = min((predicted_total - line) / prediction_std * 20, 95)
            else:
                recommendation = "UNDER"
                confidence = min((line - predicted_total) / prediction_std * 20, 95)

            # Calculate probabilities using normal distribution
            from scipy import stats
            over_prob = 1 - stats.norm.cdf(line, predicted_total, prediction_std)
            under_prob = stats.norm.cdf(line, predicted_total, prediction_std)

            # Get team analysis
            team_analysis = self._analyze_teams(team1_stats, team2_stats, is_team1_home)

            # Get feature importance
            feature_importance = self._get_feature_importance()

            # Create result
            result = PredictionResult(
                predicted_total=predicted_total,
                confidence_interval=confidence_interval,
                recommendation=recommendation,
                confidence=confidence,
                over_probability=over_prob,
                under_probability=under_prob,
                model_weights={
                    'random_forest': 0.33,
                    'xgboost': 0.5,
                    'gradient_boosting': 0.17
                },
                feature_importance=feature_importance,
                team_analysis=team_analysis,
                metadata={
                    'prediction_date': datetime.now().isoformat(),
                    'line': line,
                    'teams': f"{team1} vs {team2}",
                    'home_team': home_team,
                    'model_version': '1.0.0',
                    'data_source': 'real_nba_data',
                    'training_data_size': self.metrics.get('training_samples', 0)
                }
            )

            logger.info(f"Prediction completed: {predicted_total:.1f} vs {line} ({recommendation})")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionPipelineError(f"Prediction failed: {e}")

    def _get_team_stats(self, team_name: str) -> Optional[Dict[str, float]]:
        """Get team statistics from real data using team name mapping."""
        try:
            # Team name to ID mapping (NBA official teams)
            team_id_mapping = {
                "Los Angeles Lakers": 1610612747,
                "Boston Celtics": 1610612738,
                "Golden State Warriors": 1610612747,
                "Utah Jazz": 1610612762,
                "Miami Heat": 1610612748,
                "Denver Nuggets": 1610612743,
                "Phoenix Suns": 1610612756,
                "Milwaukee Bucks": 1610612749,
                "Philadelphia 76ers": 1610612755,
                "Brooklyn Nets": 1610612751,
                "New York Knicks": 1610612752,
                "Toronto Raptors": 1610612761,
                "Chicago Bulls": 1610612741,
                "Cleveland Cavaliers": 1610612739,
                "Detroit Pistons": 1610612765,
                "Indiana Pacers": 1610612754,
                "Atlanta Hawks": 1610612737,
                "Charlotte Hornets": 1610612766,
                "Orlando Magic": 1610612753,
                "Washington Wizards": 1610612764,
                "Dallas Mavericks": 1610612742,
                "Houston Rockets": 1610612745,
                "Memphis Grizzlies": 1610612763,
                "New Orleans Pelicans": 1610612740,
                "San Antonio Spurs": 1610612759,
                "Minnesota Timberwolves": 1610612750,
                "Oklahoma City Thunder": 1610612760,
                "Portland Trail Blazers": 1610612757,
                "Sacramento Kings": 1610612758,
                "Los Angeles Clippers": 1610612746,
                "Seattle SuperSonics": 1610612758,
                "Kansas City Kings": 1610612758
            }

            # Get team ID
            team_id = team_id_mapping.get(team_name)
            if team_id is None:
                logger.warning(f"Team '{team_name}' not found in mapping")
                return None

            # Load games dataset
            games_df = pd.read_csv(self.data_path / "nba_simple_complete_dataset.csv")

            # Filter for team (either home or away by ID)
            team_games_home = games_df[games_df['HOME_TEAM_ID'] == team_id]
            team_games_away = games_df[games_df['AWAY_TEAM_ID'] == team_id]

            if team_games_home.empty and team_games_away.empty:
                logger.warning(f"No games found for team ID {team_id}")
                return None

            # Calculate average statistics
            all_team_games = pd.concat([team_games_home, team_games_away])

            # Return statistics as dictionary
            stats = {
                'avg_home_score': team_games_home['HOME_SCORE'].mean() if not team_games_home.empty else 110.0,
                'avg_away_score': team_games_away['AWAY_SCORE'].mean() if not team_games_away.empty else 108.0,
                'avg_home_offensive_rating': team_games_home['HOME_ORtg_sAvg'].mean() if not team_games_home.empty else 110.0,
                'avg_away_offensive_rating': team_games_away['AWAY_ORtg_sAvg'].mean() if not team_games_away.empty else 110.0,
                'avg_home_defensive_rating': team_games_home['HOME_DRtg_sAvg'].mean() if not team_games_home.empty else 110.0,
                'avg_away_defensive_rating': team_games_away['AWAY_DRtg_sAvg'].mean() if not team_games_away.empty else 110.0,
                'avg_home_pace': team_games_home['HOME_PACE'].mean() if not team_games_home.empty else 100.0,
                'avg_away_pace': team_games_away['AWAY_PACE'].mean() if not team_games_away.empty else 100.0,
                'total_games': len(all_team_games),
                'team_id': team_id
            }

            logger.info(f"Found stats for {team_name}: {stats['total_games']} games")
            return stats

        except Exception as e:
            logger.error(f"Error getting stats for {team_name}: {e}")
            return None

    def _create_prediction_features(
        self,
        team1_stats: Dict[str, float],
        team2_stats: Dict[str, float],
        is_team1_home: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Create prediction features from team statistics.

        Args:
            team1_stats: Statistics for team 1
            team2_stats: Statistics for team 2
            is_team1_home: Whether team 1 is playing at home

        Returns:
            Feature dictionary or None if invalid
        """
        try:
            # Determine home/away
            home_stats = team1_stats if is_team1_home else team2_stats
            away_stats = team2_stats if is_team1_home else team1_stats

            # Extract statistics
            home_score = home_stats.get('avg_home_score', 110.0)
            away_score = away_stats.get('avg_away_score', 108.0)
            home_rtg = home_stats.get('avg_home_offensive_rating', 110.0)
            away_rtg = away_stats.get('avg_away_offensive_rating', 110.0)
            home_drtg = home_stats.get('avg_home_defensive_rating', 110.0)
            away_drtg = away_stats.get('avg_away_defensive_rating', 110.0)
            home_pace = home_stats.get('avg_home_pace', 100.0)
            away_pace = away_stats.get('avg_away_pace', 100.0)

            # Calculate efficiency differentials
            home_eff_diff = home_rtg - home_drtg
            away_eff_diff = away_rtg - away_drtg

            # Create comprehensive feature set matching training format
            features = {
                'home_score': home_score,
                'away_score': away_score,
                'home_offensive_rating': home_rtg,
                'away_offensive_rating': away_rtg,
                'home_defensive_rating': home_drtg,
                'away_defensive_rating': away_drtg,
                'home_pace': home_pace,
                'away_pace': away_pace,
                'home_eff_diff': home_eff_diff,
                'away_eff_diff': away_eff_diff,
                'home_advantage': 3.5,  # Standard NBA home advantage
                'pace_differential': home_pace - away_pace,
                'offensive_quality': (home_rtg + away_rtg) / 2,
                'defensive_quality': (home_drtg + away_drtg) / 2,
                'expected_total': home_score + away_score,
                'game_pace': (home_pace + away_pace) / 2,
                'pace_differential_actual': home_pace - away_pace,
                'home_off_vs_away_def': home_rtg - away_drtg,
                'away_off_vs_home_def': away_rtg - home_drtg,
                'total_expected_scoring': home_score + away_score
            }

            return features

        except Exception as e:
            logger.warning(f"Error creating prediction features: {e}")
            return None

    def _analyze_teams(
        self,
        team1_stats: Dict[str, float],
        team2_stats: Dict[str, float],
        is_team1_home: bool
    ) -> Dict[str, Any]:
        """Analyze team performance for prediction explanation."""

        home_stats = team1_stats if is_team1_home else team2_stats
        away_stats = team2_stats if is_team1_home else team1_stats

        return {
            'home_team': {
                'name': 'Team1' if is_team1_home else 'Team2',
                'avg_points_scored': float(home_stats.get('HOME_SCORE', 0)),
                'avg_points_allowed': float(home_stats.get('AWAY_SCORE', 0)),
                'offensive_rating': float(home_stats.get('HOME_ORtg_sAvg', 110)),
                'defensive_rating': float(home_stats.get('HOME_DRtg_sAvg', 110)),
                'pace': float(home_stats.get('HOME_PACE', 100))
            },
            'away_team': {
                'name': 'Team2' if is_team1_home else 'Team1',
                'avg_points_scored': float(away_stats.get('AWAY_SCORE', 0)),
                'avg_points_allowed': float(away_stats.get('HOME_SCORE', 0)),
                'offensive_rating': float(away_stats.get('AWAY_ORtg_sAvg', 110)),
                'defensive_rating': float(away_stats.get('AWAY_DRtg_sAvg', 110)),
                'pace': float(away_stats.get('AWAY_PACE', 100))
            }
        }

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(self.trained_model.named_steps['ensemble'], 'feature_importances_'):
                # For RandomForest-like models
                importances = self.trained_model.named_steps['ensemble'].feature_importances_
                selected_features = self.trained_model.named_steps['selector'].get_feature_names_out()

                return dict(zip(selected_features, importances))
            else:
                # For VotingRegressor, return feature names with equal importance
                selected_features = self.trained_model.named_steps['selector'].get_feature_names_out()
                return {feat: 1.0/len(selected_features) for feat in selected_features}

        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}")
            return {}

    def _save_model(self) -> None:
        """Save trained model to disk."""
        try:
            model_path = self.model_path / "nba_prediction_pipeline.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.trained_model,
                    'feature_columns': self.feature_columns,
                    'metrics': self.metrics,
                    'training_date': datetime.now().isoformat()
                }, f)

            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.warning(f"Error saving model: {e}")

    def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            model_path = self.model_path / "nba_prediction_pipeline.pkl"
            if not model_path.exists():
                return False

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self.trained_model = data['model']
            self.feature_columns = data['feature_columns']
            self.metrics = data['metrics']
            self.is_trained = True

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'feature_columns_count': len(self.feature_columns) if self.feature_columns else 0,
            'model_type': 'VotingRegressor (RandomForest + XGBoost + GradientBoosting)',
            'data_source': 'Real NBA games dataset',
            'last_training': self.metrics.get('training_date', 'Not trained')
        }