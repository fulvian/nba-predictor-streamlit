#!/usr/bin/env python3
"""
🏀 NBA Feature Engineering Pipeline
Context7-compliant feature extraction and engineering for NBA predictive analytics.
Implements advanced basketball metrics, player impact scores, team chemistry analysis,
and injury impact factors based on established sports analytics best practices.
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from nba_predictor.core.data_store import UnifiedDataStore
from nba_predictor.core.roster_injury_schemas import InjuryInfo, InjuryStatus

logger = logging.getLogger(__name__)

@dataclass
class NBAMetricsConfig:
    """Configuration for NBA metrics calculation."""

    # Rolling averages windows
    player_rolling_window: int = 10
    team_rolling_window: int = 5

    # Advanced metrics weights
    per_weight: float = 0.4
    ts_weight: float = 0.3
    eff_weight: float = 0.3

    # Team chemistry factors
    lineup_continuity_weight: float = 0.5
    experience_balance_weight: float = 0.3
    role_distribution_weight: float = 0.2

    # Injury impact factors
    star_player_impact: float = 2.0
    starter_impact: float = 1.5
    bench_impact: float = 1.0

    # Feature scaling
    scale_features: bool = True
    handle_missing: str = "interpolate"  # "interpolate", "zero", "mean"

class PlayerMetricsExtractor:
    """Extracts advanced player metrics from raw NBA data."""

    def __init__(self, config: NBAMetricsConfig):
        self.config = config
        self.team_abbreviations = self._get_team_abbreviations()

    def _get_team_abbreviations(self) -> Dict[str, str]:
        """Get NBA team abbreviation mappings."""
        return {
            'Atlanta': 'ATL', 'Boston': 'BOS', 'Brooklyn': 'BKN', 'Charlotte': 'CHA',
            'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN',
            'Detroit': 'DET', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND',
            'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis': 'MEM',
            'Miami': 'MIA', 'Milwaukee': 'MIL', 'Minnesota': 'MIN', 'New Orleans': 'NOP',
            'New York': 'NYK', 'Oklahoma City': 'OKC', 'Orlando': 'ORL', 'Philadelphia': 'PHI',
            'Phoenix': 'PHX', 'Portland': 'POR', 'Sacramento': 'SAC', 'San Antonio': 'SAS',
            'Toronto': 'TOR', 'Utah': 'UTA', 'Washington': 'WAS'
        }

    def calculate_advanced_metrics(self, player_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate advanced basketball metrics for players.

        Based on established basketball analytics formulas:
        - PER (Player Efficiency Rating)
        - True Shooting Percentage (TS%)
        - Effective Field Goal Percentage (eFG%)
        - Usage Rate (USG%)
        - Box Plus/Minus (BPM)
        """
        logger.info("📊 Calculating advanced player metrics")

        # Convert to pandas for complex calculations
        pdf = player_df.to_pandas()

        # Basic defensive metrics
        if 'PTS' in pdf.columns and 'FGA' in pdf.columns and 'FTA' in pdf.columns:
            # True Shooting Percentage
            ts_points = pdf['PTS'].astype(float)
            ts_fga = pdf['FGA'].astype(float)
            ts_fta = pdf['FTA'].astype(float)
            pdf['TS_PCT'] = ts_points / (2 * (ts_fga + 0.44 * ts_fta))
            pdf['TS_PCT'] = pdf['TS_PCT'].fillna(0.0)

        if 'FGM' in pdf.columns and 'FGA' in pdf.columns and 'FG3M' in pdf.columns:
            # Effective Field Goal Percentage
            efg_fgm = pdf['FGM'].astype(float)
            efg_fga = pdf['FGA'].astype(float)
            efg_3pm = pdf['FG3M'].astype(float)
            pdf['EFG_PCT'] = (efg_fgm + 0.5 * efg_3pm) / efg_fga
            pdf['EFG_PCT'] = pdf['EFG_PCT'].fillna(0.0)

        # Usage Rate approximation
        required_cols = ['FGA', 'FTA', 'TOV', 'MIN']
        if all(col in pdf.columns for col in required_cols):
            fga = pdf['FGA'].astype(float)
            fta = pdf['FTA'].astype(float)
            tov = pdf['TOV'].astype(float)
            minutes = pdf['MIN'].astype(float)

            # Simplified usage rate (scaled by minutes)
            pdf['USAGE_RATE'] = (fga + 0.44 * fta + tov) / minutes
            pdf['USAGE_RATE'] = pdf['USAGE_RATE'].fillna(0.0)

        # Player Efficiency Rating (simplified)
        per_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FGM', 'FTM', 'TOV', 'FGA', 'FTA']
        if all(col in pdf.columns for col in per_cols):
            pts = pdf['PTS'].astype(float)
            ast = pdf['AST'].astype(float)
            reb = pdf['REB'].astype(float)
            stl = pdf['STL'].astype(float)
            blk = pdf['BLK'].astype(float)
            fgm = pdf['FGM'].astype(float)
            ftm = pdf['FTM'].astype(float)
            tov = pdf['TOV'].astype(float)
            fga = pdf['FGA'].astype(float)
            fta = pdf['FTA'].astype(float)

            # Simplified PER calculation
            per_numerator = (pts + ast + reb + stl + blk + fgm + ftm - tov - fga - fta)
            pdf['PER'] = per_numerator / (pdf['MIN'].astype(float) + 1)
            pdf['PER'] = pdf['PER'].fillna(0.0)

        # Impact Score (weighted combination)
        if all(col in pdf.columns for col in ['TS_PCT', 'EFG_PCT', 'PER']):
            pdf['IMPACT_SCORE'] = (
                self.config.per_weight * pdf['PER'].fillna(0) +
                self.config.ts_weight * pdf['TS_PCT'].fillna(0) * 100 +
                self.config.eff_weight * pdf['EFG_PCT'].fillna(0) * 100
            )

        # Convert back to polars
        return pl.from_pandas(pdf)

    def extract_rolling_features(self, player_df: pl.DataFrame,
                                player_id: int, window: Optional[int] = None) -> pl.DataFrame:
        """
        Extract rolling average features for a specific player.
        """
        if window is None:
            window = self.config.player_rolling_window

        logger.info(f"📈 Extracting rolling features for player {player_id} (window: {window})")

        # Sort by date
        if 'GAME_DATE' in player_df.columns:
            player_df = player_df.sort('GAME_DATE')

        # Calculate rolling averages
        numeric_cols = [col for col in player_df.columns if player_df[col].dtype in [pl.Float64, pl.Int64]]

        rolling_features = {}
        for col in numeric_cols:
            if col not in ['PLAYER_ID', 'GAME_ID', 'TEAM_ID']:
                rolling_features[f'{col}_ROLLING_{window}'] = player_df[col].rolling_mean(window_size=window)

        # Add rolling features to dataframe
        result_df = player_df.with_columns([
            pl.Series(name, rolling_features[name]) for name in rolling_features
        ])

        return result_df

    def calculate_form_trends(self, player_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate player form trends (improvement/decline patterns).
        """
        logger.info("📊 Calculating player form trends")

        # Sort by date
        if 'GAME_DATE' in player_df.columns:
            player_df = player_df.sort('GAME_DATE')

        # Calculate trend indicators
        result_df = player_df.with_columns([
            # Points trend (last 5 games vs previous 5)
            pl.col('PTS').rolling_mean(window_size=5).alias('PTS_RECENT_5'),
            pl.col('PTS').rolling_mean(window_size=10).shift(5).alias('PTS_PREVIOUS_5'),
        ])

        # Calculate trend differences
        result_df = result_df.with_columns([
            (pl.col('PTS_RECENT_5') - pl.col('PTS_PREVIOUS_5')).alias('PTS_TREND'),
            ((pl.col('PTS_RECENT_5') / pl.col('PTS_PREVIOUS_5')) - 1).alias('PTS_TREND_PCT')
        ])

        # Handle division by zero
        result_df = result_df.with_columns([
            pl.when(pl.col('PTS_PREVIOUS_5') == 0)
            .then(0.0)
            .otherwise(pl.col('PTS_TREND_PCT'))
            .alias('PTS_TREND_PCT')
        ])

        return result_df

class TeamChemistryCalculator:
    """Calculates team chemistry and cohesion metrics."""

    def __init__(self, config: NBAMetricsConfig):
        self.config = config

    def calculate_lineup_continuity(self, games_df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate lineup continuity - how consistent team lineups are.
        """
        logger.info("🔄 Calculating lineup continuity metrics")

        # Group by team and calculate lineup consistency
        team_continuity = {}

        if 'TEAM_ID' in games_df.columns:
            teams = games_df['TEAM_ID'].unique()

            for team_id in teams:
                team_games = games_df.filter(pl.col('TEAM_ID') == team_id)

                # Calculate lineup diversity (simplified)
                if 'PLAYER_ID' in team_games.columns:
                    unique_players = team_games['PLAYER_ID'].n_unique()
                    total_games = len(team_games)

                    # Continuity score (higher = more consistent)
                    continuity_score = 1.0 - (unique_players / (total_games * 5))  # 5 players per game
                    team_continuity[str(team_id)] = max(0.0, continuity_score)

        return team_continuity

    def calculate_experience_balance(self, roster_df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate experience balance within teams.
        """
        logger.info("⚖️ Calculating experience balance metrics")

        experience_balance = {}

        if 'TEAM_ID' in roster_df.columns and 'EXPERIENCE' in roster_df.columns:
            teams = roster_df['TEAM_ID'].unique()

            for team_id in teams:
                team_roster = roster_df.filter(pl.col('TEAM_ID') == team_id)

                if len(team_roster) > 0:
                    # Calculate experience distribution
                    experience_values = team_roster['EXPERIENCE'].to_list()

                    # Balance score (lower std = more balanced)
                    exp_mean = np.mean(experience_values)
                    exp_std = np.std(experience_values)

                    # Normalize balance score
                    balance_score = 1.0 / (1.0 + exp_std) if exp_mean > 0 else 0.0
                    experience_balance[str(team_id)] = balance_score

        return experience_balance

    def calculate_role_distribution(self, player_stats_df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate role distribution balance within teams.
        """
        logger.info("🎭 Calculating role distribution metrics")

        role_scores = {}

        if 'TEAM_ID' in player_stats.columns:
            teams = player_stats_df['TEAM_ID'].unique()

            for team_id in teams:
                team_players = player_stats_df.filter(pl.col('TEAM_ID') == team_id)

                if len(team_players) > 0:
                    # Calculate role distribution based on minutes and usage
                    if 'MIN' in team_players.columns and 'USAGE_RATE' in team_players.columns:
                        minutes = team_players['MIN'].to_list()
                        usage = team_players['USAGE_RATE'].to_list()

                        # Calculate Gini coefficient for role distribution
                        minutes_gini = self._calculate_gini(minutes)
                        usage_gini = self._calculate_gini(usage)

                        # Balance score (lower Gini = more balanced)
                        role_balance = (2.0 - minutes_gini - usage_gini) / 2.0
                        role_scores[str(team_id)] = max(0.0, min(1.0, role_balance))

        return role_scores

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if len(values) == 0:
            return 0.0

        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)

        if cumsum[-1] == 0:
            return 0.0

        return (2 * sum((i + 1) * v for i, v in enumerate(sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

class InjuryImpactAnalyzer:
    """Analyzes the impact of injuries on team performance."""

    def __init__(self, config: NBAMetricsConfig):
        self.config = config

    def calculate_injury_impact_scores(self, injuries: List[InjuryInfo]) -> Dict[str, float]:
        """
        Calculate injury impact scores for teams.
        """
        logger.info("🏥 Calculating injury impact scores")

        team_impacts = {}

        for injury in injuries:
            team_id = str(injury.team_id)

            # Base impact by status
            status_impacts = {
                InjuryStatus.OUT: 1.0,
                InjuryStatus.DOUBTFUL: 0.75,
                InjuryStatus.QUESTIONABLE: 0.5,
                InjuryStatus.PROBABLE: 0.25,
                InjuryStatus.AVAILABLE: 0.0
            }

            base_impact = status_impacts.get(injury.injury_status, 0.0)

            # Adjust by player importance (simplified)
            player_impact = self.config.star_player_impact  # Default to star player impact

            # Calculate total impact
            total_impact = base_impact * player_impact

            if team_id not in team_impacts:
                team_impacts[team_id] = 0.0
            team_impacts[team_id] += total_impact

        return team_impacts

    def calculate_availability_scores(self, injuries: List[InjuryInfo],
                                    roster_df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate team availability scores based on injuries.
        """
        logger.info("✅ Calculating team availability scores")

        team_availability = {}

        # Group injuries by team
        team_injuries = {}
        for injury in injuries:
            team_id = str(injury.team_id)
            if team_id not in team_injuries:
                team_injuries[team_id] = []
            team_injuries[team_id].append(injury)

        # Calculate availability for each team
        for team_id, team_injury_list in team_injuries.items():
            total_players = len(roster_df.filter(pl.col('TEAM_ID') == int(team_id)))

            if total_players > 0:
                available_players = 0

                for injury in team_injury_list:
                    if injury.injury_status in [InjuryStatus.AVAILABLE, InjuryStatus.PROBABLE]:
                        available_players += 1

                # Availability score (percentage of available players)
                availability_score = available_players / total_players
                team_availability[team_id] = availability_score

        return team_availability

class NBAFeatureEngineer:
    """
    Main feature engineering pipeline for NBA predictive analytics.

    Context7-compliant implementation that transforms raw NBA data into
    comprehensive feature sets for machine learning models.
    """

    def __init__(self, data_store: UnifiedDataStore, config: Optional[NBAMetricsConfig] = None):
        self.data_store = data_store
        self.config = config or NBAMetricsConfig()

        # Initialize component processors
        self.player_extractor = PlayerMetricsExtractor(self.config)
        self.chemistry_calculator = TeamChemistryCalculator(self.config)
        self.injury_analyzer = InjuryImpactAnalyzer(self.config)

        logger.info("🏀 NBA Feature Engineer initialized")

    def process_player_features(self, season: str, team_id: Optional[int] = None) -> pl.DataFrame:
        """
        Process and engineer player-level features.
        """
        logger.info(f"👤 Processing player features for {season}")

        try:
            # Load player statistics from data store - use date range for season
            season_start = f"{season.split('-')[0]}-10-01"
            season_end = f"{season.split('-')[1]}-04-30"

            player_stats = self.data_store.get_player_stats(date_range=(season_start, season_end))

  
            # Calculate advanced metrics
            enhanced_stats = self.player_extractor.calculate_advanced_metrics(player_stats)

            # Extract rolling features for each player
            if 'PLAYER_ID' in enhanced_stats.columns:
                player_ids = enhanced_stats['PLAYER_ID'].unique()

                rolling_features = []
                for pid in player_ids:
                    player_data = enhanced_stats.filter(pl.col('PLAYER_ID') == pid)
                    player_with_rolling = self.player_extractor.extract_rolling_features(player_data, pid)
                    rolling_features.append(player_with_rolling)

                if rolling_features:
                    enhanced_stats = pl.concat(rolling_features)

            # Calculate form trends
            final_features = self.player_extractor.calculate_form_trends(enhanced_stats)

            logger.info(f"✅ Processed player features: {len(final_features)} records")
            return final_features

        except Exception as e:
            logger.error(f"❌ Error processing player features: {e}")
            return pl.DataFrame()

    def process_team_features(self, season: str) -> pl.DataFrame:
        """
        Process and engineer team-level features.
        """
        logger.info(f"🏀 Processing team features for {season}")

        try:
            # Load team statistics - use date range for season
            season_start = f"{season.split('-')[0]}-10-01"
            season_end = f"{season.split('-')[1]}-04-30"

            team_stats = self.data_store.get_team_stats(date_range=(season_start, season_end))

            if team_stats is None or len(team_stats) == 0:
                logger.warning("No team statistics available")
                return pl.DataFrame()

            # Calculate team chemistry metrics
            lineup_continuity = self.chemistry_calculator.calculate_lineup_continuity(team_stats)

            # Load roster data for experience balance - collect all team rosters
            from ..nba_predictor.core.roster_injury_store_extensions import TeamRoster
            all_rosters = []
            for team_id in range(1610612737, 1610612767):  # NBA team ID range
                try:
                    roster = self.data_store.get_team_roster(team_id, season)
                    if roster is not None:
                        all_rosters.append(roster)
                except Exception:
                    continue

            experience_balance = {}
            if all_rosters:
                # Convert rosters to dataframe for processing
                roster_data = pl.DataFrame({
                    'TEAM_ID': [r.team_id for r in all_rosters],
                    'EXPERIENCE': [sum([p.experience for p in r.players]) / len(r.players) for r in all_rosters if r.players]
                })
                experience_balance = self.chemistry_calculator.calculate_experience_balance(roster_data)

            # Combine team features
            team_features = team_stats.with_columns([
                pl.col('TEAM_ID').cast(pl.Utf64)
            ])

            # Add chemistry scores
            continuity_expr = pl.when(pl.col('TEAM_ID').cast(pl.Utf64).is_in(list(lineup_continuity.keys())))
            continuity_expr = continuity_expr.then(pl.col('TEAM_ID').cast(pl.Utf64).map_dict(lineup_continuity))
            continuity_expr = continuity_expr.otherwise(0.0)

            team_features = team_features.with_columns([
                continuity_expr.alias('LINEUP_CONTINUITY')
            ])

            experience_expr = pl.when(pl.col('TEAM_ID').cast(pl.Utf64).is_in(list(experience_balance.keys())))
            experience_expr = experience_expr.then(pl.col('TEAM_ID').cast(pl.Utf64).map_dict(experience_balance))
            experience_expr = experience_expr.otherwise(0.0)

            team_features = team_features.with_columns([
                experience_expr.alias('EXPERIENCE_BALANCE')
            ])

            logger.info(f"✅ Processed team features: {len(team_features)} records")
            return team_features

        except Exception as e:
            logger.error(f"❌ Error processing team features: {e}")
            return pl.DataFrame()

    def process_injury_features(self, season: str) -> pl.DataFrame:
        """
        Process injury-related features.
        """
        logger.info(f"🏥 Processing injury features for {season}")

        try:
            # Load injury data - collect all team injuries
            all_injuries = []
            for team_id in range(1610612737, 1610612767):  # NBA team ID range
                try:
                    team_injuries = self.data_store.get_team_injuries(team_id, season)
                    if team_injuries:
                        all_injuries.extend(team_injuries)
                except Exception:
                    continue

            if not all_injuries:
                logger.warning("No injury data available")
                return pl.DataFrame()

            # Calculate injury impacts
            injury_impacts = self.injury_analyzer.calculate_injury_impact_scores(all_injuries)

            # Create roster data for availability calculation
            roster_data = pl.DataFrame({
                'TEAM_ID': list(set([inj.team_id for inj in all_injuries])),
                'TOTAL_PLAYERS': [15] * len(set([inj.team_id for inj in all_injuries]))  # Approximate roster size
            })

            availability_scores = self.injury_analyzer.calculate_availability_scores(all_injuries, roster_data)

            # Create injury features dataframe
            injury_features = pl.DataFrame({
                'TEAM_ID': list(injury_impacts.keys()),
                'INJURY_IMPACT': list(injury_impacts.values()),
                'AVAILABILITY_SCORE': [
                    availability_scores.get(team_id, 1.0)
                    for team_id in injury_impacts.keys()
                ]
            })

            logger.info(f"✅ Processed injury features: {len(injury_features)} teams")
            return injury_features

        except Exception as e:
            logger.error(f"❌ Error processing injury features: {e}")
            return pl.DataFrame()

    def create_training_dataset(self, season: str, target_variable: str = 'WIN') -> pl.DataFrame:
        """
        Create comprehensive training dataset combining all features.
        """
        logger.info(f"🎯 Creating training dataset for {season}")

        try:
            # Process individual feature sets
            player_features = self.process_player_features(season)
            team_features = self.process_team_features(season)
            injury_features = self.process_injury_features(season)

            # Load game results for target variable - use date range
            season_start = f"{season.split('-')[0]}-10-01"
            season_end = f"{season.split('-')[1]}-04-30"

            game_results = self.data_store.get_games_data(date_range=(season_start, season_end))

            if game_results is None or len(game_results) == 0:
                logger.warning("No game results available for target variable")
                return pl.DataFrame()

            # Combine features (simplified merging)
            # In a real implementation, this would involve more complex feature alignment
            if len(team_features) > 0 and len(injury_features) > 0:
                # Merge team and injury features
                combined_features = team_features.join(
                    injury_features,
                    on='TEAM_ID',
                    how='left'
                )

                # Fill missing injury features with defaults
                combined_features = combined_features.with_columns([
                    pl.col('INJURY_IMPACT').fill_null(0.0),
                    pl.col('AVAILABILITY_SCORE').fill_null(1.0)
                ])

                logger.info(f"✅ Created training dataset: {len(combined_features)} records")
                return combined_features
            else:
                logger.warning("Insufficient features for training dataset")
                return pl.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error creating training dataset: {e}")
            return pl.DataFrame()

    def save_features(self, features: pl.DataFrame, feature_type: str, season: str) -> bool:
        """
        Save engineered features to data store.
        """
        try:
            filename = f"{feature_type}_features_{season.replace('-', '_')}.parquet"
            filepath = self.data_store.base_path / "features" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            features.write_parquet(filepath)

            logger.info(f"💾 Saved {feature_type} features: {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving features: {e}")
            return False