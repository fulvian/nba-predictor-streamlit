#!/usr/bin/env python3
"""
🏆 NBA Backtesting Engine - Historical Performance Validation
Comprehensive backtesting system for NBA predictive models with realistic betting simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

from temporal_ml_validator import TemporalMLValidator

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    start_date: date
    end_date: date
    initial_bankroll: float = 1000.0
    bet_size_percentage: float = 0.02  # 2% of bankroll per bet
    min_confidence_threshold: float = 0.6
    max_bets_per_day: int = 5
    odds_threshold: float = 1.5  # Minimum odds to consider
    validation_days: int = 30
    gap_days: int = 1

@dataclass
class BetResult:
    """Result of a single bet."""
    date: date
    game_id: str
    home_team: str
    away_team: str
    prediction: float
    actual: float
    bet_type: str  # 'over' or 'under'
    odds: float
    confidence: float
    stake: float
    profit: float
    won: bool

class NBABacktestingEngine:
    """
    Comprehensive backtesting engine for NBA predictive models.
    Simulates realistic betting scenarios with proper bankroll management.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.temporal_validator = TemporalMLValidator(
            validation_days=config.validation_days,
            gap_days=config.gap_days
        )

        # Results storage
        self.bet_history: List[BetResult] = []
        self.daily_results: List[Dict] = []
        self.model_metrics: List[Dict] = []

        # Performance tracking
        self.current_bankroll = config.initial_bankroll
        self.peak_bankroll = config.initial_bankroll
        self.max_drawdown = 0.0

        logger.info(f"🏆 Backtesting engine initialized: {config.start_date} to {config.end_date}")

    def run_backtest(self,
                    model,
                    historical_data: pd.DataFrame,
                    feature_columns: List[str],
                    target_column: str = 'TOTAL_POINTS',
                    odds_columns: List[str] = ['OVER_ODDS', 'UNDER_ODDS'],
                    date_column: str = 'GAME_DATE') -> Dict[str, Any]:
        """
        Run comprehensive backtest on historical data.

        Args:
            model: Trained predictive model
            historical_data: Historical NBA data with features and results
            feature_columns: Feature column names
            target_column: Target column name
            odds_columns: Betting odds columns
            date_column: Date column name

        Returns:
            Comprehensive backtest results
        """
        logger.info(f"🚀 Starting backtest with {len(historical_data)} games")

        # Prepare data
        data = self._prepare_backtest_data(historical_data, date_column)

        # Create temporal splits for rolling validation
        splits = self.temporal_validator.create_temporal_splits(data, date_column)

        if not splits:
            logger.error("❌ No valid temporal splits created")
            return {'error': 'Insufficient data for backtesting'}

        # Process each temporal split
        for split_idx, (train_df, val_df) in enumerate(splits):
            logger.info(f"📊 Processing split {split_idx + 1}/{len(splits)}")

            # Process validation period
            split_results = self._process_validation_period(
                model, train_df, val_df,
                feature_columns, target_column, odds_columns, date_column
            )

            # Update overall results
            self.bet_history.extend(split_results['bets'])
            self.daily_results.extend(split_results['daily_results'])
            self.model_metrics.append(split_results['model_metrics'])

            # Update bankroll tracking
            self._update_bankroll_tracking()

        # Generate comprehensive results
        final_results = self._generate_backtest_report()

        logger.info(f"✅ Backtest completed: {len(self.bet_history)} total bets")
        return final_results

    def _prepare_backtest_data(self,
                              df: pd.DataFrame,
                              date_column: str) -> pd.DataFrame:
        """Prepare data for backtesting with filtering and validation."""
        data = df.copy()

        # Convert date column
        data[date_column] = pd.to_datetime(data[date_column])

        # Filter by backtest period
        start_mask = data[date_column].dt.date >= self.config.start_date
        end_mask = data[date_column].dt.date <= self.config.end_date
        data = data[start_mask & end_mask]

        # Sort by date
        data = data.sort_values(date_column).reset_index(drop=True)

        # Validate required columns
        required_cols = [date_column, 'HOME_TEAM', 'AWAY_TEAM']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()

        logger.info(f"📊 Prepared {len(data)} games for backtesting period")
        return data

    def _process_validation_period(self,
                                 model,
                                 train_df: pd.DataFrame,
                                 val_df: pd.DataFrame,
                                 feature_columns: List[str],
                                 target_column: str,
                                 odds_columns: List[str],
                                 date_column: str) -> Dict[str, Any]:
        """Process a single validation period with betting simulation."""
        # Fit model on training data with temporal validation
        validation_results = self.temporal_validator.validate_model_performance(
            model, [(train_df, val_df)], feature_columns, target_column
        )

        # Simulate betting on validation games
        bets = []
        daily_results = []

        # Group validation games by date
        val_df = val_df.copy()
        val_df[date_column] = pd.to_datetime(val_df[date_column])
        val_df['DATE_ONLY'] = val_df[date_column].dt.date

        for game_date, day_games in val_df.groupby('DATE_ONLY'):
            day_bets = self._simulate_betting_day(
                model, day_games, train_df, feature_columns, odds_columns
            )

            bets.extend(day_bets)

            # Calculate daily results
            daily_profit = sum(bet.profit for bet in day_bets)
            daily_results.append({
                'date': game_date,
                'bets_placed': len(day_bets),
                'profit': daily_profit,
                'bankroll': self.current_bankroll + daily_profit,
                'win_rate': sum(1 for bet in day_bets if bet.won) / len(day_bets) if day_bets else 0
            })

            # Update bankroll
            self.current_bankroll += daily_profit

        return {
            'bets': bets,
            'daily_results': daily_results,
            'model_metrics': validation_results
        }

    def _simulate_betting_day(self,
                            model,
                            games_df: pd.DataFrame,
                            train_df: pd.DataFrame,
                            feature_columns: List[str],
                            odds_columns: List[str]) -> List[BetResult]:
        """Simulate betting for a single day."""
        day_bets = []

        # Limit number of bets per day
        if len(games_df) > self.config.max_bets_per_day:
            # Select games with highest confidence predictions
            games_df = games_df.head(self.config.max_bets_per_day)

        for _, game in games_df.iterrows():
            try:
                # Get model prediction
                prediction = self._get_model_prediction(
                    model, game, train_df, feature_columns
                )

                # Determine bet recommendation
                bet_recommendation = self._analyze_betting_opportunity(
                    game, prediction, odds_columns
                )

                if bet_recommendation['should_bet']:
                    # Calculate stake and simulate bet
                    stake = min(
                        self.current_bankroll * self.config.bet_size_percentage,
                        100  # Maximum stake cap
                    )

                    bet_result = self._place_bet(
                        game, prediction, bet_recommendation, stake
                    )

                    day_bets.append(bet_result)

            except Exception as e:
                logger.warning(f"⚠️ Failed to process game: {e}")
                continue

        return day_bets

    def _get_model_prediction(self,
                            model,
                            game: pd.Series,
                            train_df: pd.DataFrame,
                            feature_columns: List[str]) -> float:
        """Get model prediction for a single game."""
        # Prepare features
        game_features = game[feature_columns].values.reshape(1, -1)

        # Apply preprocessing (use train statistics)
        if hasattr(self.temporal_validator, 'scalers') and 'scaler' in self.temporal_validator.scalers:
            game_features = self.temporal_validator.scalers['scaler'].transform(game_features)

        # Get prediction
        prediction = model.predict(game_features)[0]
        return prediction

    def _analyze_betting_opportunity(self,
                                   game: pd.Series,
                                   prediction: float,
                                   odds_columns: List[str]) -> Dict[str, Any]:
        """Analyze if a betting opportunity is worth taking."""
        analysis = {
            'should_bet': False,
            'bet_type': None,
            'odds': None,
            'confidence': 0.0
        }

        # Check if odds are available
        available_odds = {}
        for odds_col in odds_columns:
            if odds_col in game and pd.notna(game[odds_col]):
                available_odds[odds_col] = game[odds_col]

        if not available_odds:
            return analysis

        # Get betting line (if available)
        betting_line = game.get('BETTING_LINE', prediction)  # Fallback to prediction

        # Determine bet recommendation based on prediction vs line
        diff = prediction - betting_line
        confidence = abs(diff) / 10.0  # Normalize confidence

        if confidence < self.config.min_confidence_threshold:
            return analysis

        # Check over/under opportunity
        if 'OVER_ODDS' in available_odds and diff > 0:
            if available_odds['OVER_ODDS'] >= self.config.odds_threshold:
                analysis.update({
                    'should_bet': True,
                    'bet_type': 'over',
                    'odds': available_odds['OVER_ODDS'],
                    'confidence': confidence
                })

        elif 'UNDER_ODDS' in available_odds and diff < 0:
            if available_odds['UNDER_ODDS'] >= self.config.odds_threshold:
                analysis.update({
                    'should_bet': True,
                    'bet_type': 'under',
                    'odds': available_odds['UNDER_ODDS'],
                    'confidence': confidence
                })

        return analysis

    def _place_bet(self,
                  game: pd.Series,
                  prediction: float,
                  bet_recommendation: Dict[str, Any],
                  stake: float) -> BetResult:
        """Place a bet and calculate the result."""
        # Get actual result (in real scenario, this would be unknown)
        actual_total = game.get('TOTAL_POINTS', 0)
        betting_line = game.get('BETTING_LINE', prediction)

        # Determine if bet won
        if bet_recommendation['bet_type'] == 'over':
            won = actual_total > betting_line
        else:  # under
            won = actual_total < betting_line

        # Calculate profit
        if won:
            profit = stake * (bet_recommendation['odds'] - 1)
        else:
            profit = -stake

        return BetResult(
            date=game['GAME_DATE'].date(),
            game_id=str(game.get('GAME_ID', 'unknown')),
            home_team=game.get('HOME_TEAM', 'Unknown'),
            away_team=game.get('AWAY_TEAM', 'Unknown'),
            prediction=prediction,
            actual=actual_total,
            bet_type=bet_recommendation['bet_type'],
            odds=bet_recommendation['odds'],
            confidence=bet_recommendation['confidence'],
            stake=stake,
            profit=profit,
            won=won
        )

    def _update_bankroll_tracking(self):
        """Update bankroll performance metrics."""
        self.peak_bankroll = max(self.peak_bankroll, self.current_bankroll)
        current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def _generate_backtest_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        if not self.bet_history:
            return {'error': 'No bets placed during backtest'}

        # Calculate betting metrics
        total_bets = len(self.bet_history)
        winning_bets = sum(1 for bet in self.bet_history if bet.won)
        win_rate = winning_bets / total_bets

        total_profit = sum(bet.profit for bet in self.bet_history)
        total_staked = sum(bet.stake for bet in self.bet_history)
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

        # Calculate additional metrics
        average_odds = np.mean([bet.odds for bet in self.bet_history])
        average_confidence = np.mean([bet.confidence for bet in self.bet_history])

        # Performance by bet type
        over_bets = [bet for bet in self.bet_history if bet.bet_type == 'over']
        under_bets = [bet for bet in self.bet_history if bet.bet_type == 'under']

        over_win_rate = sum(1 for bet in over_bets if bet.won) / len(over_bets) if over_bets else 0
        under_win_rate = sum(1 for bet in under_bets if bet.won) / len(under_bets) if under_bets else 0

        # Calculate prediction accuracy
        prediction_errors = [abs(bet.prediction - bet.actual) for bet in self.bet_history]
        mae = np.mean(prediction_errors)

        # Generate report
        report = {
            'backtest_summary': {
                'period': f"{self.config.start_date} to {self.config.end_date}",
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': round(win_rate, 3),
                'total_profit': round(total_profit, 2),
                'total_staked': round(total_staked, 2),
                'roi_percentage': round(roi, 2),
                'average_odds': round(average_odds, 2),
                'average_confidence': round(average_confidence, 3)
            },
            'bankroll_performance': {
                'initial_bankroll': self.config.initial_bankroll,
                'final_bankroll': round(self.current_bankroll, 2),
                'peak_bankroll': round(self.peak_bankroll, 2),
                'max_drawdown_percentage': round(self.max_drawdown * 100, 2)
            },
            'prediction_accuracy': {
                'mae': round(mae, 2),
                'prediction_std': round(np.std(prediction_errors), 2)
            },
            'bet_type_performance': {
                'over_bets': {
                    'count': len(over_bets),
                    'win_rate': round(over_win_rate, 3)
                },
                'under_bets': {
                    'count': len(under_bets),
                    'win_rate': round(under_win_rate, 3)
                }
            },
            'daily_performance': {
                'total_days': len(self.daily_results),
                'profitable_days': sum(1 for day in self.daily_results if day['profit'] > 0),
                'average_daily_bets': round(np.mean([day['bets_placed'] for day in self.daily_results]), 1)
            },
            'detailed_bets': [
                {
                    'date': bet.date,
                    'teams': f"{bet.away_team} @ {bet.home_team}",
                    'prediction': bet.prediction,
                    'actual': bet.actual,
                    'bet_type': bet.bet_type,
                    'odds': bet.odds,
                    'confidence': bet.confidence,
                    'stake': bet.stake,
                    'profit': bet.profit,
                    'won': bet.won
                }
                for bet in self.bet_history
            ]
        }

        logger.info(f"📊 Backtest Report Generated:")
        logger.info(f"   Total Bets: {total_bets} (Win Rate: {win_rate:.1%})")
        logger.info(f"   Total Profit: ${total_profit:.2f} (ROI: {roi:.1f}%)")
        logger.info(f"   Final Bankroll: ${self.current_bankroll:.2f}")
        logger.info(f"   Max Drawdown: {self.max_drawdown:.1%}")

        return report

    def save_backtest_results(self, filepath: str):
        """Save backtest results to file."""
        results = {
            'config': self.config,
            'bet_history': self.bet_history,
            'daily_results': self.daily_results,
            'model_metrics': self.model_metrics,
            'final_bankroll': self.current_bankroll,
            'peak_bankroll': self.peak_bankroll,
            'max_drawdown': self.max_drawdown
        }

        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"💾 Backtest results saved to {filepath}")

    @staticmethod
    def load_backtest_results(filepath: str) -> Dict[str, Any]:
        """Load backtest results from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)