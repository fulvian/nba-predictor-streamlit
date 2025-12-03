#!/usr/bin/env python3
"""
🕒 Temporal ML Validator - Robust Time-Series Validation for NBA Predictions
Addresses critical data leakage issues in sports predictions with proper temporal validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TemporalMLValidator:
    """
    Advanced temporal validation system for NBA predictive models.
    Prevents data leakage and ensures realistic performance evaluation.
    """

    def __init__(self,
                 min_train_days: int = 30,
                 validation_days: int = 7,
                 gap_days: int = 1,
                 n_splits: int = 5):
        """
        Initialize temporal validator with NBA-specific parameters.

        Args:
            min_train_days: Minimum days of training data required
            validation_days: Days for each validation fold
            gap_days: Gap between train and validation (prevents leakage)
            n_splits: Number of temporal splits
        """
        self.min_train_days = min_train_days
        self.validation_days = validation_days
        self.gap_days = gap_days
        self.n_splits = n_splits

        # Feature preprocessors (fit only on training data)
        self.scalers = {}
        self.feature_stats = {}

        logger.info(f"🕒 TemporalMLValidator initialized: {n_splits} splits, {validation_days} days validation")

    def create_temporal_splits(self,
                              df: pd.DataFrame,
                              date_column: str = 'GAME_DATE') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create temporally-aware train/validation splits.

        Args:
            df: Dataset with date column
            date_column: Name of date column

        Returns:
            List of (train_df, val_df) tuples with chronological splits
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date (critical for temporal validation)
        df = df.sort_values(date_column).reset_index(drop=True)

        splits = []
        min_date = df[date_column].min()
        max_date = df[date_column].max()

        total_days = (max_date - min_date).days
        required_days = self.min_train_days + self.gap_days + self.validation_days

        if total_days < required_days:
            logger.warning(f"⚠️ Insufficient data: {total_days} days < {required_days} days required")
            return []

        # Create chronological splits
        for split_idx in range(self.n_splits):
            # Calculate split boundaries
            train_start = min_date
            train_end = train_start + timedelta(days=self.min_train_days + split_idx * 14)  # 2-week increments
            gap_end = train_end + timedelta(days=self.gap_days)
            val_end = gap_end + timedelta(days=self.validation_days)

            # Skip if we don't have enough data
            if val_end > max_date:
                break

            # Create splits
            train_mask = (df[date_column] >= train_start) & (df[date_column] < train_end)
            val_mask = (df[date_column] >= gap_end) & (df[date_column] < val_end)

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

            if len(train_df) > 0 and len(val_df) > 0:
                splits.append((train_df, val_df))
                logger.info(f"📊 Split {split_idx + 1}: Train {len(train_df)} games ({train_start.date()} to {train_end.date()}), "
                           f"Val {len(val_df)} games ({gap_end.date()} to {val_end.date()})")

        return splits

    def fit_preprocessors(self,
                        train_df: pd.DataFrame,
                        feature_columns: List[str],
                        scaler_type: str = 'robust') -> Dict[str, Any]:
        """
        Fit preprocessing scalers ONLY on training data to prevent leakage.

        Args:
            train_df: Training data
            feature_columns: List of feature column names
            scaler_type: 'standard', 'robust', or 'none'

        Returns:
            Dictionary of fitted preprocessors
        """
        preprocessors = {}

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            logger.info("📊 No scaling applied")
            return preprocessors

        # Fit scaler ONLY on training data
        train_features = train_df[feature_columns].select_dtypes(include=[np.number])

        if len(train_features.columns) > 0:
            scaler.fit(train_features)
            preprocessors['scaler'] = scaler

            # Store feature statistics for monitoring
            self.feature_stats = {
                'train_mean': train_features.mean().to_dict(),
                'train_std': train_features.std().to_dict(),
                'feature_names': feature_columns
            }

            logger.info(f"📊 Fitted {scaler_type} scaler on {len(train_features)} features")
            logger.info(f"   Feature mean range: [{train_features.mean().min():.3f}, {train_features.mean().max():.3f}]")
            logger.info(f"   Feature std range: [{train_features.std().min():.3f}, {train_features.std().max():.3f}]")

        return preprocessors

    def apply_preprocessing(self,
                           df: pd.DataFrame,
                           feature_columns: List[str],
                           preprocessors: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply fitted preprocessors to data (train or validation).

        Args:
            df: Data to preprocess
            feature_columns: Feature column names
            preprocessors: Fitted preprocessors from fit_preprocessors()

        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()

        if 'scaler' in preprocessors:
            scaler = preprocessors['scaler']
            numeric_features = df_processed[feature_columns].select_dtypes(include=[np.number])

            if len(numeric_features.columns) > 0:
                # Apply scaling transformation
                scaled_features = scaler.transform(numeric_features)
                df_processed[numeric_features.columns] = scaled_features

                logger.debug(f"📊 Applied scaling to {len(numeric_features.columns)} features")

        return df_processed

    def validate_model_performance(self,
                                  model,
                                  splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                                  feature_columns: List[str],
                                  target_column: str,
                                  scaler_type: str = 'robust') -> Dict[str, Any]:
        """
        Comprehensive temporal validation with proper preprocessing.

        Args:
            model: ML model to validate
            splits: Temporal splits from create_temporal_splits()
            feature_columns: Feature column names
            target_column: Target column name
            scaler_type: Type of scaler to use

        Returns:
            Comprehensive validation results
        """
        all_predictions = []
        all_actuals = []
        fold_metrics = []

        logger.info(f"🚀 Starting temporal validation with {len(splits)} folds")

        for fold_idx, (train_df, val_df) in enumerate(splits):
            logger.info(f"📊 Processing Fold {fold_idx + 1}/{len(splits)}")

            try:
                # Step 1: Fit preprocessors ONLY on training data
                preprocessors = self.fit_preprocessors(train_df, feature_columns, scaler_type)

                # Step 2: Apply preprocessing to both train and validation
                train_processed = self.apply_preprocessing(train_df, feature_columns, preprocessors)
                val_processed = self.apply_preprocessing(val_df, feature_columns, preprocessors)

                # Step 3: Prepare features and targets
                X_train = train_processed[feature_columns].select_dtypes(include=[np.number])
                y_train = train_processed[target_column]
                X_val = val_processed[feature_columns].select_dtypes(include=[np.number])
                y_val = val_processed[target_column]

                # Step 4: Train model on training data only
                model.fit(X_train, y_train)

                # Step 5: Predict on validation data
                y_pred = model.predict(X_val)

                # Step 6: Calculate metrics
                fold_mae = mean_absolute_error(y_val, y_pred)
                fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                fold_r2 = r2_score(y_val, y_pred)

                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'mae': fold_mae,
                    'rmse': fold_rmse,
                    'r2': fold_r2,
                    'train_date_range': (train_df['GAME_DATE'].min().date(), train_df['GAME_DATE'].max().date()),
                    'val_date_range': (val_df['GAME_DATE'].min().date(), val_df['GAME_DATE'].max().date())
                })

                # Store predictions for overall analysis
                all_predictions.extend(y_pred)
                all_actuals.extend(y_val.tolist())

                logger.info(f"   Fold {fold_idx + 1}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}, R²={fold_r2:.3f}")

            except Exception as e:
                logger.error(f"❌ Fold {fold_idx + 1} failed: {e}")
                continue

        # Calculate overall metrics
        if all_predictions and all_actuals:
            overall_mae = mean_absolute_error(all_actuals, all_predictions)
            overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
            overall_r2 = r2_score(all_actuals, all_predictions)

            # Calculate prediction bias (important for betting)
            predictions_mean = np.mean(all_predictions)
            actuals_mean = np.mean(all_actuals)
            prediction_bias = predictions_mean - actuals_mean

            results = {
                'validation_type': 'temporal',
                'n_folds': len(splits),
                'total_predictions': len(all_predictions),
                'overall_metrics': {
                    'mae': overall_mae,
                    'rmse': overall_rmse,
                    'r2': overall_r2,
                    'prediction_bias': prediction_bias,
                    'predictions_mean': predictions_mean,
                    'actuals_mean': actuals_mean
                },
                'fold_metrics': fold_metrics,
                'feature_stats': self.feature_stats,
                'validation_params': {
                    'min_train_days': self.min_train_days,
                    'validation_days': self.validation_days,
                    'gap_days': self.gap_days,
                    'scaler_type': scaler_type
                }
            }

            logger.info(f"✅ Temporal validation completed:")
            logger.info(f"   Overall MAE: {overall_mae:.3f}")
            logger.info(f"   Overall RMSE: {overall_rmse:.3f}")
            logger.info(f"   Overall R²: {overall_r2:.3f}")
            logger.info(f"   Prediction Bias: {prediction_bias:.3f} ({'Positive bias' if prediction_bias > 0 else 'Negative bias'})")

            return results

        else:
            logger.error("❌ No valid predictions generated")
            return {'validation_type': 'temporal', 'error': 'No valid predictions'}

    def create_time_series_cv(self, df: pd.DataFrame, date_column: str = 'GAME_DATE'):
        """
        Create sklearn-compatible TimeSeriesSplit for cross-validation.

        Args:
            df: Dataset with date column
            date_column: Name of date column

        Returns:
            TimeSeriesSplit object configured for NBA data
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Calculate number of splits based on data size
        n_samples = len(df)
        max_splits = min(self.n_splits, n_samples // (self.min_train_days // 2))

        tscv = TimeSeriesSplit(
            n_splits=max_splits,
            test_size=max(self.validation_days // 2, 5),  # Minimum 5 games per test
            gap=self.gap_days
        )

        logger.info(f"🕒 Created TimeSeriesSplit: {max_splits} splits, test_size={max(self.validation_days // 2, 5)}")
        return tscv

    def detect_data_leakage(self,
                           df: pd.DataFrame,
                           feature_columns: List[str],
                           target_column: str,
                           date_column: str = 'GAME_DATE') -> Dict[str, Any]:
        """
        Detect potential data leakage in the dataset.

        Args:
            df: Dataset to analyze
            feature_columns: Feature column names
            target_column: Target column name
            date_column: Date column name

        Returns:
            Leakage detection report
        """
        leakage_report = {
            'potential_leakage': [],
            'warnings': [],
            'recommendations': []
        }

        # Check 1: Future information in features
        future_keywords = ['next_', 'future_', 'upcoming_', 'subsequent_']
        for col in feature_columns:
            if any(keyword in col.lower() for keyword in future_keywords):
                leakage_report['potential_leakage'].append(f"Future information in feature: {col}")

        # Check 2: Target-related features
        target_related = ['total_points', 'final_score', 'game_result']
        for col in feature_columns:
            if any(keyword in col.lower() for keyword in target_related):
                leakage_report['potential_leakage'].append(f"Target-related feature: {col}")

        # Check 3: Post-game information
        post_game_keywords = ['post_', 'final_', 'actual_', 'result_']
        for col in feature_columns:
            if any(keyword in col.lower() for keyword in post_game_keywords):
                leakage_report['warnings'].append(f"Post-game information: {col}")

        # Check 4: Date consistency
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            if not df[date_column].is_monotonic_increasing:
                leakage_report['warnings'].append("Data not sorted by date - potential leakage")

        # Recommendations
        if leakage_report['potential_leakage']:
            leakage_report['recommendations'].append("Remove features containing future information")
        if leakage_report['warnings']:
            leakage_report['recommendations'].append("Review suspicious features and ensure temporal consistency")
        leakage_report['recommendations'].append("Use temporal validation instead of random split")

        logger.info(f"🔍 Data leakage analysis: {len(leakage_report['potential_leakage'])} issues found")
        return leakage_report