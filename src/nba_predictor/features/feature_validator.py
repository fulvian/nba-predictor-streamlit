"""NBA Feature Validation and Optimization System.

This module implements comprehensive feature validation, optimization, and analysis
for the NBA Predictor system, ensuring feature quality and performance.
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class FeatureValidationResult:
    """Result of feature validation analysis."""

    feature_name: str
    is_valid: bool
    validation_issues: List[str]
    correlation_score: float
    importance_score: float
    nba_realism_score: float
    recommendation: str
    optimization_suggestions: List[str]


@dataclass
class FeatureOptimizationReport:
    """Comprehensive feature optimization report."""

    total_features: int
    valid_features: int
    removed_features: int
    optimized_features: int
    correlation_matrix: pd.DataFrame
    feature_importance: Dict[str, float]
    validation_results: List[FeatureValidationResult]
    optimization_summary: Dict[str, Any]
    performance_impact: Dict[str, float]


class NBAFeatureValidator:
    """
    Comprehensive NBA feature validation and optimization system.

    Implements feature analysis including:
    - SHAP importance analysis
    - Correlation analysis and redundancy detection
    - NBA realism validation
    - Performance optimization recommendations
    """

    def __init__(self, data_store: Optional[UnifiedDataStore] = None):
        """
        Initialize feature validator.

        Args:
            data_store: UnifiedDataStore instance for data access
        """
        self.data_store = data_store

        # NBA realistic feature ranges (based on real NBA data analysis)
        self.NBA_FEATURE_RANGES = {
            # Four Factors Features
            "HOME_eFG_PCT_sAvg": (0.450, 0.580),
            "AWAY_eFG_PCT_sAvg": (0.450, 0.580),
            "HOME_TOV_PCT_sAvg": (0.100, 0.180),
            "AWAY_TOV_PCT_sAvg": (0.100, 0.180),
            "HOME_OREB_PCT_sAvg": (0.200, 0.320),
            "AWAY_OREB_PCT_sAvg": (0.200, 0.320),
            "HOME_FT_RATE_sAvg": (0.150, 0.300),
            "AWAY_FT_RATE_sAvg": (0.150, 0.300),
            # Efficiency Features
            "HOME_ORtg_sAvg": (105.0, 125.0),
            "AWAY_ORtg_sAvg": (105.0, 125.0),
            "HOME_DRtg_sAvg": (105.0, 125.0),
            "AWAY_DRtg_sAvg": (105.0, 125.0),
            # Pace Features
            "HOME_PACE": (95.0, 105.0),
            "AWAY_PACE": (95.0, 105.0),
            "GAME_PACE": (95.0, 105.0),
            "PACE_DIFFERENTIAL": (-10.0, 10.0),
            "AVG_PACE": (95.0, 105.0),
            # Differential Features
            "HOME_OFF_vs_AWAY_DEF": (-20.0, 20.0),
            "AWAY_OFF_vs_HOME_DEF": (-20.0, 20.0),
            "TOTAL_EXPECTED_SCORING": (200.0, 290.0),
            "LgAvg_ORtg_season": (110.0, 120.0),
        }

        # Feature importance thresholds
        self.IMPORTANCE_THRESHOLDS = {"high": 0.05, "medium": 0.02, "low": 0.01}

        # Correlation thresholds for redundancy detection
        self.CORRELATION_THRESHOLDS = {
            "high_redundancy": 0.95,
            "medium_redundancy": 0.85,
            "low_redundancy": 0.70,
        }

        logger.info("🔍 NBA Feature Validator initialized")

    def validate_feature_set(
        self,
        features_df: pd.DataFrame,
        target_column: str = "total_score",
        use_shap: bool = True,
    ) -> FeatureOptimizationReport:
        """
        Validate and optimize a complete feature set.

        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column
            use_shap: Whether to use SHAP for importance analysis

        Returns:
            Comprehensive optimization report
        """
        logger.info(f"🔍 Validating feature set: {len(features_df.columns)} features")

        try:
            # Separate features and target
            feature_columns = [
                col for col in features_df.columns if col != target_column
            ]
            X = features_df[feature_columns]
            y = features_df[target_column]

            # 1. Basic validation
            validation_results = []
            for feature_name in feature_columns:
                result = self._validate_single_feature(feature_name, X[feature_name], y)
                validation_results.append(result)

            # 2. Correlation analysis
            correlation_matrix = self._analyze_correlations(X)

            # 3. Feature importance analysis
            if use_shap:
                try:
                    feature_importance = self._calculate_shap_importance(X, y)
                except Exception as e:
                    logger.warning(
                        f"SHAP analysis failed, using correlation importance: {e}"
                    )
                    feature_importance = self._calculate_correlation_importance(X, y)
            else:
                feature_importance = self._calculate_correlation_importance(X, y)

            # 4. Redundancy analysis
            redundant_features = self._identify_redundant_features(correlation_matrix)

            # 5. Generate optimization recommendations
            optimization_summary = self._generate_optimization_recommendations(
                validation_results, feature_importance, redundant_features
            )

            # 6. Calculate performance impact
            performance_impact = self._estimate_performance_impact(
                validation_results, feature_importance
            )

            # 7. Create optimization report
            report = FeatureOptimizationReport(
                total_features=len(feature_columns),
                valid_features=sum(1 for r in validation_results if r.is_valid),
                removed_features=len(redundant_features),
                optimized_features=len(
                    [r for r in validation_results if r.recommendation != "KEEP"]
                ),
                correlation_matrix=correlation_matrix,
                feature_importance=feature_importance,
                validation_results=validation_results,
                optimization_summary=optimization_summary,
                performance_impact=performance_impact,
            )

            logger.info(
                f"✅ Feature validation completed: "
                f"{report.valid_features}/{report.total_features} valid, "
                f"{report.removed_features} redundant, "
                f"{report.optimized_features} optimized"
            )

            return report

        except Exception as e:
            logger.error(f"❌ Feature validation failed: {e}")
            raise ValidationError(f"Failed to validate feature set: {e}") from e

    def _validate_single_feature(
        self, feature_name: str, feature_values: pd.Series, target_values: pd.Series
    ) -> FeatureValidationResult:
        """Validate a single feature against NBA standards."""
        validation_issues = []
        optimization_suggestions = []

        try:
            # 1. Check for missing values
            missing_ratio = feature_values.isnull().sum() / len(feature_values)
            if missing_ratio > 0.1:
                validation_issues.append(
                    f"High missing value ratio: {missing_ratio:.2%}"
                )
                optimization_suggestions.append(
                    "Impute missing values or consider removal"
                )

            # 2. Check variance
            if feature_values.var() < 0.001:
                validation_issues.append("Very low variance (near-constant)")
                optimization_suggestions.append(
                    "Consider removing low-variance feature"
                )

            # 3. Check NBA realism
            realism_score = self._validate_nba_realism(feature_name, feature_values)
            if realism_score < 0.7:
                validation_issues.append(f"Low NBA realism score: {realism_score:.2f}")
                optimization_suggestions.append(
                    "Review feature calculation for NBA accuracy"
                )

            # 4. Check correlation with target
            correlation = abs(feature_values.corr(target_values))
            if correlation < 0.05:
                validation_issues.append(
                    f"Very low target correlation: {correlation:.3f}"
                )
                optimization_suggestions.append(
                    "Consider feature engineering or removal"
                )

            # 5. Check for outliers
            q1, q3 = feature_values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = (
                (feature_values < (q1 - 1.5 * iqr))
                | (feature_values > (q3 + 1.5 * iqr))
            ).sum()
            outlier_ratio = outliers / len(feature_values)
            if outlier_ratio > 0.1:
                validation_issues.append(f"High outlier ratio: {outlier_ratio:.2%}")
                optimization_suggestions.append("Consider outlier treatment")

            # 6. Determine validity and recommendation
            is_valid = (
                len(validation_issues) <= 2
                and missing_ratio < 0.2
                and realism_score > 0.5
            )

            if is_valid and len(validation_issues) == 0:
                recommendation = "KEEP"
            elif is_valid:
                recommendation = "OPTIMIZE"
            else:
                recommendation = "REMOVE"

            # 7. Calculate importance score (simplified)
            importance_score = correlation * realism_score

            return FeatureValidationResult(
                feature_name=feature_name,
                is_valid=is_valid,
                validation_issues=validation_issues,
                correlation_score=correlation,
                importance_score=importance_score,
                nba_realism_score=realism_score,
                recommendation=recommendation,
                optimization_suggestions=optimization_suggestions,
            )

        except Exception as e:
            logger.error(f"Error validating feature {feature_name}: {e}")
            return FeatureValidationResult(
                feature_name=feature_name,
                is_valid=False,
                validation_issues=[f"Validation error: {e}"],
                correlation_score=0.0,
                importance_score=0.0,
                nba_realism_score=0.0,
                recommendation="ERROR",
                optimization_suggestions=["Fix feature calculation"],
            )

    def _validate_nba_realism(
        self, feature_name: str, feature_values: pd.Series
    ) -> float:
        """
        Validate feature values against realistic NBA ranges.

        Args:
            feature_name: Name of the feature
            feature_values: Feature values to validate

        Returns:
            Realism score between 0 and 1
        """
        try:
            # Get expected range for this feature
            expected_range = self.NBA_FEATURE_RANGES.get(feature_name)

            if expected_range is None:
                # Unknown feature - assume moderate realism
                return 0.7

            min_val, max_val = expected_range

            # Calculate percentage of values within realistic range
            within_range = (
                (feature_values >= min_val) & (feature_values <= max_val)
            ).sum()
            realism_score = within_range / len(feature_values)

            # Penalize extreme outliers more heavily
            extreme_outliers = (
                (feature_values < min_val * 0.5) | (feature_values > max_val * 1.5)
            ).sum()
            extreme_penalty = extreme_outliers / len(feature_values) * 0.5

            return max(0.0, realism_score - extreme_penalty)

        except Exception as e:
            logger.warning(f"Error validating NBA realism for {feature_name}: {e}")
            return 0.5

    def _analyze_correlations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlation matrix for feature relationships."""
        try:
            # Calculate correlation matrix
            correlation_matrix = features_df.corr(method="pearson")

            # Fill diagonal with NaN (self-correlation not useful)
            np.fill_diagonal(correlation_matrix.values, np.nan)

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return pd.DataFrame()

    def _identify_redundant_features(
        self, correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Identify redundant features based on high correlations."""
        redundant_features = []

        try:
            # Find feature pairs with high correlation
            high_corr_pairs = []

            for i, feature1 in enumerate(correlation_matrix.columns):
                for j, feature2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = correlation_matrix.iloc[i, j]
                        if (
                            not pd.isna(corr_value)
                            and abs(corr_value)
                            > self.CORRELATION_THRESHOLDS["high_redundancy"]
                        ):
                            high_corr_pairs.append((feature1, feature2, corr_value))

            # For each high-correlation pair, keep the more important one
            # (simplified: keep the one that comes first alphabetically)
            # In practice, this should be based on importance scores
            for feature1, feature2, corr_value in high_corr_pairs:
                # Simple heuristic: keep the feature that comes first alphabetically
                # In practice, this should be based on importance scores
                if feature1 < feature2:
                    redundant_features.append(feature2)
                else:
                    redundant_features.append(feature1)

            return list(set(redundant_features))

        except Exception as e:
            logger.error(f"Error identifying redundant features: {e}")
            return []

    def _calculate_shap_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Calculate SHAP-based feature importance."""
        try:
            # Use a simple model for SHAP analysis
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train simple model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            # Calculate SHAP values
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_values).mean(axis=0)

            # Create importance dictionary
            importance_dict = dict(zip(X.columns, mean_shap_values))

            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {
                    k: v / total_importance for k, v in importance_dict.items()
                }

            return importance_dict

        except Exception as e:
            logger.warning(f"SHAP importance calculation failed: {e}")
            return self._calculate_correlation_importance(X, y)

    def _calculate_correlation_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Calculate feature importance based on correlation with target."""
        try:
            correlations = {}
            for feature in X.columns:
                corr = abs(X[feature].corr(y))
                correlations[feature] = corr if not pd.isna(corr) else 0.0

            # Normalize to sum to 1
            total_corr = sum(correlations.values())
            if total_corr > 0:
                correlations = {k: v / total_corr for k, v in correlations.items()}

            return correlations

        except Exception as e:
            logger.error(f"Error calculating correlation importance: {e}")
            return {}

    def _generate_optimization_recommendations(
        self,
        validation_results: List[FeatureValidationResult],
        feature_importance: Dict[str, float],
        redundant_features: List[str],
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations."""
        recommendations = {
            "features_to_remove": [],
            "features_to_optimize": [],
            "features_to_keep": [],
            "priority_actions": [],
            "estimated_performance_gain": 0.0,
        }

        # Categorize features
        for result in validation_results:
            if (
                result.recommendation == "REMOVE"
                or result.feature_name in redundant_features
            ):
                recommendations["features_to_remove"].append(result.feature_name)
            elif result.recommendation == "OPTIMIZE":
                recommendations["features_to_optimize"].append(result.feature_name)
            else:
                recommendations["features_to_keep"].append(result.feature_name)

        # Generate priority actions
        if recommendations["features_to_remove"]:
            recommendations["priority_actions"].append(
                f"Remove {len(recommendations['features_to_remove'])} redundant/invalid features"
            )

        if recommendations["features_to_optimize"]:
            recommendations["priority_actions"].append(
                f"Optimize {len(recommendations['features_to_optimize'])} underperforming features"
            )

        # Estimate performance gain (simplified)
        total_importance = sum(feature_importance.values())
        removed_importance = sum(
            feature_importance.get(f, 0) for f in recommendations["features_to_remove"]
        )
        recommendations["estimated_performance_gain"] = (
            (removed_importance / total_importance * 100) if total_importance > 0 else 0
        )

        return recommendations

    def _estimate_performance_impact(
        self,
        validation_results: List[FeatureValidationResult],
        feature_importance: Dict[str, float],
    ) -> Dict[str, float]:
        """Estimate performance impact of optimizations."""
        try:
            # Calculate current feature quality score
            total_importance = sum(feature_importance.values())
            valid_importance = sum(
                feature_importance.get(r.feature_name, 0)
                for r in validation_results
                if r.is_valid
            )

            current_quality = (
                valid_importance / total_importance if total_importance > 0 else 0
            )

            # Calculate potential quality after optimization
            optimized_importance = sum(
                feature_importance.get(r.feature_name, 0)
                for r in validation_results
                if r.recommendation in ["KEEP", "OPTIMIZE"]
            )

            optimized_quality = (
                optimized_importance / total_importance if total_importance > 0 else 0
            )

            return {
                "current_quality_score": current_quality,
                "optimized_quality_score": optimized_quality,
                "quality_improvement": optimized_quality - current_quality,
                "quality_improvement_percent": (
                    (optimized_quality - current_quality) / current_quality * 100
                )
                if current_quality > 0
                else 0,
            }

        except Exception as e:
            logger.error(f"Error estimating performance impact: {e}")
            return {
                "current_quality_score": 0.5,
                "optimized_quality_score": 0.5,
                "quality_improvement": 0.0,
                "quality_improvement_percent": 0.0,
            }

    def optimize_feature_set(
        self,
        features_df: pd.DataFrame,
        target_column: str = "total_score",
        optimization_level: str = "moderate",
    ) -> Tuple[pd.DataFrame, FeatureOptimizationReport]:
        """
        Optimize feature set based on validation results.

        Args:
            features_df: Original feature DataFrame
            target_column: Target column name
            optimization_level: 'conservative', 'moderate', or 'aggressive'

        Returns:
            Tuple of (optimized DataFrame, optimization report)
        """
        logger.info(f"🔧 Optimizing feature set (level: {optimization_level})")

        try:
            # Validate features
            report = self.validate_feature_set(features_df, target_column)

            # Determine features to keep based on optimization level
            features_to_keep = set(report.optimization_summary["features_to_keep"])

            if optimization_level == "conservative":
                # Only remove clearly invalid features
                features_to_keep.update(
                    f for f in report.optimization_summary["features_to_optimize"][:5]
                )
            elif optimization_level == "moderate":
                # Keep top 75% of optimize features
                optimize_features = report.optimization_summary["features_to_optimize"]
                top_optimize = sorted(
                    optimize_features,
                    key=lambda f: report.feature_importance.get(f, 0),
                    reverse=True,
                )[: int(len(optimize_features) * 0.75)]
                features_to_keep.update(top_optimize)
            elif optimization_level == "aggressive":
                # Only keep high-importance features
                high_importance_features = [
                    f
                    for f, imp in report.feature_importance.items()
                    if imp >= self.IMPORTANCE_THRESHOLDS["medium"]
                ]
                features_to_keep.update(high_importance_features)

            # Create optimized DataFrame
            columns_to_keep = list(features_to_keep) + [target_column]
            optimized_df = features_df[columns_to_keep].copy()

            logger.info(
                f"✅ Feature optimization completed: "
                f"{len(optimized_df.columns) - 1}/{len(features_df.columns) - 1} features retained"
            )

            return optimized_df, report

        except Exception as e:
            logger.error(f"❌ Feature optimization failed: {e}")
            raise ValidationError(f"Failed to optimize feature set: {e}") from e

    def save_optimization_report(
        self,
        report: FeatureOptimizationReport,
        filepath: str = "feature_optimization_report.json",
    ) -> None:
        """Save optimization report to file."""
        try:
            # Convert report to serializable format
            report_dict = {
                "summary": {
                    "total_features": report.total_features,
                    "valid_features": report.valid_features,
                    "removed_features": report.removed_features,
                    "optimized_features": report.optimized_features,
                },
                "optimization_recommendations": report.optimization_summary,
                "performance_impact": report.performance_impact,
                "feature_importance": report.feature_importance,
                "validation_results": [
                    {
                        "feature_name": r.feature_name,
                        "is_valid": r.is_valid,
                        "recommendation": r.recommendation,
                        "correlation_score": r.correlation_score,
                        "importance_score": r.importance_score,
                        "nba_realism_score": r.nba_realism_score,
                        "validation_issues": r.validation_issues,
                        "optimization_suggestions": r.optimization_suggestions,
                    }
                    for r in report.validation_results
                ],
                "correlation_matrix": report.correlation_matrix.to_dict(),
                "generated_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(report_dict, f, indent=2)

            logger.info(f"💾 Optimization report saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving optimization report: {e}")


def create_feature_validator(
    data_store: Optional[UnifiedDataStore] = None,
) -> NBAFeatureValidator:
    """
    Create and configure NBA feature validator.

    Args:
        data_store: Optional data store for validation

    Returns:
        Configured NBAFeatureValidator instance
    """
    return NBAFeatureValidator(data_store)
