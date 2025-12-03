#!/usr/bin/env python3
"""
🛡️ NBA Data Validator - Comprehensive Data Quality Framework

Provides robust data validation and sanitization for NBA data from multiple sources.
Ensures data quality, consistency, and reliability for the predictive analytics system.

Author: NBA Predictive Analytics System
Task ID: nba-data-validator-phase1
Date: 2025-01-10
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Data cannot be used
    ERROR = "error"       # Significant issues, should be fixed
    WARNING = "warning"   # Minor issues, can be used with caution
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    severity: ValidationSeverity
    field: str
    message: str
    row_count: int = 0
    affected_rows: Optional[List[int]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""
    is_valid: bool
    total_rows: int
    issues: List[ValidationIssue]
    quality_score: float  # 0.0 to 1.0
    warnings_count: int
    errors_count: int
    critical_count: int
    recommendations: List[str]
    validation_timestamp: datetime


class NBADataValidator:
    """
    Comprehensive NBA data validation and sanitization framework.

    Provides data quality validation, cleaning, and consistency checks
    for NBA data from multiple sources with intelligent error reporting.
    """

    def __init__(self):
        """Initialize NBA data validator with comprehensive rules."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Define required fields for each data type
        self.required_fields = {
            'games': ['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score'],
            'boxscores': ['GAME_ID', 'TEAM_ID', 'PTS', 'REB', 'AST'],
            'teams': ['team_id', 'team_name', 'team_abbreviation'],
            'players': ['player_id', 'player_name', 'team_id'],
            'features': ['game_id', 'home_team', 'away_team', 'game_date']
        }

        # Define expected data types
        self.data_types = {
            'game_id': str,
            'GAME_ID': str,
            'game_date': 'datetime64[ns]',
            'date': 'datetime64[ns]',
            'home_score': int,
            'away_score': int,
            'home_team': str,
            'away_team': str,
            'team_id': int,
            'TEAM_ID': int,
            'player_id': int,
            'PTS': int,
            'REB': int,
            'AST': int,
            'team_name': str,
            'player_name': str,
            'team_abbreviation': str
        }

        # Define value constraints
        self.value_constraints = {
            'scores': {'min': 0, 'max': 200},  # Reasonable NBA score ranges
            'team_id': {'min': 1, 'max': 999999},
            'player_id': {'min': 1, 'max': 9999999},
            'minutes': {'min': 0, 'max': 48},
            'field_goals': {'min': 0, 'max': 100},
            'three_pointers': {'min': 0, 'max': 50}
        }

        # Regular expressions for format validation
        self.patterns = {
            'game_id': r'^\d{10}$',  # 10-digit game ID format
            'team_abbreviation': r'^[A-Z]{3}$',  # 3-letter team codes
            'player_name': r'^[A-Za-z\.\'\-\s]+$',  # Valid player name characters
        }

    def validate_games_data(self, games_df: pd.DataFrame) -> ValidationReport:
        """
        Validate NBA games DataFrame structure and content.

        Args:
            games_df: DataFrame containing games data

        Returns:
            Comprehensive validation report
        """
        self.logger.info(f"Starting validation of games data with {len(games_df)} rows")
        issues = []
        recommendations = []

        # Check DataFrame structure
        structure_issues = self._validate_dataframe_structure(games_df, 'games')
        issues.extend(structure_issues)

        # Check for required fields
        missing_fields = set(self.required_fields['games']) - set(games_df.columns)
        if missing_fields:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field='structure',
                message=f"Missing required fields: {missing_fields}",
                suggestion="Ensure all required fields are present in the data source"
            ))

        # Data type validation
        type_issues = self._validate_data_types(games_df, 'games')
        issues.extend(type_issues)

        # Check for duplicates
        duplicate_issues = self._check_duplicates(games_df, ['game_id'])
        issues.extend(duplicate_issues)

        # Score validation
        score_issues = self._validate_scores(games_df)
        issues.extend(score_issues)

        # Date validation
        date_issues = self._validate_dates(games_df)
        issues.extend(date_issues)

        # Team validation
        team_issues = self._validate_teams(games_df)
        issues.extend(team_issues)

        # Game logic validation
        logic_issues = self._validate_game_logic(games_df)
        issues.extend(logic_issues)

        # Calculate quality score
        quality_score = self._calculate_quality_score(games_df, issues)

        # Count severity levels
        warnings_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        errors_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        # Determine overall validity
        is_valid = critical_count == 0

        report = ValidationReport(
            is_valid=is_valid,
            total_rows=len(games_df),
            issues=issues,
            quality_score=quality_score,
            warnings_count=warnings_count,
            errors_count=errors_count,
            critical_count=critical_count,
            recommendations=recommendations,
            validation_timestamp=datetime.now()
        )

        self.logger.info(f"Games validation completed: Quality Score {quality_score:.2f}, "
                        f"Issues: {len(issues)} (Critical: {critical_count}, Errors: {errors_count})")

        return report

    def validate_boxscores_data(self, boxscores_df: pd.DataFrame) -> ValidationReport:
        """
        Validate NBA boxscores DataFrame structure and content.

        Args:
            boxscores_df: DataFrame containing boxscores data

        Returns:
            Comprehensive validation report
        """
        self.logger.info(f"Starting validation of boxscores data with {len(boxscores_df)} rows")
        issues = []

        # Check DataFrame structure
        structure_issues = self._validate_dataframe_structure(boxscores_df, 'boxscores')
        issues.extend(structure_issues)

        # Check for required fields
        required_boxscore_fields = self.required_fields['boxscores']
        missing_fields = set(required_boxscore_fields) - set(boxscores_df.columns)
        if missing_fields:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field='structure',
                message=f"Missing required boxscore fields: {missing_fields}",
                suggestion="Ensure all required boxscore fields are present"
            ))

        # Data type validation
        type_issues = self._validate_data_types(boxscores_df, 'boxscores')
        issues.extend(type_issues)

        # Check for statistical reasonableness
        stat_issues = self._validate_boxscore_statistics(boxscores_df)
        issues.extend(stat_issues)

        # Calculate quality score
        quality_score = self._calculate_quality_score(boxscores_df, issues)

        # Count severity levels
        warnings_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        errors_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        # Determine overall validity
        is_valid = critical_count == 0

        report = ValidationReport(
            is_valid=is_valid,
            total_rows=len(boxscores_df),
            issues=issues,
            quality_score=quality_score,
            warnings_count=warnings_count,
            errors_count=errors_count,
            critical_count=critical_count,
            recommendations=recommendations,
            validation_timestamp=datetime.now()
        )

        self.logger.info(f"Boxscores validation completed: Quality Score {quality_score:.2f}")

        return report

    def sanitize_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Clean and sanitize DataFrame based on validation rules.

        Args:
            df: DataFrame to clean
            data_type: Type of data ('games', 'boxscores', etc.)

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Starting data sanitization for {data_type} with {len(df)} rows")
        cleaned_df = df.copy()

        # Remove duplicates
        if 'game_id' in cleaned_df.columns:
            before_dedup = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['game_id'], keep='last')
            removed_duplicates = before_dedup - len(cleaned_df)
            if removed_duplicates > 0:
                self.logger.info(f"Removed {removed_duplicates} duplicate game records")

        # Fill missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in cleaned_df.columns:
                missing_before = cleaned_df[col].isna().sum()
                if missing_before > 0:
                    # Use appropriate fill strategy based on column
                    if 'score' in col.lower():
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                    elif col in ['PTS', 'REB', 'AST']:
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

                    self.logger.info(f"Filled {missing_before} missing values in {col}")

        # Remove rows with missing critical fields
        critical_fields = ['game_id', 'GAME_ID', 'team_id', 'TEAM_ID']
        for field in critical_fields:
            if field in cleaned_df.columns:
                before_removal = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=[field])
                removed_rows = before_removal - len(cleaned_df)
                if removed_rows > 0:
                    self.logger.info(f"Removed {removed_rows} rows with missing {field}")

        # Correct data types
        cleaned_df = self._correct_data_types(cleaned_df, data_type)

        # Apply business logic corrections
        cleaned_df = self._apply_business_logic_corrections(cleaned_df, data_type)

        self.logger.info(f"Data sanitization completed: {len(cleaned_df)} rows remaining")
        return cleaned_df.reset_index(drop=True)

    def _validate_dataframe_structure(self, df: pd.DataFrame, data_type: str) -> List[ValidationIssue]:
        """Validate basic DataFrame structure."""
        issues = []

        if df.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field='structure',
                message="DataFrame is empty",
                suggestion="Check data source and connection"
            ))
            return issues

        # Check for minimum required columns
        required_cols = self.required_fields.get(data_type, [])
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field='columns',
                    message=f"Missing required columns: {missing_cols}",
                    suggestion="Ensure data source provides all required columns"
                ))

        return issues

    def _validate_data_types(self, df: pd.DataFrame, data_type: str) -> List[ValidationIssue]:
        """Validate data types of DataFrame columns."""
        issues = []

        for column in df.columns:
            expected_type = self.data_types.get(column)
            if expected_type is None:
                continue

            try:
                if expected_type == 'datetime64[ns]':
                    # Validate datetime columns
                    invalid_dates = pd.to_datetime(df[column], errors='coerce').isna()
                    if invalid_dates.any():
                        count = invalid_dates.sum()
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            field=column,
                            message=f"Found {count} invalid date values",
                            row_count=count,
                            suggestion="Check date format and ensure valid dates"
                        ))
                else:
                    # Validate other data types
                    try:
                        df[column].astype(expected_type)
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            field=column,
                            message=f"Column {column} has incompatible data type",
                            suggestion=f"Convert to {expected_type} if possible"
                        ))

            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=column,
                    message=f"Could not validate data type for {column}: {str(e)}",
                    suggestion="Manually review data in this column"
                ))

        return issues

    def _check_duplicates(self, df: pd.DataFrame, key_columns: List[str]) -> List[ValidationIssue]:
        """Check for duplicate records based on key columns."""
        issues = []

        for col in key_columns:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field=col,
                        message=f"Found {duplicates} duplicate {col} values",
                        suggestion="Remove duplicate records or investigate data source"
                    ))

        return issues

    def _validate_scores(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate game scores for reasonableness."""
        issues = []

        score_columns = ['home_score', 'away_score', 'PTS']
        score_constraints = self.value_constraints['scores']

        for col in score_columns:
            if col in df.columns:
                # Check for negative scores
                negative_scores = (df[col] < 0).sum()
                if negative_scores > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=col,
                        message=f"Found {negative_scores} negative score values",
                        suggestion="Verify score data source"
                    ))

                # Check for unreasonably high scores
                high_scores = (df[col] > score_constraints['max']).sum()
                if high_scores > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field=col,
                        message=f"Found {high_scores} unusually high scores (> {score_constraints['max']})",
                        suggestion="Verify if scores are correct"
                    ))

        return issues

    def _validate_dates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate date fields for reasonableness."""
        issues = []

        date_columns = ['game_date', 'date']
        current_year = datetime.now().year

        for col in date_columns:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')

                    # Check for invalid dates
                    invalid_dates = dates.isna().sum()
                    if invalid_dates > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            field=col,
                            message=f"Found {invalid_dates} invalid date values",
                            suggestion="Check date format"
                        ))

                    # Check for future dates
                    valid_dates = dates.dropna()
                    future_dates = (valid_dates.dt.year > current_year + 1).sum()
                    if future_dates > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            field=col,
                            message=f"Found {future_dates} future dates",
                            suggestion="Verify if dates are correct"
                        ))

                    # Check for very old dates
                    old_dates = (valid_dates.dt.year < 2000).sum()
                    if old_dates > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            field=col,
                            message=f"Found {old_dates} dates before year 2000",
                            suggestion="Verify historical data is correct"
                        ))

                except Exception as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=col,
                        message=f"Could not validate dates: {str(e)}",
                        suggestion="Check date column format"
                    ))

        return issues

    def _validate_teams(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate team names and abbreviations."""
        issues = []

        team_columns = ['home_team', 'away_team', 'team_name']

        for col in team_columns:
            if col in df.columns:
                # Check for empty team names
                empty_teams = df[col].isna().sum() + (df[col] == '').sum()
                if empty_teams > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=col,
                        message=f"Found {empty_teams} empty team names",
                        suggestion="Ensure all team names are provided"
                ))

        # Validate team abbreviations if present
        if 'team_abbreviation' in df.columns:
            pattern = self.patterns['team_abbreviation']
            invalid_abbrevs = ~df['team_abbreviation'].str.match(pattern, na=False)
            invalid_count = invalid_abbrevs.sum()

            if invalid_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field='team_abbreviation',
                    message=f"Found {invalid_count} invalid team abbreviation formats",
                    suggestion="Ensure abbreviations follow XXX format"
                ))

        return issues

    def _validate_game_logic(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate game logic and consistency."""
        issues = []

        # Check for games where home and away teams are the same
        if 'home_team' in df.columns and 'away_team' in df.columns:
            same_teams = (df['home_team'] == df['away_team']).sum()
            if same_teams > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field='game_logic',
                    message=f"Found {same_teams} games where home and away teams are the same",
                    suggestion="Check team assignment logic"
                ))

        # Check for tied games (shouldn't happen in NBA regular season)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            tied_games = (df['home_score'] == df['away_score']).sum()
            if tied_games > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field='game_logic',
                    message=f"Found {tied_games} tied games",
                    suggestion="Verify if these are playoff games or check scoring data"
                ))

        return issues

    def _validate_boxscore_statistics(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate boxscore statistics for reasonableness."""
        issues = []

        # Check basic stats
        stat_constraints = {
            'PTS': self.value_constraints['scores'],
            'REB': {'min': 0, 'max': 50},
            'AST': {'min': 0, 'max': 30}
        }

        for stat, constraints in stat_constraints.items():
            if stat in df.columns:
                # Check for negative values
                negative = (df[stat] < 0).sum()
                if negative > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=stat,
                        message=f"Found {negative} negative {stat} values",
                        suggestion="Check data source for calculation errors"
                    ))

                # Check for unreasonably high values
                high = (df[stat] > constraints['max']).sum()
                if high > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field=stat,
                        message=f"Found {high} unusually high {stat} values (> {constraints['max']})",
                        suggestion="Verify if these are correct or data entry errors"
                    ))

        return issues

    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0.0 to 1.0)."""
        if df.empty:
            return 0.0

        # Base score starts at 1.0
        score = 1.0

        # Deduct points for issues
        severity_penalties = {
            ValidationSeverity.CRITICAL: 0.5,
            ValidationSeverity.ERROR: 0.2,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.INFO: 0.01
        }

        for issue in issues:
            # Scale penalty by proportion of affected rows
            if issue.row_count > 0 and len(df) > 0:
                affected_ratio = min(issue.row_count / len(df), 1.0)
                penalty = severity_penalties[issue.severity] * affected_ratio
                score = max(0.0, score - penalty)

        return round(score, 3)

    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on validation issues."""
        recommendations = []

        # Group issues by type
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]

        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address critical issues before using data")

        if error_issues:
            recommendations.append("⚠️ ERRORS: Fix data quality issues to improve reliability")

        if warning_issues:
            recommendations.append("ℹ️ WARNINGS: Review and validate unusual values")

        # Specific recommendations based on issue types
        issue_fields = [issue.field for issue in issues]

        if 'structure' in issue_fields:
            recommendations.append("📋 STRUCTURE: Verify data source provides required fields")

        if 'game_logic' in issue_fields:
            recommendations.append("🏀 LOGIC: Review game data for logical consistency")

        if any('score' in field for field in issue_fields):
            recommendations.append("📊 SCORES: Validate score data accuracy")

        return recommendations

    def _correct_data_types(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Correct data types based on validation rules."""
        corrected_df = df.copy()

        for column in corrected_df.columns:
            expected_type = self.data_types.get(column)
            if expected_type is None:
                continue

            try:
                if expected_type == 'datetime64[ns]':
                    corrected_df[column] = pd.to_datetime(corrected_df[column], errors='coerce')
                elif expected_type == int:
                    corrected_df[column] = pd.to_numeric(corrected_df[column], errors='coerce').astype('Int64')
                elif expected_type == float:
                    corrected_df[column] = pd.to_numeric(corrected_df[column], errors='coerce')
                elif expected_type == str:
                    corrected_df[column] = corrected_df[column].astype(str)
            except Exception as e:
                self.logger.warning(f"Could not convert {column} to {expected_type}: {e}")

        return corrected_df

    def _apply_business_logic_corrections(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply business logic corrections to clean data."""
        corrected_df = df.copy()

        if data_type == 'games':
            # Ensure home_score and away_score are integers
            score_cols = ['home_score', 'away_score']
            for col in score_cols:
                if col in corrected_df.columns:
                    corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce').fillna(0).astype(int)

        elif data_type == 'boxscores':
            # Ensure basic stats are integers
            stat_cols = ['PTS', 'REB', 'AST']
            for col in stat_cols:
                if col in corrected_df.columns:
                    corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce').fillna(0).astype(int)

        return corrected_df


def create_nba_data_validator() -> NBADataValidator:
    """
    Factory function to create NBA data validator instance.

    Returns:
        NBADataValidator instance
    """
    return NBADataValidator()


# Example usage and testing
if __name__ == "__main__":
    # Create validator
    validator = create_nba_data_validator()

    # Example validation
    print("NBA Data Validator initialized successfully")
    print("Available validation methods:")
    print("- validate_games_data()")
    print("- validate_boxscores_data()")
    print("- sanitize_data()")
    print("\nReady for data quality validation!")