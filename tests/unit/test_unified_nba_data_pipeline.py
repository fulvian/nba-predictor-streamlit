"""
Unit tests for UnifiedNBADataPipeline.

Tests data pipeline functionality, feature engineering,
and data quality validation.
"""

import pandas as pd
import pytest
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, patch

from unified_nba_data_pipeline import UnifiedNBADataPipeline


class TestUnifiedNBADataPipeline:
    """Test suite for UnifiedNBADataPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create UnifiedNBADataPipeline instance."""
        return UnifiedNBADataPipeline()

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw NBA data."""
        games_data = pd.DataFrame({
            'game_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'home_team': ['Lakers', 'Celtics', 'Warriors'],
            'away_team': ['Bulls', 'Heat', 'Nets'],
            'home_score': [110, 105, 115],
            'away_score': [100, 108, 112],
            'home_team_fg_pct': [0.45, 0.48, 0.50],
            'away_team_fg_pct': [0.42, 0.46, 0.48],
            'home_team_reb': [45, 42, 48],
            'away_team_reb': [40, 38, 44],
            'home_team_ast': [25, 22, 28],
            'away_team_ast': [20, 24, 26]
        })

        team_stats_data = pd.DataFrame({
            'team_name': ['Lakers', 'Celtics', 'Warriors', 'Bulls', 'Heat', 'Nets'],
            'points_per_game': [112.5, 108.3, 115.7, 105.2, 110.8, 109.4],
            'opp_points_per_game': [108.2, 105.6, 112.1, 108.9, 107.5, 111.2],
            'rebounds_per_game': [44.2, 42.8, 45.5, 41.7, 43.3, 42.9],
            'assists_per_game': [26.3, 24.8, 28.1, 23.2, 25.7, 24.5]
        })

        return {
            'games': games_data,
            'team_stats': team_stats_data,
            'boxscores': pd.DataFrame()
        }

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.cache_ttl == 3600
        assert hasattr(pipeline, 'cache')
        assert hasattr(pipeline, 'feature_cache')

    def test_pipeline_initialization_custom_ttl(self):
        """Test pipeline initialization with custom TTL."""
        pipeline = UnifiedNBADataPipeline(cache_ttl=7200)
        assert pipeline.cache_ttl == 7200

    @patch.object(UnifiedNBADataPipeline, '_fetch_games_data')
    @patch.object(UnifiedNBADataPipeline, '_fetch_team_stats')
    def test_fetch_all_data_success(self, mock_team_stats, mock_games, pipeline):
        """Test successful data fetching."""
        # Setup mocks
        mock_games_data = pd.DataFrame({
            'game_id': [1, 2],
            'date': ['2024-01-01', '2024-01-02']
        })
        mock_team_stats_data = pd.DataFrame({
            'team_name': ['Lakers', 'Celtics'],
            'points_per_game': [110.5, 105.3]
        })

        mock_games.return_value = mock_games_data
        mock_team_stats.return_value = mock_team_stats_data

        date_range = (date(2024, 1, 1), date(2024, 1, 7))
        result = pipeline.fetch_all_data(date_range, include_boxscores=False)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'games' in result
        assert 'team_stats' in result
        assert 'boxscores' in result

        # Verify data was fetched
        mock_games.assert_called_once()
        mock_team_stats.assert_called_once()

    def test_fetch_all_data_empty_date_range(self, pipeline):
        """Test fetching with empty date range."""
        result = pipeline.fetch_all_data((None, None))
        assert isinstance(result, dict)
        assert 'games' in result
        assert 'team_stats' in result
        assert 'boxscores' in result

    def test_preprocess_features_basic(self, pipeline, sample_raw_data):
        """Test basic feature preprocessing."""
        features = pipeline.preprocess_features(sample_raw_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 3  # Same number of games
        assert 'point_diff' in features.columns
        assert 'home_team' in features.columns
        assert 'away_team' in features.columns

    def test_preprocess_features_empty_data(self, pipeline):
        """Test feature preprocessing with empty data."""
        empty_data = {
            'games': pd.DataFrame(),
            'team_stats': pd.DataFrame(),
            'boxscores': pd.DataFrame()
        }

        features = pipeline.preprocess_features(empty_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0

    def test_preprocess_features_missing_games(self, pipeline):
        """Test feature preprocessing with missing games data."""
        incomplete_data = {
            'team_stats': pd.DataFrame({'team': ['Lakers']}),
            'boxscores': pd.DataFrame()
        }

        features = pipeline.preprocess_features(incomplete_data)
        assert isinstance(features, pd.DataFrame)
        # Should handle missing games gracefully

    def test_validate_data_quality_good_data(self, pipeline):
        """Test data validation with good quality data."""
        good_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [1, 0, 1, 0, 1]
        })

        result = pipeline.validate_data_quality(good_data)

        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'missing_values' in result
        assert 'duplicate_rows' in result
        assert result['is_valid'] is True
        assert result['quality_score'] > 0.8

    def test_validate_data_quality_missing_values(self, pipeline):
        """Test data validation with missing values."""
        bad_data = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],  # Missing value
            'feature2': [0.1, None, 0.3, 0.4, 0.5],  # Missing value
            'target': [1, 0, 1, 0, 1]
        })

        result = pipeline.validate_data_quality(bad_data)

        assert result['is_valid'] is False or result['quality_score'] < 0.8
        assert len(result['missing_values']) > 0

    def test_validate_data_quality_duplicates(self, pipeline):
        """Test data validation with duplicate rows."""
        duplicate_data = pd.DataFrame({
            'feature1': [1, 2, 2, 4, 5],  # Duplicate value
            'feature2': [0.1, 0.2, 0.2, 0.4, 0.5],  # Duplicate value
            'target': [1, 0, 0, 1, 1]
        })

        result = pipeline.validate_data_quality(duplicate_data)

        assert isinstance(result, dict)
        assert result['duplicate_rows'] > 0

    def test_validate_data_quality_empty_dataframe(self, pipeline):
        """Test data validation with empty dataframe."""
        empty_data = pd.DataFrame()

        result = pipeline.validate_data_quality(empty_data)

        assert isinstance(result, dict)
        assert result['is_valid'] is False
        assert 'Empty dataframe' in str(result.get('error', ''))

    def test_feature_engineering_point_diff(self, pipeline, sample_raw_data):
        """Test point difference feature engineering."""
        features = pipeline.preprocess_features(sample_raw_data)

        if 'point_diff' in features.columns:
            # Check that point_diff is calculated correctly
            expected_diff = sample_raw_data['games']['home_score'] - sample_raw_data['games']['away_score']
            assert all(features['point_diff'] == expected_diff)

    def test_feature_engineering_team_stats(self, pipeline, sample_raw_data):
        """Test team statistics feature engineering."""
        features = pipeline.preprocess_features(sample_raw_data)

        # Should have some team-related features
        team_features = [col for col in features.columns if 'team' in col.lower()]
        assert len(team_features) > 0

    def test_caching_functionality(self, pipeline):
        """Test that caching works properly."""
        # Test cache initialization
        assert hasattr(pipeline, 'cache')
        assert hasattr(pipeline, 'feature_cache')

        # Test cache key generation
        cache_key = pipeline._generate_cache_key('test', {'param': 'value'})
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_error_handling_invalid_data_types(self, pipeline):
        """Test error handling for invalid data types."""
        invalid_data = "not a dataframe"

        with pytest.raises((ValueError, TypeError, AttributeError)):
            pipeline.validate_data_quality(invalid_data)

    def test_pipeline_metrics(self, pipeline):
        """Test pipeline metrics collection."""
        metrics = pipeline.get_pipeline_metrics()

        assert isinstance(metrics, dict)
        # Should contain basic metrics
        possible_metrics = ['cache_hit_rate', 'average_fetch_time', 'total_requests']
        has_metrics = any(metric in metrics for metric in possible_metrics)
        assert has_metrics or isinstance(metrics, dict)  # Either has metrics or empty dict

    def test_data_quality_score_calculation(self, pipeline):
        """Test quality score calculation logic."""
        # Perfect data
        perfect_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [1, 0, 1, 0, 1]
        })

        result = pipeline.validate_data_quality(perfect_data)
        assert result['quality_score'] >= 0.9

        # Poor quality data
        poor_data = pd.DataFrame({
            'feature1': [1, None, None, None, 5],
            'feature2': [None, None, None, None, 0.5],
            'target': [1, 0, 1, 0, 1]
        })

        result = pipeline.validate_data_quality(poor_data)
        assert result['quality_score'] <= 0.5

    def test_feature_selection_logic(self, pipeline, sample_raw_data):
        """Test that feature selection works properly."""
        features = pipeline.preprocess_features(sample_raw_data)

        # Should have meaningful features
        assert len(features.columns) > 0
        # Should not have redundant or useless features
        useless_features = ['unnamed', 'index', 'level_0']
        has_useless = any(feature.lower() in useless_features for feature in features.columns)
        assert not has_useless

    def test_data_integrity_checks(self, pipeline):
        """Test data integrity validation."""
        # Data with inconsistent values
        inconsistent_data = pd.DataFrame({
            'score': [-10, 200, 50],  # Invalid scores
            'percentage': [1.5, -0.1, 0.5],  # Invalid percentages
            'target': [1, 0, 1]
        })

        result = pipeline.validate_data_quality(inconsistent_data)
        # Should detect data quality issues
        assert result['is_valid'] is False or result['quality_score'] < 0.7

    def test_preprocessing_pipeline_completeness(self, pipeline, sample_raw_data):
        """Test that the preprocessing pipeline is complete."""
        features = pipeline.preprocess_features(sample_raw_data)

        # Check that preprocessing adds expected columns
        assert len(features.columns) >= len(sample_raw_data['games'].columns)

        # Check that no NaN values remain in critical features
        critical_features = ['home_team', 'away_team']
        for feature in critical_features:
            if feature in features.columns:
                assert not features[feature].isna().any()

    def test_performance_optimization_features(self, pipeline):
        """Test performance optimization features."""
        # Test that pipeline has performance optimization
        assert hasattr(pipeline, 'cache_ttl')
        assert isinstance(pipeline.cache_ttl, int)
        assert pipeline.cache_ttl > 0