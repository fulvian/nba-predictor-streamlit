"""
Unit tests for NBAExplainabilityEngine.

Tests SHAP-based model explainability functionality,
feature importance calculations, and visualization generation.
"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import Mock, patch

from nba_explainability_engine import NBAExplainabilityEngine


class TestNBAExplainabilityEngine:
    """Test suite for NBAExplainabilityEngine class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock trained model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 0, 1]))
        model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]))
        return model

    @pytest.fixture
    def feature_names(self):
        """Create sample feature names."""
        return ['home_team_score', 'away_team_score', 'home_team_fg_pct',
                'away_team_fg_pct', 'home_team_reb', 'away_team_reb']

    @pytest.fixture
    def sample_data(self):
        """Create sample data for explanation."""
        np.random.seed(42)
        return pd.DataFrame({
            'home_team_score': np.random.randint(80, 140, 10),
            'away_team_score': np.random.randint(80, 140, 10),
            'home_team_fg_pct': np.random.uniform(0.35, 0.55, 10),
            'away_team_fg_pct': np.random.uniform(0.35, 0.55, 10),
            'home_team_reb': np.random.randint(30, 60, 10),
            'away_team_reb': np.random.randint(30, 60, 10)
        })

    @pytest.fixture
    def explainability_engine(self, mock_model, feature_names):
        """Create NBAExplainabilityEngine instance."""
        return NBAExplainabilityEngine(mock_model, feature_names)

    def test_engine_initialization(self, explainability_engine, mock_model, feature_names):
        """Test engine initialization."""
        assert explainability_engine.model == mock_model
        assert explainability_engine.feature_names == feature_names
        assert hasattr(explainability_engine, 'explainer')
        assert hasattr(explainability_engine, '_explanation_cache')

    def test_engine_initialization_with_background_data(self, mock_model, feature_names, sample_data):
        """Test engine initialization with background data."""
        engine = NBAExplainabilityEngine(mock_model, feature_names, background_data=sample_data)
        assert engine.background_data is not None
        assert len(engine.background_data) == len(sample_data)

    @patch('nba_explainability_engine.shap.TreeExplainer')
    def test_shap_explainer_creation(self, mock_tree_explainer, explainability_engine):
        """Test SHAP explainer creation."""
        mock_explainer = Mock()
        mock_tree_explainer.return_value = mock_explainer

        # Re-initialize to trigger explainer creation
        NBAExplainabilityEngine(explainability_engine.model, explainability_engine.feature_names)

        mock_tree_explainer.assert_called_once()

    @patch('nba_explainability_engine.shap.TreeExplainer')
    def test_calculate_shap_values_success(self, mock_tree_explainer, explainability_engine, sample_data):
        """Test successful SHAP value calculation."""
        # Setup mock explainer
        mock_explainer = Mock()
        mock_shap_values = np.array([
            [[0.1, -0.2, 0.3, -0.1, 0.2, -0.3],  # Class 0
             [0.2, -0.1, 0.1, -0.2, 0.3, -0.3]],  # Class 1
            [[-0.1, 0.2, -0.3, 0.1, -0.2, 0.3],  # Class 0
             [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]]   # Class 1
        ])
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer

        # Recreate engine with mock
        engine = NBAExplainabilityEngine(explainability_engine.model, explainability_engine.feature_names)
        engine.explainer = mock_explainer

        shap_values = engine.calculate_shap_values(sample_data)

        assert isinstance(shap_values, (np.ndarray, list))
        assert len(shap_values) > 0

    @patch('nba_explainability_engine.shap.TreeExplainer')
    def test_calculate_shap_values_with_cache(self, mock_tree_explainer, explainability_engine, sample_data):
        """Test SHAP value calculation with caching."""
        mock_explainer = Mock()
        mock_shap_values = np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]])
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer

        engine = NBAExplainabilityEngine(explainability_engine.model, explainability_engine.feature_names)
        engine.explainer = mock_explainer

        # First call should calculate
        result1 = engine.calculate_shap_values(sample_data, use_cache=True)
        # Second call should use cache
        result2 = engine.calculate_shap_values(sample_data, use_cache=True)

        # Should return same result
        np.testing.assert_array_equal(result1, result2)

    @patch('nba_explainability_engine.shap.TreeExplainer')
    def test_calculate_shap_values_error(self, mock_tree_explainer, explainability_engine, sample_data):
        """Test SHAP value calculation error handling."""
        mock_explainer = Mock()
        mock_explainer.shap_values.side_effect = Exception("SHAP calculation failed")
        mock_tree_explainer.return_value = mock_explainer

        engine = NBAExplainabilityEngine(explainability_engine.model, explainability_engine.feature_names)
        engine.explainer = mock_explainer

        with pytest.raises(Exception):
            engine.calculate_shap_values(sample_data)

    def test_generate_global_explanation_basic(self, explainability_engine):
        """Test basic global explanation generation."""
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.3],
            [-0.1, 0.2, -0.3, 0.1, -0.2, 0.3]
        ])

        result = explainability_engine.generate_global_explanation(mock_shap_values)

        assert isinstance(result, dict)
        assert 'feature_importance' in result
        assert 'summary_plot_data' in result
        assert isinstance(result['feature_importance'], (dict, list))

    @patch('nba_explainability_engine.plt')
    def test_generate_global_explanation_with_plot(self, mock_plt, explainability_engine):
        """Test global explanation with plot generation."""
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.3]
        ])

        result = explainability_engine.generate_global_explanation(
            mock_shap_values,
            plot_type="beeswarm"
        )

        assert isinstance(result, dict)
        # Should attempt to create plot
        assert 'feature_importance' in result

    def test_explain_single_prediction_basic(self, explainability_engine, sample_data):
        """Test single prediction explanation."""
        game_features = sample_data.iloc[0]
        prediction = 1

        # Mock SHAP values for single prediction
        with patch.object(explainability_engine, 'calculate_shap_values') as mock_shap:
            mock_shap.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]])

            result = explainability_engine.explain_single_prediction(game_features, prediction)

            assert isinstance(result, dict)
            assert 'base_value' in result
            assert 'shap_values' in result
            assert 'feature_contributions' in result

    def test_explain_single_prediction_empty_features(self, explainability_engine):
        """Test single prediction explanation with empty features."""
        empty_features = pd.Series()
        prediction = 1

        with pytest.raises((ValueError, IndexError)):
            explainability_engine.explain_single_prediction(empty_features, prediction)

    @patch('nba_explainability_engine.shap.force_plot')
    def test_explain_single_prediction_with_force_plot(self, mock_force_plot, explainability_engine, sample_data):
        """Test single prediction explanation with force plot."""
        game_features = sample_data.iloc[0]
        prediction = 1

        with patch.object(explainability_engine, 'calculate_shap_values') as mock_shap:
            mock_shap.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]])

            result = explainability_engine.explain_single_prediction(game_features, prediction)

            assert isinstance(result, dict)
            assert 'shap_values' in result

    def test_feature_importance_calculation(self, explainability_engine):
        """Test feature importance calculation from SHAP values."""
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.3],
            [-0.1, 0.2, -0.3, 0.1, -0.2, 0.3]
        ])

        importance = explainability_engine._calculate_feature_importance(mock_shap_values)

        assert isinstance(importance, dict)
        assert len(importance) == len(explainability_engine.feature_names)

        # Check that all feature names are present
        for feature_name in explainability_engine.feature_names:
            assert feature_name in importance

    def test_feature_importance_empty_shap_values(self, explainability_engine):
        """Test feature importance calculation with empty SHAP values."""
        empty_shap_values = np.array([])

        with pytest.raises((ValueError, IndexError)):
            explainability_engine._calculate_feature_importance(empty_shap_values)

    def test_cache_functionality(self, explainability_engine, sample_data):
        """Test caching functionality."""
        # Initially cache should be empty
        assert len(explainability_engine._explanation_cache) == 0

        # Add something to cache
        cache_key = "test_key"
        explainability_engine._explanation_cache[cache_key] = {"test": "data"}

        # Cache should contain the item
        assert cache_key in explainability_engine._explanation_cache
        assert explainability_engine._explanation_cache[cache_key] == {"test": "data"}

    def test_get_feature_importance_ranking(self, explainability_engine):
        """Test getting feature importance ranking."""
        mock_shap_values = np.array([
            [0.5, -0.3, 0.2, -0.1, 0.4, -0.2],
            [0.3, -0.4, 0.1, -0.2, 0.2, -0.1]
        ])

        importance = explainability_engine._calculate_feature_importance(mock_shap_values)
        ranked_features = explainability_engine._get_feature_ranking(importance)

        assert isinstance(ranked_features, list)
        assert len(ranked_features) == len(explainability_engine.feature_names)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked_features)

    def test_explanation_data_validation(self, explainability_engine):
        """Test explanation data validation."""
        # Test with invalid data
        invalid_data = "not a dataframe"

        with pytest.raises((TypeError, ValueError)):
            explainability_engine.calculate_shap_values(invalid_data)

    def test_multi_class_shap_handling(self, explainability_engine, sample_data):
        """Test handling of multi-class SHAP values."""
        # Mock multi-class SHAP values (3 classes, 6 features, 2 samples)
        multi_class_shap = np.array([
            [[0.1, -0.2, 0.3], [0.2, -0.1, 0.0]],  # Sample 1, features 1-3
            [[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]]   # Sample 2, features 1-3
        ])

        # Should handle multi-dimensional SHAP values
        result = explainability_engine._process_multi_class_shap(multi_class_shap)

        assert isinstance(result, (np.ndarray, list))
        assert len(result) > 0

    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model."""
        invalid_model = None
        feature_names = ['feature1', 'feature2']

        with pytest.raises((ValueError, TypeError, AttributeError)):
            NBAExplainabilityEngine(invalid_model, feature_names)

    def test_background_data_handling(self, explainability_engine, sample_data):
        """Test background data handling for SHAP explanations."""
        # Test with background data
        engine_with_bg = NBAExplainabilityEngine(
            explainability_engine.model,
            explainability_engine.feature_names,
            background_data=sample_data
        )

        assert engine_with_bg.background_data is not None
        assert len(engine_with_bg.background_data) == len(sample_data)

    def test_shap_value_aggregation_methods(self, explainability_engine):
        """Test different SHAP value aggregation methods."""
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.3],
            [-0.1, 0.2, -0.3, 0.1, -0.2, 0.3]
        ])

        # Test mean absolute values aggregation
        importance_mean = explainability_engine._calculate_feature_importance(mock_shap_values)

        # Test different aggregation
        importance_max = explainability_engine._calculate_feature_importance_max(mock_shap_values)

        assert isinstance(importance_mean, dict)
        assert isinstance(importance_max, dict)
        assert len(importance_mean) == len(importance_max)

    def test_explanation_output_formats(self, explainability_engine):
        """Test different explanation output formats."""
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.3]
        ])

        # Test JSON format
        result_json = explainability_engine.generate_global_explanation(
            mock_shap_values,
            output_format='json'
        )
        assert isinstance(result_json, dict)

        # Test DataFrame format
        result_df = explainability_engine.generate_global_explanation(
            mock_shap_values,
            output_format='dataframe'
        )
        # Should return dict containing DataFrame or DataFrame directly