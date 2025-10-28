#!/usr/bin/env python3
"""
🧪 SHAP Explainer Unit Tests
Test suite for SHAP-based model explainability functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from nba_predictor.explainability.shap_explainer import (
    create_nba_shap_explainer,
    validate_explainer_inputs,
    create_tree_explainer,
    calculate_global_shap_values,
    calculate_local_shap_values,
    get_feature_importance_from_shap,
    create_shap_summary_plot,
    create_waterfall_plot,
    generate_nba_explanation_report
)


class TestShapExplainer(unittest.TestCase):
    """Test cases for SHAP explainer module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample NBA data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'team1_score': np.random.randint(80, 140, 100),
            'team2_score': np.random.randint(80, 140, 100),
            'efg_pct': np.random.uniform(0.45, 0.65, 100),
            'tov_pct': np.random.uniform(0.10, 0.20, 100),
            'orb_pct': np.random.uniform(0.20, 0.35, 100),
            'ftr': np.random.uniform(0.15, 0.35, 100),
            'pace_possessions': np.random.uniform(180, 220, 100),
            'four_factors_product': np.random.uniform(0.02, 0.08, 100)
        })

        self.target = np.random.randint(160, 280, 100)
        self.X_background = self.sample_data.iloc[:50]
        self.X_test = self.sample_data.iloc[50:]

        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.randn(len(self.X_test))
        self.mock_model.__class__.__name__ = "MockXGBRegressor"

    def test_create_nba_shap_explainer_basic(self):
        """Test basic SHAP explainer creation."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            explainer = create_nba_shap_explainer(self.mock_model, self.X_background)

            self.assertIsNotNone(explainer)
            mock_tree_explainer.assert_called_once()

    def test_create_nba_shap_explainer_missing_shap(self):
        """Test error handling when SHAP is not available."""
        with patch('nba_predictor.explainability.shap_explainer._shap_available', False):
            with self.assertRaises(ImportError) as context:
                create_nba_shap_explainer(self.mock_model, self.X_background)

            self.assertIn("SHAP is required", str(context.exception))

    def test_validate_explainer_inputs_success(self):
        """Test successful input validation."""
        # Should not raise exception
        validate_explainer_inputs(self.mock_model, self.X_background, "raw")

    def test_validate_explainer_inputs_none_model(self):
        """Test validation with None model."""
        with self.assertRaises(ValueError) as context:
            validate_explainer_inputs(None, self.X_background, "raw")

        self.assertIn("Model cannot be None", str(context.exception))

    def test_validate_explainer_inputs_empty_background(self):
        """Test validation with empty background dataset."""
        empty_df = pd.DataFrame()

        with self.assertRaises(ValueError) as context:
            validate_explainer_inputs(self.mock_model, empty_df, "raw")

        self.assertIn("Background dataset cannot be None or empty", str(context.exception))

    def test_validate_explainer_inputs_insufficient_samples(self):
        """Test validation with insufficient background samples."""
        small_df = self.X_background.iloc[:5]  # Only 5 samples

        with self.assertRaises(ValueError) as context:
            validate_explainer_inputs(self.mock_model, small_df, "raw")

        self.assertIn("at least 10 samples", str(context.exception))

    def test_validate_explainer_inputs_invalid_model_output(self):
        """Test validation with invalid model output type."""
        with self.assertRaises(ValueError) as context:
            validate_explainer_inputs(self.mock_model, self.X_background, "invalid")

        self.assertIn("model_output must be one of", str(context.exception))

    def test_validate_explainer_inputs_no_predict_method(self):
        """Test validation with model that has no predict method."""
        bad_model = object()  # Object without predict method

        with self.assertRaises(ValueError) as context:
            validate_explainer_inputs(bad_model, self.X_background, "raw")

        self.assertIn("must have a 'predict' method", str(context.exception))

    def test_create_tree_explainer_xgboost(self):
        """Test TreeExplainer creation for XGBoost model."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            self.mock_model.__class__.__name__ = "XGBRegressor"
            explainer = create_tree_explainer(self.mock_model, self.X_background, "raw")

            self.assertIsNotNone(explainer)
            mock_tree_explainer.assert_called_once()

            # Check that XGBoost-specific parameters were set
            call_args = mock_tree_explainer.call_args[1]
            self.assertEqual(call_args['model_output'], "raw")
            self.assertEqual(call_args['feature_perturbation'], "interventional")

    def test_create_tree_explainer_lightgbm(self):
        """Test TreeExplainer creation for LightGBM model."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            self.mock_model.__class__.__name__ = "LGBMRegressor"
            explainer = create_tree_explainer(self.mock_model, self.X_background, "raw")

            self.assertIsNotNone(explainer)
            mock_tree_explainer.assert_called_once()

            # Check that LightGBM-specific parameters were set
            call_args = mock_tree_explainer.call_args[1]
            self.assertEqual(call_args['model_output'], "raw")

    def test_create_tree_explainer_stacking_regressor(self):
        """Test TreeExplainer creation for StackingRegressor."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            # Create StackingRegressor mock
            stacking_model = MagicMock()
            stacking_model.__class__.__name__ = "StackingRegressor"
            stacking_model.final_estimator_ = self.mock_model

            explainer = create_tree_explainer(stacking_model, self.X_background, "raw")

            self.assertIsNotNone(explainer)
            mock_tree_explainer.assert_called_once()

            # Should use final_estimator_
            call_args = mock_tree_explainer.call_args[1]
            self.assertEqual(call_args['model'], self.mock_model)

    def test_create_tree_explainer_unfitted_stacking(self):
        """Test error handling for unfitted StackingRegressor."""
        stacking_model = MagicMock()
        stacking_model.__class__.__name__ = "StackingRegressor"
        # Make it so accessing final_estimator_ raises AttributeError
        del stacking_model.final_estimator_

        with self.assertRaises(ValueError) as context:
            create_tree_explainer(stacking_model, self.X_background, "raw")

        self.assertIn("must be fitted", str(context.exception))

    def test_calculate_global_shap_values(self):
        """Test global SHAP values calculation."""
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(len(self.X_test), len(self.X_test.columns))
        mock_explainer.return_value = mock_shap_values

        result = calculate_global_shap_values(mock_explainer, self.X_test)

        self.assertIsNotNone(result)
        mock_explainer.assert_called_once_with(self.X_test)

    def test_calculate_global_shap_values_failure(self):
        """Test error handling in global SHAP calculation."""
        mock_explainer = MagicMock()
        mock_explainer.side_effect = Exception("SHAP calculation failed")

        with self.assertRaises(ValueError) as context:
            calculate_global_shap_values(mock_explainer, self.X_test)

        self.assertIn("Failed to calculate global SHAP values", str(context.exception))

    def test_calculate_local_shap_values(self):
        """Test local SHAP values calculation."""
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(1, len(self.X_test.columns))
        mock_explainer.return_value = mock_shap_values

        single_instance = self.X_test.iloc[[0]]
        result = calculate_local_shap_values(mock_explainer, single_instance)

        self.assertIsNotNone(result)
        mock_explainer.assert_called_once_with(single_instance)

    def test_calculate_local_shap_values_multiple_rows(self):
        """Test error handling for local calculation with multiple rows."""
        mock_explainer = MagicMock()

        with self.assertRaises(ValueError) as context:
            calculate_local_shap_values(mock_explainer, self.X_test.iloc[:2])

        self.assertIn("exactly one row", str(context.exception))

    def test_get_feature_importance_from_shap(self):
        """Test feature importance extraction."""
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(len(self.X_test), len(self.X_test.columns))
        mock_shap_values.feature_names = self.X_test.columns.tolist()

        importance = get_feature_importance_from_shap(mock_shap_values)

        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), len(self.X_test.columns))
        self.assertIn('feature', importance.columns)
        self.assertIn('mean_abs_shap_value', importance.columns)

    def test_get_feature_importance_from_shap_custom_names(self):
        """Test feature importance extraction with custom feature names."""
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(len(self.X_test), len(self.X_test.columns))

        # Use correct number of feature names to match the data
        custom_names = [f'feature_{i}' for i in range(len(self.X_test.columns))]
        importance = get_feature_importance_from_shap(mock_shap_values, custom_names)

        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), len(custom_names))
        # Check that all custom names are present (order may differ due to sorting by importance)
        self.assertCountEqual(importance['feature'].tolist(), custom_names)

    def test_get_feature_importance_no_values(self):
        """Test error handling when SHAP values have no values attribute."""
        mock_shap_values = MagicMock()
        del mock_shap_values.values  # Remove values attribute

        with self.assertRaises(ValueError) as context:
            get_feature_importance_from_shap(mock_shap_values, self.X_test.columns.tolist())

        self.assertIn("does not contain 'values' attribute", str(context.exception))

    def test_create_shap_summary_plot(self):
        """Test SHAP summary plot creation."""
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(len(self.X_test), len(self.X_test.columns))

        with patch('nba_predictor.explainability.shap_explainer.shap.summary_plot') as mock_summary_plot:
            with patch('nba_predictor.explainability.shap_explainer.plt.subplots') as mock_subplots:
                mock_fig, mock_ax = MagicMock(), MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                fig = create_shap_summary_plot(mock_shap_values, self.X_test, plot_type="bar")

                self.assertIsNotNone(fig)
                mock_summary_plot.assert_called_once()
                mock_subplots.assert_called_once()

    def test_create_shap_summary_plot_no_matplotlib(self):
        """Test error handling when matplotlib is not available."""
        mock_shap_values = MagicMock()

        with patch('nba_predictor.explainability.shap_explainer.plt', None):
            with self.assertRaises(ValueError) as context:
                create_shap_summary_plot(mock_shap_values, self.X_test)

            self.assertIn("matplotlib is required", str(context.exception))

    def test_create_waterfall_plot(self):
        """Test waterfall plot creation."""
        mock_shap_values = MagicMock()
        mock_shap_values.__getitem__ = MagicMock(return_value=MagicMock())
        mock_shap_values.__len__ = MagicMock(return_value=len(self.X_test))

        with patch('nba_predictor.explainability.shap_explainer.shap.waterfall_plot') as mock_waterfall:
            with patch('nba_predictor.explainability.shap_explainer.plt.figure') as mock_figure:
                mock_fig = MagicMock()
                mock_figure.return_value = mock_fig

                fig = create_waterfall_plot(mock_shap_values, instance_idx=0)

                self.assertIsNotNone(fig)
                mock_waterfall.assert_called_once()

    def test_create_waterfall_plot_out_of_range(self):
        """Test error handling for out-of-range instance index."""
        mock_shap_values = MagicMock()
        mock_shap_values.__len__ = MagicMock(return_value=5)  # Only 5 instances

        with self.assertRaises(ValueError) as context:
            create_waterfall_plot(mock_shap_values, instance_idx=10)

        self.assertIn("out of range", str(context.exception))

    def test_generate_nba_explanation_report(self):
        """Test comprehensive NBA explanation report generation."""
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.randn(len(self.X_test), len(self.X_test.columns))

        with patch('nba_predictor.explainability.shap_explainer.calculate_global_shap_values') as mock_global:
            with patch('nba_predictor.explainability.shap_explainer.get_feature_importance_from_shap') as mock_importance:
                mock_global.return_value = mock_shap_values
                mock_importance.return_value = pd.DataFrame({
                    'feature': self.X_test.columns,
                    'mean_abs_shap_value': np.random.rand(len(self.X_test.columns))
                })

                predictions = np.random.randn(len(self.X_test))
                report = generate_nba_explanation_report(
                    mock_explainer, self.X_test,
                    pd.Series(self.target[:len(self.X_test)]),
                    predictions
                )

                self.assertIsInstance(report, dict)
                self.assertIn('model_performance', report)
                self.assertIn('feature_importance', report)
                self.assertIn('shap_summary', report)
                self.assertIn('data_info', report)

                # Check model performance metrics
                self.assertIn('mae', report['model_performance'])
                self.assertIn('mse', report['model_performance'])
                self.assertIn('rmse', report['model_performance'])
                self.assertIn('samples', report['model_performance'])

    def test_generate_nba_explanation_report_failure(self):
        """Test error handling in report generation."""
        mock_explainer = MagicMock()

        with patch('nba_predictor.explainability.shap_explainer.calculate_global_shap_values') as mock_global:
            mock_global.side_effect = Exception("Report generation failed")

            with self.assertRaises(ValueError) as context:
                generate_nba_explanation_report(
                    mock_explainer, self.X_test,
                    pd.Series(self.target[:len(self.X_test)]),
                    np.random.randn(len(self.X_test))
                )

            self.assertIn("Failed to generate NBA explanation report", str(context.exception))

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            explainer = create_nba_shap_explainer(self.mock_model, self.X_background)

            self.assertIsNotNone(explainer)
            mock_tree_explainer.assert_called_once()

    def test_different_model_outputs(self):
        """Test different model output types."""
        for output_type in ["raw", "probability", "log_odds"]:
            with self.subTest(output_type=output_type):
                with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
                    mock_explainer = MagicMock()
                    mock_tree_explainer.return_value = mock_explainer

                    # Use a generic model (not XGBoost) to avoid model_output override
                    generic_model = MagicMock()
                    generic_model.__class__.__name__ = "GenericModel"
                    generic_model.predict.return_value = np.random.randn(len(self.X_background))

                    explainer = create_nba_shap_explainer(
                        generic_model, self.X_background, model_output=output_type
                    )
                    self.assertIsNotNone(explainer)

                    # Check that the model_output was passed correctly
                    call_args = mock_tree_explainer.call_args[1]
                    self.assertEqual(call_args['model_output'], output_type)

    def test_warning_for_unsupported_model(self):
        """Test warning generation for unsupported model types."""
        self.mock_model.__class__.__name__ = "UnsupportedModel"

        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            with patch('nba_predictor.explainability.shap_explainer.logger') as mock_logger:
                mock_explainer = MagicMock()
                mock_tree_explainer.return_value = mock_explainer

                create_tree_explainer(self.mock_model, self.X_background, "raw")

                # Should log a warning for unsupported model
                mock_logger.warning.assert_called()

    def test_nba_specific_feature_perturbation(self):
        """Test that NBA-specific feature perturbation is used."""
        with patch('nba_predictor.explainability.shap_explainer.shap.TreeExplainer') as mock_tree_explainer:
            mock_explainer = MagicMock()
            mock_tree_explainer.return_value = mock_explainer

            create_nba_shap_explainer(self.mock_model, self.X_background)

            # Check that intervention perturbation is used for NBA stability
            call_args = mock_tree_explainer.call_args[1]
            self.assertEqual(call_args['feature_perturbation'], "interventional")


if __name__ == '__main__':
    unittest.main()