"""
NBA Explainability Engine with SHAP Integration.

Context7 compliant SHAP-based explainability system for NBA predictive models.
Provides global and local explanations using SHAP values and visualizations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, overload
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logger = logging.getLogger(__name__)


class ExplainabilityError(Exception):
    """Custom exception for explainability engine errors."""
    pass


class NBAExplainabilityEngine:
    """
    SHAP-based explainability engine for NBA predictive models.

    Provides global and local explanations for model predictions
    using SHAP values and interactive visualizations.

    Attributes:
        explainer: SHAP explainer instance
        model: Trained predictive model
        feature_names: List of feature names
        background_data: Background dataset for SHAP explanations
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize the explainability engine.

        Args:
            model: Trained predictive model (tree-based ensemble recommended)
            feature_names: List of feature names for explanations
            background_data: Background dataset for SHAP explanations

        Example:
            >>> engine = NBAExplainabilityEngine(
            ...     model=trained_model,
            ...     feature_names=feature_columns,
            ...     background_data=X_train.sample(100)
            ... )
            >>> shap_values = engine.calculate_shap_values(X_test)
        """
        try:
            logger.info("Initializing NBA Explainability Engine")

            self.model = model
            self.feature_names = feature_names
            self.background_data = background_data

            # Initialize SHAP explainer based on model type
            self.explainer = self._create_shap_explainer()

            # Cache for computed explanations
            self._explanation_cache: Dict[str, Any] = {}

            logger.info("NBA Explainability Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize explainability engine: {e}")
            raise ExplainabilityError(f"Explainability engine initialization failed: {str(e)}") from e

    def _create_shap_explainer(self) -> shap.Explainer:
        """
        Create appropriate SHAP explainer based on model type.

        Returns:
            Configured SHAP explainer instance

        Raises:
            ExplainabilityError: If no suitable explainer can be created
        """
        try:
            logger.info("Creating SHAP explainer")

            # Use TreeExplainer for tree-based models (XGBoost, RandomForest, etc.)
            if hasattr(self.model, 'feature_importances_') or 'xgboost' in str(type(self.model)).lower():
                logger.info("Using TreeExplainer for tree-based model")
                if self.background_data is not None:
                    return shap.TreeExplainer(
                        self.model,
                        data=self.background_data.values,
                        feature_names=self.feature_names
                    )
                else:
                    return shap.TreeExplainer(self.model, feature_names=self.feature_names)

            # Fallback to generic Explainer
            logger.info("Using generic Explainer")
            if self.background_data is not None:
                return shap.Explainer(
                    self.model,
                    data=self.background_data.values,
                    feature_names=self.feature_names
                )
            else:
                return shap.Explainer(self.model, feature_names=self.feature_names)

        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise ExplainabilityError(f"SHAP explainer creation failed: {str(e)}") from e

    def calculate_shap_values(
        self,
        data: pd.DataFrame,
        use_cache: bool = True
    ) -> Union[np.ndarray, Any]:
        """
        Calculate SHAP values for given data.

        Args:
            data: Input data for explanation
            use_cache: Whether to use cached explanations

        Returns:
            SHAP values array

        Raises:
            ExplainabilityError: If SHAP calculation fails
        """
        try:
            logger.info(f"Calculating SHAP values for {len(data)} samples")

            # Create cache key
            data_hash = hash(str(data.values.tobytes()))
            cache_key = f"shap_values_{data_hash}_{datetime.now().strftime('%Y%m%d')}"

            # Check cache
            if use_cache and cache_key in self._explanation_cache:
                logger.info("Using cached SHAP values")
                return self._explanation_cache[cache_key]

            # Calculate SHAP values
            shap_values = self.explainer(data)

            # Extract values from Explanation object (SHAP 0.44+)
            if hasattr(shap_values, 'values'):
                values_array = shap_values.values
            else:
                values_array = np.asarray(shap_values)

            # Cache results
            if use_cache:
                self._explanation_cache[cache_key] = values_array

            logger.info(f"SHAP values calculated successfully. Shape: {values_array.shape}")
            return values_array

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            raise ExplainabilityError(f"SHAP calculation failed: {str(e)}") from e

    def generate_global_explanation(
        self,
        shap_values: np.ndarray,
        plot_type: str = "beeswarm"
    ) -> Dict[str, Any]:
        """
        Generate global feature importance explanations.

        Args:
            shap_values: Calculated SHAP values
            plot_type: Type of plot ('beeswarm', 'bar', 'violin')

        Returns:
            Dictionary containing global explanation results

        Raises:
            ExplainabilityError: If global explanation generation fails
        """
        try:
            logger.info(f"Generating global explanation with plot type: {plot_type}")

            # Handle multi-class SHAP values (3D array: samples x features x classes)
            if len(shap_values.shape) == 3:
                # For binary classification, use positive class
                if shap_values.shape[2] == 2:
                    shap_values_2d = shap_values[:, :, 1]  # Use positive class
                else:
                    shap_values_2d = shap_values[:, :, 0]  # Use first class
            else:
                shap_values_2d = shap_values

            # Calculate mean absolute SHAP values for feature importance
            mean_shap_values = np.mean(np.abs(shap_values_2d), axis=0)

            # Ensure we have a 1D array for DataFrame creation
            if len(mean_shap_values.shape) > 1:
                mean_shap_values = mean_shap_values.flatten()

            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap_values
            }).sort_values('importance', ascending=False)

            # Generate visualization based on plot type
            fig = self._create_global_plot(shap_values, plot_type, feature_importance)

            # Calculate summary statistics
            summary_stats = {
                'top_features': feature_importance.head(10).to_dict('records'),
                'total_features': len(self.feature_names),
                'mean_importance': np.mean(mean_shap_values),
                'max_importance': np.max(mean_shap_values),
                'min_importance': np.min(mean_shap_values)
            }

            result = {
                'feature_importance': feature_importance,
                'plot': fig,
                'summary_stats': summary_stats,
                'plot_type': plot_type
            }

            logger.info("Global explanation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate global explanation: {e}")
            raise ExplainabilityError(f"Global explanation generation failed: {str(e)}") from e

    def _create_global_plot(
        self,
        shap_values: np.ndarray,
        plot_type: str,
        feature_importance: pd.DataFrame
    ) -> Figure:
        """
        Create global visualization plot.

        Args:
            shap_values: SHAP values array
            plot_type: Type of plot to create
            feature_importance: Feature importance DataFrame

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))

        if plot_type == "bar":
            # Bar plot of feature importance
            top_features = feature_importance.head(15)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Global Feature Importance (Top 15 Features)')

        elif plot_type == "beeswarm":
            # Beeswarm plot using SHAP's built-in visualization
            # Create a simple version for compatibility
            ax.scatter(feature_importance['importance'], range(len(feature_importance)))
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Feature Importance (Beeswarm Style)')

        elif plot_type == "violin":
            # Violin plot style for feature distribution
            ax.violinplot([shap_values[:, i] for i in range(min(10, len(self.feature_names)))])
            ax.set_xticks(range(1, min(11, len(self.feature_names) + 1)))
            ax.set_xticklabels(self.feature_names[:10], rotation=45)
            ax.set_title('SHAP Value Distribution (Top 10 Features)')

        plt.tight_layout()
        return fig

  
    def explain_single_prediction(
        self,
        game_features: Union[pd.Series, List[Any], np.ndarray],
        prediction: float,
        feature_values: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Explain a single game prediction with detailed SHAP analysis.

        Args:
            game_features: Feature values for the game
            prediction: Model prediction for the game
            feature_values: Original feature values (if different from game_features)

        Returns:
            Dictionary containing detailed explanation for single prediction

        Raises:
            ExplainabilityError: If single prediction explanation fails
        """
        try:
            logger.info("Generating single prediction explanation")

            # Context7: Convert to DataFrame using cast-based type handling
            # This approach avoids mypy --strict unreachable statement false positives
            if isinstance(game_features, pd.Series):
                # Handle Series input with proper column preservation
                game_df = pd.DataFrame([game_features.values], columns=game_features.index)
            else:
                # Context7: Use cast for remaining Union types per Context7 mypy best practices
                # This avoids unreachable statement warnings with complex Union types
                from typing import cast

                # Cast to list-like type and handle uniformly
                # mypy: ignore[unreachable]  # Context7: False positive due to Union type analysis limitations
                features_array = cast(list[Any], game_features)
                game_df = pd.DataFrame([features_array], columns=self.feature_names)

            # Calculate SHAP values for this instance
            shap_values = self.calculate_shap_values(game_df, use_cache=False)

            # Handle multi-class SHAP values for single instance
            if len(shap_values.shape) == 3:
                # For binary classification, use positive class
                if shap_values.shape[2] == 2:
                    instance_shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
                else:
                    instance_shap_values = shap_values[0, :, 0]  # First sample, all features, first class
            elif len(shap_values.shape) == 2:
                instance_shap_values = shap_values[0]  # First sample, all features
            else:
                instance_shap_values = shap_values

            # Ensure we have a 1D array for DataFrame creation
            if len(instance_shap_values.shape) > 1:
                instance_shap_values = instance_shap_values.flatten()

            # Create feature contribution DataFrame
            contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': instance_shap_values,
                'feature_value': game_df.iloc[0].values
            })

            # Sort by absolute contribution
            contributions['abs_shap'] = np.abs(contributions['shap_value'])
            contributions = contributions.sort_values('abs_shap', ascending=False)

            # Create force plot data
            force_plot_data = self._create_force_plot_data(
                contributions, prediction
            )

            # Calculate explanation summary
            explanation_summary = {
                'prediction': prediction,
                'base_value': getattr(self.explainer, 'expected_value', 0.0),
                'top_positive_features': contributions[contributions['shap_value'] > 0].head(5).to_dict('records'),
                'top_negative_features': contributions[contributions['shap_value'] < 0].tail(5).to_dict('records'),
                'total_positive_impact': contributions[contributions['shap_value'] > 0]['shap_value'].sum(),
                'total_negative_impact': contributions[contributions['shap_value'] < 0]['shap_value'].sum()
            }

            result = {
                'contributions': contributions,
                'force_plot_data': force_plot_data,
                'explanation_summary': explanation_summary,
                'game_features': game_df.iloc[0].to_dict()
            }

            logger.info("Single prediction explanation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to explain single prediction: {e}")
            raise ExplainabilityError(f"Single prediction explanation failed: {str(e)}") from e

    def _create_force_plot_data(
        self,
        contributions: pd.DataFrame,
        prediction: float
    ) -> Dict[str, Any]:
        """
        Create data for force plot visualization.

        Args:
            contributions: Feature contributions DataFrame
            prediction: Model prediction value

        Returns:
            Dictionary containing force plot data
        """
        # Prepare data for force plot
        base_value = getattr(self.explainer, 'expected_value', 0.0)

        # Create force plot segments
        segments = []
        current_value = base_value

        for _, row in contributions.iterrows():
            next_value = current_value + row['shap_value']
            segments.append({
                'feature': row['feature'],
                'value': row['feature_value'],
                'shap_value': row['shap_value'],
                'start': current_value,
                'end': next_value,
                'contribution_type': 'positive' if row['shap_value'] > 0 else 'negative'
            })
            current_value = next_value

        return {
            'base_value': base_value,
            'prediction': prediction,
            'segments': segments,
            'total_positive': sum(s['shap_value'] for s in segments if s['shap_value'] > 0),
            'total_negative': sum(s['shap_value'] for s in segments if s['shap_value'] < 0)
        }

    def create_interactive_plotly_explanation(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame,
        plot_type: str = "waterfall"
    ) -> go.Figure:
        """
        Create interactive Plotly visualization of SHAP explanations.

        Args:
            shap_values: SHAP values array
            data: Input data
            plot_type: Type of plot ('waterfall', 'scatter', 'bar')

        Returns:
            Plotly figure object

        Raises:
            ExplainabilityError: If interactive plot creation fails
        """
        try:
            logger.info(f"Creating interactive Plotly explanation: {plot_type}")

            if plot_type == "waterfall":
                fig = self._create_waterfall_plot(shap_values, data)
            elif plot_type == "scatter":
                fig = self._create_scatter_plot(shap_values, data)
            elif plot_type == "bar":
                fig = self._create_bar_plot(shap_values)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            logger.info("Interactive Plotly explanation created successfully")
            return fig

        except Exception as e:
            logger.error(f"Failed to create interactive plot: {e}")
            raise ExplainabilityError(f"Interactive plot creation failed: {str(e)}") from e

    def _create_waterfall_plot(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame
    ) -> go.Figure:
        """Create waterfall plot for SHAP values."""
        # Use first instance for waterfall plot
        instance_idx = 0
        instance_shap = shap_values[instance_idx]
        instance_data = data.iloc[instance_idx]

        # Calculate base value and prediction
        base_value = getattr(self.explainer, 'expected_value', 0.0)
        prediction = base_value + instance_shap.sum()

        # Prepare waterfall data
        features = []
        values = []
        colors = []

        # Add base value
        features.append('Base Value')
        values.append(base_value)
        colors.append('lightgray')

        # Add features sorted by absolute SHAP value
        abs_shap = np.abs(instance_shap)
        sorted_indices = np.argsort(abs_shap)[::-1]

        for idx in sorted_indices[:15]:  # Top 15 features
            feature_name = self.feature_names[idx]
            shap_val = instance_shap[idx]

            features.append(feature_name)
            values.append(shap_val)
            colors.append('red' if shap_val > 0 else 'blue')

        # Add final prediction
        features.append('Prediction')
        values.append(prediction)
        colors.append('darkgreen')

        # Create waterfall plot
        fig = go.Figure(go.Waterfall(
            name="SHAP Waterfall",
            orientation="v",
            measure=["relative"] * (len(features) - 1) + ["total"],
            x=features,
            y=values,
            textposition="outside",
            text=[f"{val:.3f}" for val in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="SHAP Waterfall Plot - Single Game Prediction",
            xaxis_title="Features",
            yaxis_title="SHAP Values",
            height=600
        )

        return fig

    def _create_scatter_plot(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame
    ) -> go.Figure:
        """Create scatter plot of SHAP values vs feature values."""
        # Select top features by importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:]  # Top 10 features

        fig = go.Figure()

        for idx in top_indices:
            feature_name = self.feature_names[idx]
            feature_vals = data.iloc[:, idx].values
            shap_vals = shap_values[:, idx]

            fig.add_trace(go.Scatter(
                x=feature_vals,
                y=shap_vals,
                mode='markers',
                name=feature_name,
                opacity=0.7
            ))

        fig.update_layout(
            title="SHAP Values vs Feature Values (Top 10 Features)",
            xaxis_title="Feature Value",
            yaxis_title="SHAP Value",
            height=500
        )

        return fig

    def _create_bar_plot(self, shap_values: np.ndarray) -> go.Figure:
        """Create bar plot of mean absolute SHAP values."""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:15]  # Top 15

        fig = go.Figure(go.Bar(
            x=[self.feature_names[i] for i in sorted_indices],
            y=[mean_abs_shap[i] for i in sorted_indices],
            orientation='v'
        ))

        fig.update_layout(
            title="Global Feature Importance (Mean |SHAP Value|)",
            xaxis_title="Features",
            yaxis_title="Mean |SHAP Value|",
            height=500
        )

        fig.update_xaxes(tickangle=45)

        return fig

    def get_explanation_summary(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get comprehensive summary of SHAP explanations.

        Args:
            shap_values: SHAP values array
            data: Input data

        Returns:
            Dictionary containing comprehensive explanation summary
        """
        try:
            logger.info("Generating comprehensive explanation summary")

            # Basic statistics
            summary = {
                'data_info': {
                    'num_samples': len(data),
                    'num_features': len(self.feature_names),
                    'feature_names': self.feature_names
                },
                'shap_statistics': {
                    'mean_abs_shap': np.mean(np.abs(shap_values)),
                    'max_abs_shap': np.max(np.abs(shap_values)),
                    'min_shap': np.min(shap_values),
                    'max_shap': np.max(shap_values),
                    'shap_std': np.std(shap_values)
                },
                'feature_importance': {},
                'interaction_insights': {}
            }

            # Feature importance ranking
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_ranking = sorted(
                zip(self.feature_names, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )

            summary['feature_importance'] = {
                'top_5_features': feature_ranking[:5],
                'bottom_5_features': feature_ranking[-5:],
                'feature_ranking': feature_ranking
            }

            # Prediction distribution insights
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                predictions = base_value + np.sum(shap_values, axis=1)

                summary['prediction_insights'] = {
                    'base_value': base_value,
                    'prediction_mean': np.mean(predictions),
                    'prediction_std': np.std(predictions),
                    'prediction_range': [np.min(predictions), np.max(predictions)]
                }

            logger.info("Explanation summary generated successfully")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate explanation summary: {e}")
            raise ExplainabilityError(f"Explanation summary generation failed: {str(e)}") from e

    def clear_cache(self) -> None:
        """Clear explanation cache to free memory."""
        self._explanation_cache.clear()
        logger.info("Explanation cache cleared")