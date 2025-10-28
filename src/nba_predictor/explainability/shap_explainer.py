#!/usr/bin/env python3
"""
🏀 SHAP Explainer - Context7 Compliant
SHAP-based model explainability system for NBA predictions.

This module implements:
- TreeExplainer for ensemble models (XGBoost, LightGBM, RandomForest)
- Global and local explanation methods
- Visualization integration for NBA predictions
- Proper error handling for missing dependencies
- Research-based explanation strategies
"""

import logging
from typing import Optional, Any, Dict, Union, List, Tuple
import warnings

try:
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    _shap_available = True
    SHAP_IMPORT_ERROR: Optional[str] = None
except ImportError as e:
    shap = None
    plt = None  # type: ignore
    np = None  # type: ignore
    pd = None  # type: ignore
    _shap_available = False
    SHAP_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings for cleaner output
if plt is not None:
    plt.rcParams['figure.max_open_warning'] = 0


def create_nba_shap_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_output: str = "raw"
) -> 'shap.Explainer':
    """
    Create SHAP explainer for NBA prediction models.

    Args:
        model: Trained model to explain
        X_background: Background dataset for explanation
        model_output: Type of model output to explain

    Returns:
        Configured SHAP explainer

    Raises:
        ImportError: If SHAP not installed
        ValueError: If model type unsupported

    Example:
        >>> explainer = create_nba_shap_explainer(model, X_train)
        >>> shap_values = explainer(X_test)
        >>> shap.plots.waterfall(shap_values[0])
    """
    if not _shap_available or shap is None:
        logger.error(
            "SHAP not available",
            extra={"error": SHAP_IMPORT_ERROR}
        )
        raise ImportError(
            f"SHAP is required but not installed. Install with: "
            f"/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/pip install shap"
        )

    try:
        # Validate inputs
        validate_explainer_inputs(model, X_background, model_output)

        # Determine explainer type based on model
        explainer = create_tree_explainer(model, X_background, model_output)

        logger.info(
            "NBA SHAP explainer created successfully",
            extra={
                "model_type": type(model).__name__,
                "background_size": len(X_background),
                "model_output": model_output
            }
        )

        return explainer

    except (ValueError, TypeError) as e:
        logger.error(
            "SHAP explainer creation failed",
            extra={
                "model_type": type(model).__name__,
                "background_shape": X_background.shape,
                "model_output": model_output,
                "error": str(e)
            }
        )
        raise ValueError(f"SHAP explainer creation failed: {e}") from e


def validate_explainer_inputs(
    model: Any,
    X_background: pd.DataFrame,
    model_output: str
) -> None:
    """
    Validate inputs for SHAP explainer creation.

    Args:
        model: Trained model to validate
        X_background: Background dataset to validate
        model_output: Model output type to validate

    Raises:
        ValueError: If inputs are invalid
    """
    if model is None:
        raise ValueError("Model cannot be None")

    if X_background is None or X_background.empty:
        raise ValueError("Background dataset cannot be None or empty")

    if not isinstance(X_background, pd.DataFrame):
        raise ValueError("Background dataset must be a pandas DataFrame")

    if len(X_background) < 10:
        raise ValueError("Background dataset should have at least 10 samples for stable explanations")

    if model_output not in ["raw", "probability", "log_odds"]:
        raise ValueError("model_output must be one of: 'raw', 'probability', 'log_odds'")

    # Check if model is fitted (has required attributes)
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a 'predict' method")

    logger.debug(
        "Explainer inputs validated",
        extra={
            "model_type": type(model).__name__,
            "background_features": len(X_background.columns),
            "background_samples": len(X_background),
            "model_output": model_output
        }
    )


def create_tree_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_output: str
) -> 'shap.Explainer':
    """
    Create TreeExplainer for tree-based models.

    Args:
        model: Tree-based model (XGBoost, LightGBM, RandomForest)
        X_background: Background dataset
        model_output: Type of model output

    Returns:
        Configured TreeExplainer

    Raises:
        ValueError: If model type is not supported
    """
    model_name = type(model).__name__.lower()

    # Check for supported tree models
    supported_models = ['xgbregressor', 'lgbmregressor', 'randomforestregressor', 'decisiontreeregressor']

    if not any(supported in model_name for supported in supported_models):
        logger.warning(
            f"Model type {type(model).__name__} may not be optimally supported by TreeExplainer"
        )

    # Handle StackingRegressor (special case)
    if 'stackingregressor' in model_name:
        logger.info("Creating explainer for StackingRegressor - using final estimator")
        if hasattr(model, 'final_estimator_'):
            model = model.final_estimator_
        else:
            raise ValueError("StackingRegressor must be fitted before creating explainer")

    # Create explainer with appropriate parameters
    explainer_kwargs = {
        'model': model,
        'data': X_background,
        'model_output': model_output,
        'feature_perturbation': "interventional"  # More stable for NBA predictions
    }

    # Add model-specific parameters
    if 'xgb' in model_name:
        explainer_kwargs['model_output'] = "raw"  # XGBoost works best with raw output
    elif 'lgb' in model_name:
        explainer_kwargs['model_output'] = "raw"  # LightGBM works best with raw output

    explainer = shap.TreeExplainer(**explainer_kwargs)

    logger.debug(
        "TreeExplainer created",
        extra={
            "explainer_type": "TreeExplainer",
            "model_output": explainer_kwargs['model_output'],
            "feature_perturbation": explainer_kwargs['feature_perturbation']
        }
    )

    return explainer


def calculate_global_shap_values(
    explainer: 'shap.Explainer',
    X_test: pd.DataFrame
) -> 'shap.Explanation':
    """
    Calculate global SHAP values for test dataset.

    Args:
        explainer: Configured SHAP explainer
        X_test: Test dataset for explanation

    Returns:
        SHAP values object for global explanations

    Raises:
        ValueError: If calculation fails
    """
    try:
        shap_values = explainer(X_test)

        logger.info(
            "Global SHAP values calculated",
            extra={
                "test_samples": len(X_test),
                "features": len(X_test.columns),
                "shap_values_shape": shap_values.values.shape if hasattr(shap_values, 'values') else "N/A"
            }
        )

        return shap_values

    except Exception as e:
        logger.error(
            "Global SHAP calculation failed",
            extra={
                "test_shape": X_test.shape,
                "error": str(e)
            }
        )
        raise ValueError(f"Failed to calculate global SHAP values: {e}") from e


def calculate_local_shap_values(
    explainer: 'shap.Explainer',
    single_instance: pd.DataFrame
) -> 'shap.Explanation':
    """
    Calculate local SHAP values for single prediction.

    Args:
        explainer: Configured SHAP explainer
        single_instance: Single row DataFrame for local explanation

    Returns:
        SHAP values object for local explanation

    Raises:
        ValueError: If calculation fails
    """
    try:
        if len(single_instance) != 1:
            raise ValueError("single_instance must contain exactly one row")

        shap_values = explainer(single_instance)

        logger.debug(
            "Local SHAP values calculated",
            extra={
                "instance_shape": single_instance.shape,
                "shap_values_shape": shap_values.values.shape if hasattr(shap_values, 'values') else "N/A"
            }
        )

        return shap_values

    except Exception as e:
        logger.error(
            "Local SHAP calculation failed",
            extra={
                "instance_shape": single_instance.shape,
                "error": str(e)
            }
        )
        raise ValueError(f"Failed to calculate local SHAP values: {e}") from e


def get_feature_importance_from_shap(
    shap_values: 'shap.Explanation',
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract feature importance from SHAP values.

    Args:
        shap_values: SHAP values object
        feature_names: Optional list of feature names

    Returns:
        DataFrame with feature importance rankings

    Raises:
        ValueError: If extraction fails
    """
    try:
        if feature_names is None and hasattr(shap_values, 'feature_names'):
            feature_names = shap_values.feature_names

        if feature_names is None:
            raise ValueError("Feature names must be provided or available in shap_values")

        # Calculate mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        else:
            raise ValueError("SHAP values object does not contain 'values' attribute")

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap_value': mean_shap_values
        }).sort_values('mean_abs_shap_value', ascending=False)

        logger.info(
            "Feature importance extracted from SHAP",
            extra={
                "total_features": len(importance_df),
                "top_feature": importance_df.iloc[0]['feature'] if len(importance_df) > 0 else None
            }
        )

        return importance_df

    except Exception as e:
        logger.error(
            "Feature importance extraction failed",
            extra={"error": str(e)}
        )
        raise ValueError(f"Failed to extract feature importance: {e}") from e


def create_shap_summary_plot(
    shap_values: 'shap.Explanation',
    X_test: pd.DataFrame,
    plot_type: str = "bar",
    max_display: int = 20,
    show_plot: bool = False
) -> Optional[Any]:
    """
    Create SHAP summary plot for NBA predictions.

    Args:
        shap_values: SHAP values object
        X_test: Test dataset
        plot_type: Type of plot ("bar", "dot", "violin")
        max_display: Maximum number of features to display
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object (optional)

    Raises:
        ValueError: If plot creation fails
    """
    try:
        if plt is None:
            raise ImportError("matplotlib is required for plotting")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "bar":
            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="bar",
                max_display=max_display,
                show=False,
                plot_size=(10, 6)
            )
        elif plot_type == "dot":
            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="dot",
                max_display=max_display,
                show=False,
                plot_size=(10, 6)
            )
        elif plot_type == "violin":
            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="violin",
                max_display=max_display,
                show=False,
                plot_size=(10, 6)
            )
        else:
            raise ValueError("plot_type must be one of: 'bar', 'dot', 'violin'")

        plt.title(f"NBA Prediction Feature Importance ({plot_type.capitalize()} Plot)", fontsize=14, pad=20)
        plt.tight_layout()

        if show_plot:
            plt.show()

        logger.info(
            "SHAP summary plot created",
            extra={
                "plot_type": plot_type,
                "max_display": max_display,
                "features_shown": min(max_display, len(X_test.columns))
            }
        )

        return fig

    except Exception as e:
        logger.error(
            "SHAP summary plot creation failed",
            extra={
                "plot_type": plot_type,
                "error": str(e)
            }
        )
        raise ValueError(f"Failed to create SHAP summary plot: {e}") from e


def create_waterfall_plot(
    shap_values: 'shap.Explanation',
    instance_idx: int = 0,
    show_plot: bool = False
) -> Optional[Any]:
    """
    Create waterfall plot for single prediction explanation.

    Args:
        shap_values: SHAP values object
        instance_idx: Index of instance to explain
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object (optional)

    Raises:
        ValueError: If plot creation fails
    """
    try:
        if plt is None:
            raise ImportError("matplotlib is required for plotting")

        if instance_idx >= len(shap_values):
            raise ValueError(f"instance_idx {instance_idx} is out of range")

        # Create waterfall plot
        fig = plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap_values[instance_idx],
            max_display=20,
            show=False
        )
        plt.title(f"NBA Prediction Explanation - Instance {instance_idx}", fontsize=14, pad=20)
        plt.tight_layout()

        if show_plot:
            plt.show()

        logger.info(
            "SHAP waterfall plot created",
            extra={
                "instance_idx": instance_idx,
                "base_value": float(shap_values[instance_idx].base_values) if hasattr(shap_values[instance_idx], 'base_values') else None
            }
        )

        return fig

    except Exception as e:
        logger.error(
            "SHAP waterfall plot creation failed",
            extra={
                "instance_idx": instance_idx,
                "error": str(e)
            }
        )
        raise ValueError(f"Failed to create SHAP waterfall plot: {e}") from e


def generate_nba_explanation_report(
    explainer: 'shap.Explainer',
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    top_features: int = 10
) -> Dict[str, Any]:
    """
    Generate comprehensive NBA prediction explanation report.

    Args:
        explainer: Configured SHAP explainer
        X_test: Test dataset
        y_test: True values
        predictions: Model predictions
        top_features: Number of top features to highlight

    Returns:
        Dictionary containing comprehensive explanation report

    Raises:
        ValueError: If report generation fails
    """
    try:
        # Calculate SHAP values
        shap_values = calculate_global_shap_values(explainer, X_test)

        # Get feature importance
        feature_importance = get_feature_importance_from_shap(shap_values, X_test.columns.tolist())

        # Calculate prediction accuracy metrics
        mae = np.mean(np.abs(predictions - y_test))
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)

        # Generate report
        report = {
            "model_performance": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "samples": len(X_test)
            },
            "feature_importance": {
                "top_features": feature_importance.head(top_features).to_dict('records'),
                "total_features": len(feature_importance)
            },
            "shap_summary": {
                "mean_absolute_shap": float(np.abs(shap_values.values).mean()) if hasattr(shap_values, 'values') else None,
                "max_shap_value": float(np.abs(shap_values.values).max()) if hasattr(shap_values, 'values') else None,
                "feature_count": len(X_test.columns)
            },
            "data_info": {
                "background_size": getattr(explainer, 'background_data', None) is not None and len(getattr(explainer, 'background_data', [])),
                "test_features": list(X_test.columns),
                "prediction_range": [float(predictions.min()), float(predictions.max())]
            }
        }

        logger.info(
            "NBA explanation report generated",
            extra={
                "top_features": top_features,
                "mae": float(mae),
                "rmse": float(rmse)
            }
        )

        return report

    except Exception as e:
        logger.error(
            "NBA explanation report generation failed",
            extra={"error": str(e)}
        )
        raise ValueError(f"Failed to generate NBA explanation report: {e}") from e