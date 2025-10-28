#!/usr/bin/env python3
"""
🏀 NBA Predictors Explainability Module

This module provides SHAP-based model explainability for NBA predictions.
"""

from .shap_explainer import (
    create_nba_shap_explainer,
    calculate_global_shap_values,
    calculate_local_shap_values,
    get_feature_importance_from_shap,
    create_shap_summary_plot,
    create_waterfall_plot,
    generate_nba_explanation_report,
    validate_explainer_inputs,
    create_tree_explainer
)

__all__ = [
    "create_nba_shap_explainer",
    "calculate_global_shap_values",
    "calculate_local_shap_values",
    "get_feature_importance_from_shap",
    "create_shap_summary_plot",
    "create_waterfall_plot",
    "generate_nba_explanation_report",
    "validate_explainer_inputs",
    "create_tree_explainer"
]