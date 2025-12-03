#!/usr/bin/env python3
"""
🤖 NBA Machine Learning Models
Context7-compliant ensemble models for NBA predictive analytics.

DEPRECATED: This module is deprecated and will be removed.
Use production models from:
- src/nba_predictor/models/lightgbm_model.py
- src/nba_predictor/ensemble/nba_ensemble_predictor.py
- src/nba_predictor/models/stacked_ensemble.py
"""

import warnings

warnings.warn(
    "src.models is deprecated. Use src.nba_predictor.models and src.nba_predictor.ensemble instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Redirect to production models
from nba_predictor.ensemble.nba_ensemble_predictor import (
    NBAEnsemblePredictor as NBAEnsembleModel,
)
from nba_predictor.models.lightgbm_model import create_nba_lightgbm_model
from nba_predictor.models.stacked_ensemble import create_research_stacked_ensemble

__all__ = [
    "NBAEnsembleModel",
    "create_nba_lightgbm_model",
    "create_research_stacked_ensemble",
]
