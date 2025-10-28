#!/usr/bin/env python3
"""
🏀 LightGBM Model - Context7 Compliant
LightGBM model wrapper optimized for NBA over/under predictions.

This module implements:
- NBA-optimized LightGBM hyperparameters
- Prevention of overfitting on limited NBA data
- Proper error handling for missing dependencies
- Research-based parameter configuration
"""

import logging
from typing import Optional, Any, Dict, Union

try:
    import lightgbm as lgb
    _lightgbm_available = True
    LIGHTGBM_IMPORT_ERROR: Optional[str] = None
except ImportError as e:
    lgb = None  # type: ignore
    _lightgbm_available = False
    LIGHTGBM_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_nba_lightgbm_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    random_state: int = 42
) -> Union[Any, 'lgb.LGBMRegressor']:
    """
    Create LightGBM model optimized for NBA over/under predictions.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for shrinkage
        num_leaves: Maximum number of leaves in one tree
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility

    Returns:
        Configured LightGBM regressor

    Raises:
        ImportError: If LightGBM not installed
        ValueError: If parameters are invalid

    Example:
        >>> model = create_nba_lightgbm_model(n_estimators=200, learning_rate=0.05)
        >>> model.fit(X_train, y_train)
    """
    if not _lightgbm_available or lgb is None:
        logger.error(
            "LightGBM not available",
            extra={"error": LIGHTGBM_IMPORT_ERROR}
        )
        raise ImportError(
            f"LightGBM is required but not installed. Install with: "
            f"/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/pip install lightgbm"
        )

    try:
        # Validate parameters first
        validate_lightgbm_parameters(n_estimators, learning_rate, num_leaves, max_depth, random_state)

        # NBA-optimized parameters based on research
        params: Dict[str, Any] = {
            'objective': 'regression',
            'metric': ['l1', 'l2'],
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)

        logger.info(
            "LightGBM model created successfully",
            extra={
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "random_state": random_state
            }
        )

        return model

    except ValueError as e:
        logger.error(
            "LightGBM model creation failed",
            extra={
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "random_state": random_state,
                "error": str(e)
            }
        )
        raise ValueError(f"Invalid LightGBM parameters: {e}") from e


def validate_lightgbm_parameters(
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    random_state: int
) -> None:
    """
    Validate LightGBM parameters for NBA data.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for shrinkage
        num_leaves: Maximum number of leaves in one tree
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility

    Raises:
        ValueError: If parameters are invalid for NBA data
    """
    if n_estimators < 10:
        raise ValueError("n_estimators must be at least 10 for meaningful NBA predictions")

    if n_estimators > 10000:
        logger.warning("Very large n_estimators (%d) may cause overfitting", n_estimators)

    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError("learning_rate must be between 0 and 1")

    if learning_rate > 0.3:
        logger.warning("High learning rate (%.3f) may cause overfitting", learning_rate)

    if num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")

    if num_leaves > 1000:
        logger.warning("Very high num_leaves (%d) may cause overfitting", num_leaves)

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    if max_depth > 20:
        logger.warning("Very deep trees (max_depth=%d) may overfit", max_depth)

    if random_state < 0:
        raise ValueError("random_state must be non-negative")


def get_nba_optimized_params() -> Dict[str, Any]:
    """
    Get NBA-optimized LightGBM parameters based on research.

    Returns:
        Dictionary of optimized parameters for NBA predictions
    """
    return {
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'verbose': -1
    }


def create_lightgbm_for_time_series(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    random_state: int = 42
) -> Union[Any, 'lgb.LGBMRegressor']:
    """
    Create LightGBM model specifically configured for time series NBA data.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for shrinkage
        random_state: Random seed for reproducibility

    Returns:
        LightGBM model configured for time series data

    Raises:
        ImportError: If LightGBM not installed
        ValueError: If parameters are invalid
    """
    if not _lightgbm_available or lgb is None:
        raise ImportError(
            f"LightGBM is required but not installed. Install with: "
            f"/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/pip install lightgbm"
        )

    # More conservative parameters for time series data
    params: Dict[str, Any] = {
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': 15,  # More conservative
        'max_depth': 4,    # More conservative
        'min_child_samples': 30,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.2,  # Stronger regularization
        'reg_lambda': 0.2, # Stronger regularization
        'random_state': random_state,
        'n_jobs': -1,
        'verbose': -1
    }

    return lgb.LGBMRegressor(**params)