#!/usr/bin/env python3
"""
🏀 Stacked Ensemble Model - Context7 Compliant
Research-based stacked ensemble for NBA over/under predictions.

This module implements:
- XGBoost + LightGBM + Random Forest + Ridge + MLP meta-learner
- Time series cross-validation integration
- NBA-optimized base model configurations
- Proper error handling for missing dependencies
- Research-based stacking architecture
"""

import logging
from typing import Optional, Any, Dict, Union, List

try:
    import xgboost as xgb

    _xgboost_available = True
except ImportError:
    xgb = None  # type: ignore
    _xgboost_available = False

try:
    import lightgbm as lgb

    _lightgbm_available = True
except ImportError:
    lgb = None  # type: ignore
    _lightgbm_available = False

try:
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import TimeSeriesSplit

    _sklearn_available = True
except ImportError:
    _sklearn_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_research_stacked_ensemble(
    cv_strategy: Optional[Any] = None, n_jobs: int = -1
) -> "StackingRegressor":
    """
    Create research-based stacked ensemble for NBA predictions.

    Args:
        cv_strategy: Cross-validation strategy for stacking
        n_jobs: Number of parallel jobs

    Returns:
        Configured StackingRegressor with optimized base models

    Raises:
        ImportError: If required models not installed
        ValueError: If cv_strategy is invalid

    Example:
        >>> ensemble = create_research_stacked_ensemble()
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    if not _sklearn_available:
        logger.error("scikit-learn not available")
        raise ImportError(
            f"scikit-learn is required but not installed. Install with: "
            f"/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/pip install scikit-learn"
        )

    try:
        # Validate dependencies
        validate_ensemble_dependencies()

        # Create default CV strategy if not provided
        if cv_strategy is None:
            # Use standard KFold for compatibility with StackingRegressor
            from sklearn.model_selection import KFold

            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

        # Validate CV strategy
        if not hasattr(cv_strategy, "split"):
            raise ValueError("cv_strategy must have a 'split' method")

        # Create base estimators with NBA-optimized configurations
        base_estimators = create_base_estimators(n_jobs)

        # Create MLP meta-learner optimized for NBA predictions
        meta_learner = create_mlp_meta_learner()

        # Build stacked ensemble
        stacked_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv_strategy,
            n_jobs=n_jobs,
            passthrough=True,  # Pass original features to meta-learner
        )

        logger.info(
            "Research stacked ensemble created successfully",
            extra={
                "base_models": len(base_estimators),
                "cv_strategy": type(cv_strategy).__name__,
                "n_jobs": n_jobs,
                "passthrough": True,
            },
        )

        return stacked_model

    except (ImportError, ValueError) as e:
        logger.error(
            "Stacked ensemble creation failed",
            extra={
                "cv_strategy": type(cv_strategy).__name__ if cv_strategy else None,
                "n_jobs": n_jobs,
                "error": str(e),
            },
        )
        raise ValueError(f"Invalid stacked ensemble parameters: {e}") from e


def create_base_estimators(n_jobs: int = -1) -> List[tuple[str, Any]]:
    """
    Create NBA-optimized base estimators for stacking.

    Args:
        n_jobs: Number of parallel jobs

    Returns:
        List of (name, estimator) tuples for base models

    Raises:
        ImportError: If required models not installed
    """
    base_estimators = []

    # 1. XGBoost - NBA optimized
    if _xgboost_available:
        xgb_params: Dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        base_estimators.append(("xgb", xgb_model))
        logger.info("Added XGBoost base estimator")
    else:
        logger.warning("XGBoost not available, skipping")

    # 2. LightGBM - NBA optimized
    if _lightgbm_available:
        lgbm_params: Dict[str, Any] = {
            "objective": "regression",
            "metric": ["l1", "l2"],
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        base_estimators.append(("lgbm", lgbm_model))
        logger.info("Added LightGBM base estimator")
    else:
        logger.warning("LightGBM not available, skipping")

    # 3. Random Forest - Conservative configuration
    rf_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 10,  # Conservative to prevent overfitting
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": n_jobs,
    }
    rf_model = RandomForestRegressor(**rf_params)
    base_estimators.append(("rf", rf_model))
    logger.info("Added Random Forest base estimator")

    # 4. Ridge Regression - Linear baseline
    ridge_params: Dict[str, Any] = {
        "alphas": [0.1, 1.0, 10.0],
        "cv": 5,  # Use standard CV for Ridge (not time series)
        "scoring": "neg_mean_absolute_error",
    }
    ridge_model = RidgeCV(**ridge_params)
    base_estimators.append(("ridge", ridge_model))
    logger.info("Added Ridge Regression base estimator")

    if len(base_estimators) < 2:
        raise ImportError(
            "At least 2 base models are required for stacking. "
            "Please install xgboost and/or lightgbm."
        )

    return base_estimators


def create_mlp_meta_learner() -> "MLPRegressor":
    """
    Create MLP meta-learner optimized for NBA predictions.

    Returns:
        Configured MLPRegressor for meta-learning

    Raises:
        ImportError: If scikit-learn not available
    """
    if not _sklearn_available:
        raise ImportError("scikit-learn is required for MLP meta-learner")

    # MLP architecture research-based for NBA data
    meta_params: Dict[str, Any] = {
        "hidden_layer_sizes": (64, 32),  # Two layers, decreasing size
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,  # L2 regularization
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "tol": 1e-4,
    }

    meta_learner = MLPRegressor(**meta_params)
    logger.info(
        "Created MLP meta-learner",
        extra={
            "hidden_layers": meta_params["hidden_layer_sizes"],
            "activation": meta_params["activation"],
            "early_stopping": meta_params["early_stopping"],
        },
    )

    return meta_learner


def validate_ensemble_dependencies() -> None:
    """
    Validate that required dependencies are available.

    Raises:
        ImportError: If required dependencies are missing
    """
    missing_deps = []

    if not _sklearn_available:
        missing_deps.append("scikit-learn")

    if not _xgboost_available:
        missing_deps.append("xgboost")

    if not _lightgbm_available:
        missing_deps.append("lightgbm")

    if missing_deps:
        logger.warning("Some dependencies missing", extra={"missing": missing_deps})


def get_ensemble_feature_importance(
    stacked_model: "StackingRegressor",
) -> Dict[str, float]:
    """
    Extract feature importance from stacked ensemble base models.

    Args:
        stacked_model: Fitted StackingRegressor model

    Returns:
        Dictionary mapping model names to feature importance scores

    Raises:
        ValueError: If model is not fitted or doesn't support feature importance
    """
    if not hasattr(stacked_model, "estimators_"):
        raise ValueError(
            "Stacked model must be fitted before extracting feature importance"
        )

    importance_scores = {}

    try:
        # StackingRegressor.estimators_ is a list of fitted estimators
        # StackingRegressor.estimators is a list of (name, estimator) tuples
        # We handle cases where structure might differ

        # Get names safely
        names = []
        if hasattr(stacked_model, "estimators"):
            for item in stacked_model.estimators:
                if isinstance(item, tuple) and len(item) >= 1:
                    names.append(str(item[0]))
                else:
                    names.append(f"estimator_{len(names)}")

        # Iterate fitted estimators
        for i, estimator in enumerate(stacked_model.estimators_):
            try:
                # Determine name
                name = names[i] if i < len(names) else f"estimator_{i}"

                if hasattr(estimator, "feature_importances_"):
                    # Get mean importance across all features
                    importance = float(estimator.feature_importances_.mean())
                    importance_scores[name] = importance
                    logger.debug(
                        f"Extracted feature importance for {name}: {importance:.4f}"
                    )
                else:
                    logger.debug(f"Model {name} does not support feature importance")
            except Exception as e:
                logger.warning(f"Failed to extract importance from estimator {i}: {e}")

    except Exception as e:
        logger.warning(f"Failed to iterate estimators for importance: {e}")

    return importance_scores


def create_conservative_stacked_ensemble(
    cv_strategy: Optional[Any] = None, n_jobs: int = -1
) -> "StackingRegressor":
    """
    Create conservative stacked ensemble for limited NBA data.

    Args:
        cv_strategy: Cross-validation strategy for stacking
        n_jobs: Number of parallel jobs

    Returns:
        Configured StackingRegressor with conservative base models

    Raises:
        ImportError: If required models not installed
        ValueError: If cv_strategy is invalid
    """
    if not _sklearn_available:
        raise ImportError("scikit-learn is required for conservative ensemble")

    try:
        # Create more conservative CV strategy
        if cv_strategy is None:
            from sklearn.model_selection import KFold

            cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

        # Conservative base estimators
        conservative_estimators = []

        # More conservative XGBoost
        if _xgboost_available:
            xgb_conservative = xgb.XGBRegressor(
                n_estimators=100,  # Fewer trees
                learning_rate=0.03,  # Slower learning
                max_depth=4,  # Shallower trees
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.3,  # Stronger regularization
                reg_lambda=0.3,
                random_state=42,
                n_jobs=n_jobs,
                verbosity=0,
            )
            conservative_estimators.append(("xgb_conservative", xgb_conservative))

        # More conservative LightGBM
        if _lightgbm_available:
            lgbm_conservative = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=15,  # Fewer leaves
                max_depth=4,
                min_child_samples=30,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=n_jobs,
                verbose=-1,
            )
            conservative_estimators.append(("lgbm_conservative", lgbm_conservative))

        # Simple Random Forest
        rf_conservative = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=n_jobs,
        )
        conservative_estimators.append(("rf_conservative", rf_conservative))

        # Simple Ridge
        ridge_conservative = RidgeCV(alphas=[1.0, 10.0, 100.0], cv=3)
        conservative_estimators.append(("ridge_conservative", ridge_conservative))

        # Conservative meta-learner
        meta_conservative = MLPRegressor(
            hidden_layer_sizes=(32,),  # Single smaller layer
            activation="relu",
            alpha=0.01,  # Stronger regularization
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
        )

        conservative_ensemble = StackingRegressor(
            estimators=conservative_estimators,
            final_estimator=meta_conservative,
            cv=cv_strategy,
            n_jobs=n_jobs,
            passthrough=True,
        )

        logger.info(
            "Conservative stacked ensemble created",
            extra={
                "base_models": len(conservative_estimators),
                "cv_strategy": type(cv_strategy).__name__,
            },
        )

        return conservative_ensemble

    except Exception as e:
        logger.error("Conservative ensemble creation failed", extra={"error": str(e)})
        raise ValueError(f"Failed to create conservative ensemble: {e}") from e
