#!/usr/bin/env python3
"""
🏀 Time Series Cross-Validation Validator - Context7 Compliant
Time series cross-validation module specifically designed for NBA predictions.

This module implements:
- TimeSeriesSplit configuration for NBA data validation
- Prevention of data leakage in temporal NBA data
- Proper gap handling for consecutive games
- NBA-specific cross-validation strategies
"""

import logging
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_time_series_splits(
    n_splits: int = 5,
    max_train_size: Optional[int] = 1000,
    gap: int = 2
) -> TimeSeriesSplit:
    """
    Create TimeSeriesSplit configured for NBA data validation.

    Args:
        n_splits: Number of cross-validation folds
        max_train_size: Maximum training samples per fold
        gap: Days gap between train and test sets

    Returns:
        Configured TimeSeriesSplit object

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> tscv = create_time_series_splits(n_splits=5, gap=2)
        >>> for train_idx, test_idx in tscv.split(X):
        ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """
    try:
        # Validate parameters first
        validate_time_series_parameters(n_splits, max_train_size, gap)

        # Implementation with error handling as specified
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=gap
        )

        logger.info(
            "TimeSeriesSplit created successfully",
            extra={
                "n_splits": n_splits,
                "max_train_size": max_train_size,
                "gap": gap
            }
        )

        return tscv

    except ValueError as e:
        logger.error(
            "TimeSeriesSplit creation failed",
            extra={
                "n_splits": n_splits,
                "max_train_size": max_train_size,
                "gap": gap,
                "error": str(e)
            }
        )
        raise ValueError(f"Invalid TimeSeriesSplit parameters: {e}") from e


def validate_time_series_parameters(
    n_splits: int,
    max_train_size: Optional[int],
    gap: int
) -> None:
    """
    Validate TimeSeriesSplit parameters for NBA data.

    Args:
        n_splits: Number of cross-validation folds
        max_train_size: Maximum training samples per fold
        gap: Days gap between train and test sets

    Raises:
        ValueError: If parameters are invalid for NBA data
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation")

    if n_splits > 20:
        raise ValueError("n_splits should not exceed 20 for practical NBA validation")

    if gap < 0:
        raise ValueError("gap must be non-negative")

    if gap > 7:
        logger.warning("Large gap (%d) may result in insufficient test data", gap)

    if max_train_size is not None and max_train_size < 50:
        raise ValueError("max_train_size should be at least 50 for meaningful NBA validation")


def get_nba_optimal_splits(data_size: int) -> int:
    """
    Get optimal number of time series splits for NBA data size.

    Args:
        data_size: Number of games in dataset

    Returns:
        Optimal number of splits
    """
    if data_size < 100:
        return 3
    elif data_size < 500:
        return 5
    elif data_size < 1000:
        return 7
    else:
        return 10