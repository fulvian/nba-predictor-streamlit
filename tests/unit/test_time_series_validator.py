#!/usr/bin/env python3
"""
🧪 Time Series Validator Unit Tests
Test suite for time series cross-validation functionality.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from nba_predictor.core.time_series_validator import (
    create_time_series_splits,
    validate_time_series_parameters,
    get_nba_optimal_splits
)


class TestTimeSeriesValidator(unittest.TestCase):
    """Test cases for time series validation module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample NBA data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'team1_score': np.random.randint(80, 140, 100),
            'team2_score': np.random.randint(80, 140, 100),
            'total_score': np.random.randint(160, 280, 100)
        })

    def test_create_time_series_splits_default(self):
        """Test creating time series splits with default parameters."""
        tscv = create_time_series_splits()

        self.assertIsInstance(tscv, TimeSeriesSplit)
        self.assertEqual(tscv.get_n_splits(), 5)
        self.assertEqual(tscv.max_train_size, 1000)
        self.assertEqual(tscv.test_size, 2)

    def test_create_time_series_splits_custom(self):
        """Test creating time series splits with custom parameters."""
        tscv = create_time_series_splits(
            n_splits=3,
            max_train_size=500,
            gap=3
        )

        self.assertIsInstance(tscv, TimeSeriesSplit)
        self.assertEqual(tscv.get_n_splits(), 3)
        self.assertEqual(tscv.max_train_size, 500)
        self.assertEqual(tscv.test_size, 3)

    def test_create_time_series_splits_no_max_train_size(self):
        """Test creating time series splits without max_train_size limit."""
        tscv = create_time_series_splits(
            n_splits=3,
            max_train_size=None,
            gap=1
        )

        self.assertIsInstance(tscv, TimeSeriesSplit)
        self.assertEqual(tscv.get_n_splits(), 3)
        self.assertIsNone(tscv.max_train_size)
        self.assertEqual(tscv.test_size, 1)

    def test_create_time_series_splits_invalid_n_splits(self):
        """Test error handling for invalid n_splits."""
        with self.assertRaises(ValueError) as context:
            create_time_series_splits(n_splits=1)

        self.assertIn("Invalid TimeSeriesSplit parameters", str(context.exception))

    def test_create_time_series_splits_negative_gap(self):
        """Test error handling for negative gap."""
        # TimeSeriesSplit accepts negative gap in constructor but validation function catches it
        with self.assertRaises(ValueError):
            validate_time_series_parameters(5, 1000, -1)

    def test_validate_time_series_parameters_valid(self):
        """Test validation with valid parameters."""
        # Should not raise exception
        validate_time_series_parameters(5, 1000, 2)

    def test_validate_time_series_parameters_invalid_n_splits(self):
        """Test validation with invalid n_splits."""
        with self.assertRaises(ValueError):
            validate_time_series_parameters(1, 1000, 2)

        with self.assertRaises(ValueError):
            validate_time_series_parameters(25, 1000, 2)

    def test_validate_time_series_parameters_invalid_max_train_size(self):
        """Test validation with invalid max_train_size."""
        with self.assertRaises(ValueError):
            validate_time_series_parameters(5, 25, 2)

    def test_validate_time_series_parameters_negative_gap(self):
        """Test validation with negative gap."""
        with self.assertRaises(ValueError):
            validate_time_series_parameters(5, 1000, -1)

    def test_get_nba_optimal_splits_small_dataset(self):
        """Test optimal splits calculation for small dataset."""
        splits = get_nba_optimal_splits(50)
        self.assertEqual(splits, 3)

    def test_get_nba_optimal_splits_medium_dataset(self):
        """Test optimal splits calculation for medium dataset."""
        splits = get_nba_optimal_splits(300)
        self.assertEqual(splits, 5)

    def test_get_nba_optimal_splits_large_dataset(self):
        """Test optimal splits calculation for large dataset."""
        splits = get_nba_optimal_splits(800)
        self.assertEqual(splits, 7)

    def test_get_nba_optimal_splits_very_large_dataset(self):
        """Test optimal splits calculation for very large dataset."""
        splits = get_nba_optimal_splits(1500)
        self.assertEqual(splits, 10)

    def test_time_series_split_functionality(self):
        """Test actual time series splitting functionality."""
        tscv = create_time_series_splits(n_splits=3, gap=2)
        X = np.arange(20).reshape(10, 2)

        splits = list(tscv.split(X))
        self.assertEqual(len(splits), 3)

        # Check that splits are chronological
        for i, (train_idx, test_idx) in enumerate(splits):
            max_train_idx = max(train_idx) if len(train_idx) > 0 else -1
            min_test_idx = min(test_idx) if len(test_idx) > 0 else float('inf')
            self.assertLess(max_train_idx, min_test_idx,
                           f"Split {i}: train max index {max_train_idx} >= test min index {min_test_idx}")

    def test_example_from_docstring(self):
        """Test the example from the function docstring."""
        # Create sample data
        X = pd.DataFrame(np.random.randn(20, 3))

        tscv = create_time_series_splits(n_splits=3, gap=2)
        split_count = 0

        for train_idx, test_idx in tscv.split(X):
            train_size = len(train_idx)
            test_size = len(test_idx)
            # Test that sizes are reasonable
            self.assertGreater(train_size, 0)
            self.assertGreater(test_size, 0)
            split_count += 1

        self.assertEqual(split_count, 3)


if __name__ == '__main__':
    unittest.main()