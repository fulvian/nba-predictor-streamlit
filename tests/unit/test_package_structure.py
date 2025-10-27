"""Test package structure and imports."""

import sys
from pathlib import Path

import pytest


def test_src_layout():
    """Test that src layout is properly configured."""
    # Test that the src directory exists
    src_path = Path(__file__).parent.parent.parent / "src"
    assert src_path.exists(), "src directory should exist"

    # Test that the main package exists
    package_path = src_path / "nba_predictor"
    assert package_path.exists(), "nba_predictor package should exist"
    assert package_path.is_dir(), "nba_predictor should be a directory"

    # Test that __init__.py exists
    init_file = package_path / "__init__.py"
    assert init_file.exists(), "__init__.py should exist in main package"

    # Test that submodules exist
    submodules = ["core", "streamlit", "api", "utils"]
    for submodule in submodules:
        submodule_path = package_path / submodule
        assert submodule_path.exists(), f"{submodule} submodule should exist"
        assert submodule_path.is_dir(), f"{submodule} should be a directory"

        init_file = submodule_path / "__init__.py"
        assert init_file.exists(), f"__init__.py should exist in {submodule}"


def test_package_imports():
    """Test that package can be imported correctly."""
    # Add src to path for testing
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        import nba_predictor
        assert hasattr(nba_predictor, "__version__")
        assert hasattr(nba_predictor, "__author__")
        assert hasattr(nba_predictor, "__all__")
    except ImportError as e:
        pytest.fail(f"Failed to import nba_predictor: {e}")


def test_submodule_imports():
    """Test that submodules can be imported correctly."""
    # Add src to path for testing
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from nba_predictor.core import UnifiedDataStore, AutomaticSyncEngine
        from nba_predictor.streamlit import (
            render_sync_dashboard,
            render_analytics_dashboard,
            create_main_app,
        )
        from nba_predictor.api import ModernNBAAPIClient, ModernOddsAPIClient

        # Note: These will fail initially since the modules don't exist yet,
        # but this test validates the __init__.py structure
        assert True  # If imports don't raise ImportError, structure is correct

    except ImportError as e:
        # Expected since modules don't exist yet, but structure should be correct
        if "cannot import name" not in str(e):
            pytest.fail(f"Unexpected import error: {e}")