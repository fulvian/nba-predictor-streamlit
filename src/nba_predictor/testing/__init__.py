"""NBA Predictor Testing Module.

This module provides comprehensive testing and validation capabilities
for the NBA Predictor system, including:
- System validation and integration testing
- Performance benchmarking
- User acceptance testing
- Component unit testing
"""

from .system_validator import (
    SystemValidator,
    TestResult,
    ValidationReport,
    create_system_validator,
)

__all__ = [
    "SystemValidator",
    "TestResult",
    "ValidationReport",
    "create_system_validator",
]
