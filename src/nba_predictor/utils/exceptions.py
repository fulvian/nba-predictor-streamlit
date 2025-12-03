"""Custom exceptions for the NBA predictor system.

This module defines custom exception classes used throughout the
NBA predictor system for better error handling and debugging.
"""

from typing import Optional, Any


class NBAPredictorError(Exception):
    """Base exception class for NBA predictor system."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> None:
        """
        Initialize base exception.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """String representation of the exception."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            base_msg += f" | Context: {self.context}"
        return base_msg


class FileNotFoundError(NBAPredictorError):
    """Exception raised when a required file or directory is not found."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs) -> None:
        """
        Initialize file not found exception.

        Args:
            message: Error message
            file_path: Path to the missing file
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if file_path:
            context["file_path"] = file_path
        kwargs["context"] = context
        super().__init__(message, error_code="FILE_NOT_FOUND", **kwargs)


class DatabaseError(NBAPredictorError):
    """Exception raised for database-related errors."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs) -> None:
        """
        Initialize database error.

        Args:
            message: Error message
            operation: Database operation that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        kwargs["context"] = context
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)


class ValidationError(NBAPredictorError):
    """Exception raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        kwargs["context"] = context
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class APIError(NBAPredictorError):
    """Exception raised for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if status_code:
            context["status_code"] = status_code
        if endpoint:
            context["endpoint"] = endpoint
        kwargs["context"] = context
        super().__init__(message, error_code="API_ERROR", **kwargs)


class SyncError(NBAPredictorError):
    """Exception raised during data synchronization operations."""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        target: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize sync error.

        Args:
            message: Error message
            source: Source of sync operation
            target: Target of sync operation
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if source:
            context["source"] = source
        if target:
            context["target"] = target
        kwargs["context"] = context
        super().__init__(message, error_code="SYNC_ERROR", **kwargs)


class ConfigurationError(NBAPredictorError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)
        kwargs["context"] = context
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class StreamlitError(NBAPredictorError):
    """Exception raised for Streamlit-related errors."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs) -> None:
        """
        Initialize Streamlit error.

        Args:
            message: Error message
            component: Streamlit component that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if component:
            context["component"] = component
        kwargs["context"] = context
        super().__init__(message, error_code="STREAMLIT_ERROR", **kwargs)


class PredictionError(NBAPredictorError):
    """Exception raised for prediction-related errors."""

    def __init__(
        self,
        message: str,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize prediction error.

        Args:
            message: Error message
            prediction_type: Type of prediction that failed
            model_name: Name of the model that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if prediction_type:
            context["prediction_type"] = prediction_type
        if model_name:
            context["model_name"] = model_name
        kwargs["context"] = context
        super().__init__(message, error_code="PREDICTION_ERROR", **kwargs)


class CacheError(NBAPredictorError):
    """Exception raised for cache-related errors."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize cache error.

        Args:
            message: Error message
            cache_key: Cache key that caused the error
            operation: Cache operation that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if cache_key:
            context["cache_key"] = cache_key
        if operation:
            context["operation"] = operation
        kwargs["context"] = context
        super().__init__(message, error_code="CACHE_ERROR", **kwargs)


class FeatureEngineeringError(NBAPredictorError):
    """Exception raised for feature engineering errors."""

    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize feature engineering error.

        Args:
            message: Error message
            feature_name: Name of the feature that caused the error
            operation: Feature engineering operation that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if feature_name:
            context["feature_name"] = feature_name
        if operation:
            context["operation"] = operation
        kwargs["context"] = context
        super().__init__(message, error_code="FEATURE_ENGINEERING_ERROR", **kwargs)


class OptimizationError(NBAPredictorError):
    """Exception raised for model optimization errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        optimization_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize optimization error.

        Args:
            message: Error message
            model_name: Name of the model being optimized
            optimization_type: Type of optimization that failed
            **kwargs: Additional context
        """
        context = kwargs.get("context", {})
        if model_name:
            context["model_name"] = model_name
        if optimization_type:
            context["optimization_type"] = optimization_type
        kwargs["context"] = context
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)
