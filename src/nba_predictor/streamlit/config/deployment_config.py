"""Deployment configuration management for NBA Predictor Streamlit.

Context7-compliant configuration management supporting multiple environments
with secure defaults and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from ...utils.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data/nba_predictor.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: Optional[str] = None
    default_ttl: int = 300  # 5 minutes
    max_memory_mb: int = 1024
    enabled: bool = True


@dataclass
class APIConfig:
    """API configuration settings."""
    nba_api_key: Optional[str] = None
    odds_api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class StreamlitConfig:
    """Streamlit-specific configuration."""
    port: int = 8501
    host: str = "0.0.0.0"
    base_url: str = ""
    enable_cors: bool = True
    max_upload_size_mb: int = 200
    enable_xsrf_protection: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    enable_tracing: bool = False
    health_check_interval: int = 30


@dataclass
class DeploymentConfig:
    """
    Comprehensive deployment configuration for NBA Streamlit application.

    Context7-compliant configuration with environment-specific settings,
    security defaults, and validation.
    """

    # Environment settings
    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Security settings
    secret_key: Optional[str] = field(default_factory=lambda: os.getenv("SECRET_KEY"))
    session_timeout_minutes: int = 30
    enable_auth: bool = False

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Feature flags
    enable_predictions: bool = True
    enable_analytics: bool = True
    enable_real_time_data: bool = True
    enable_ml_explanations: bool = True

    # Performance settings
    max_concurrent_users: int = 100
    cache_enabled: bool = True
    background_sync_enabled: bool = True
    sync_interval_minutes: int = 60

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env.lower() in ("development", "dev", "local")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() in ("production", "prod")

    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.env.lower() in ("staging", "stage", "test")

    def get_log_level(self) -> str:
        """Get appropriate log level based on environment."""
        if self.debug:
            return "DEBUG"
        if self.is_production:
            return "WARNING"
        return "INFO"

    def get_database_url(self) -> str:
        """Get complete database URL with environment-specific settings."""
        base_url = os.getenv("DATABASE_URL", self.database.url)

        if self.is_development and not base_url.startswith("sqlite"):
            return base_url.replace("sslmode=require", "sslmode=disable")

        return base_url

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL with environment defaults."""
        return os.getenv("REDIS_URL", self.cache.redis_url)

    def get_nba_api_key(self) -> Optional[str]:
        """Get NBA API key from environment or config."""
        return os.getenv("NBA_API_KEY", self.api.nba_api_key)

    def validate(self) -> None:
        """
        Validate configuration settings.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []

        # Validate environment
        valid_envs = {"development", "dev", "local", "production", "prod", "staging", "stage", "test"}
        if self.env.lower() not in valid_envs:
            errors.append(f"Invalid environment: {self.env}. Valid: {valid_envs}")

        # Validate required settings for production
        if self.is_production:
            if not self.secret_key:
                errors.append("SECRET_KEY is required in production")
            if not self.get_nba_api_key():
                errors.append("NBA_API_KEY is required in production")
            if self.debug:
                errors.append("DEBUG should not be enabled in production")

        # Validate port ranges
        if not (1024 <= self.streamlit.port <= 65535):
            errors.append(f"Streamlit port {self.streamlit.port} must be between 1024 and 65535")

        # Validate performance settings
        if self.max_concurrent_users < 1:
            errors.append("max_concurrent_users must be >= 1")

        if self.sync_interval_minutes < 1:
            errors.append("sync_interval_minutes must be >= 1")

        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "env": self.env,
            "debug": self.debug,
            "is_production": self.is_production,
            "database": {
                "url": self.get_database_url(),
                "echo": self.database.echo
            },
            "cache": {
                "enabled": self.cache.enabled,
                "default_ttl": self.cache.default_ttl,
                "redis_url": self.get_redis_url()
            },
            "api": {
                "rate_limit_per_minute": self.api.rate_limit_per_minute,
                "timeout_seconds": self.api.timeout_seconds,
                "has_nba_key": bool(self.get_nba_api_key())
            },
            "streamlit": {
                "port": self.streamlit.port,
                "host": self.streamlit.host,
                "base_url": self.streamlit.base_url
            },
            "monitoring": {
                "log_level": self.get_log_level(),
                "log_format": self.monitoring.log_format,
                "enable_metrics": self.monitoring.enable_metrics
            },
            "features": {
                "enable_predictions": self.enable_predictions,
                "enable_analytics": self.enable_analytics,
                "enable_real_time_data": self.enable_real_time_data,
                "enable_ml_explanations": self.enable_ml_explanations
            }
        }

    @classmethod
    def from_environment(cls) -> "DeploymentConfig":
        """
        Create configuration from environment variables.

        Returns:
            DeploymentConfig instance populated from environment

        Examples:
            >>> config = DeploymentConfig.from_environment()
            >>> print(f"Environment: {config.env}")
            Environment: development
        """
        return cls()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "DeploymentConfig":
        """
        Load configuration from file (JSON/YAML support).

        Args:
            config_path: Path to configuration file

        Returns:
            DeploymentConfig instance

        Raises:
            ConfigurationError: If file loading fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix.lower() == ".json":
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in (".yml", ".yaml"):
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")

            return cls(**config_data)

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}") from e

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.

        Args:
            config_path: Path where to save configuration

        Raises:
            ConfigurationError: If file saving fails
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_data = self.to_dict()

            if config_path.suffix.lower() == ".json":
                import json
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif config_path.suffix.lower() in (".yml", ".yaml"):
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}") from e

    def get_streamlit_config_dict(self) -> Dict[str, any]:
        """
        Get Streamlit-specific configuration dictionary.

        Returns:
            Dictionary with Streamlit configuration settings
        """
        return {
            "server.port": self.streamlit.port,
            "server.address": self.streamlit.host,
            "server.baseUrlPath": self.streamlit.base_url,
            "server.enableCORS": self.streamlit.enable_cors,
            "server.enableXsrfProtection": self.streamlit.enable_xsrf_protection,
            "server.maxUploadSize": self.streamlit.max_upload_size,
            "logger.level": self.get_log_level(),
            "browser.gatherUsageStats": False,
            "global.developmentMode": self.debug
        }


def load_config(config_path: Optional[Union[str, Path]] = None) -> DeploymentConfig:
    """
    Load deployment configuration with fallback to environment variables.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Loaded DeploymentConfig instance

    Examples:
        >>> config = load_config()
        >>> print(f"Loaded config for environment: {config.env}")
        Loaded config for environment: development
    """
    if config_path:
        try:
            config = DeploymentConfig.from_file(config_path)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except ConfigurationError:
            logger.warning(f"Failed to load config from {config_path}, falling back to environment")

    return DeploymentConfig.from_environment()


def setup_environment_config(config: DeploymentConfig) -> None:
    """
    Setup environment variables and configuration for the application.

    Args:
        config: Deployment configuration to apply

    Raises:
        ConfigurationError: If setup fails
    """
    try:
        # Set environment variables for external libraries
        os.environ["NBA_PREDICTOR_ENV"] = config.env
        os.environ["NBA_PREDICTOR_DEBUG"] = str(config.debug).lower()

        if config.get_nba_api_key():
            os.environ["NBA_API_KEY"] = config.get_nba_api_key()

        # Configure logging
        import logging
        log_level = getattr(logging, config.get_log_level().upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Environment configured for {config.env}")
        if config.debug:
            logger.debug(f"Configuration: {config.to_dict()}")

    except Exception as e:
        raise ConfigurationError(f"Failed to setup environment: {e}") from e