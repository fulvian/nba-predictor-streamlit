#!/usr/bin/env python3
"""
🔧 NBA Model Version Manager - DevStream SuperPowered

Advanced model versioning and rollback system for NBA Ensemble Predictor.
Provides semantic versioning, performance tracking, and automatic rollback capabilities.

Architecture: DevStream SuperPowered with ContextSet compliance
Date: 2025-01-12
"""

import os
import json
import pickle
import shutil
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import pandas as pd

# Model management imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("joblib not available. Using pickle for model serialization.")

class ModelStatus(Enum):
    """Model deployment status"""
    ACTIVE = "active"
    STAGED = "staged"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class ModelType(Enum):
    """Model type classification"""
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    SCALER = "scaler"
    CONFIDENCE_CALCULATOR = "confidence_calculator"
    PREDICTION_EXPLAINER = "prediction_explainer"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 0.0
    calibration_score: float = 0.0
    confidence_quality: float = 0.0
    explanation_quality: float = 0.0
    prediction_latency_ms: float = 0.0
    model_size_mb: float = 0.0
    training_time_minutes: float = 0.0

    # NBA-specific metrics
    nba_accuracy: float = 0.0  # Accuracy on NBA games
    home_win_prediction_accuracy: float = 0.0
    away_win_prediction_accuracy: float = 0.0
    close_game_accuracy: float = 0.0  # Games within 5 points
    blowout_prediction_accuracy: float = 0.0  # Games > 15 points

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    created_by: str
    description: str
    metrics: ModelMetrics = field(default_factory=ModelMetrics)

    # Model file information
    model_hash: str = ""
    file_path: str = ""
    config_path: str = ""

    # Training information
    training_data_hash: str = ""
    training_samples: int = 0
    training_features: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # NBA-specific information
    nba_season: str = ""
    training_date_range: Tuple[str, str] = ("", "")
    team_coverage: List[str] = field(default_factory=list)

    # Versioning metadata
    parent_version: Optional[str] = None
    changelog: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to strings
        data['created_at'] = self.created_at.isoformat()
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        data['metrics'] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        # Convert string fields back to proper types
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['metrics'] = ModelMetrics.from_dict(data['metrics'])
        return cls(**data)

@dataclass
class RollbackConfig:
    """Rollback configuration"""
    enabled: bool = True
    auto_rollback: bool = False
    performance_threshold: float = 0.02  # 2% performance degradation threshold
    max_rollback_versions: int = 5
    rollback_cooldown_hours: int = 24
    monitoring_window_days: int = 7
    min_predictions_before_rollback: int = 100

class NBAModelVersionManager:
    """
    Advanced NBA Model Version Manager

    Provides:
    - Semantic versioning for models
    - Performance tracking and comparison
    - Automatic rollback capabilities
    - Model registry and metadata management
    - NBA-specific versioning features
    """

    def __init__(self,
                 model_registry_path: str = "data/models/registry",
                 models_path: str = "data/models/versions",
                 rollback_config: Optional[RollbackConfig] = None):
        """
        Initialize Model Version Manager

        Args:
            model_registry_path: Path to model registry directory
            models_path: Path to model versions directory
            rollback_config: Rollback configuration
        """
        self.logger = logging.getLogger(__name__)
        self.model_registry_path = Path(model_registry_path)
        self.models_path = Path(models_path)
        self.rollback_config = rollback_config or RollbackConfig()

        # Create directories
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Model registry storage
        self.registry_file = self.model_registry_path / "model_registry.json"
        self.performance_log_file = self.model_registry_path / "performance_log.json"
        self.rollback_log_file = self.model_registry_path / "rollback_log.json"

        # Load existing registry
        self._load_registry()

        # Performance tracking
        self._load_performance_log()

        # Rollback tracking
        self._load_rollback_log()

        self.logger.info(f"🔧 NBA Model Version Manager initialized")
        self.logger.info(f"   Registry: {self.model_registry_path}")
        self.logger.info(f"   Models: {self.models_path}")
        self.logger.info(f"   Auto-rollback: {self.rollback_config.enabled}")

    def _load_registry(self) -> None:
        """Load model registry from file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.model_registry = {
                        version: ModelVersion.from_dict(version_data)
                        for version, version_data in data.items()
                    }
            else:
                self.model_registry = {}

            self.logger.info(f"📋 Loaded {len(self.model_registry)} model versions from registry")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model registry: {e}")
            self.model_registry = {}

    def _save_registry(self) -> None:
        """Save model registry to file"""
        try:
            data = {
                version: version_data.to_dict()
                for version, version_data in self.model_registry.items()
            }

            # Create backup before saving
            if self.registry_file.exists():
                backup_file = self.registry_file.with_suffix('.json.backup')
                shutil.copy2(self.registry_file, backup_file)

            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.debug(f"💾 Saved model registry with {len(self.model_registry)} versions")

        except Exception as e:
            self.logger.error(f"❌ Failed to save model registry: {e}")

    def _load_performance_log(self) -> None:
        """Load performance log from file"""
        try:
            if self.performance_log_file.exists():
                with open(self.performance_log_file, 'r') as f:
                    self.performance_log = json.load(f)
            else:
                self.performance_log = []

        except Exception as e:
            self.logger.error(f"❌ Failed to load performance log: {e}")
            self.performance_log = []

    def _save_performance_log(self) -> None:
        """Save performance log to file"""
        try:
            with open(self.performance_log_file, 'w') as f:
                json.dump(self.performance_log, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"❌ Failed to save performance log: {e}")

    def _load_rollback_log(self) -> None:
        """Load rollback log from file"""
        try:
            if self.rollback_log_file.exists():
                with open(self.rollback_log_file, 'r') as f:
                    self.rollback_log = json.load(f)
            else:
                self.rollback_log = []

        except Exception as e:
            self.logger.error(f"❌ Failed to load rollback log: {e}")
            self.rollback_log = []

    def _save_rollback_log(self) -> None:
        """Save rollback log to file"""
        try:
            with open(self.rollback_log_file, 'w') as f:
                json.dump(self.rollback_log, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"❌ Failed to save rollback log: {e}")

    def _generate_model_hash(self, model_path: str) -> str:
        """Generate hash for model file"""
        try:
            hash_md5 = hashlib.md5()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Failed to generate model hash: {e}")
            return ""

    def _generate_version_number(self, model_type: ModelType, parent_version: Optional[str] = None) -> str:
        """Generate semantic version number"""
        try:
            # Get existing versions for this model type
            existing_versions = [
                v for v in self.model_registry.keys()
                if self.model_registry[v].model_type == model_type
            ]

            if not existing_versions:
                return "1.0.0"

            # Parse version numbers
            version_numbers = []
            for v in existing_versions:
                try:
                    major, minor, patch = map(int, v.split('.'))
                    version_numbers.append((major, minor, patch))
                except:
                    continue

            if not version_numbers:
                return "1.0.0"

            # Get highest version
            max_version = max(version_numbers)
            major, minor, patch = max_version

            # If parent version is specified, create patch version
            if parent_version and parent_version in existing_versions:
                try:
                    p_major, p_minor, p_patch = map(int, parent_version.split('.'))
                    if (p_major, p_minor) == (major, minor):
                        return f"{major}.{minor}.{patch + 1}"
                except:
                    pass

            # Otherwise, increment minor version
            return f"{major}.{minor + 1}.0"

        except Exception as e:
            self.logger.error(f"❌ Failed to generate version number: {e}")
            return "1.0.0"

    def register_model(self,
                      model: Any,
                      model_type: ModelType,
                      description: str,
                      created_by: str,
                      metrics: Optional[ModelMetrics] = None,
                      hyperparameters: Optional[Dict[str, Any]] = None,
                      nba_season: str = "",
                      training_date_range: Tuple[str, str] = ("", ""),
                      team_coverage: Optional[List[str]] = None,
                      parent_version: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      config: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new model version

        Args:
            model: The trained model object
            model_type: Type of model
            description: Model description
            created_by: Who created this model
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            nba_season: NBA season covered
            training_date_range: Training data date range
            team_coverage: Teams covered by this model
            parent_version: Parent model version
            tags: Version tags
            config: Model configuration

        Returns:
            Version string
        """
        try:
            # Generate version number
            version = self._generate_version_number(model_type, parent_version)

            # Create version directory
            version_dir = self.models_path / model_type.value / version
            version_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = version_dir / "model.pkl"
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_file, compress=3)
            else:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)

            # Save configuration if provided
            config_file = version_dir / "config.json"
            config_data = {
                'model_type': model_type.value,
                'version': version,
                'hyperparameters': hyperparameters or {},
                'config': config or {},
                'nba_metadata': {
                    'season': nba_season,
                    'training_date_range': training_date_range,
                    'team_coverage': team_coverage or []
                }
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

            # Generate model hash
            model_hash = self._generate_model_hash(str(model_file))

            # Create model version entry
            model_version = ModelVersion(
                version=version,
                model_type=model_type,
                status=ModelStatus.STAGED,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                metrics=metrics or ModelMetrics(),
                model_hash=model_hash,
                file_path=str(model_file),
                config_path=str(config_file),
                hyperparameters=hyperparameters or {},
                nba_season=nba_season,
                training_date_range=training_date_range,
                team_coverage=team_coverage or [],
                parent_version=parent_version,
                tags=tags or [],
                training_features=len(hyperparameters or {})
            )

            # Add to registry
            self.model_registry[version] = model_version
            self._save_registry()

            self.logger.info(f"✅ Model version {version} registered successfully")
            self.logger.info(f"   Type: {model_type.value}")
            self.logger.info(f"   Status: {ModelStatus.STAGED.value}")
            self.logger.info(f"   Path: {model_file}")

            return version

        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise

    def load_model(self, version: str) -> Tuple[Any, ModelVersion]:
        """
        Load model by version

        Args:
            version: Model version

        Returns:
            Tuple of (model, model_version_info)
        """
        try:
            if version not in self.model_registry:
                raise ValueError(f"Version {version} not found in registry")

            model_version = self.model_registry[version]
            model_file = Path(model_version.file_path)

            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            # Load model
            if JOBLIB_AVAILABLE:
                model = joblib.load(model_file)
            else:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

            self.logger.info(f"✅ Model version {version} loaded successfully")

            return model, model_version

        except Exception as e:
            self.logger.error(f"❌ Failed to load model version {version}: {e}")
            raise

    def activate_model(self, version: str) -> bool:
        """
        Activate a model version

        Args:
            version: Model version to activate

        Returns:
            Success status
        """
        try:
            if version not in self.model_registry:
                self.logger.error(f"❌ Version {version} not found in registry")
                return False

            model_version = self.model_registry[version]
            model_type = model_version.model_type

            # Deactivate other models of same type
            for v, mv in self.model_registry.items():
                if mv.model_type == model_type and mv.status == ModelStatus.ACTIVE:
                    mv.status = ModelStatus.DEPRECATED

            # Activate requested version
            model_version.status = ModelStatus.ACTIVE
            self._save_registry()

            self.logger.info(f"✅ Model version {version} activated")
            self.logger.info(f"   Type: {model_type.value}")
            self.logger.info(f"   Previous active models deprecated")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to activate model version {version}: {e}")
            return False

    def get_active_version(self, model_type: ModelType) -> Optional[str]:
        """
        Get active version for model type

        Args:
            model_type: Model type

        Returns:
            Active version string or None
        """
        try:
            for version, model_version in self.model_registry.items():
                if model_version.model_type == model_type and model_version.status == ModelStatus.ACTIVE:
                    return version
            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to get active version: {e}")
            return None

    def list_versions(self, model_type: Optional[ModelType] = None, status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """
        List model versions with optional filtering

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of model versions
        """
        try:
            versions = list(self.model_registry.values())

            if model_type:
                versions = [v for v in versions if v.model_type == model_type]

            if status:
                versions = [v for v in versions if v.status == status]

            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)

            return versions

        except Exception as e:
            self.logger.error(f"❌ Failed to list versions: {e}")
            return []

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison results
        """
        try:
            if version1 not in self.model_registry or version2 not in self.model_registry:
                raise ValueError("One or both versions not found in registry")

            v1 = self.model_registry[version1]
            v2 = self.model_registry[version2]

            comparison = {
                'version1': version1,
                'version2': version2,
                'created_at': {
                    version1: v1.created_at.isoformat(),
                    version2: v2.created_at.isoformat()
                },
                'metrics_comparison': {},
                'improvement_areas': [],
                'degradation_areas': [],
                'nba_metrics_comparison': {}
            }

            # Compare general metrics
            for field in v1.metrics.__dataclass_fields__:
                v1_val = getattr(v1.metrics, field)
                v2_val = getattr(v2.metrics, field)

                if v1_val > 0 and v2_val > 0:  # Only compare if both have valid values
                    diff = v2_val - v1_val
                    pct_change = (diff / v1_val) * 100 if v1_val != 0 else 0

                    comparison['metrics_comparison'][field] = {
                        'v1': v1_val,
                        'v2': v2_val,
                        'diff': diff,
                        'pct_change': pct_change
                    }

                    # For metrics where higher is better (accuracy, precision, etc.)
                    if field in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                                'calibration_score', 'confidence_quality', 'explanation_quality',
                                'nba_accuracy', 'home_win_prediction_accuracy',
                                'away_win_prediction_accuracy', 'close_game_accuracy',
                                'blowout_prediction_accuracy']:
                        if pct_change > 0:
                            comparison['improvement_areas'].append(field)
                        elif pct_change < -self.rollback_config.performance_threshold * 100:
                            comparison['degradation_areas'].append(field)

                    # For metrics where lower is better (loss, latency, size)
                    elif field in ['log_loss', 'prediction_latency_ms', 'model_size_mb', 'training_time_minutes']:
                        if pct_change < 0:
                            comparison['improvement_areas'].append(field)
                        elif pct_change > self.rollback_config.performance_threshold * 100:
                            comparison['degradation_areas'].append(field)

            return comparison

        except Exception as e:
            self.logger.error(f"❌ Failed to compare versions: {e}")
            return {}

    def log_performance(self, version: str, performance_data: Dict[str, Any]) -> None:
        """
        Log performance data for a model version

        Args:
            version: Model version
            performance_data: Performance metrics
        """
        try:
            log_entry = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'performance': performance_data
            }

            self.performance_log.append(log_entry)
            self._save_performance_log()

            self.logger.debug(f"📊 Performance logged for version {version}")

            # Check for auto-rollback conditions
            if self.rollback_config.auto_rollback:
                self._check_auto_rollback_conditions(version, performance_data)

        except Exception as e:
            self.logger.error(f"❌ Failed to log performance: {e}")

    def _check_auto_rollback_conditions(self, version: str, performance_data: Dict[str, Any]) -> None:
        """Check if auto-rollback should be triggered"""
        try:
            if version not in self.model_registry:
                return

            model_version = self.model_registry[version]
            if model_version.status != ModelStatus.ACTIVE:
                return

            # Get baseline metrics from registry
            baseline_metrics = model_version.metrics

            # Check performance degradation
            degradation_detected = False
            degraded_metrics = []

            for metric_name, current_value in performance_data.items():
                if hasattr(baseline_metrics, metric_name):
                    baseline_value = getattr(baseline_metrics, metric_name)

                    if baseline_value > 0:
                        pct_change = ((current_value - baseline_value) / baseline_value) * 100

                        # For metrics where higher is better
                        if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                            if pct_change < -self.rollback_config.performance_threshold * 100:
                                degradation_detected = True
                                degraded_metrics.append(metric_name)

                        # For metrics where lower is better
                        elif metric_name in ['log_loss', 'prediction_latency_ms']:
                            if pct_change > self.rollback_config.performance_threshold * 100:
                                degradation_detected = True
                                degraded_metrics.append(metric_name)

            # Trigger auto-rollback if degradation detected
            if degradation_detected:
                self.logger.warning(f"🚨 Performance degradation detected in version {version}")
                self.logger.warning(f"   Degraded metrics: {degraded_metrics}")

                # Check cooldown period
                if self._can_rollback(version):
                    self.logger.info(f"🔄 Triggering auto-rollback for version {version}")
                    rollback_success = self.rollback_model(version)

                    if rollback_success:
                        self._log_rollback_event(version, "auto_rollback", degraded_metrics)
                    else:
                        self.logger.error(f"❌ Auto-rollback failed for version {version}")
                else:
                    self.logger.info(f"⏰ Auto-rollback blocked by cooldown period")

        except Exception as e:
            self.logger.error(f"❌ Failed to check auto-rollback conditions: {e}")

    def _can_rollback(self, version: str) -> bool:
        """Check if rollback is allowed (cooldown period)"""
        try:
            cooldown_hours = self.rollback_config.rollback_cooldown_hours
            cutoff_time = datetime.now() - timedelta(hours=cooldown_hours)

            # Check recent rollbacks for this model type
            if version in self.model_registry:
                model_type = self.model_registry[version].model_type

                for rollback_event in self.rollback_log:
                    if rollback_event.get('model_type') == model_type.value:
                        rollback_time = datetime.fromisoformat(rollback_event['timestamp'])
                        if rollback_time > cutoff_time:
                            return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to check rollback cooldown: {e}")
            return False

    def rollback_model(self, version: str, target_version: Optional[str] = None) -> bool:
        """
        Rollback model to previous version

        Args:
            version: Current version to rollback
            target_version: Target version to rollback to (if None, uses previous)

        Returns:
            Success status
        """
        try:
            if version not in self.model_registry:
                self.logger.error(f"❌ Version {version} not found in registry")
                return False

            current_version = self.model_registry[version]
            model_type = current_version.model_type

            # Find target version
            if target_version is None:
                # Get previous active version of same type
                candidates = [
                    v for v in self.model_registry.values()
                    if v.model_type == model_type and v.version != version
                    and v.status in [ModelStatus.ACTIVE, ModelStatus.DEPRECATED]
                ]
                candidates.sort(key=lambda v: v.created_at, reverse=True)

                if candidates:
                    target_version = candidates[0].version
                else:
                    self.logger.error(f"❌ No previous version found for rollback of {version}")
                    return False

            if target_version not in self.model_registry:
                self.logger.error(f"❌ Target version {target_version} not found")
                return False

            # Perform rollback
            target_model_version = self.model_registry[target_version]

            # Mark current version as rolled back
            current_version.status = ModelStatus.ROLLED_BACK

            # Activate target version
            target_model_version.status = ModelStatus.ACTIVE

            # Save registry
            self._save_registry()

            self.logger.info(f"✅ Rollback completed successfully")
            self.logger.info(f"   From: {version} ({model_type.value})")
            self.logger.info(f"   To: {target_version}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to rollback model: {e}")
            return False

    def _log_rollback_event(self, version: str, reason: str, details: List[str]) -> None:
        """Log rollback event"""
        try:
            if version in self.model_registry:
                model_type = self.model_registry[version].model_type

                rollback_event = {
                    'version': version,
                    'model_type': model_type.value,
                    'timestamp': datetime.now().isoformat(),
                    'reason': reason,
                    'details': details
                }

                self.rollback_log.append(rollback_event)
                self._save_rollback_log()

        except Exception as e:
            self.logger.error(f"❌ Failed to log rollback event: {e}")

    def cleanup_old_versions(self, keep_versions: int = 10) -> int:
        """
        Cleanup old model versions

        Args:
            keep_versions: Number of versions to keep per model type

        Returns:
            Number of versions cleaned up
        """
        try:
            cleaned_count = 0

            for model_type in ModelType:
                versions = self.list_versions(model_type)

                if len(versions) > keep_versions:
                    # Sort by creation date (newest first)
                    versions.sort(key=lambda v: v.created_at, reverse=True)

                    # Keep newest N versions
                    versions_to_keep = versions[:keep_versions]
                    versions_to_remove = versions[keep_versions:]

                    for version in versions_to_remove:
                        try:
                            # Remove version from registry
                            if version.version in self.model_registry:
                                del self.model_registry[version.version]

                            # Remove model files
                            version_dir = Path(version.file_path).parent
                            if version_dir.exists():
                                shutil.rmtree(version_dir)

                            cleaned_count += 1
                            self.logger.info(f"🗑️ Cleaned up old version: {version.version}")

                        except Exception as e:
                            self.logger.error(f"❌ Failed to cleanup version {version.version}: {e}")

            if cleaned_count > 0:
                self._save_registry()
                self.logger.info(f"✅ Cleanup completed: {cleaned_count} versions removed")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old versions: {e}")
            return 0

    def get_version_summary(self) -> Dict[str, Any]:
        """Get summary of all model versions"""
        try:
            summary = {
                'total_versions': len(self.model_registry),
                'by_model_type': {},
                'by_status': {},
                'recent_versions': [],
                'nba_seasons': set(),
                'active_versions': {}
            }

            # Group by model type and status
            for version in self.model_registry.values():
                model_type = version.model_type.value
                status = version.status.value

                summary['by_model_type'][model_type] = summary['by_model_type'].get(model_type, 0) + 1
                summary['by_status'][status] = summary['by_status'].get(status, 0) + 1

                # Track NBA seasons
                if version.nba_season:
                    summary['nba_seasons'].add(version.nba_season)

                # Track active versions
                if version.status == ModelStatus.ACTIVE:
                    summary['active_versions'][model_type] = version.version

            # Recent versions (last 10)
            recent_versions = list(self.model_registry.values())
            recent_versions.sort(key=lambda v: v.created_at, reverse=True)
            summary['recent_versions'] = [
                {
                    'version': v.version,
                    'type': v.model_type.value,
                    'status': v.status.value,
                    'created_at': v.created_at.isoformat(),
                    'description': v.description
                }
                for v in recent_versions[:10]
            ]

            summary['nba_seasons'] = list(summary['nba_seasons'])

            return summary

        except Exception as e:
            self.logger.error(f"❌ Failed to get version summary: {e}")
            return {}

    def export_registry(self, export_path: str) -> bool:
        """Export model registry to file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'model_registry': {
                    version: version_data.to_dict()
                    for version, version_data in self.model_registry.items()
                },
                'performance_log': self.performance_log,
                'rollback_log': self.rollback_log,
                'summary': self.get_version_summary()
            }

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"✅ Registry exported to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to export registry: {e}")
            return False