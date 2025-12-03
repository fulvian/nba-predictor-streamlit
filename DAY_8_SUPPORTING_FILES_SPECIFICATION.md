# 🎯 Day 8 Supporting Files Implementation Specification

**Document Purpose**: Complete implementation specifications for all supporting files required for Day 8 ML State Management Centralization
**Target Audience**: General-purpose agents implementing Day 8 features
**Implementation Priority**: Core files first, then supporting validation, persistence, and synchronization layers

---

## 📂 FILE IMPLEMENTATION ORDER

### Priority 1: Core Infrastructure (Immediate)
1. `state_validators/__init__.py`
2. `state_validators/state_schema.py`
3. `state_validators/ml_state_validator.py`
4. `state_validators/consistency_checker.py`

### Priority 2: Persistence Layer (After Core)
1. `persistence/__init__.py`
2. `persistence/session_storage.py`
3. `persistence/file_storage.py`
4. `persistence/encryption_handler.py`

### Priority 3: Synchronization Layer (After Persistence)
1. `synchronization/__init__.py`
2. `synchronization/event_bus.py`
3. `synchronization/conflict_resolver.py`
4. `synchronization/background_sync.py`

### Priority 4: Testing (Development Phase)
1. `tests/state_management/test_state_manager.py`
2. `tests/state_management/test_state_validation.py`
3. `tests/state_management/test_integration/test_dashboard_integration.py`

---

## 🔧 PRIORITY 1: CORE INFRASTRUCTURE

### 1. state_validators/__init__.py

```python
"""
State Validators Module
Provides validation and consistency checking for ML system state.

This module includes state schema definitions, ML-specific validators,
and cross-component consistency checking.
"""

from .state_schema import StateSchema, ComponentStateSchema
from .ml_state_validator import MLStateValidator, ValidationSeverity
from .consistency_checker import ConsistencyChecker, ConsistencyIssue

__all__ = [
    'StateSchema',
    'ComponentStateSchema',
    'MLStateValidator',
    'ValidationSeverity',
    'ConsistencyChecker',
    'ConsistencyIssue'
]

# Module version for compatibility checking
__version__ = '1.0.0'
```

### 2. state_validators/state_schema.py

```python
"""
State Schema Definitions
Defines the structure and validation rules for ML system state.

This module provides schema definitions that ensure state consistency
and data integrity across all components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import json
import logging

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types for state fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"

class ValidationRule(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    LENGTH_CHECK = "length_check"
    PATTERN_CHECK = "pattern_check"
    CUSTOM = "custom"

@dataclass
class FieldDefinition:
    """Definition of a state field with validation rules."""
    name: str
    data_type: DataType
    required: bool = False
    default_value: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[Set[Any]] = None
    description: str = ""

@dataclass
class ComponentStateSchema:
    """Schema definition for a component's state."""
    component_id: str
    version: str = "1.0.0"
    required_fields: List[FieldDefinition] = field(default_factory=list)
    optional_fields: List[FieldDefinition] = field(default_factory=list)
    data_fields: List[FieldDefinition] = field(default_factory=list)
    metadata_fields: List[FieldDefinition] = field(default_factory=list)
    validation_rules: Dict[str, List[ValidationRule]] = field(default_factory=dict)

class StateSchema:
    """
    Central registry for state schemas and validation rules.

    This class provides a comprehensive schema system that defines
    the expected structure and validation rules for all ML system
    component states.
    """

    def __init__(self):
        """Initialize the state schema registry."""
        self._schemas: Dict[str, ComponentStateSchema] = {}
        self._initialize_core_schemas()

    def _initialize_core_schemas(self):
        """Initialize schemas for core ML system components."""

        # Data Pipeline Schema
        self._schemas['data_pipeline'] = ComponentStateSchema(
            component_id='data_pipeline',
            version='1.0.0',
            required_fields=[
                FieldDefinition('last_fetch_time', DataType.DATETIME, required=True),
                FieldDefinition('data_quality_score', DataType.FLOAT, required=True,
                              min_value=0.0, max_value=1.0),
                FieldDefinition('record_count', DataType.INTEGER, required=True,
                              min_value=0),
                FieldDefinition('status', DataType.STRING, required=True,
                              allowed_values={'active', 'idle', 'error', 'maintenance'})
            ],
            optional_fields=[
                FieldDefinition('fetch_duration_ms', DataType.INTEGER, min_value=0),
                FieldDefinition('data_sources', DataType.ARRAY),
                FieldDefinition('last_error', DataType.STRING),
                FieldDefinition('processing_queue_size', DataType.INTEGER, min_value=0)
            ]
        )

        # ML Models Schema
        self._schemas['ml_models'] = ComponentStateSchema(
            component_id='ml_models',
            version='1.0.0',
            required_fields=[
                FieldDefinition('model_versions', DataType.JSON, required=True),
                FieldDefinition('last_prediction_time', DataType.DATETIME, required=True),
                FieldDefinition('prediction_count', DataType.INTEGER, required=True, min_value=0),
                FieldDefinition('accuracy_score', DataType.FLOAT, required=True,
                              min_value=0.0, max_value=1.0)
            ],
            optional_fields=[
                FieldDefinition('average_confidence', DataType.FLOAT, min_value=0.0, max_value=1.0),
                FieldDefinition('model_load_time_ms', DataType.INTEGER, min_value=0),
                FieldDefinition('feature_importance', DataType.JSON),
                FieldDefinition('training_data_version', DataType.STRING),
                FieldDefinition('drift_score', DataType.FLOAT, min_value=0.0)
            ]
        )

        # Model Monitoring Schema
        self._schemas['model_monitoring'] = ComponentStateSchema(
            component_id='model_monitoring',
            version='1.0.0',
            required_fields=[
                FieldDefinition('last_check_time', DataType.DATETIME, required=True),
                FieldDefinition('monitoring_enabled', DataType.BOOLEAN, required=True),
                FieldDefinition('alert_count', DataType.INTEGER, required=True, min_value=0)
            ],
            optional_fields=[
                FieldDefinition('performance_metrics', DataType.JSON),
                FieldDefinition('drift_detected', DataType.BOOLEAN),
                FieldDefinition('last_drift_time', DataType.DATETIME),
                FieldDefinition('threshold_violations', DataType.INTEGER, min_value=0),
                FieldDefinition('health_score', DataType.FLOAT, min_value=0.0, max_value=1.0)
            ]
        )

        # Betting System Schema
        self._schemas['betting_system'] = ComponentStateSchema(
            component_id='betting_system',
            version='1.0.0',
            required_fields=[
                FieldDefinition('total_bankroll', DataType.FLOAT, required=True, min_value=0.0),
                FieldDefinition('active_bets_count', DataType.INTEGER, required=True, min_value=0),
                FieldDefinition('last_bet_time', DataType.DATETIME, required=True)
            ],
            optional_fields=[
                FieldDefinition('current_odds_source', DataType.STRING),
                FieldDefinition('total_winnings', DataType.FLOAT),
                FieldDefinition('total_losses', DataType.FLOAT),
                FieldDefinition('win_rate', DataType.FLOAT, min_value=0.0, max_value=1.0),
                FieldDefinition('risk_level', DataType.STRING,
                              allowed_values={'low', 'medium', 'high'})
            ]
        )

        # User Preferences Schema
        self._schemas['user_preferences'] = ComponentStateSchema(
            component_id='user_preferences',
            version='1.0.0',
            required_fields=[
                FieldDefinition('user_id', DataType.STRING, required=True),
                FieldDefinition('preferences_updated', DataType.DATETIME, required=True)
            ],
            optional_fields=[
                FieldDefinition('theme', DataType.STRING, allowed_values={'light', 'dark', 'auto'}),
                FieldDefinition('notification_enabled', DataType.BOOLEAN),
                FieldDefinition('risk_tolerance', DataType.STRING,
                              allowed_values={'conservative', 'moderate', 'aggressive'}),
                FieldDefinition('default_bet_amount', DataType.FLOAT, min_value=0.0),
                FieldDefinition('favorite_teams', DataType.ARRAY),
                FieldDefinition('dashboard_layout', DataType.JSON)
            ]
        )

        # System Health Schema
        self._schemas['system_health'] = ComponentStateSchema(
            component_id='system_health',
            version='1.0.0',
            required_fields=[
                FieldDefinition('overall_status', DataType.STRING, required=True,
                              allowed_values={'healthy', 'degraded', 'error', 'maintenance'}),
                FieldDefinition('uptime_percentage', DataType.FLOAT, required=True,
                              min_value=0.0, max_value=100.0),
                FieldDefinition('last_health_check', DataType.DATETIME, required=True)
            ],
            optional_fields=[
                FieldDefinition('memory_usage_mb', DataType.FLOAT, min_value=0.0),
                FieldDefinition('cpu_usage_percent', DataType.FLOAT, min_value=0.0, max_value=100.0),
                FieldDefinition('disk_usage_percent', DataType.FLOAT, min_value=0.0, max_value=100.0),
                FieldDefinition('active_connections', DataType.INTEGER, min_value=0),
                FieldDefinition('error_rate_24h', DataType.FLOAT, min_value=0.0)
            ]
        )

        logger.info(f"Initialized {len(self._schemas)} component schemas")

    def get_schema(self, component_id: str) -> Optional[ComponentStateSchema]:
        """
        Get schema for a specific component.

        Args:
            component_id: ID of the component

        Returns:
            ComponentStateSchema or None if not found
        """
        return self._schemas.get(component_id)

    def register_schema(self, schema: ComponentStateSchema) -> bool:
        """
        Register a new component schema.

        Args:
            schema: Component schema to register

        Returns:
            True if registered successfully, False if already exists
        """
        if schema.component_id in self._schemas:
            logger.warning(f"Schema for {schema.component_id} already exists")
            return False

        self._schemas[schema.component_id] = schema
        logger.info(f"Registered schema for component {schema.component_id}")
        return True

    def validate_field_value(self, field: FieldDefinition, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a field value against its definition.

        Args:
            field: Field definition
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Type check
        if not self._check_data_type(field.data_type, value):
            return False, f"Expected {field.data_type.value}, got {type(value).__name__}"

        # Range check for numeric values
        if field.min_value is not None and value < field.min_value:
            return False, f"Value {value} is below minimum {field.min_value}"

        if field.max_value is not None and value > field.max_value:
            return False, f"Value {value} is above maximum {field.max_value}"

        # Length check for strings and arrays
        if isinstance(value, (str, list)):
            if field.min_length is not None and len(value) < field.min_length:
                return False, f"Length {len(value)} is below minimum {field.min_length}"

            if field.max_length is not None and len(value) > field.max_length:
                return False, f"Length {len(value)} is above maximum {field.max_length}"

        # Pattern check for strings
        if field.pattern and isinstance(value, str):
            import re
            if not re.match(field.pattern, value):
                return False, f"Value does not match pattern {field.pattern}"

        # Allowed values check
        if field.allowed_values and value not in field.allowed_values:
            return False, f"Value {value} not in allowed values {field.allowed_values}"

        return True, None

    def _check_data_type(self, expected_type: DataType, value: Any) -> bool:
        """Check if value matches expected data type."""
        type_mapping = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: (int, float),
            DataType.BOOLEAN: bool,
            DataType.DATETIME: datetime,
            DataType.JSON: (dict, list),
            DataType.ARRAY: (list, tuple)
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return False

        return isinstance(value, expected_python_type)

    def get_all_schemas(self) -> Dict[str, ComponentStateSchema]:
        """Get all registered schemas."""
        return self._schemas.copy()

    def validate_schema_compatibility(self,
                                    old_schema: ComponentStateSchema,
                                    new_schema: ComponentStateSchema) -> tuple[bool, List[str]]:
        """
        Check if new schema is compatible with old schema.

        Args:
            old_schema: Existing schema
            new_schema: New schema to validate

        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []

        # Check for removed required fields
        old_required = {field.name for field in old_schema.required_fields}
        new_required = {field.name for field in new_schema.required_fields}

        removed_required = old_required - new_required
        if removed_required:
            issues.append(f"Removed required fields: {removed_required}")

        # Check for incompatible type changes
        all_old_fields = {field.name: field for field in old_schema.required_fields + old_schema.optional_fields}
        all_new_fields = {field.name: field for field in new_schema.required_fields + new_schema.optional_fields}

        for field_name, old_field in all_old_fields.items():
            if field_name in all_new_fields:
                new_field = all_new_fields[field_name]
                if old_field.data_type != new_field.data_type:
                    issues.append(f"Incompatible type change for {field_name}: "
                                f"{old_field.data_type.value} -> {new_field.data_type.value}")

        return len(issues) == 0, issues
```

### 3. state_validators/ml_state_validator.py

```python
"""
ML State Validator
Comprehensive validation for ML system component states.

This module provides specialized validation logic for ML-specific
components including models, data pipelines, and monitoring systems.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .state_schema import StateSchema, ComponentStateSchema, FieldDefinition

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Individual validation issue."""
    component_id: str
    field_name: Optional[str]
    severity: ValidationSeverity
    message: str
    timestamp: datetime
    suggestion: Optional[str] = None

class MLStateValidator:
    """
    Specialized validator for ML system state.

    This class provides comprehensive validation for all ML system
    components, including business logic validation, performance
    checks, and consistency verification.
    """

    def __init__(self, schema_registry: StateSchema):
        """
        Initialize the ML state validator.

        Args:
            schema_registry: Schema registry for structural validation
        """
        self.schema_registry = schema_registry
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> Dict[str, List[callable]]:
        """Initialize validation rules for different components."""
        return {
            'data_pipeline': [
                self._validate_data_pipeline_freshness,
                self._validate_data_quality_score,
                self._validate_record_count_consistency,
                self._validate_processing_performance
            ],
            'ml_models': [
                self._validate_model_version_consistency,
                self._validate_prediction_frequency,
                self._validate_accuracy_thresholds,
                self._validate_confidence_scores
            ],
            'model_monitoring': [
                self._validate_monitoring_frequency,
                self._validate_alert_thresholds,
                self._validate_drift_detection
            ],
            'betting_system': [
                self._validate_bankroll_consistency,
                self._validate_bet_limits,
                self._validate_odds_consistency
            ],
            'system_health': [
                self._validate_system_metrics,
                self._validate_uptime_consistency,
                self._validate_error_rates
            ]
        }

    def validate_component_state(self, component_id: str, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate component state comprehensively.

        Args:
            component_id: ID of the component to validate
            state_data: State data to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Get schema for component
        schema = self.schema_registry.get_schema(component_id)
        if schema is None:
            issues.append(ValidationIssue(
                component_id=component_id,
                field_name=None,
                severity=ValidationSeverity.WARNING,
                message=f"No schema found for component {component_id}",
                timestamp=datetime.now(),
                suggestion="Register a schema for this component"
            ))
            return issues

        # Structural validation against schema
        schema_issues = self._validate_against_schema(schema, state_data)
        issues.extend(schema_issues)

        # Business logic validation
        if component_id in self.validation_rules:
            for rule in self.validation_rules[component_id]:
                try:
                    rule_issues = rule(state_data)
                    issues.extend(rule_issues)
                except Exception as e:
                    logger.error(f"Error in validation rule for {component_id}: {e}")
                    issues.append(ValidationIssue(
                        component_id=component_id,
                        field_name=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule error: {e}",
                        timestamp=datetime.now()
                    ))

        return issues

    def _validate_against_schema(self, schema: ComponentStateSchema, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate state data against component schema.

        Args:
            schema: Component schema
            state_data: State data to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Check required fields
        for field_def in schema.required_fields:
            if field_def.name not in state_data:
                issues.append(ValidationIssue(
                    component_id=schema.component_id,
                    field_name=field_def.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field_def.name}' is missing",
                    timestamp=datetime.now(),
                    suggestion=f"Add field '{field_def.name}' with appropriate {field_def.data_type.value} value"
                ))
            else:
                # Validate field value
                is_valid, error_msg = self.schema_registry.validate_field_value(
                    field_def, state_data[field_def.name]
                )
                if not is_valid:
                    issues.append(ValidationIssue(
                        component_id=schema.component_id,
                        field_name=field_def.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_def.name}' validation failed: {error_msg}",
                        timestamp=datetime.now()
                    ))

        # Check optional fields (if present)
        for field_def in schema.optional_fields:
            if field_def.name in state_data:
                is_valid, error_msg = self.schema_registry.validate_field_value(
                    field_def, state_data[field_def.name]
                )
                if not is_valid:
                    issues.append(ValidationIssue(
                        component_id=schema.component_id,
                        field_name=field_def.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Optional field '{field_def.name}' validation failed: {error_msg}",
                        timestamp=datetime.now()
                    ))

        return issues

    def _validate_data_pipeline_freshness(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data freshness for data pipeline."""
        issues = []

        if 'last_fetch_time' in state_data:
            last_fetch = state_data['last_fetch_time']
            if isinstance(last_fetch, str):
                try:
                    last_fetch = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
                except ValueError:
                    issues.append(ValidationIssue(
                        component_id='data_pipeline',
                        field_name='last_fetch_time',
                        severity=ValidationSeverity.ERROR,
                        message="Invalid datetime format for last_fetch_time",
                        timestamp=datetime.now(),
                        suggestion="Use ISO 8601 format"
                    ))
                    return issues

            # Check if data is stale (older than 1 hour)
            age_minutes = (datetime.now() - last_fetch).total_seconds() / 60
            if age_minutes > 60:
                severity = ValidationSeverity.CRITICAL if age_minutes > 180 else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='last_fetch_time',
                    severity=severity,
                    message=f"Data is {age_minutes:.1f} minutes old",
                    timestamp=datetime.now(),
                    suggestion="Trigger data refresh"
                ))

        return issues

    def _validate_data_quality_score(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data quality scores."""
        issues = []

        if 'data_quality_score' in state_data:
            score = state_data['data_quality_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='data_quality_score',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid data quality score: {score}",
                    timestamp=datetime.now(),
                    suggestion="Score must be between 0.0 and 1.0"
                ))
            elif score < 0.8:
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='data_quality_score',
                    severity=ValidationSeverity.WARNING,
                    message=f"Low data quality score: {score:.3f}",
                    timestamp=datetime.now(),
                    suggestion="Investigate data quality issues"
                ))

        return issues

    def _validate_record_count_consistency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate record count consistency."""
        issues = []

        if 'record_count' in state_data:
            count = state_data['record_count']
            if not isinstance(count, int) or count < 0:
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='record_count',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid record count: {count}",
                    timestamp=datetime.now(),
                    suggestion="Record count must be a non-negative integer"
                ))
            elif count == 0:
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='record_count',
                    severity=ValidationSeverity.WARNING,
                    message="No records found",
                    timestamp=datetime.now(),
                    suggestion="Check data ingestion process"
                ))

        return issues

    def _validate_processing_performance(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data processing performance."""
        issues = []

        if 'fetch_duration_ms' in state_data:
            duration = state_data['fetch_duration_ms']
            if not isinstance(duration, (int, float)) or duration < 0:
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='fetch_duration_ms',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid fetch duration: {duration}",
                    timestamp=datetime.now()
                ))
            elif duration > 30000:  # 30 seconds
                issues.append(ValidationIssue(
                    component_id='data_pipeline',
                    field_name='fetch_duration_ms',
                    severity=ValidationSeverity.WARNING,
                    message=f"Slow data fetch: {duration}ms",
                    timestamp=datetime.now(),
                    suggestion="Optimize data fetching process"
                ))

        return issues

    def _validate_model_version_consistency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate model version consistency."""
        issues = []

        if 'model_versions' in state_data:
            versions = state_data['model_versions']
            if not isinstance(versions, dict):
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='model_versions',
                    severity=ValidationSeverity.ERROR,
                    message="model_versions must be a dictionary",
                    timestamp=datetime.now(),
                    suggestion="Use format: {'model_name': 'version'}"
                ))
            elif len(versions) == 0:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='model_versions',
                    severity=ValidationSeverity.WARNING,
                    message="No model versions specified",
                    timestamp=datetime.now(),
                    suggestion="Load ML models and specify versions"
                ))

        return issues

    def _validate_prediction_frequency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate prediction generation frequency."""
        issues = []

        if 'last_prediction_time' in state_data:
            last_prediction = state_data['last_prediction_time']
            if isinstance(last_prediction, str):
                try:
                    last_prediction = datetime.fromisoformat(last_prediction.replace('Z', '+00:00'))
                except ValueError:
                    issues.append(ValidationIssue(
                        component_id='ml_models',
                        field_name='last_prediction_time',
                        severity=ValidationSeverity.ERROR,
                        message="Invalid datetime format",
                        timestamp=datetime.now()
                    ))
                    return issues

            # Check if predictions are stale
            age_minutes = (datetime.now() - last_prediction).total_seconds() / 60
            if age_minutes > 15:  # 15 minutes
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='last_prediction_time',
                    severity=ValidationSeverity.WARNING,
                    message=f"Predictions are {age_minutes:.1f} minutes old",
                    timestamp=datetime.now(),
                    suggestion="Generate new predictions"
                ))

        return issues

    def _validate_accuracy_thresholds(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate model accuracy thresholds."""
        issues = []

        if 'accuracy_score' in state_data:
            accuracy = state_data['accuracy_score']
            if not isinstance(accuracy, (int, float)) or accuracy < 0 or accuracy > 1:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='accuracy_score',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid accuracy score: {accuracy}",
                    timestamp=datetime.now(),
                    suggestion="Accuracy must be between 0.0 and 1.0"
                ))
            elif accuracy < 0.6:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='accuracy_score',
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Very low model accuracy: {accuracy:.3f}",
                    timestamp=datetime.now(),
                    suggestion="Retrain models or investigate data quality"
                ))
            elif accuracy < 0.75:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='accuracy_score',
                    severity=ValidationSeverity.WARNING,
                    message=f"Low model accuracy: {accuracy:.3f}",
                    timestamp=datetime.now(),
                    suggestion="Consider model retraining"
                ))

        return issues

    def _validate_confidence_scores(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate confidence score ranges."""
        issues = []

        if 'average_confidence' in state_data:
            confidence = state_data['average_confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='average_confidence',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid confidence score: {confidence}",
                    timestamp=datetime.now(),
                    suggestion="Confidence must be between 0.0 and 1.0"
                ))
            elif confidence < 0.5:
                issues.append(ValidationIssue(
                    component_id='ml_models',
                    field_name='average_confidence',
                    severity=ValidationSeverity.WARNING,
                    message=f"Low confidence scores: {confidence:.3f}",
                    timestamp=datetime.now(),
                    suggestion="Review prediction confidence thresholds"
                ))

        return issues

    def _validate_monitoring_frequency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate monitoring system frequency."""
        issues = []

        if 'last_check_time' in state_data:
            last_check = state_data['last_check_time']
            if isinstance(last_check, str):
                try:
                    last_check = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                except ValueError:
                    issues.append(ValidationIssue(
                        component_id='model_monitoring',
                        field_name='last_check_time',
                        severity=ValidationSeverity.ERROR,
                        message="Invalid datetime format",
                        timestamp=datetime.now()
                    ))
                    return issues

            # Check if monitoring is active
            age_minutes = (datetime.now() - last_check).total_seconds() / 60
            if 'monitoring_enabled' in state_data and state_data['monitoring_enabled']:
                if age_minutes > 5:  # 5 minutes
                    issues.append(ValidationIssue(
                        component_id='model_monitoring',
                        field_name='last_check_time',
                        severity=ValidationSeverity.WARNING,
                        message=f"Monitoring check overdue by {age_minutes:.1f} minutes",
                        timestamp=datetime.now(),
                        suggestion="Check monitoring system status"
                    ))

        return issues

    def _validate_alert_thresholds(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate alert thresholds and counts."""
        issues = []

        if 'alert_count' in state_data:
            alert_count = state_data['alert_count']
            if not isinstance(alert_count, int) or alert_count < 0:
                issues.append(ValidationIssue(
                    component_id='model_monitoring',
                    field_name='alert_count',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid alert count: {alert_count}",
                    timestamp=datetime.now()
                ))
            elif alert_count > 10:
                issues.append(ValidationIssue(
                    component_id='model_monitoring',
                    field_name='alert_count',
                    severity=ValidationSeverity.WARNING,
                    message=f"High number of alerts: {alert_count}",
                    timestamp=datetime.now(),
                    suggestion="Review and address active alerts"
                ))

        return issues

    def _validate_drift_detection(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate drift detection status."""
        issues = []

        if 'drift_detected' in state_data and state_data['drift_detected']:
            issues.append(ValidationIssue(
                component_id='model_monitoring',
                field_name='drift_detected',
                severity=ValidationSeverity.CRITICAL,
                message="Model drift detected",
                timestamp=datetime.now(),
                suggestion="Trigger model retraining pipeline"
            ))

        return issues

    def _validate_bankroll_consistency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate betting bankroll consistency."""
        issues = []

        if 'total_bankroll' in state_data:
            bankroll = state_data['total_bankroll']
            if not isinstance(bankroll, (int, float)) or bankroll < 0:
                issues.append(ValidationIssue(
                    component_id='betting_system',
                    field_name='total_bankroll',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid bankroll amount: {bankroll}",
                    timestamp=datetime.now(),
                    suggestion="Bankroll must be a non-negative number"
                ))

        return issues

    def _validate_bet_limits(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate betting limits and amounts."""
        issues = []

        # Check if bet amounts are reasonable compared to bankroll
        if 'total_bankroll' in state_data and 'default_bet_amount' in state_data:
            bankroll = state_data['total_bankroll']
            default_bet = state_data['default_bet_amount']

            if isinstance(default_bet, (int, float)) and isinstance(bankroll, (int, float)):
                if default_bet > bankroll:
                    issues.append(ValidationIssue(
                        component_id='betting_system',
                        field_name='default_bet_amount',
                        severity=ValidationSeverity.ERROR,
                        message=f"Default bet amount (${default_bet}) exceeds bankroll (${bankroll})",
                        timestamp=datetime.now(),
                        suggestion="Adjust default bet amount"
                    ))
                elif default_bet > bankroll * 0.1:  # 10% of bankroll
                    issues.append(ValidationIssue(
                        component_id='betting_system',
                        field_name='default_bet_amount',
                        severity=ValidationSeverity.WARNING,
                        message=f"High default bet amount: {default_bet/bankroll*100:.1f}% of bankroll",
                        timestamp=datetime.now(),
                        suggestion="Consider more conservative bet sizing"
                    ))

        return issues

    def _validate_odds_consistency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate odds consistency."""
        issues = []

        if 'current_odds_source' in state_data:
            odds_source = state_data['current_odds_source']
            valid_sources = {'manual', 'api', 'scraped', 'calculated'}
            if odds_source not in valid_sources:
                issues.append(ValidationIssue(
                    component_id='betting_system',
                    field_name='current_odds_source',
                    severity=ValidationSeverity.WARNING,
                    message=f"Unrecognized odds source: {odds_source}",
                    timestamp=datetime.now(),
                    suggestion=f"Use one of: {valid_sources}"
                ))

        return issues

    def _validate_system_metrics(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate system health metrics."""
        issues = []

        # CPU usage validation
        if 'cpu_usage_percent' in state_data:
            cpu_usage = state_data['cpu_usage_percent']
            if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0 or cpu_usage > 100:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='cpu_usage_percent',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid CPU usage: {cpu_usage}%",
                    timestamp=datetime.now()
                ))
            elif cpu_usage > 90:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='cpu_usage_percent',
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High CPU usage: {cpu_usage}%",
                    timestamp=datetime.now(),
                    suggestion="Scale resources or optimize processes"
                ))

        # Memory usage validation
        if 'memory_usage_mb' in state_data:
            memory_usage = state_data['memory_usage_mb']
            if not isinstance(memory_usage, (int, float)) or memory_usage < 0:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='memory_usage_mb',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid memory usage: {memory_usage}MB",
                    timestamp=datetime.now()
                ))

        # Disk usage validation
        if 'disk_usage_percent' in state_data:
            disk_usage = state_data['disk_usage_percent']
            if not isinstance(disk_usage, (int, float)) or disk_usage < 0 or disk_usage > 100:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='disk_usage_percent',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid disk usage: {disk_usage}%",
                    timestamp=datetime.now()
                ))
            elif disk_usage > 85:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='disk_usage_percent',
                    severity=ValidationSeverity.WARNING,
                    message=f"High disk usage: {disk_usage}%",
                    timestamp=datetime.now(),
                    suggestion="Clean up disk space or expand storage"
                ))

        return issues

    def _validate_uptime_consistency(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate uptime consistency."""
        issues = []

        if 'uptime_percentage' in state_data:
            uptime = state_data['uptime_percentage']
            if not isinstance(uptime, (int, float)) or uptime < 0 or uptime > 100:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='uptime_percentage',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid uptime percentage: {uptime}%",
                    timestamp=datetime.now()
                ))
            elif uptime < 95:
                severity = ValidationSeverity.CRITICAL if uptime < 90 else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='uptime_percentage',
                    severity=severity,
                    message=f"Low uptime: {uptime}%",
                    timestamp=datetime.now(),
                    suggestion="Investigate system stability issues"
                ))

        return issues

    def _validate_error_rates(self, state_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate error rates."""
        issues = []

        if 'error_rate_24h' in state_data:
            error_rate = state_data['error_rate_24h']
            if not isinstance(error_rate, (int, float)) or error_rate < 0:
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='error_rate_24h',
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid error rate: {error_rate}",
                    timestamp=datetime.now()
                ))
            elif error_rate > 5:  # 5% error rate
                severity = ValidationSeverity.CRITICAL if error_rate > 10 else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    component_id='system_health',
                    field_name='error_rate_24h',
                    severity=severity,
                    message=f"High error rate: {error_rate}%",
                    timestamp=datetime.now(),
                    suggestion="Investigate and fix error sources"
                ))

        return issues
```

### 4. state_validators/consistency_checker.py

```python
"""
Consistency Checker
Cross-component consistency validation for ML system state.

This module provides comprehensive consistency checking across all
ML system components to ensure state synchronization and data integrity.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from .ml_state_validator import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

@dataclass
class ConsistencyIssue:
    """Cross-component consistency issue."""
    component_ids: List[str]
    severity: ValidationSeverity
    message: str
    timestamp: datetime
    suggestion: Optional[str] = None
    affected_data: Dict[str, Any] = None

class ConsistencyChecker:
    """
    Cross-component consistency checker.

    This class provides comprehensive consistency validation across
    all ML system components, detecting data synchronization issues,
    logical inconsistencies, and state conflicts.
    """

    def __init__(self):
        """Initialize the consistency checker."""
        self.consistency_rules = self._initialize_consistency_rules()

    def _initialize_consistency_rules(self) -> List[callable]:
        """Initialize consistency checking rules."""
        return [
            self._check_ml_pipeline_consistency,
            self._check_prediction_bet_consistency,
            self._check_monitoring_model_consistency,
            self._check_user_preferences_consistency,
            self._check_system_resource_consistency,
            self._check_temporal_consistency,
            self._check_data_flow_consistency
        ]

    def check_system_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """
        Check consistency across all system components.

        Args:
            all_states: Dictionary of all component states

        Returns:
            List of consistency issues
        """
        issues = []

        for rule in self.consistency_rules:
            try:
                rule_issues = rule(all_states)
                issues.extend(rule_issues)
            except Exception as e:
                logger.error(f"Error in consistency rule: {e}")
                issues.append(ConsistencyIssue(
                    component_ids=['system'],
                    severity=ValidationSeverity.ERROR,
                    message=f"Consistency check error: {e}",
                    timestamp=datetime.now()
                ))

        return issues

    def _check_ml_pipeline_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check consistency between ML models and data pipeline."""
        issues = []

        data_state = all_states.get('data_pipeline', {})
        model_state = all_states.get('ml_models', {})

        # Check if models are healthy but data pipeline has errors
        if (model_state.get('status') == 'healthy' and
            data_state.get('status') in ['error', 'offline']):

            issues.append(ConsistencyIssue(
                component_ids=['data_pipeline', 'ml_models'],
                severity=ValidationSeverity.WARNING,
                message="ML models healthy but data pipeline has issues",
                timestamp=datetime.now(),
                suggestion="Investigate data pipeline status",
                affected_data={
                    'data_status': data_state.get('status'),
                    'model_status': model_state.get('status')
                }
            ))

        # Check data quality vs model accuracy correlation
        data_quality = data_state.get('data_quality_score')
        model_accuracy = model_state.get('accuracy_score')

        if (data_quality is not None and model_accuracy is not None and
            data_quality > 0.9 and model_accuracy < 0.6):

            issues.append(ConsistencyIssue(
                component_ids=['data_pipeline', 'ml_models'],
                severity=ValidationSeverity.WARNING,
                message="High data quality but low model accuracy",
                timestamp=datetime.now(),
                suggestion="Review model training or feature engineering",
                affected_data={
                    'data_quality': data_quality,
                    'model_accuracy': model_accuracy
                }
            ))

        # Check record count consistency
        record_count = data_state.get('record_count', 0)
        prediction_count = model_state.get('prediction_count', 0)

        if record_count > 0 and prediction_count > record_count:
            issues.append(ConsistencyIssue(
                component_ids=['data_pipeline', 'ml_models'],
                severity=ValidationSeverity.ERROR,
                message="More predictions than data records",
                timestamp=datetime.now(),
                suggestion="Check data-prediction synchronization",
                affected_data={
                    'record_count': record_count,
                    'prediction_count': prediction_count
                }
            ))

        return issues

    def _check_prediction_bet_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check consistency between predictions and betting system."""
        issues = []

        model_state = all_states.get('ml_models', {})
        betting_state = all_states.get('betting_system', {})

        # Check if betting is active but no recent predictions
        last_prediction = model_state.get('last_prediction_time')
        last_bet = betting_state.get('last_bet_time')

        if last_prediction and last_bet:
            try:
                pred_time = datetime.fromisoformat(last_prediction.replace('Z', '+00:00'))
                bet_time = datetime.fromisoformat(last_bet.replace('Z', '+00:00'))

                # If last bet is much newer than last prediction
                if (bet_time - pred_time).total_seconds() > 300:  # 5 minutes
                    issues.append(ConsistencyIssue(
                        component_ids=['ml_models', 'betting_system'],
                        severity=ValidationSeverity.WARNING,
                        message="Bets placed without recent predictions",
                        timestamp=datetime.now(),
                        suggestion="Generate new predictions before betting",
                        affected_data={
                            'last_prediction': last_prediction,
                            'last_bet': last_bet
                        }
                    ))

            except ValueError:
                # Invalid datetime format
                pass

        # Check bankroll consistency with betting activity
        active_bets = betting_state.get('active_bets_count', 0)
        total_bankroll = betting_state.get('total_bankroll', 0)

        if active_bets > 0 and total_bankroll <= 0:
            issues.append(ConsistencyIssue(
                component_ids=['betting_system'],
                severity=ValidationSeverity.ERROR,
                message="Active bets with zero bankroll",
                timestamp=datetime.now(),
                suggestion="Update bankroll or settle bets",
                affected_data={
                    'active_bets': active_bets,
                    'bankroll': total_bankroll
                }
            ))

        return issues

    def _check_monitoring_model_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check consistency between monitoring and ML models."""
        issues = []

        model_state = all_states.get('ml_models', {})
        monitoring_state = all_states.get('model_monitoring', {})

        # Check if monitoring is disabled for healthy models
        if (model_state.get('status') == 'healthy' and
            not monitoring_state.get('monitoring_enabled', True)):

            issues.append(ConsistencyIssue(
                component_ids=['ml_models', 'model_monitoring'],
                severity=ValidationSeverity.WARNING,
                message="Monitoring disabled for healthy models",
                timestamp=datetime.now(),
                suggestion="Consider enabling monitoring for proactive detection",
                affected_data={
                    'model_status': model_state.get('status'),
                    'monitoring_enabled': monitoring_state.get('monitoring_enabled')
                }
            ))

        # Check drift detection consistency
        drift_detected = monitoring_state.get('drift_detected', False)
        model_accuracy = model_state.get('accuracy_score')

        if drift_detected and model_accuracy and model_accuracy > 0.8:
            issues.append(ConsistencyIssue(
                component_ids=['ml_models', 'model_monitoring'],
                severity=ValidationSeverity.WARNING,
                message="Drift detected but high model accuracy",
                timestamp=datetime.now(),
                suggestion="Review drift detection thresholds",
                affected_data={
                    'drift_detected': drift_detected,
                    'model_accuracy': model_accuracy
                }
            ))

        # Check monitoring frequency consistency
        last_prediction = model_state.get('last_prediction_time')
        last_check = monitoring_state.get('last_check_time')

        if last_prediction and last_check:
            try:
                pred_time = datetime.fromisoformat(last_prediction.replace('Z', '+00:00'))
                check_time = datetime.fromisoformat(last_check.replace('Z', '+00:00'))

                # If monitoring hasn't checked since last prediction
                if check_time < pred_time:
                    issues.append(ConsistencyIssue(
                        component_ids=['ml_models', 'model_monitoring'],
                        severity=ValidationSeverity.WARNING,
                        message="Monitoring check older than last prediction",
                        timestamp=datetime.now(),
                        suggestion="Run monitoring check after model updates",
                        affected_data={
                            'last_prediction': last_prediction,
                            'last_check': last_check
                        }
                    ))

            except ValueError:
                pass

        return issues

    def _check_user_preferences_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check consistency in user preferences."""
        issues = []

        user_prefs = all_states.get('user_preferences', {})
        betting_state = all_states.get('betting_system', {})

        # Check risk tolerance vs betting behavior
        risk_tolerance = user_prefs.get('risk_tolerance')
        default_bet = user_prefs.get('default_bet_amount', 0)
        total_bankroll = betting_state.get('total_bankroll', 1000)

        if risk_tolerance and default_bet > 0:
            risk_bet_ratios = {
                'conservative': 0.02,  # 2% max
                'moderate': 0.05,      # 5% max
                'aggressive': 0.10     # 10% max
            }

            max_ratio = risk_bet_ratios.get(risk_tolerance, 0.05)
            actual_ratio = default_bet / total_bankroll if total_bankroll > 0 else 0

            if actual_ratio > max_ratio:
                issues.append(ConsistencyIssue(
                    component_ids=['user_preferences', 'betting_system'],
                    severity=ValidationSeverity.WARNING,
                    message=f"Bet amount ({actual_ratio*100:.1f}%) exceeds {risk_tolerance} risk tolerance ({max_ratio*100:.1f}%)",
                    timestamp=datetime.now(),
                    suggestion="Adjust default bet amount or risk tolerance",
                    affected_data={
                        'risk_tolerance': risk_tolerance,
                        'bet_ratio': actual_ratio,
                        'max_ratio': max_ratio
                    }
                ))

        return issues

    def _check_system_resource_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check system resource consistency."""
        issues = []

        system_health = all_states.get('system_health', {})
        model_state = all_states.get('ml_models', {})
        betting_state = all_states.get('betting_system', {})

        # Check if system resources are strained but components are healthy
        cpu_usage = system_health.get('cpu_usage_percent', 0)
        memory_usage = system_health.get('memory_usage_mb', 0)
        overall_status = system_health.get('overall_status', 'unknown')

        if (cpu_usage > 90 or memory_usage > 1000) and overall_status == 'healthy':
            issues.append(ConsistencyIssue(
                component_ids=['system_health'],
                severity=ValidationSeverity.WARNING,
                message=f"System resources strained but status is healthy (CPU: {cpu_usage}%, Memory: {memory_usage}MB)",
                timestamp=datetime.now(),
                suggestion="Update system health status to reflect resource strain",
                affected_data={
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'overall_status': overall_status
                }
            ))

        # Check activity vs resource usage correlation
        active_components = 0
        if model_state.get('status') == 'healthy':
            active_components += 1
        if betting_state.get('active_bets_count', 0) > 0:
            active_components += 1

        if active_components > 0 and cpu_usage < 10:
            issues.append(ConsistencyIssue(
                component_ids=['system_health', 'ml_models', 'betting_system'],
                severity=ValidationSeverity.INFO,
                message="Low CPU usage despite active components",
                timestamp=datetime.now(),
                suggestion="Verify resource monitoring accuracy",
                affected_data={
                    'active_components': active_components,
                    'cpu_usage': cpu_usage
                }
            ))

        return issues

    def _check_temporal_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check temporal consistency across components."""
        issues = []

        # Collect all timestamps
        timestamps = {}
        for component_id, state in all_states.items():
            for field_name, field_value in state.items():
                if 'time' in field_name.lower() and isinstance(field_value, str):
                    try:
                        timestamp = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                        timestamps[f"{component_id}.{field_name}"] = timestamp
                    except ValueError:
                        # Invalid timestamp format
                        issues.append(ConsistencyIssue(
                            component_ids=[component_id],
                            severity=ValidationSeverity.WARNING,
                            message=f"Invalid timestamp format in {field_name}: {field_value}",
                            timestamp=datetime.now(),
                            suggestion="Use ISO 8601 datetime format"
                        ))

        # Check for future timestamps
        now = datetime.now()
        for location, timestamp in timestamps.items():
            if timestamp > now + timedelta(minutes=5):  # 5 minute tolerance
                issues.append(ConsistencyIssue(
                    component_ids=[location.split('.')[0]],
                    severity=ValidationSeverity.ERROR,
                    message=f"Future timestamp detected in {location}: {timestamp}",
                    timestamp=now,
                    suggestion="Check system clock synchronization",
                    affected_data={'timestamp': timestamp.isoformat()}
                ))

        # Check for very old timestamps (stale data)
        cutoff_time = now - timedelta(hours=24)
        stale_fields = [
            location for location, timestamp in timestamps.items()
            if timestamp < cutoff_time
        ]

        if stale_fields:
            issues.append(ConsistencyIssue(
                component_ids=list(set(loc.split('.')[0] for loc in stale_fields)),
                severity=ValidationSeverity.WARNING,
                message=f"Stale data detected in {len(stale_fields)} fields (older than 24 hours)",
                timestamp=now,
                suggestion="Update stale data or investigate data pipeline",
                affected_data={'stale_fields': stale_fields}
            ))

        return issues

    def _check_data_flow_consistency(self, all_states: Dict[str, Dict[str, Any]]) -> List[ConsistencyIssue]:
        """Check data flow consistency across the pipeline."""
        issues = []

        data_state = all_states.get('data_pipeline', {})
        model_state = all_states.get('ml_models', {})
        monitoring_state = all_states.get('model_monitoring', {})

        # Check data pipeline flow: fetch -> process -> predict -> monitor
        data_fetch_time = data_state.get('last_fetch_time')
        prediction_time = model_state.get('last_prediction_time')
        monitoring_time = monitoring_state.get('last_check_time')

        timestamps = []
        if data_fetch_time:
            try:
                timestamps.append(('data_fetch', datetime.fromisoformat(data_fetch_time.replace('Z', '+00:00'))))
            except ValueError:
                pass

        if prediction_time:
            try:
                timestamps.append(('prediction', datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))))
            except ValueError:
                pass

        if monitoring_time:
            try:
                timestamps.append(('monitoring', datetime.fromisoformat(monitoring_time.replace('Z', '+00:00'))))
            except ValueError:
                pass

        # Check chronological order
        if len(timestamps) >= 2:
            timestamps.sort(key=lambda x: x[1])

            expected_order = ['data_fetch', 'prediction', 'monitoring']
            actual_order = [t[0] for t in timestamps]

            # Find out-of-order elements
            for i, (expected, actual) in enumerate(zip(expected_order, actual_order)):
                if actual != expected and actual in expected_order[i+1:]:
                    issues.append(ConsistencyIssue(
                        component_ids=['data_pipeline', 'ml_models', 'model_monitoring'],
                        severity=ValidationSeverity.WARNING,
                        message=f"Data flow order issue: {actual} came before {expected}",
                        timestamp=datetime.now(),
                        suggestion="Check pipeline execution sequence",
                        affected_data={
                            'expected_order': expected_order,
                            'actual_order': actual_order,
                            'timestamps': {t[0]: t[1].isoformat() for t in timestamps}
                        }
                    ))

        return issues
```

---

## 🎯 IMPLEMENTATION EXECUTION PLAN

This specification provides complete implementation details for all supporting files needed for Day 8. General-purpose agents should:

1. **Start with Priority 1 files** (Core Infrastructure)
2. **Follow exact code specifications** provided
3. **Test each component individually** before integration
4. **Maintain X7 Compliant patterns** throughout implementation
5. **Ensure thread safety** in all operations

### Implementation Checklist

#### Core Infrastructure
- [ ] Create directory structure
- [ ] Implement `state_validators/__init__.py`
- [ ] Implement `state_validators/state_schema.py`
- [ ] Implement `state_validators/ml_state_validator.py`
- [ ] Implement `state_validators/consistency_checker.py`

#### Testing
- [ ] Create test files for each component
- [ ] Run individual component tests
- [ ] Run integration tests
- [ ] Validate performance benchmarks

#### Integration
- [ ] Integrate with existing `state_manager.py`
- [ ] Connect to dashboard components
- [ ] Test end-to-end functionality
- [ ] Validate performance under load

This comprehensive specification ensures precise implementation by general-purpose agents while maintaining the highest standards of code quality, thread safety, and X7 compliance.