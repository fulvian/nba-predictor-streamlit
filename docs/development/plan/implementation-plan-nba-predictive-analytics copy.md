# Implementation Plan: NBA Predictive Analytics System

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `nba-predictive-analytics-2024`
**Phase**: Development
**Priority**: 9/10
**Estimated Duration**: 50 hours

---

## 🎯 EXECUTION PROFILE FOR GLM-4.6

You are an **expert coding agent** specialized in **precise execution** of well-defined tasks.

**YOUR STRENGTHS** (leverage these):
- ✅ Tool calling accuracy 90.6% (best-in-class)
- ✅ Efficient token usage (15% fewer than alternatives)
- ✅ Standard coding patterns excellence
- ✅ Integration with Claude Code ecosystem

**YOUR CONSTRAINTS** (respect these):
- ⚠️ AVOID prolonged reasoning (thinking mode costly - 18K tokens)
- ⚠️ FOCUS on execution over exploration
- ⚠️ FOLLOW provided patterns exactly (framework knowledge gaps)
- ⚠️ CHECK syntax precision (13% error rate - mitigate with type hints)
- ⚠️ COMPLETE micro-tasks fully (no early quit - acceptance criteria mandatory)

---

## 📋 MICRO-TASK BREAKDOWN

### Task 1: Unified Data Pipeline Implementation (Duration: 480 min)

**File**: `unified_nba_data_pipeline.py` (Lines: 1-300)

**ACTION**: Create centralized data orchestration system

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class UnifiedNBADataPipeline:
    """
    Centralized data pipeline for NBA predictive analytics.

    This class orchestrates data collection from multiple sources,
    performs quality validation, and provides unified access to
    NBA data for machine learning models.

    Attributes:
        data_provider: NBA data provider instance
        feature_engineer: Feature engineering module
        cache: Data caching mechanism
    """

    def __init__(
        self,
        data_provider: Optional[NBADataProvider] = None,
        cache_ttl: int = 3600
    ) -> None:
        """Initialize the unified data pipeline."""

    def fetch_all_data(
        self,
        date_range: Tuple[date, date],
        include_boxscores: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive NBA data for specified date range."""

    def preprocess_features(
        self,
        raw_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Preprocess and engineer features from raw NBA data."""

    def validate_data_quality(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate data quality and completeness."""
```

**PATTERN REFERENCE**: See `data_provider_june2025.py:50-100` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    result = self._fetch_with_retry(endpoint, params)
except (ConnectionError, TimeoutError) as e:
    logger.error(
        "Data fetch failed",
        extra={"endpoint": endpoint, "error": str(e)}
    )
    raise DataPipelineError(f"Failed to fetch data from {endpoint}") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need pandas/numpy optimization patterns
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="pandas")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="data optimization caching",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_unified_nba_data_pipeline.py::test_data_pipeline_initialization`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_unified_nba_data_pipeline.py -v
.venv/bin/python -m mypy unified_nba_data_pipeline.py --strict
```

### Task 2: Advanced Ensemble System (Duration: 600 min)

**File**: `advanced_predictive_model.py` (Lines: 1-400)

**ACTION**: Implement ensemble ML system with XGBoost and voting

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class AdvancedPredictiveModel:
    """
    Advanced predictive model using ensemble methods for NBA games.

    Combines multiple ML models with weighted voting to improve
    prediction accuracy and provide confidence intervals.

    Attributes:
        models: Dictionary of trained models
        ensemble_weights: Weights for voting
        feature_columns: List of feature column names
    """

    def __init__(
        self,
        model_configs: Optional[Dict[str, Dict]] = None
    ) -> None:
        """Initialize the advanced predictive model."""

    def train_predictive_models(
        self,
        training_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Train multiple predictive models on NBA data."""

    def predict_game_outcome(
        self,
        game_features: pd.DataFrame,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """Predict game outcomes with confidence intervals."""
```

**PATTERN REFERENCE**: See `momentum_ml_trainer.py:100-200` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Model training
    model.fit(X_train, y_train)
except ValueError as e:
    logger.error(
        "Model training failed",
        extra={"model": model_name, "error": str(e)}
    )
    raise ModelTrainingError(f"Failed to train {model_name}") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need scikit-learn VotingClassifier patterns
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="scikit-learn")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="VotingClassifier ensemble methods",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_advanced_predictive_model.py::test_ensemble_prediction`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_advanced_predictive_model.py -v
.venv/bin/python -m mypy advanced_predictive_model.py --strict
```

### Task 3: SHAP Explainability Engine (Duration: 450 min)

**File**: `nba_explainability_engine.py` (Lines: 1-350)

**ACTION**: Implement SHAP-based model explainability system

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class NBAExplainabilityEngine:
    """
    SHAP-based explainability engine for NBA predictive models.

    Provides global and local explanations for model predictions
    using SHAP values and visualization tools.

    Attributes:
        explainer: SHAP explainer instance
        model: Trained predictive model
        feature_names: List of feature names
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str]
    ) -> None:
        """Initialize the explainability engine."""

    def calculate_shap_values(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate SHAP values for given data."""

    def generate_global_explanation(
        self,
        shap_values: np.ndarray,
        plot_type: str = "beeswarm"
    ) -> Dict[str, Any]:
        """Generate global feature importance explanations."""

    def explain_single_prediction(
        self,
        game_features: pd.Series,
        prediction: float
    ) -> Dict[str, Any]:
        """Explain a single game prediction with SHAP force plot."""
```

**PATTERN REFERENCE**: See existing ML trainers for pattern consistency

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # SHAP calculation
    shap_values = self.explainer.shap_values(data)
except ValueError as e:
    logger.error(
        "SHAP calculation failed",
        extra={"data_shape": data.shape, "error": str(e)}
    )
    raise ExplainabilityError("Failed to calculate SHAP values") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need SHAP library patterns
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="shap")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="TreeExplainer force plot visualization",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_nba_explainability_engine.py::test_shap_value_calculation`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_nba_explainability_engine.py -v
.venv/bin/python -m mypy nba_explainability_engine.py --strict
```

### Task 4: Predictive Analytics Dashboard (Duration: 480 min)

**File**: `predictive_analytics_dashboard.py` (Lines: 1-500)

**ACTION**: Create advanced Streamlit dashboard with ML insights

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class PredictiveAnalyticsDashboard:
    """
    Advanced Streamlit dashboard for NBA predictive analytics.

    Provides real-time predictions, model explanations, and
    performance monitoring through interactive visualizations.

    Attributes:
        pipeline: Unified data pipeline
        model: Trained predictive model
        explainability: SHAP explainability engine
    """

    def __init__(
        self,
        pipeline: UnifiedNBADataPipeline,
        model: AdvancedPredictiveModel,
        explainability: NBAExplainabilityEngine
    ) -> None:
        """Initialize the predictive analytics dashboard."""

    def render_dashboard(self) -> None:
        """Render the main dashboard interface."""

    def display_game_predictions(
        self,
        games_data: pd.DataFrame
    ) -> None:
        """Display game predictions with confidence intervals."""

    def show_feature_importance(
        self,
        shap_values: np.ndarray
    ) -> None:
        """Display SHAP feature importance visualizations."""
```

**PATTERN REFERENCE**: See `app_advanced.py:50-150` for Streamlit patterns

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Dashboard rendering
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    logger.error(
        "Dashboard rendering failed",
        extra={"component": component_name, "error": str(e)}
    )
    st.error("Failed to render dashboard component")
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need plotly/Streamlit visualization patterns
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="plotly")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="interactive charts confidence intervals",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_predictive_analytics_dashboard.py::test_dashboard_rendering`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_predictive_analytics_dashboard.py -v
.venv/bin/python -m mypy predictive_analytics_dashboard.py --strict
```

### Task 5: Auto Model Retrainer (Duration: 360 min)

**File**: `auto_model_retrainer.py` (Lines: 1-300)

**ACTION**: Implement automated model retraining system

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class AutoModelRetrainer:
    """
    Automated model retraining system for NBA predictions.

    Monitors model performance and triggers retraining
    when accuracy degrades below specified thresholds.

    Attributes:
        model: Current trained model
        performance_threshold: Minimum accuracy threshold
        retrain_interval: Days between retraining checks
    """

    def __init__(
        self,
        model: AdvancedPredictiveModel,
        performance_threshold: float = 0.75,
        retrain_interval: int = 7
    ) -> None:
        """Initialize the auto retrainer."""

    def check_retrain_needed(
        self,
        recent_predictions: pd.DataFrame,
        actual_results: pd.DataFrame
    ) -> bool:
        """Check if model retraining is needed."""

    def retrain_models(
        self,
        new_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Retrain models with new data."""

    def validate_retrained_models(
        self,
        validation_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate newly retrained models."""
```

**PATTERN REFERENCE**: See existing ML trainers for retraining patterns

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Model retraining
    retrained_model = self.model.train_predictive_models(new_data, target)
except Exception as e:
    logger.error(
        "Model retraining failed",
        extra={"data_size": len(new_data), "error": str(e)}
    )
    raise RetrainingError("Failed to retrain models") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need scikit-learn model persistence patterns
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="scikit-learn")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="joblib model persistence incremental learning",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_auto_model_retrainer.py::test_retrain_trigger`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_auto_model_retrainer.py -v
.venv/bin/python -m mypy auto_model_retrainer.py --strict
```

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

**Library**: scikit-learn 1.7.1
**Trust Score**: 8.5/10
**Context7 ID**: /scikit-learn/scikit-learn

**Key Pattern 1**: VotingClassifier Ensemble
```python
from sklearn.ensemble import VotingClassifier

# Soft voting with weights
ensemble = VotingClassifier(
    estimators=[
        ('xgboost', xgb_model),
        ('logistic', lr_model),
        ('random_forest', rf_model)
    ],
    voting='soft',
    weights=[2, 1.5, 1]
)
```
**When to use**: Combining multiple models for improved accuracy

**Key Pattern 2**: Model Persistence
```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')
# Load model
model = joblib.load('model.pkl')
```

**Library**: XGBoost latest
**Trust Score**: 8.9/10
**Context7 ID**: /dmlc/xgboost

**Key Pattern 1**: Feature Importance
```python
importance_weight = model.get_score(importance_type='weight')
importance_gain = model.get_score(importance_type='total_gain')
```
**When to use**: Understanding model feature contributions

**Library**: SHAP latest
**Trust Score**: 5.7/10
**Context7 ID**: /shap/shap

**Key Pattern 1**: TreeExplainer
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
```
**When to use**: Explaining tree-based model predictions

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** feature removal to "fix" problems
- ❌ **NO** workarounds instead of proper solutions
- ❌ **NO** simplifications that reduce functionality
- ❌ **NO** skipping error handling
- ❌ **NO** marking task complete with failing tests

**REQUIRED ACTIONS**:
- ✅ **YES** use Context7 for unknowns (tools provided above)
- ✅ **YES** maintain ALL existing functionality
- ✅ **YES** follow exact error handling pattern
- ✅ **YES** full docstrings + type hints EVERY function
- ✅ **YES** check acceptance criteria per micro-task

---

## ✅ QUALITY GATES (MANDATORY BEFORE COMPLETION)

### 1. Test Coverage
```bash
.venv/bin/python -m pytest tests/ -v \
    --cov=unified_nba_data_pipeline \
    --cov=advanced_predictive_model \
    --cov=nba_explainability_engine \
    --cov=predictive_analytics_dashboard \
    --cov=auto_model_retrainer \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
.venv/bin/python -m mypy unified_nba_data_pipeline.py advanced_predictive_model.py nba_explainability_engine.py predictive_analytics_dashboard.py auto_model_retrainer.py --strict

# REQUIREMENT: Zero errors
```

### 3. Performance Benchmark (if applicable)
```bash
.venv/bin/python -c "
from time import time
import pandas as pd
from unified_nba_data_pipeline import UnifiedNBADataPipeline

# Test pipeline performance
pipeline = UnifiedNBADataPipeline()
start = time()
data = pipeline.fetch_all_data(('2024-01-01', '2024-01-07'))
end = time()
print(f'Pipeline fetch time: {end-start:.2f}s')

# TARGET: < 5 seconds for 7 days of data
"
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat(ml): implement NBA predictive analytics system with ensemble methods

Implement comprehensive ML system with:
- Unified data pipeline with multi-source orchestration
- Advanced ensemble models with XGBoost, Logistic Regression, Random Forest
- SHAP-based explainability engine for model transparency
- Interactive Streamlit dashboard with real-time predictions
- Automated model retraining with performance monitoring

Technical Implementation:
- Used scikit-learn VotingClassifier for weighted ensemble voting
- Implemented SHAP TreeExplainer for feature importance analysis
- Created automated pipeline with data quality validation
- Added comprehensive error handling and logging
- Integrated with existing NBA data provider infrastructure

Quality Validation:
- ✅ Tests: 25 tests passing, 97% coverage
- ✅ Type safety: mypy --strict passed
- ✅ Performance: Pipeline fetch < 3s for 7 days data
- ✅ Integration: Compatible with existing data_provider_june2025.py

Task ID: nba-predictive-analytics-2024

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: Pipeline fetch < 5s for 7 days data
- **Code Review**: @code-reviewer validation passed

---

**READY TO START?**
1. Mark first TodoWrite task as "in_progress"
2. Search DevStream memory for context
3. Implement according to specification
4. Run tests + type check
5. Mark "completed" when all acceptance criteria met
6. Proceed to next micro-task

**REMEMBER**: Execute, don't explore. Follow patterns, don't invent. Complete tasks, don't quit early. 🚀