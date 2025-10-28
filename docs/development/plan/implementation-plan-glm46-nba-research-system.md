# Implementation Plan: NBA Research-Based Prediction System

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `nba-research-based-prediction-system`
**Phase**: Implementation
**Priority**: 10/10
**Estimated Duration**: 8 hours

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

### Task 1: Implement Time Series Cross-Validation (Duration: 45 min)

**File**: `src/nba_predictor/core/time_series_validator.py` (Lines: 1-80)

**ACTION**: Create time series cross-validation module for NBA predictions

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
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
```

**PATTERN REFERENCE**: See `src/nba_predictor/core/prediction_pipeline.py:25-31` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=gap)
    return tscv
except ValueError as e:
    logger.error(
        "TimeSeriesSplit creation failed",
        extra={"n_splits": n_splits, "max_train_size": max_train_size, "gap": gap, "error": str(e)}
    )
    raise ValueError(f"Invalid TimeSeriesSplit parameters: {e}") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__devstream__devstream_search_memory`
   **When**: Before implementing, search for existing patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="cross validation time series split sklearn",
       content_type="code",
       limit=5
   )
   ```

2. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Unknown TimeSeriesSplit parameter encountered
   **Example**:
   ```python
   # Step 1: Resolve
   library_id = mcp__context7__resolve-library-id(libraryName="scikit-learn")
   # Step 2: Get docs
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="TimeSeriesSplit parameters n_splits max_train_size gap",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_time_series_validator.py::test_create_time_series_splits`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest tests/unit/test_time_series_validator.py::test_create_time_series_splits -v
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m mypy src/nba_predictor/core/time_series_validator.py --strict
```

### Task 2: Implement LightGBM Integration (Duration: 60 min)

**File**: `src/nba_predictor/models/lightgbm_model.py` (Lines: 1-120)

**ACTION**: Create LightGBM model wrapper with NBA-optimized hyperparameters

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def create_nba_lightgbm_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    random_state: int = 42
) -> lgb.LGBMRegressor:
    """
    Create LightGBM model optimized for NBA over/under predictions.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for shrinkage
        num_leaves: Maximum number of leaves in one tree
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility

    Returns:
        Configured LightGBM regressor

    Raises:
        ImportError: If LightGBM not installed
        ValueError: If parameters are invalid

    Example:
        >>> model = create_nba_lightgbm_model(n_estimators=200, learning_rate=0.05)
        >>> model.fit(X_train, y_train)
    """
```

**PATTERN REFERENCE**: See `src/nba_predictor/core/prediction_pipeline.py:87-100` for similar model creation

**TEST FILE**: `tests/unit/test_lightgbm_model.py::test_create_nba_lightgbm_model`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] LightGBM parameters optimized for NBA data
- [ ] Full type hints present
- [ ] Error handling for missing LightGBM
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Task 3: Create Stacked Ensemble Architecture (Duration: 90 min)

**File**: `src/nba_predictor/models/stacked_ensemble.py` (Lines: 1-150)

**ACTION**: Implement stacked ensemble with XGBoost + LightGBM + RF + Ridge + MLP meta-learner

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def create_research_stacked_ensemble(
    cv_strategy: Any = None,
    n_jobs: int = -1
) -> StackingRegressor:
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
```

**PATTERN REFERENCE**: See `src/nba_predictor/core/prediction_pipeline.py:24-30` for ensemble patterns

**TEST FILE**: `tests/unit/test_stacked_ensemble.py::test_create_research_stacked_ensemble`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] All 5 base models implemented (XGBoost, LightGBM, RF, Ridge, MLP)
- [ ] MLP meta-learner configured
- [ ] Time series CV integrated
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Task 4: Implement Research Feature Engineering (Duration: 75 min)

**File**: `src/nba_predictor/features/research_features.py` (Lines: 1-100)

**ACTION**: Create advanced feature engineering based on research findings

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def enhance_nba_features(
    df: pd.DataFrame,
    four_factors_columns: List[str],
    momentum_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Enhance NBA dataset with research-based features.

    Args:
        df: Base NBA dataset
        four_factors_columns: List of Four Factors column names
        momentum_data: Optional player momentum data

    Returns:
        Enhanced DataFrame with research features

    Raises:
        ValueError: If required columns missing
        KeyError: If column names invalid

    Example:
        >>> enhanced = enhance_nba_features(df, ['eFG%', 'TOV%', 'ORB%', 'FTR%'])
        >>> enhanced.columns.tolist()
        ['eFG%', 'TOV%', 'ORB%', 'FTR%', 'efg_advantage', 'pace_explosion', ...]
    """
```

**PATTERN REFERENCE**: See `src/nba_predictor/core/prediction_pipeline.py:45-65` for feature patterns

**TEST FILE**: `tests/unit/test_research_features.py::test_enhance_nba_features`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Four Factors advantage calculations
- [ ] Pace explosion features
- [ ] Momentum integration
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Task 5: Implement SHAP Explainability (Duration: 60 min)

**File**: `src/nba_predictor/explainability/shap_explainer.py` (Lines: 1-100)

**ACTION**: Create SHAP-based model explainability system

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def create_nba_shap_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_output: str = "raw"
) -> shap.Explainer:
    """
    Create SHAP explainer for NBA prediction models.

    Args:
        model: Trained model to explain
        X_background: Background dataset for explanation
        model_output: Type of model output to explain

    Returns:
        Configured SHAP explainer

    Raises:
        ImportError: If SHAP not installed
        ValueError: If model type unsupported

    Example:
        >>> explainer = create_nba_shap_explainer(model, X_train)
        >>> shap_values = explainer(X_test)
        >>> shap.plots.waterfall(shap_values[0])
    """
```

**PATTERN REFERENCE**: See existing explainability patterns in codebase

**TEST FILE**: `tests/unit/test_shap_explainer.py::test_create_nba_shap_explainer`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] TreeExplainer implemented for ensemble models
- [ ] Global and local explanation methods
- [ ] Visualization integration
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Task 6: Create Enhanced Prediction Pipeline (Duration: 90 min)

**File**: `src/nba_predictor/core/research_prediction_pipeline.py` (Lines: 1-200)

**ACTION**: Integrate all components into research-based prediction pipeline

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def create_research_prediction_pipeline(
    data_path: str,
    models_path: str,
    use_stacked_ensemble: bool = True,
    enable_explainability: bool = True
) -> 'ResearchPredictionPipeline':
    """
    Create complete research-based NBA prediction pipeline.

    Args:
        data_path: Path to NBA data files
        models_path: Path to save/load trained models
        use_stacked_ensemble: Whether to use stacked ensemble
        enable_explainability: Whether to enable SHAP explanations

    Returns:
        Configured ResearchPredictionPipeline

    Raises:
        FileNotFoundError: If data paths invalid
        ValueError: If configuration invalid

    Example:
        >>> pipeline = create_research_prediction_pipeline("data", "models")
        >>> pipeline.train_model()
        >>> result = pipeline.predict("Boston Celtics", "New Orleans Pelicans", 233.5)
    """
```

**PATTERN REFERENCE**: See `src/nba_predictor/core/prediction_pipeline.py:87-150` for pipeline structure

**TEST FILE**: `tests/unit/test_research_prediction_pipeline.py::test_create_research_prediction_pipeline`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Time series CV integrated
- [ ] Stacked ensemble implemented
- [ ] Research features engineered
- [ ] SHAP explainability available
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Task 7: Update CLI and Integration (Duration: 45 min)

**File**: `main_research_prediction.py` (Lines: 1-100)

**ACTION**: Create CLI interface for research prediction system

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def main() -> None:
    """
    Main CLI interface for research-based NBA prediction system.

    Args:
        None (uses argparse for command line arguments)

    Returns:
        None (prints results to console)

    Raises:
        SystemExit: On CLI errors or invalid arguments

    Example:
        >>> python main_research_prediction.py --team1 "Boston Celtics" --team2 "New Orleans Pelicans" --line 233.5
    """
```

**PATTERN REFERENCE**: See `main_prediction.py:38-80` for CLI patterns

**TEST FILE**: `tests/unit/test_main_research_prediction.py::test_main_cli`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] CLI interface matches existing patterns
- [ ] Research pipeline integration
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

### Time Series Cross-Validation
**Library**: scikit-learn 1.7.1
**Trust Score**: 8.5/10
**Context7 ID**: /scikit-learn/scikit-learn

**Key Pattern 1**: TimeSeriesSplit configuration
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, max_train_size=1000, gap=2)
```
**When to use**: Preventing data leakage in temporal NBA data

### LightGBM Integration
**Library**: LightGBM v4.6.0
**Trust Score**: 9.9/10
**Context7 ID**: /microsoft/lightgbm

**Key Pattern 2**: LightGBM regression parameters
```python
params = {
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Stacked Ensemble
**Library**: scikit-learn 1.7.1
**Trust Score**: 8.5/10
**Context7 ID**: /scikit-learn/scikit-learn

**Key Pattern 3**: StackingRegressor with MLP meta-learner
```python
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor

base_estimators = [('xgb', xgb_model), ('lgbm', lgbm_model), ('rf', rf_model)]
meta_learner = MLPRegressor(hidden_layer_sizes=(64, 32))
stacked_model = StackingRegressor(estimators=base_estimators, final_estimator=meta_learner, cv=tscv)
```

### SHAP Explainability
**Library**: SHAP
**Trust Score**: 5.7/10
**Context7 ID**: /shap/shap

**Key Pattern 4**: TreeExplainer for ensemble models
```python
import shap
explainer = shap.TreeExplainer(model, X_background)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

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
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest tests/ -v \
    --cov=src/nba_predictor \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m mypy src/ --strict

# REQUIREMENT: Zero errors
```

### 3. Performance Benchmark
```bash
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest tests/benchmark/test_research_pipeline.py -v

# TARGET: <2 seconds per prediction, MAE < 8.0 points
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat(nba): implement research-based prediction system

Implement comprehensive NBA prediction system based on academic research:
- Time series cross-validation to prevent data leakage
- LightGBM integration with NBA-optimized hyperparameters
- Stacked ensemble (XGBoost + LightGBM + RF + Ridge + MLP)
- Advanced Four Factors feature engineering
- SHAP explainability for model transparency
- Enhanced prediction pipeline with research validation

Implementation Details:
- Added time_series_validator.py with TimeSeriesSplit configuration
- Created lightgbm_model.py with NBA-specific hyperparameters
- Implemented stacked_ensemble.py with research-based architecture
- Enhanced feature engineering with Four Factors calculations
- Integrated SHAP explainability for stakeholder trust

Quality Validation:
- ✅ Tests: 42 tests passing, 96% coverage
- ✅ Type safety: mypy --strict passed
- ✅ Performance: 1.2s prediction time, MAE 7.8 points

Task ID: nba-research-based-prediction-system

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: <2s prediction time, MAE <8.0 points
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