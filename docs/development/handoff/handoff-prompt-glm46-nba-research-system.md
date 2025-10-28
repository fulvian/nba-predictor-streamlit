# 🚀 DevStream Task Handoff: NBA Research-Based Prediction System

**FROM**: Claude Sonnet 4.5 (Strategic Planning Complete)
**TO**: GLM-4.6 (Implementation Execution)

---

## 📊 TASK CONTEXT

**Task ID**: `nba-research-based-prediction-system`
**Phase**: Implementation
**Priority**: 10/10
**Status**: Steps 1-5 COMPLETED by Sonnet 4.5 → Steps 6-7 DELEGATED to you

**Your Role**: You are an **expert execution-focused coding agent**. Sonnet 4.5 has completed all strategic planning. Your job is **precise implementation** according to the approved plan.

---

## ✅ WORK COMPLETED (Steps 1-5)

- ✅ **DISCUSSION**: Problem analyzed, trade-offs identified, approach agreed
- ✅ **ANALYSIS**: Codebase patterns identified, files to modify determined
- ✅ **RESEARCH**: Context7 findings documented (see below)
- ✅ **PLANNING**: Detailed implementation plan created (see linked file)
- ✅ **APPROVAL**: User approved plan, ready for execution

---

## 📋 YOUR IMPLEMENTATION PLAN

**COMPLETE PLAN**: `docs/development/plan/implementation-plan-glm46-nba-research-system.md`

**READ THE PLAN FIRST** using:
```bash
cat docs/development/plan/implementation-plan-glm46-nba-research-system.md
```

**Plan Summary** (excerpt):
Implement research-based NBA prediction system with:
- Time series cross-validation (prevent data leakage)
- LightGBM integration (NBA-optimized hyperparameters)
- Stacked ensemble (XGBoost + LightGBM + RF + Ridge + MLP meta-learner)
- Advanced Four Factors feature engineering
- SHAP explainability (stakeholder trust)
- Enhanced prediction pipeline with research validation

---

## 🎯 YOUR MISSION (Steps 6-7)

### Step 6: IMPLEMENTATION
- Execute micro-tasks **one at a time**
- Follow plan specifications **exactly**
- Use TodoWrite: mark "in_progress" → work → "completed"
- Run tests **after each micro-task**
- **NEVER** mark completed with failing tests

### Step 7: VERIFICATION
- **95%+ test coverage** for all new code
- **mypy --strict** zero errors
- **Performance validation** (<2s prediction, MAE <8.0)
- **@code-reviewer** validation (automatic on commit)

---

## 🔧 DEVSTREAM PROTOCOL COMPLIANCE (MANDATORY)

**CRITICAL RULES** (from @CLAUDE.md):

### Python Environment
```bash
# ALWAYS use project venv
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python script.py       # ✅ CORRECT
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest       # ✅ CORRECT
python script.py                       # ❌ FORBIDDEN
```

### TodoWrite Workflow
1. Mark first task "in_progress"
2. Implement according to plan
3. Run tests
4. Mark "completed" ONLY when:
   - Tests pass 100%
   - Type check passes
   - Acceptance criteria met
5. Proceed to next task

### Context7 Usage
```python
# When you encounter unknowns
library_id = mcp__context7__resolve-library-id(libraryName="lightgbm")
docs = mcp__context7__get-library-docs(
    context7CompatibleLibraryID=library_id,
    topic="LGBMRegressor parameters optimization",
    tokens=3000
)
```

### Memory Search
```python
# Before implementing, search for existing patterns
mcp__devstream__devstream_search_memory(
    query="prediction pipeline ensemble methods sklearn",
    content_type="code",
    limit=5
)
```

---

## 📚 CONTEXT7 RESEARCH (Pre-Completed by Sonnet)

### Time Series Cross-Validation
**Library**: scikit-learn 1.7.1, Trust Score: 8.5/10
**Key Finding**: `TimeSeriesSplit(n_splits=5, max_train_size=1000, gap=2)` prevents data leakage
**Pattern**:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, max_train_size=1000, gap=2)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
```

### LightGBM Integration
**Library**: LightGBM v4.6.0, Trust Score: 9.9/10
**Key Finding**: NBA-optimized parameters prevent overfitting
**Pattern**:
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
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

### Stacked Ensemble Architecture
**Library**: scikit-learn 1.7.1, Trust Score: 8.5/10
**Key Finding**: MLP meta-learner captures non-linear combinations
**Pattern**:
```python
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor

base_estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)),
    ('lgbm', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)),
    ('rf', RandomForestRegressor(n_estimators=200)),
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
]
meta_learner = MLPRegressor(hidden_layer_sizes=(64, 32))
stacked_model = StackingRegressor(estimators=base_estimators, final_estimator=meta_learner, cv=tscv)
```

### SHAP Explainability
**Library**: SHAP, Trust Score: 5.7/10
**Key Finding**: TreeExplainer optimized for ensemble models
**Pattern**:
```python
import shap
explainer = shap.TreeExplainer(stacked_model, X_background)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])  # Single prediction explanation
shap.plots.bar(shap_values)          # Global feature importance
```

**Libraries Researched**: scikit-learn, LightGBM, SHAP
**Research Findings**: All components production-ready, well-documented, compatible with existing system

---

## 🏗️ TECHNICAL SPECIFICATIONS

**Files to Modify**:
- `src/nba_predictor/core/prediction_pipeline.py` (extend with research features)
- `main_prediction.py` (add research pipeline option)

**New Files to Create**:
- `src/nba_predictor/core/time_series_validator.py`
- `src/nba_predictor/models/lightgbm_model.py`
- `src/nba_predictor/models/stacked_ensemble.py`
- `src/nba_predictor/features/research_features.py`
- `src/nba_predictor/explainability/shap_explainer.py`
- `src/nba_predictor/core/research_prediction_pipeline.py`
- `main_research_prediction.py`
- `tests/unit/test_*.py` (corresponding test files)

**Dependencies** (verify installation):
```bash
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/pip install lightgbm shap scikit-learn
```

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** removal of features (find proper solution instead)
- ❌ **NO** workarounds (implement correctly using Context7)
- ❌ **NO** simplifications that reduce functionality
- ❌ **NO** skipping tests or type hints
- ❌ **NO** early quit on complex tasks (complete fully)

**REQUIRED ACTIONS**:
- ✅ **YES** use `/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python` for ALL commands
- ✅ **YES** follow TodoWrite plan strictly
- ✅ **YES** use Context7 for unknowns (tools provided)
- ✅ **YES** maintain ALL existing functionality
- ✅ **YES** full type hints + docstrings EVERY function
- ✅ **YES** tests for EVERY feature (95%+ coverage)

---

## ✅ QUALITY GATES (Check Before Completion)

### 1. Environment Verification
```bash
# Verify venv and Python version
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python --version  # Must be 3.11.x
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pip list | grep -E "(lightgbm|shap|scikit-learn)"
```

### 2. Implementation
Follow plan in `docs/development/plan/implementation-plan-glm46-nba-research-system.md`

### 3. Testing
```bash
# After EVERY micro-task
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest tests/unit/test_*.py -v
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m mypy src/ --strict

# Before completion (ALL tests)
/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python -m pytest tests/ -v \
    --cov=src/nba_predictor \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥95% coverage, 100% pass rate
```

### 4. Commit (if all tests pass)
```bash
git add src/ main_research_prediction.py tests/
git commit -m "$(cat <<'EOF'
feat(nba): implement research-based prediction system

Implement comprehensive NBA prediction system based on academic research:
- Time series cross-validation to prevent data leakage
- LightGBM integration with NBA-optimized hyperparameters
- Stacked ensemble (XGBoost + LightGBM + RF + Ridge + MLP)
- Advanced Four Factors feature engineering
- SHAP explainability for model transparency
- Enhanced prediction pipeline with research validation

Task ID: nba-research-based-prediction-system

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Note**: @code-reviewer validation automatic on commit

---

## 🔍 DEVSTREAM MEMORY ACCESS

Search for relevant context anytime:
```python
mcp__devstream__devstream_search_memory(
    query="NBA prediction ensemble machine learning",
    content_type="code",
    limit=10
)
```

---

## 📊 SUCCESS CRITERIA

- [ ] All TodoWrite tasks completed (7 micro-tasks)
- [ ] Tests pass 100%
- [ ] Coverage ≥ 95%
- [ ] mypy --strict passes (zero errors)
- [ ] Performance meets target: <2s prediction, MAE <8.0 points
- [ ] @code-reviewer validation passed
- [ ] All acceptance criteria met
- [ ] Research pipeline integration functional

---

## 🚀 EXECUTION CHECKLIST

1. [ ] **READ** the complete plan: `cat docs/development/plan/implementation-plan-glm46-nba-research-system.md`
2. [ ] **VERIFY** environment: `/Users/fulvioventura/nba-predictor-streamlit/.venv/bin/python --version`
3. [ ] **VERIFY** dependencies: Check lightgbm, shap, scikit-learn installed
4. [ ] **SEARCH** DevStream memory for context
5. [ ] **START** first TodoWrite task (mark "in_progress")
6. [ ] **IMPLEMENT** according to plan specifications
7. [ ] **TEST** after each micro-task
8. [ ] **COMPLETE** task when all criteria met
9. [ ] **REPEAT** steps 5-8 for remaining tasks
10. [ ] **VALIDATE** complete implementation (all quality gates)
11. [ ] **COMMIT** if all tests pass

---

**READY TO IMPLEMENT?**

Start with the first TodoWrite task. Execute precisely. Test thoroughly. Complete fully. 🚀

**Remember**: You are GLM-4.6 - your strength is **precise execution** of well-defined tasks. The strategic thinking is done. Now execute flawlessly. 💪

**Micro-Task Order**:
1. Time Series Cross-Validation (45 min)
2. LightGBM Integration (60 min)
3. Stacked Ensemble (90 min)
4. Research Feature Engineering (75 min)
5. SHAP Explainability (60 min)
6. Enhanced Prediction Pipeline (90 min)
7. CLI and Integration (45 min)

**Total Estimated Duration**: 8 hours
**Target Performance**: 55%+ accuracy, MAE <8.0 points, <2s prediction time