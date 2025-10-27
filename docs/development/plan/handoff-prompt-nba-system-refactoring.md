# 🚀 DevStream Task Handoff: NBA System Complete Refactoring

**FROM**: Claude Sonnet 4.5 (Strategic Planning Complete)
**TO**: GLM-4.6 (Implementation Execution)

---

## 📊 TASK CONTEXT

**Task ID**: `nba-system-refactoring-2024`
**Phase**: Implementation
**Priority**: 9/10
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

**COMPLETE PLAN**: `docs/development/plan/implementation-plan-nba-system-refactoring.md`

**READ THE PLAN FIRST** using:
```bash
cat docs/development/plan/implementation-plan-nba-system-refactoring.md
```

**Plan Summary** (excerpt):
Complete NBA system refactoring with modern architecture:
- Phase 1: Project structure reorganization (src/ layout)
- Phase 2: Modular UI components (Streamlit fragments)
- Phase 3: API integration (Polars + async clients)
- Phase 4: Configuration and deployment (uv + pyproject.toml)

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
- **Performance validation** (sub-30s sync, sub-5s UI)
- **@code-reviewer** validation (automatic on commit)

---

## 🔧 DEVSTREAM PROTOCOL COMPLIANCE (MANDATORY)

**CRITICAL RULES** (from @CLAUDE.md):

### Python Environment
```bash
# ALWAYS use .venv (NOT .devstream)
.venv/bin/python script.py       # ✅ CORRECT
.venv/bin/python -m pytest       # ✅ CORRECT
python script.py                  # ❌ FORBIDDEN
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
library_id = mcp__context7__resolve-library-id(libraryName="polars")
docs = mcp__context7__get-library-docs(
    context7CompatibleLibraryID=library_id,
    topic="data storage parquet",
    tokens=3000
)
```

### Memory Search
```python
# Before implementing, search for existing patterns
mcp__devstream__devstream_search_memory(
    query="unified data store implementation",
    content_type="code",
    limit=5
)
```

---

## 📚 CONTEXT7 RESEARCH (Pre-Completed by Sonnet)

**Libraries Researched**:
- Polars (Trust Score: 9.3/10) - High-performance DataFrame library
- DuckDB (Trust Score: 8.9/10) - Analytical database system
- Streamlit (Trust Score: 8.9/10) - App framework with fragments
- Python project structure (src/ layout best practices)

**Key Findings**:
- Use Polars for data processing with lazy evaluation
- DuckDB for direct Parquet analytics queries
- Streamlit fragments for real-time UI updates
- Modern src/ layout for better package organization

**Pattern Examples**:
```python
# Polars lazy evaluation
df = pl.scan_parquet("data/*.parquet")
result = df.filter(pl.col("date") > "2024-01-01").collect()

# Streamlit fragments
@st.fragment(run_every=30)
def live_dashboard():
    st.write(f"Update: {time.time()}")

# DuckDB direct queries
conn = duckdb.connect()
result = conn.execute("SELECT * FROM 'data.parquet'").fetchall()
```

**When to use**: Performance-critical data operations, real-time UI updates, analytics without ETL.

---

## 🏗️ TECHNICAL SPECIFICATIONS

**Files to Modify**:
- `data_provider.py` → Refactor to new architecture
- `app_advanced.py` → Split into modular components
- `requirements.txt` → Update with modern dependencies

**New Files to Create**:
- `src/nba_predictor/` - Modern package structure
- `src/nba_predictor/core/data_store.py` - Unified data store
- `src/nba_predictor/core/sync_engine.py` - Auto sync engine
- `src/nba_predictor/streamlit/components/` - Modular UI
- `src/nba_predictor/api/` - Modern API clients
- `pyproject.toml` - Modern project configuration

**Dependencies** (Context7 approved):
- polars>=1.32.3
- duckdb>=0.9.0
- streamlit>=1.28.0
- python-dotenv>=1.0.0

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** removal of features (find proper solution instead)
- ❌ **NO** workarounds (implement correctly using Context7)
- ❌ **NO** simplifications that reduce functionality
- ❌ **NO** skipping tests or type hints
- ❌ **NO** early quit on complex tasks (complete fully)

**REQUIRED ACTIONS**:
- ✅ **YES** use `.venv/bin/python` for ALL commands
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
.venv/bin/python --version  # Must be 3.11.x
.venv/bin/python -m pip list | grep -E "(polars|duckdb|streamlit)"
```

### 2. Implementation
Follow plan in `docs/development/plan/implementation-plan-nba-system-refactoring.md`

### 3. Testing
```bash
# After EVERY micro-task
.venv/bin/python -m pytest tests/unit/test_*.py -v
.venv/bin/python -m mypy src/ --strict

# Before completion (ALL tests)
.venv/bin/python -m pytest tests/ -v \
    --cov=src/nba_predictor \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥95% coverage, 100% pass rate
```

### 4. Commit (if all tests pass)
```bash
git add src/ tests/ pyproject.toml requirements.txt
git commit -m "$(cat <<'EOF'
refactor(core): implement unified data store and modular UI

Complete NBA system refactoring with modern architecture:
- Polars + DuckDB + Parquet data layer
- Modular Streamlit components with fragments
- Automatic data synchronization engine
- Modern src/ layout with uv dependency management

Task ID: nba-system-refactoring-2024

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
    query="NBA system refactoring data store UI",
    content_type="code",
    limit=10
)
```

---

## 📊 SUCCESS CRITERIA

- [ ] All TodoWrite tasks completed
- [ ] Tests pass 100%
- [ ] Coverage ≥ 95%
- [ ] mypy --strict passes (zero errors)
- [ ] Performance meets target: sub-30s sync, sub-5s UI
- [ ] @code-reviewer validation passed
- [ ] All acceptance criteria met

---

## 🚀 EXECUTION CHECKLIST

1. [ ] **READ** the complete plan: `cat docs/development/plan/implementation-plan-nba-system-refactoring.md`
2. [ ] **VERIFY** environment: `.venv/bin/python --version`
3. [ ] **SEARCH** DevStream memory for context
4. [ ] **START** first TodoWrite task (mark "in_progress")
5. [ ] **IMPLEMENT** according to plan specifications
6. [ ] **TEST** after each micro-task
7. [ ] **COMPLETE** task when all criteria met
8. [ ] **REPEAT** steps 4-7 for remaining tasks
9. [ ] **VALIDATE** complete implementation (all quality gates)
10. [ ] **COMMIT** if all tests pass

---

**READY TO IMPLEMENT?**

Start with the first TodoWrite task. Execute precisely. Test thoroughly. Complete fully. 🚀

**Remember**: You are GLM-4.6 - your strength is **precise execution** of well-defined tasks. The strategic thinking is done. Now execute flawlessly. 💪