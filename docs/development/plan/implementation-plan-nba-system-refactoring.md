# Implementation Plan: NBA System Complete Refactoring

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `nba-system-refactoring-2024`
**Phase**: Implementation
**Priority**: 9/10
**Estimated Duration**: 24 hours

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

### Phase 1: Project Structure Reorganization (Duration: 120 min)

**Task 1.1**: Create src/ layout structure (Duration: 30 min)
**File**: `src/nba_predictor/__init__.py` (Lines: 1-20)

**ACTION**: Create modern Python package structure with src/ layout

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def __init__() -> None:
    """NBA Predictor package initialization."""
```

**PATTERN REFERENCE**: See `/johnthagen/python-blueprint:src layout` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    result = operation()
except SpecificException as e:
    logger.error(
        "Operation failed",
        extra={"context": value, "error": str(e)}
    )
    raise CustomException("User-friendly message") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__devstream__devstream_search_memory`
   **When**: Before implementing, search for existing patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="project structure python src layout",
       content_type="code",
       limit=5
   )
   ```

2. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Unknown library/pattern encountered
   **Example**:
   ```python
   # Step 1: Resolve
   library_id = mcp__context7__resolve-library-id(libraryName="polars")
   # Step 2: Get docs
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="data storage parquet",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_package_structure.py::test_src_layout`

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
.devstream/bin/python -m pytest tests/unit/test_package_structure.py::test_src_layout -v
.devstream/bin/python -m mypy src/nba_predictor/__init__.py --strict
```

**Task 1.2**: Implement unified data store core (Duration: 45 min)
**File**: `src/nba_predictor/core/data_store.py` (Lines: 1-200)

**ACTION**: Create unified data store with Polars + DuckDB + Parquet integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class UnifiedDataStore:
    """Unified data store for NBA system with Polars + DuckDB + Parquet."""

    def __init__(
        self,
        base_path: str,
        duckdb_path: Optional[str] = None,
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize the unified data store.

        Args:
            base_path: Base directory for data storage
            duckdb_path: Path to DuckDB database file
            cache_enabled: Enable caching for performance

        Returns:
            None

        Raises:
            FileNotFoundError: If base_path doesn't exist
            DatabaseError: If DuckDB initialization fails

        Example:
            >>> store = UnifiedDataStore("/data", cache_enabled=True)
            >>> store.initialize()
        """
```

**PATTERN REFERENCE**: See `/pola-rs/polars:parquet operations` for similar implementation

**TEST FILE**: `tests/unit/test_data_store.py::test_unified_data_store_init`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**Task 1.3**: Implement automatic sync engine (Duration: 45 min)
**File**: `src/nba_predictor/core/sync_engine.py` (Lines: 1-250)

**ACTION**: Create automatic data synchronization engine for startup and continuous updates

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class AutomaticSyncEngine:
    """Automatic data synchronization engine for NBA system."""

    async def sync_all_data(
        self,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize all NBA data automatically.

        Args:
            force_refresh: Force refresh even if data is fresh
            progress_callback: Callback for progress updates

        Returns:
            Dict with sync results and statistics

        Raises:
            SyncError: If synchronization fails
            APIError: If API calls fail

        Example:
            >>> engine = AutomaticSyncEngine(data_store)
            >>> result = await engine.sync_all_data()
            >>> print(f"Synced {result['games_count']} games")
        """
```

**PATTERN REFERENCE**: See `/streamlit/streamlit:fragments` for similar implementation

**TEST FILE**: `tests/unit/test_sync_engine.py::test_automatic_sync`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Phase 2: Modular UI Components (Duration: 180 min)

**Task 2.1**: Create data sync UI component (Duration: 60 min)
**File**: `src/nba_predictor/streamlit/components/sync_dashboard.py` (Lines: 1-150)

**ACTION**: Create Streamlit component for data synchronization visualization

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
@st.fragment(run_every=30)
def render_sync_dashboard(
    sync_engine: AutomaticSyncEngine,
    data_store: UnifiedDataStore
) -> None:
    """
    Render data synchronization dashboard with real-time updates.

    Args:
        sync_engine: Automatic sync engine instance
        data_store: Unified data store instance

    Returns:
        None (renders to Streamlit)

    Raises:
        StreamlitError: If rendering fails

    Example:
        >>> render_sync_dashboard(engine, store)
        # Dashboard appears in Streamlit app
    """
```

**PATTERN REFERENCE**: See `/streamlit/streamlit:fragment patterns` for similar implementation

**TEST FILE**: `tests/unit/test_sync_dashboard.py::test_render_sync_dashboard`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**Task 2.2**: Create analytics dashboard component (Duration: 60 min)
**File**: `src/nba_predictor/streamlit/components/analytics_dashboard.py` (Lines: 1-200)

**ACTION**: Create analytics dashboard with DuckDB integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def render_analytics_dashboard(
    data_store: UnifiedDataStore,
    date_range: Optional[Tuple[date, date]] = None
) -> None:
    """
    Render analytics dashboard with interactive charts and metrics.

    Args:
        data_store: Unified data store instance
        date_range: Optional date range for filtering

    Returns:
        None (renders to Streamlit)

    Raises:
        DatabaseError: If DuckDB queries fail
        StreamlitError: If rendering fails

    Example:
        >>> render_analytics_dashboard(store, (date(2024,1,1), date(2024,12,31)))
        # Analytics dashboard appears
    """
```

**PATTERN REFERENCE**: See `/duckdb/duckdb:parquet queries` for similar implementation

**TEST FILE**: `tests/unit/test_analytics_dashboard.py::test_render_analytics`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**Task 2.3**: Create main app navigation (Duration: 60 min)
**File**: `src/nba_predictor/streamlit/app.py` (Lines: 1-100)

**ACTION**: Create main Streamlit app with modular navigation

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def create_main_app() -> None:
    """
    Create main Streamlit application with modular navigation.

    Returns:
        None (runs Streamlit app)

    Raises:
        ImportError: If required components are missing
        StreamlitError: If app initialization fails

    Example:
        >>> create_main_app()
        # Streamlit app starts running
    """
```

**PATTERN REFERENCE**: See `/streamlit/streamlit:navigation patterns` for similar implementation

**TEST FILE**: `tests/unit/test_main_app.py::test_create_main_app`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Phase 3: API Integration (Duration: 120 min)

**Task 3.1**: Refactor NBA API client (Duration: 60 min)
**File**: `src/nba_predictor/api/nba_client.py` (Lines: 1-200)

**ACTION**: Refactor existing NBA API client for unified data store integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class ModernNBAAPIClient:
    """Modern NBA API client with unified data store integration."""

    async def fetch_all_games(
        self,
        start_date: date,
        end_date: date,
        include_stats: bool = True
    ) -> pl.DataFrame:
        """
        Fetch all NBA games in date range with optional statistics.

        Args:
            start_date: Start date for games
            end_date: End date for games
            include_stats: Include detailed statistics

        Returns:
            Polars DataFrame with games data

        Raises:
            APIError: If API calls fail
            ValidationError: If response format is invalid

        Example:
            >>> client = ModernNBAAPIClient()
            >>> games = await client.fetch_all_games(
            ...     date(2024,1,1), date(2024,1,31)
            ... )
            >>> print(f"Fetched {len(games)} games")
        """
```

**PATTERN REFERENCE**: See existing `data_provider.py:1-100` for similar implementation

**TEST FILE**: `tests/unit/test_nba_client.py::test_fetch_all_games`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**Task 3.2**: Refactor odds API client (Duration: 60 min)
**File**: `src/nba_predictor/api/odds_client.py` (Lines: 1-200)

**ACTION**: Refactor odds API client for unified data store integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class ModernOddsAPIClient:
    """Modern odds API client with unified data store integration."""

    async def fetch_odds_for_games(
        self,
        game_ids: List[str],
        bookmakers: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Fetch betting odds for specified games.

        Args:
            game_ids: List of game IDs to fetch odds for
            bookmakers: Optional list of bookmakers to filter

        Returns:
            Polars DataFrame with odds data

        Raises:
            APIError: If API calls fail
            ValidationError: If response format is invalid

        Example:
            >>> client = ModernOddsAPIClient()
            >>> odds = await client.fetch_odds_for_games(["0012400001"])
            >>> print(f"Fetched odds for {len(odds)} games")
        """
```

**PATTERN REFERENCE**: See existing `data_provider.py:100-200` for similar implementation

**TEST FILE**: `tests/unit/test_odds_client.py::test_fetch_odds`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

### Phase 4: Configuration and Deployment (Duration: 60 min)

**Task 4.1**: Create pyproject.toml (Duration: 30 min)
**File**: `pyproject.toml` (Lines: 1-50)

**ACTION**: Create modern Python project configuration

**PATTERN REFERENCE**: See `/johnthagen/python-blueprint:pyproject.toml` for similar implementation

**TEST FILE**: `tests/unit/test_project_config.py::test_dependencies`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Configuration follows modern standards
- [ ] All dependencies correctly specified
- [ ] Development dependencies included
- [ ] mypy --strict passes (zero errors)

**Task 4.2**: Update requirements and lock dependencies (Duration: 30 min)
**File**: `requirements.txt` (Lines: 1-30)

**ACTION**: Update requirements and create uv.lock

**PATTERN REFERENCE**: See `/johnthagen/python-blueprint:uv workflow` for similar implementation

**TEST FILE**: `tests/unit/test_dependencies.py::test_all_imports`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] All dependencies import correctly
- [ ] No version conflicts
- [ ] uv.lock generated successfully
- [ ] Tests pass with new dependencies

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

**Library**: Polars 1.32.3
**Trust Score**: 9.3/10
**Context7 ID**: /pola-rs/polars

**Key Pattern 1**: Lazy evaluation with predicate pushdown
```python
import polars as pl

# Scan Parquet files with lazy evaluation
df = pl.scan_parquet("s3://bucket/nba_data/*.parquet")
filtered = df.filter(pl.col("date") > "2024-01-01")
result = filtered.collect()  # Only executes now
```
**When to use**: Large dataset processing, performance-critical operations

**Key Pattern 2**: DuckDB integration for analytics
```python
import duckdb

# Query Parquet files directly with DuckDB
conn = duckdb.connect()
result = conn.execute("""
    SELECT * FROM 'data/nba_games.parquet'
    WHERE game_date > '2024-10-01'
""").fetchall()
```

**Library**: Streamlit
**Trust Score**: 8.9/10
**Context7 ID**: /streamlit/streamlit

**Key Pattern 1**: Fragments for partial updates
```python
@st.fragment(run_every=30)  # Auto-update every 30 seconds
def live_sync_dashboard():
    st.write(f"Last sync: {time.time()}")
```
**When to use**: Real-time dashboards, performance optimization

**Key Pattern 2**: Multi-page navigation
```python
from streamlit import Page

pages = [
    Page("pages/data_sync.py", title="Data Sync", icon="🔄"),
    Page("pages/analytics.py", title="Analytics", icon="📊")
]
pg = st.navigation(pages)
```

**Library**: DuckDB
**Trust Score**: 8.9/10
**Context7 ID**: /duckdb/duckdb

**Key Pattern 1**: Direct Parquet querying
```sql
SELECT * FROM 'myfile.parquet';
```
**When to use**: Analytics without ETL, ad-hoc queries

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
.devstream/bin/python -m pytest tests/ -v \
    --cov=src/nba_predictor \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
.devstream/bin/python -m mypy src/ tests/ --strict

# REQUIREMENT: Zero errors
```

### 3. Performance Benchmark
```bash
.devstream/bin/python -m pytest tests/performance/ -v

# TARGET: Data sync < 30 seconds, UI load < 5 seconds
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
refactor(core): implement unified data store and modular UI

Complete NBA system refactoring with modern architecture:
- Polars + DuckDB + Parquet data layer
- Modular Streamlit components with fragments
- Automatic data synchronization engine
- Modern src/ layout with uv dependency management

Implementation Details:
- Created src/nba_predictor package structure
- Implemented UnifiedDataStore with Polars integration
- Built AutomaticSyncEngine for startup synchronization
- Created modular UI components with real-time updates
- Refactored API clients for modern integration

Quality Validation:
- ✅ Tests: 95%+ coverage, all passing
- ✅ Type safety: mypy --strict passed
- ✅ Performance: Sub-30s sync times

Task ID: nba-system-refactoring-2024

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: Sub-30s data sync, sub-5s UI load
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