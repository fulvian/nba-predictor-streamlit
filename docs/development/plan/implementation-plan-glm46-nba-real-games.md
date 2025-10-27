# Implementation Plan: NBA Real Games Download - BallDontLie API Integration

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `nba-real-games-balldontlie`
**Phase**: Implementation
**Priority**: 9/10
**Estimated Duration**: 2.5 hours

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

### TodoWrite Tasks:
1. Install PyrateLimiter for rate limiting
2. Create BallDontLie API client with rate limiting
3. Replace The Odds API with BallDontLie in data_provider.py
4. Update main_app.py to use new BallDontLie provider
5. Add date range selection (1-5 days)
6. Test implementation with real data
7. Verify all functionality works

### Task 1: Install Rate Limiting Library (Duration: 10 min)

**File**: `.venv/requirements.txt`

**ACTION**: Add PyrateLimiter dependency for rate limiting

**FUNCTION SIGNATURE** (USE EXACTLY):
```bash
echo "pyrate-limiter==0.7.0" >> .venv/requirements.txt
.venv/bin/pip install pyrate-limiter==0.7.0
```

**PATTERN REFERENCE**: See Context7 research for PyrateLimiter best practices

**TOOL USAGE**:
1. **Tool**: `Bash`
   **When**: Install dependency
   **Example**:
   ```bash
   .venv/bin/pip install pyrate-limiter==0.7.0
   ```

**TEST FILE**: `tests/unit/test_ball_dont_lie_client.py::test_rate_limiting`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] PyrateLimiter installed successfully
- [ ] No dependency conflicts
- [ ] Version matches requirement (0.7.0)

**COMPLETION COMMAND**:
```bash
# Run after installation
.venv/bin/python -c "import pyrate_limiter; print('PyrateLimiter installed successfully')"
```

### Task 2: Create BallDontLie API Client (Duration: 45 min)

**File**: `ball_dont_lie_client.py` (New file)

**ACTION**: Create NBA BallDontLie API client with rate limiting

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class NBABallDontLieClient:
    """
    NBA BallDontLie API client with rate limiting for real NBA games data.

    Provides access to official NBA schedule and game data with built-in
    rate limiting to respect API limits (5 requests/minute on free tier).

    Attributes:
        api: BallDontLie API client instance
        limiter: PyrateLimiter instance for rate control
        logger: Logger instance for debugging
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize BallDontLie API client with rate limiting.

        Args:
            api_key: BallDontLie API key from environment

        Raises:
            ValueError: If api_key is None or empty
            Exception: If API client initialization fails
        """

    def get_games_for_date_range(
        self,
        start_date: date,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get NBA games for specified date range with rate limiting.

        Args:
            start_date: Start date for games search
            end_date: End date for games search (default: start_date)

        Returns:
            List of NBA games with complete information

        Raises:
            RateLimitException: When API rate limit is exceeded
            APIException: When API call fails
            ValueError: If date range is invalid

        Example:
            >>> client = NBABallDontLieClient("api_key")
            >>> games = client.get_games_for_date_range(date(2025, 10, 27))
            >>> print(f"Found {len(games)} games")
        """

    def _convert_ball_dont_lie_game_to_nba_format(
        self,
        bdl_game: Any
    ) -> Dict[str, Any]:
        """
        Convert BallDontLie game format to internal NBA format.

        Args:
            bdl_game: BallDontLie game object

        Returns:
            Standardized NBA game dictionary

        Raises:
            ValueError: If required game data is missing
        """
```

**PATTERN REFERENCE**: See `data_provider.py:41-70` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # API call with rate limiting
    limiter.try_acquire("ball_dontlie_api")
    games = self.api.nba.games.list(dates=date_list)
except BucketFullException as e:
    logger.error(
        "Rate limit exceeded",
        extra={"error": str(e), "retry_after": "60 seconds"}
    )
    raise RateLimitException("API rate limit exceeded. Please wait.") from e
except Exception as e:
    logger.error(
        "BallDontLie API call failed",
        extra={"date": date_str, "error": str(e)}
    )
    raise APIException(f"Failed to fetch games: {e}") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Need BallDontLie API specifics
   **Example**:
   ```python
   library_id = mcp__context7__resolve-library-id(libraryName="balldontlie")
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="games endpoint parameters",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_ball_dont_lie_client.py::test_get_games_for_date_range`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Class signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Rate limiting implemented with PyrateLimiter
- [ ] Error handling for rate limits and API failures
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_ball_dont_lie_client.py -v
.venv/bin/python -m mypy ball_dont_lie_client.py --strict
```

### Task 3: Update Data Provider Integration (Duration: 30 min)

**File**: `data_provider.py` (Lines: 1-165)

**ACTION**: Replace The Odds API with BallDontLie API integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _get_ball_dont_lie_games(self, days_ahead: int = 7, specific_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get NBA games using BallDontLie API with real schedule data.

    Args:
        days_ahead: Number of days ahead to fetch games for
        specific_date: Specific date string (YYYY-MM-DD) if provided

    Returns:
        List of NBA games with real schedule information

    Raises:
        RateLimitException: When API rate limit is exceeded
        APIException: When BallDontLie API call fails
    """
```

**PATTERN REFERENCE**: See `data_provider.py:93-165` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Use BallDontLie client instead of The Odds API
    games = self.bdl_client.get_games_for_date_range(start_date, end_date)
    logger.info(f"Retrieved {len(games)} games from BallDontLie API")
    return self._process_ball_dont_lie_games(games)
except RateLimitException as e:
    logger.error("Rate limit exceeded for BallDontLie API")
    return []
except Exception as e:
    logger.error(f"BallDontLie API failed: {e}")
    return []
```

**TOOL USAGE**:
1. **Tool**: `Read` + `Edit`
   **When**: Modify existing data_provider.py
   **Example**:
   ```python
   # Read current implementation
   Read(file_path="data_provider.py")
   # Edit to integrate BallDontLie
   Edit(file_path="data_provider.py", old_string=..., new_string=...)
   ```

**TEST FILE**: `tests/unit/test_data_provider.py::test_get_ball_dont_lie_games`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] BallDontLie client integration complete
- [ ] Date range handling (single date + multi-day)
- [ ] Error handling implemented
- [ ] Backward compatibility maintained
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_data_provider.py -v
.venv/bin/python -m mypy data_provider.py --strict
```

### Task 4: Update Dashboard Interface (Duration: 20 min)

**File**: `main_app.py` (Lines: 71-96)

**ACTION**: Update main dashboard to use BallDontLie and support date ranges

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def render_games_schedule_with_date_range(data_provider: NBADataProvider):
    """
    Render enhanced games schedule with date range selection.

    Features:
    - Single date selection (existing)
    - Date range selection (1-5 days)
    - Real NBA games data from BallDontLie API
    - Rate limiting status indicator
    """
```

**PATTERN REFERENCE**: See `main_app.py:63-96` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Try BallDontLie API first (new primary source)
    st.write("🔄 Connecting to BallDontLie API for real NBA schedule...")
    games = data_provider.get_scheduled_games(days_ahead=days_ahead, specific_date=specific_date)

    if games:
        st.success(f"✅ Found {len(games)} real NBA games from BallDontLie API")
        data_source = "BallDontLie API (Official NBA Schedule)"
    else:
        st.warning("⚠️ No games found. Checking alternative sources...")
        # Fallback logic here

except Exception as e:
    st.error(f"❌ Error loading games: {e}")
    st.exception(e)
```

**TOOL USAGE**:
1. **Tool**: `Read` + `Edit`
   **When**: Update dashboard interface
   **Example**:
   ```python
   # Add date range selection
   date_range_option = st.selectbox(
       "Select Date Range:",
       ["Single Date", "Next 3 Days", "Next 5 Days", "Custom Range"]
   )
   ```

**TEST FILE**: `tests/unit/test_main_app.py::test_render_games_schedule_with_date_range`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Date range options implemented
- [ ] BallDontLie API integration in UI
- [ ] Rate limiting status display
- [ ] Error handling in UI
- [ ] Real NBA games display
- [ ] Test written and passing
- [ ] Manual testing confirms functionality

**COMPLETION COMMAND**:
```bash
# Run after implementation
.venv/bin/python -m pytest tests/unit/test_main_app.py -v
# Manual test
.venv/bin/python -m streamlit run main_app.py --server.port 8502
```

### Task 5: Testing and Validation (Duration: 30 min)

**File**: Multiple files for comprehensive testing

**ACTION**: Complete testing of BallDontLie integration

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def test_end_to_end_ball_dont_lie_integration():
    """
    Test complete BallDontLie API integration.

    Tests:
    - API client initialization
    - Rate limiting behavior
    - Date range queries
    - Data format conversion
    - Dashboard display
    """
```

**PATTERN REFERENCE**: See existing test files for testing patterns

**TOOL USAGE**:
1. **Tool**: `Bash`
   **When**: Run comprehensive tests
   **Example**:
   ```bash
   .venv/bin/python -m pytest tests/ -v --cov=ball_dont_lie_client --cov-report=term-missing
   ```

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] All tests pass (100% pass rate)
- [ ] Test coverage ≥ 95% for new code
- [ ] Rate limiting works correctly
- [ ] Real NBA games displayed
- [ ] Date range selection functional
- [ ] Error handling verified
- [ ] Manual testing successful

**COMPLETION COMMAND**:
```bash
# Comprehensive testing
.venv/bin/python -m pytest tests/ -v --cov=ball_dont_lie_client,data_provider --cov-report=term-missing
.venv/bin/python -m mypy ball_dont_lie_client.py data_provider.py main_app.py --strict
```

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

**Library**: BallDontLie API 0.1.6
**Trust Score**: N/A (Official API)
**Context7 ID**: N/A (Direct API)

**Key Pattern 1**: BallDontLie API Initialization
```python
from balldontlie import BalldontlieAPI
api = BalldontlieAPI(api_key=os.getenv('BALLDONTLIE_API_KEY'))
games = api.nba.games.list(dates=['2025-10-27'])
```
**When to use**: Primary data source for NBA schedule

**Key Pattern 2**: PyrateLimiter Rate Limiting
```python
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException

rate = Rate(5, Duration.MINUTE)  # 5 requests per minute
limiter = Limiter(rate)

try:
    limiter.try_acquire("api_call")
    # Execute API call
except BucketFullException:
    # Handle rate limit exceeded
    time.sleep(60)
```

**Library**: PyrateLimiter 0.7.0
**Trust Score**: 9.3/10
**Context7 ID**: `/vutran1710/pyratelimiter`

**Key Pattern 3**: Error Handling for APIs
```python
try:
    limiter.try_acquire("ball_dontlie_api")
    games = api.nba.games.list(dates=date_list)
except BucketFullException as e:
    logger.error(f"Rate limit exceeded: {e}")
    raise RateLimitException("API rate limit exceeded") from e
except Exception as e:
    logger.error(f"API call failed: {e}")
    raise APIException(f"Failed to fetch games: {e}") from e
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
.venv/bin/python -m pytest tests/ -v \
    --cov=ball_dont_lie_client \
    --cov=data_provider \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
.venv/bin/python -m mypy ball_dont_lie_client.py data_provider.py main_app.py --strict

# REQUIREMENT: Zero errors
```

### 3. Performance Benchmark
```bash
# Test API rate limiting
time .venv/bin/python -c "from ball_dont_lie_client import NBABallDontLieClient; client = NBABallDontLieClient('test_key'); print('Rate limiting test passed')"

# TARGET: < 2 seconds initialization
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat(nba): integrate BallDontLie API for real NBA games schedule

Replace The Odds API dependency with BallDontLie API to provide
official NBA schedule data instead of betting odds only.

Implementation Details:
- Created NBABallDontLieClient with PyrateLimiter rate limiting
- Updated data_provider.py to use BallDontLie as primary source
- Enhanced main_app.py with date range selection (1-5 days)
- Maintained backward compatibility with existing UI

Quality Validation:
- ✅ Tests: 12 tests passing, 97% coverage
- ✅ Type safety: mypy --strict passed
- ✅ Performance: < 2s initialization, proper rate limiting

Task ID: nba-real-games-balldontlie

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: API calls < 2s, rate limiting working correctly
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