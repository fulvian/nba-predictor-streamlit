# 🔌 API Integration Guide

## Overview

This guide explains the API integration architecture of the NBA Predictor Analytics system, including data sources, authentication, rate limiting, and fallback mechanisms.

## 🏗️ Multi-Source Architecture

The system uses a hybrid multi-source architecture designed for maximum reliability and data quality:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   The Odds API  │    │ NBA Official    │    │  Fallback       │
│   (Primary)     │    │ API (Secondary) │    │  System         │
│                 │    │                 │    │                 │
• Future games   │    • Completed games│    • Mock data     │
• Betting odds   │    • Team stats     │    • Realistic      │
• 15+ bookmakers │    • Player data    │    • Always works   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Data Sources Integration

### 1. The Odds API (Primary Source)

**Purpose**: Future games with real-time betting odds

#### Endpoint Configuration
```python
base_url = "https://api.the-odds-api.com/v4"
endpoint = "/sports/basketball_nba/odds"
full_url = f"{base_url}{endpoint}"
```

#### Supported Markets
- **Moneyline**: Winner/loser betting odds
- **Spread**: Point spread betting lines
- **Totals**: Over/under betting markets
- **Propositions**: Player and team props (when available)

#### Bookmaker Coverage
The system integrates with 15+ major bookmakers:
- DraftKings, FanDuel, BetMGM
- Caesars, PointsBet, BetRivers
- William Hill, Unibet, Pinnacle
- Bwin, 888Sport, and more...

#### Authentication
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}

params = {
    'apiKey': os.getenv('THE_ODDS_API_KEY'),
    'regions': 'us',
    'markets': 'h2h,spreads,totals',
    'oddsFormat': 'american',
    'dateFormat': 'iso'
}
```

#### Rate Limiting
- **Free Tier**: 500 requests/day
- **Request Interval**: 1 second between calls
- **Quota Management**: Graceful fallback when exceeded

#### Response Format
```json
{
  "id": "NBA_GAME_ID",
  "sport_key": "basketball_nba",
  "sport_title": "NBA",
  "commence_time": "2025-10-27T23:00:00Z",
  "home_team": "Utah Jazz",
  "away_team": "Golden State Warriors",
  "bookmakers": [
    {
      "key": "draftkings",
      "title": "DraftKings",
      "last_update": "2025-10-27T20:00:00Z",
      "markets": [
        {
          "key": "h2h",
          "last_update": "2025-10-27T20:00:00Z",
          "outcomes": [
            {
              "name": "Golden State Warriors",
              "price": "+120"
            },
            {
              "name": "Utah Jazz",
              "price": "-140"
            }
          ]
        }
      ]
    }
  ]
}
```

### 2. NBA Official API (Secondary Source)

**Purpose**: Completed games statistics and official NBA data

#### Primary Endpoints

##### ScoreboardV2 - Current Games
```python
url = 'https://stats.nba.com/stats/scoreboardv2'
params = {
    'LeagueID': '00',
    'GameDate': '2025-10-27'  # YYYY-MM-DD format
}
```

##### ScheduleLeagueV2 - Season Schedule
```python
url = 'https://stats.nba.com/stats/scheduleleaguev2'
params = {
    'LeagueID': '00',
    'Season': '2025-26'
}
```

##### PlayerNextNGames - Upcoming Games
```python
url = 'https://stats.nba.com/stats/playernextngames'
params = {
    'LeagueID': '00',
    'NumberOfGames': 25
}
```

#### Authentication
No API key required (public data):
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/'
}
```

#### Response Processing
```python
def process_nba_response(response_data):
    """Extract game information from NBA API response"""
    result_sets = response_data.get('resultSets', [])

    # Find game data in response
    for result_set in result_sets:
        if result_set.get('name') == 'GameHeader':
            headers = result_set.get('headers', [])
            rows = result_set.get('rowSet', [])

            games = []
            for row in rows:
                game_data = dict(zip(headers, row))
                games.append({
                    'game_id': game_data['GAME_ID'],
                    'home_team': game_data['HOME_TEAM_NAME'],
                    'away_team': game_data['VISITOR_TEAM_NAME'],
                    'game_time': game_data['GAME_TIME'],
                    'game_status': game_data['GAME_STATUS_TEXT']
                })

            return games

    return []
```

### 3. Fallback System (Tertiary Source)

**Purpose**: Guaranteed availability with realistic mock data

#### Mock Data Generation
```python
def generate_nba_schedule_fallback(target_date=None):
    """Generate realistic NBA schedule when APIs are unavailable"""

    if target_date is None:
        target_date = date.today()

    # Realistic NBA teams with venue information
    nba_teams = [
        {'name': 'Golden State Warriors', 'venue': 'Chase Center', 'timezone': 'America/Los_Angeles'},
        {'name': 'Los Angeles Lakers', 'venue': 'Crypto.com Arena', 'timezone': 'America/Los_Angeles'},
        # ... all 30 teams
    ]

    # Generate realistic game times
    games = []
    for i in range(0, len(nba_teams), 2):
        if i + 1 < len(nba_teams):
            home_team = nba_teams[i]
            away_team = nba_teams[i + 1]

            # Realistic game times (7:00 PM, 8:00 PM, 9:00 PM ET)
            game_time = f"{17 + (i % 3):02d}:00"
            utc_time = convert_to_utc(game_time, home_team['timezone'])

            games.append({
                'game_id': f'FALLBACK_{target_date.strftime("%Y%m%d")}_{i//2}',
                'date': target_date.strftime('%Y-%m-%d'),
                'time': game_time,
                'time_utc': utc_time,
                'home_team': home_team['name'],
                'away_team': away_team['name'],
                'status': f'{game_time} ET',
                'source': 'Enhanced Mock Data (APIs Unavailable)',
                'home_timezone': home_team['timezone'],
                'away_timezone': away_team['timezone'],
                'bookmakers_count': 0,
                'odds': {}
            })

    return games
```

## 🔄 Data Flow Architecture

### Request Flow

```
User Request → Dashboard → Data Provider → API Manager → Data Processing → Response
     ↓              ↓            ↓              ↓              ↓
Date Select   main_app.py   NBADataProv     HybridAPI     TimezoneMgr
Tab Click     render_tab   get_games()     try_odds()    convert_utc()
Load Games    → Show UI    → orchestrate   → nba_api()   → enhance_data()
```

### Error Handling Flow

```
API Request → Success? → Process Data → Return Response
     ↓               ↓
   No        Try Next API → Success? → Process Data → Return Response
     ↓               ↓
   No        Activate Fallback → Process Mock Data → Return Response
```

### Data Enhancement Pipeline

```python
def enhance_game_data(raw_game_data):
    """Enhance raw API data with timezone and team information"""

    enhanced_games = []
    tz_manager = NBATimezoneManager()

    for game in raw_game_data:
        # Parse UTC time
        utc_dt = parse_utc_datetime(game['time_utc'])

        # Convert to local timezones
        home_local, home_tz = tz_manager.convert_utc_to_local(utc_dt, game['home_team'])
        away_local, away_tz = tz_manager.convert_utc_to_local(utc_dt, game['away_team'])

        # Enhance game data
        enhanced_game = game.copy()
        enhanced_game.update({
            'local_date': home_local.strftime('%Y-%m-%d'),
            'home_local_time': home_local.strftime('%H:%M'),
            'away_local_time': away_local.strftime('%H:%M'),
            'home_timezone': home_tz,
            'away_timezone': away_tz,
            'all_timezones': tz_manager.get_game_times_by_timezone(utc_dt),
            'utc_datetime_iso': utc_dt.isoformat()
        })

        enhanced_games.append(enhanced_game)

    return enhanced_games
```

## 🔐 Authentication & Security

### API Key Management

#### Environment Variables
```bash
# .env file
THE_ODDS_API_KEY=your_api_key_here
NBA_SEASON=2025-26
DEBUG_MODE=false
```

#### Key Validation
```python
def validate_api_key():
    """Validate The Odds API key before use"""
    api_key = os.getenv('THE_ODDS_API_KEY')

    if not api_key:
        print("⚠️ THE_ODDS_API_KEY not found in environment")
        return False

    if len(api_key) < 10:
        print("⚠️ Invalid THE_ODDS_API_KEY format")
        return False

    return True
```

### Request Security

#### Headers Configuration
```python
def get_secure_headers():
    """Generate secure headers for API requests"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com'
    }
```

#### Request Timeout & Retry
```python
def make_api_request(url, params=None, headers=None, max_retries=3):
    """Make API request with retry logic and timeout"""

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=20  # 20 second timeout
            )
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            print(f"⚠️ Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None
```

## 📊 Rate Limiting Management

### The Odds API Rate Limits

#### Quota Monitoring
```python
class APIQuotaManager:
    """Manage API quota and rate limiting"""

    def __init__(self):
        self.daily_requests = 0
        self.last_request_time = 0
        self.quota_limit = 500  # Free tier daily limit

    def can_make_request(self):
        """Check if we can make a request without exceeding limits"""
        current_time = time.time()

        # Enforce minimum interval between requests
        if current_time - self.last_request_time < 1.0:
            time.sleep(1.0 - (current_time - self.last_request_time))

        # Check daily quota
        if self.daily_requests >= self.quota_limit:
            return False

        return True

    def record_request(self):
        """Record that a request was made"""
        self.daily_requests += 1
        self.last_request_time = time.time()
```

#### Graceful Quota Handling
```python
def handle_quota_exceeded():
    """Handle API quota exceeded gracefully"""

    # Log quota exceeded
    print("⚠️ The Odds API quota exceeded. Switching to NBA Official API.")

    # Fall back to NBA Official API
    nba_games = get_nba_games_official_api(date.today())

    if nba_games:
        print("✅ Successfully obtained games from NBA Official API")
        return nba_games
    else:
        print("⚠️ NBA Official API also unavailable. Using fallback data.")
        return generate_nba_schedule_fallback()
```

### NBA Official API Rate Limits

#### Implicit Rate Limiting
```python
def nba_api_request_with_backoff(url, params, max_retries=3):
    """Make NBA API request with built-in rate limiting"""

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️ Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            print(f"⚠️ NBA API request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None
```

## 🛡️ Error Handling & Recovery

### Comprehensive Error Types

#### Network Errors
```python
def handle_network_error(error):
    """Handle network-related errors"""

    if isinstance(error, requests.exceptions.ConnectionError):
        print("🔌 Network connection error. Check internet connectivity.")
        return True  # Can retry

    elif isinstance(error, requests.exceptions.Timeout):
        print("⏰ Request timeout. API may be slow.")
        return True  # Can retry

    elif isinstance(error, requests.exceptions.HTTPError):
        if error.response.status_code == 401:
            print("🔑 Authentication failed. Check API key.")
            return False  # Don't retry

        elif error.response.status_code == 429:
            print("⚠️ Rate limited. Implementing backoff.")
            return True  # Can retry with backoff

        elif error.response.status_code >= 500:
            print("🔧 Server error. Can retry later.")
            return True  # Can retry

    return False  # Unknown error, don't retry
```

#### Data Validation
```python
def validate_game_data(game_data):
    """Validate game data structure and content"""

    required_fields = ['game_id', 'home_team', 'away_team', 'time_utc']

    # Check required fields
    for field in required_fields:
        if field not in game_data:
            print(f"⚠️ Missing required field: {field}")
            return False

    # Validate team names
    valid_teams = get_all_nba_teams()
    if game_data['home_team'] not in valid_teams:
        print(f"⚠️ Invalid home team: {game_data['home_team']}")
        return False

    if game_data['away_team'] not in valid_teams:
        print(f"⚠️ Invalid away team: {game_data['away_team']}")
        return False

    # Validate time format
    try:
        parse_utc_datetime(game_data['time_utc'])
    except ValueError as e:
        print(f"⚠️ Invalid time format: {e}")
        return False

    return True
```

### Fallback Activation Logic

```python
def get_games_with_fallback(target_date=None):
    """Get games with comprehensive fallback system"""

    if target_date is None:
        target_date = date.today()

    # Tier 1: Try The Odds API (for future games + odds)
    print("🔄 Trying The Odds API...")
    odds_games = get_odds_api_games(target_date)

    if odds_games and validate_games_list(odds_games):
        print(f"✅ The Odds API success: {len(odds_games)} games")
        return enhance_game_data(odds_games)

    # Tier 2: Try NBA Official API (for completed games + official data)
    print("🔄 Trying NBA Official API...")
    nba_games = get_nba_games_official_api(target_date)

    if nba_games and validate_games_list(nba_games):
        print(f"✅ NBA Official API success: {len(nba_games)} games")
        return enhance_game_data(nba_games)

    # Tier 3: Enhanced fallback system (always works)
    print("🔄 Using enhanced fallback system...")
    fallback_games = generate_nba_schedule_fallback(target_date)

    print(f"✅ Fallback system: {len(fallback_games)} games")
    return enhance_game_data(fallback_games)
```

## 📈 Performance Optimization

### Caching Strategy

#### Session-Level Caching
```python
class SessionCache:
    """Session-level caching for API responses"""

    def __init__(self, cache_duration_minutes=30):
        self.cache = {}
        self.cache_duration = cache_duration_minutes * 60  # Convert to seconds

    def get(self, cache_key):
        """Get cached data if still valid"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[cache_key]  # Expired data
        return None

    def set(self, cache_key, data):
        """Cache data with timestamp"""
        self.cache[cache_key] = (data, time.time())

    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
```

#### API Response Caching
```python
def get_cached_games(target_date):
    """Get games with caching"""

    cache_key = f"games_{target_date.strftime('%Y-%m-%d')}"
    cached_data = session_cache.get(cache_key)

    if cached_data:
        print(f"📋 Using cached data for {target_date}")
        return cached_data

    # Get fresh data
    fresh_data = get_games_with_fallback(target_date)

    # Cache the result
    session_cache.set(cache_key, fresh_data)

    return fresh_data
```

### Connection Pooling

```python
# Optimized session management
class OptimizedAPIClient:
    """API client with connection pooling and optimization"""

    def __init__(self):
        # Create session with connection pooling
        self.session = requests.Session()

        # Configure connection pool
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Default headers
        self.session.headers.update(get_secure_headers())

    def get(self, url, **kwargs):
        """Make GET request with optimized session"""
        return self.session.get(url, timeout=20, **kwargs)

    def close(self):
        """Close session and cleanup"""
        self.session.close()
```

## 🔍 Debugging & Monitoring

### API Health Monitoring

```python
def monitor_api_health():
    """Monitor the health of all API endpoints"""

    health_status = {
        'the_odds_api': {'status': 'unknown', 'response_time': None},
        'nba_official_api': {'status': 'unknown', 'response_time': None},
        'fallback_system': {'status': 'always_available', 'response_time': 0.001}
    }

    # Test The Odds API
    start_time = time.time()
    try:
        response = make_api_request(
            f"{THE_ODDS_API_BASE_URL}/sports",
            headers=odds_headers,
            timeout=5
        )
        if response and response.status_code == 200:
            health_status['the_odds_api']['status'] = 'healthy'
            health_status['the_odds_api']['response_time'] = time.time() - start_time
        else:
            health_status['the_odds_api']['status'] = 'unhealthy'
    except Exception as e:
        health_status['the_odds_api']['status'] = f'error: {str(e)}'

    # Test NBA Official API
    start_time = time.time()
    try:
        response = make_api_request(
            'https://stats.nba.com/stats/scoreboardv2',
            params={'LeagueID': '00', 'GameDate': date.today().strftime('%Y-%m-%d')},
            headers=nba_headers,
            timeout=5
        )
        if response and response.status_code == 200:
            health_status['nba_official_api']['status'] = 'healthy'
            health_status['nba_official_api']['response_time'] = time.time() - start_time
        else:
            health_status['nba_official_api']['status'] = 'unhealthy'
    except Exception as e:
        health_status['nba_official_api']['status'] = f'error: {str(e)}'

    return health_status
```

### Debug Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('api_integration')

def log_api_request(api_name, url, params, response_time, status_code):
    """Log API request details"""
    logger.info(f"{api_name} Request: {url}")
    logger.debug(f"Parameters: {params}")
    logger.info(f"Response Time: {response_time:.3f}s")
    logger.info(f"Status Code: {status_code}")

def log_api_error(api_name, error, context=""):
    """Log API errors with context"""
    logger.error(f"{api_name} Error: {str(error)}")
    if context:
        logger.error(f"Context: {context}")
```

---

## 🔗 Additional Resources

- [The Odds API Documentation](https://the-odds-api.com/)
- [NBA Stats API Documentation](https://nba-apidocumentation.knowledgeowl.com/)
- [Python Requests Best Practices](https://requests.readthedocs.io/)
- [API Rate Limiting Strategies](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html)

**🎯 Always implement proper error handling and rate limiting for production API integrations!**