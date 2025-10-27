# NBA Timezone Utils API Documentation

## Overview

`nba_timezone_utils.py` provides comprehensive timezone management and NBA schedule data integration with Context7-compliant pytz usage.

## Classes

### NBATimezoneManager

Main class for timezone conversion and NBA team information management.

#### Methods

##### `__init__(self)`

Initialize the timezone manager with NBA team timezone mappings.

```python
from nba_timezone_utils import NBATimezoneManager
tz_manager = NBATimezoneManager()
```

##### `convert_utc_to_local(self, utc_datetime: datetime, team_name: str) -> Tuple[datetime, str]`

Convert UTC datetime to local timezone for a specific NBA team.

**Parameters:**
- `utc_datetime`: UTC datetime object
- `team_name`: NBA team name (full or short)

**Returns:**
- Tuple of (local_datetime, timezone_name)

**Example:**
```python
from datetime import datetime, timezone
from nba_timezone_utils import NBATimezoneManager

tz_manager = NBATimezoneManager()
utc_time = datetime(2025, 10, 27, 23, 0, tzinfo=timezone.utc)
local_time, tz_name = tz_manager.convert_utc_to_local(utc_time, "Golden State Warriors")
print(f"Local time: {local_time} ({tz_name})")
# Output: Local time: 2025-10-27 16:00:00-07:00 (America/Los_Angeles)
```

##### `_get_team_name_by_id(self, team_id) -> str`

Convert NBA team ID to team name.

**Parameters:**
- `team_id`: NBA team ID (integer or string)

**Returns:**
- Team name string

**Example:**
```python
team_name = tz_manager._get_team_name_by_id("1610612747")
print(team_name)  # "Los Angeles Lakers"
```

##### `get_game_times_by_timezone(self, utc_datetime: datetime) -> Dict[str, str]`

Get game time in all relevant timezones.

**Parameters:**
- `utc_datetime`: UTC datetime of the game

**Returns:**
- Dictionary mapping timezone names to formatted times

## Functions

### `get_nba_games_official_api(target_date) -> List[Dict]`

Get NBA games using the official NBA.com API.

**Parameters:**
- `target_date`: Date to get games for (datetime.date or string)

**Returns:**
- List of game dictionaries with complete information

**Example:**
```python
from nba_timezone_utils import get_nba_games_official_api
from datetime import date

games = get_nba_games_official_api(date(2025, 10, 27))
print(f"Found {len(games)} games")
for game in games[:3]:
    print(f"{game['away_team']} @ {game['home_team']} - {game['time']}")
```

**Response Format:**
```python
{
    'game_id': 'NBA_0022500007',
    'date': '2025-10-27',
    'time': '17:00',
    'time_utc': '2025-10-27T23:00:00Z',
    'away_team': 'Golden State Warriors',
    'home_team': 'Utah Jazz',
    'status': '7:00 pm ET',
    'season': '2025-26',
    'home_timezone': 'America/Denver',
    'away_timezone': 'America/Los_Angeles',
    'source': 'NBA Official API - ScoreboardV2 (Direct)',
    'api_endpoint': 'stats.nba.com/stats/scoreboardv2',
    'bookmakers_count': 0,
    'odds': {}
}
```

### `generate_nba_schedule_fallback(target_date=None) -> List[Dict]`

Generate NBA schedule using multiple API sources with automatic fallback.

**Parameters:**
- `target_date`: Target date for games (optional, defaults to today)

**Returns:**
- List of game dictionaries with timezone processing

**Strategy:**
1. Try PlayerNextNGames for future games
2. Try ScoreboardV2 for today's games
3. Use enhanced mock data as final fallback

## Team Timezone Mapping

The system includes comprehensive timezone mappings for all 30 NBA teams:

### Eastern Conference
- Atlanta Hawks: America/New_York
- Boston Celtics: America/New_York
- Brooklyn Nets: America/New_York
- Charlotte Hornets: America/New_York
- Chicago Bulls: America/Chicago
- Cleveland Cavaliers: America/New_York
- Detroit Pistons: America/Detroit
- Indiana Pacers: America/Indianapolis
- Miami Heat: America/New_York
- Milwaukee Bucks: America/Chicago
- New York Knicks: America/New_York
- Orlando Magic: America/New_York
- Philadelphia 76ers: America/New_York
- Toronto Raptors: America/Toronto
- Washington Wizards: America/New_York

### Western Conference
- Dallas Mavericks: America/Chicago
- Denver Nuggets: America/Denver
- Golden State Warriors: America/Los_Angeles
- Houston Rockets: America/Chicago
- Los Angeles Clippers: America/Los_Angeles
- Los Angeles Lakers: America/Los_Angeles
- Memphis Grizzlies: America/Chicago
- Minnesota Timberwolves: America/Chicago
- New Orleans Pelicans: America/Chicago
- Oklahoma City Thunder: America/Chicago
- Phoenix Suns: America/Phoenix
- Portland Trail Blazers: America/Los_Angeles
- Sacramento Kings: America/Los_Angeles
- San Antonio Spurs: America/Chicago
- Seattle SuperSonics: America/Los_Angeles (historical)
- Utah Jazz: America/Denver

## Error Handling

The system includes comprehensive error handling:

### API Failures
- Automatic retry with exponential backoff
- Graceful degradation to fallback sources
- Detailed logging for debugging

### Data Validation
- Type checking for all inputs
- Validation of team names and IDs
- Timezone conversion error handling

### Rate Limiting
- Built-in protection against API rate limits
- Quota management for The Odds API
- Request throttling for NBA.com APIs

## Context7 Compliance

This module follows Context7 best practices:

- **Type Hints**: Complete type annotations for all methods
- **Documentation**: Comprehensive docstrings following Google style
- **Error Handling**: Robust exception management
- **Testing**: Designed for comprehensive test coverage
- **Code Style**: Follows PEP 8 and modern Python conventions

## Dependencies

- `pytz`: Timezone database and utilities
- `requests`: HTTP client for API calls
- `nba_api`: Official NBA API client
- `datetime`: Python datetime handling

## Performance Notes

- Timezone conversions are cached for performance
- Team ID mappings are pre-loaded for fast lookups
- API calls include proper timeout handling
- Memory usage optimized for large datasets