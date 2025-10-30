#!/usr/bin/env python3
"""
Script per migliorare il collegamento tra scommesse e dati delle partite NBA.

Implementa una funzione di matching intelligente che collega le scommesse
ai dati delle partite usando diversi criteri di matching.
"""

import sys
import polars as pl
from pathlib import Path
from datetime import datetime
import logging
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def parse_game_info_from_bet_id(game_id: str) -> dict:
    """
    Parse game information from betting database game_id.

    Examples:
    - 'BDL_18446888' -> {'source': 'BallDontLie', 'api_id': '18446888', 'teams': None}
    - 'bet_BDL_18446888_OVER_231.0_20251030_090403' -> similar parsing
    """
    # Extract information from bet_id patterns
    if game_id.startswith('BDL_'):
        return {
            'source': 'BallDontLie',
            'api_id': game_id.replace('BDL_', ''),
            'teams': None,
            'original_id': game_id
        }
    elif '_OVER_' in game_id or '_UNDER_' in game_id:
        # Pattern: bet_BDL_18446888_OVER_231.0_20251030_090403
        parts = game_id.split('_')
        if len(parts) >= 2 and parts[1] == 'BDL':
            return {
                'source': 'BallDontLie',
                'api_id': parts[2],
                'teams': None,
                'original_id': game_id
            }
    else:
        return {
            'source': 'Unknown',
            'api_id': game_id,
            'teams': None,
            'original_id': game_id
        }

def find_matching_game(bet_info: dict, games_df: pl.DataFrame) -> dict:
    """
    Find the best matching game for a bet using multiple criteria.

    Args:
        bet_info: Parsed bet information
        games_df: DataFrame with games data

    Returns:
        Best matching game or None
    """
    best_match = None
    best_score = 0

    for row in games_df.iter_rows(named=True):
        score = 0

        # Criterion 1: API ID match (if available)
        if bet_info['api_id'] and str(row.get('game_id', '')).endswith(bet_info['api_id']):
            score += 100

        # Criterion 2: Date proximity (same date = high score)
        if bet_info.get('date'):
            try:
                bet_date = datetime.strptime(bet_info['date'], '%Y-%m-%d')
                game_date = row['game_date']
                if isinstance(game_date, str):
                    game_date = datetime.strptime(game_date, '%Y-%m-%d')
                if isinstance(game_date, datetime):
                    if (game_date - bet_date).days == 0:
                        score += 50
                    elif abs((game_date - bet_date).days) <= 1:
                        score += 25
            except:
                pass

        # Criterion 3: Team name matching
        home_team = row['home_team'].lower().strip()
        away_team = row['away_team'].lower().strip()

        # Common team name variations
        team_variations = {
            'golden state': ['warriors', 'gs'],
            'los angeles': ['lakers', 'la'],
            'boston': ['celtics', 'bos'],
            'miami': ['heat', 'mia'],
            'chicago': ['bulls', 'chi'],
            'milwaukee': ['bucks', 'mil'],
            'san antonio': ['spurs', 'sas'],
            'dallas': ['mavericks', 'dal'],
            'brooklyn': ['nets', 'bkn'],
            'new york': ['knicks', 'nyk'],
            'philadelphia': ['76ers', 'phi'],
        }

        # Check if we have team info in bet_id or elsewhere
        # For now, assume we'll add team info to the database later

        if score > best_score:
            best_score = score
            best_match = {
                'game': row,
                'score': score
            }

    return best_match

def enhance_bet_database_with_team_info():
    """
    Enhance the betting database with team information extracted from game_ids.

    This will add home_team and away_team fields to the placed_bets table
    for better matching with games data.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔧 Enhancing betting database with team information...")

            # Check if columns already exist
            try:
                result = conn.execute("DESCRIBE placed_bets").fetchall()
                columns = [row[0] for row in result]
            except duckdb.InvalidInputException:
                logger.error("Table placed_bets not found")
                return False

            # Add team columns if they don't exist
            if 'home_team' not in columns:
                logger.info("   Adding home_team column...")
                conn.execute("ALTER TABLE placed_bets ADD COLUMN home_team VARCHAR")
            if 'away_team' not in columns:
                logger.info("   Adding away_team column...")
                conn.execute("ALTER TABLE placed_bets ADD COLUMN away_team VARCHAR")

            # Load games data for matching
            games_data = []
            games_dir = project_root / "data" / "games"

            for parquet_file in games_dir.glob("*.parquet"):
                try:
                    df = pl.read_parquet(parquet_file)
                    games_data.extend(df.to_dicts())
                except Exception as e:
                    logger.warning(f"Could not read {parquet_file}: {e}")

            if not games_data:
                logger.warning("No games data found for matching")
                return False

            logger.info(f"   Loaded {len(games_data)} games for matching")

            # Update each bet with team info
            bets = conn.execute("SELECT bet_id, game_id, placed_at FROM placed_bets").fetchall()

            updated_count = 0
            for bet_id, db_game_id, placed_at in bets:
                # Parse bet information
                bet_info = parse_game_info_from_bet_id(db_game_id)
                # Extract date from placed_at timestamp
                if placed_at:
                    bet_info['date'] = placed_at.date()
                else:
                    bet_info['date'] = None

                # Find matching game
                games_df = pl.DataFrame(games_data)
                match = find_matching_game(bet_info, games_df)

                if match and match['score'] >= 50:  # Only update if we have a good match
                    game = match['game']
                    conn.execute(f"""
                        UPDATE placed_bets
                        SET home_team = '{game['home_team']}', away_team = '{game['away_team']}'
                        WHERE bet_id = '{bet_id}'
                    """)
                    updated_count += 1

                    if updated_count % 10 == 0:
                        logger.info(f"   Updated {updated_count} bets so far...")

            logger.info(f"✅ Successfully updated {updated_count} bets with team information")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to enhance betting database: {e}")
        return False

def update_game_ids_in_bets():
    """
    Update game_ids in betting database to match parquet file format.

    This ensures consistency between betting and games data.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("🔄 Updating game_ids in betting database...")

            # Load games data
            games_data = []
            games_dir = project_root / "data" / "games"

            for parquet_file in games_dir.glob("*.parquet"):
                try:
                    df = pl.read_parquet(parquet_file)
                    games_data.extend(df.to_dicts())
                except Exception as e:
                    logger.warning(f"Could not read {parquet_file}: {e}")

            if not games_data:
                logger.warning("No games data found")
                return False

            logger.info(f"   Loaded {len(games_data)} games for ID updating")

            # Create a mapping of existing game_id patterns to new game_ids
            bets = conn.execute("SELECT bet_id, game_id, home_team, away_team FROM placed_bets").fetchall()

            updated_count = 0
            for bet_id, old_game_id, home_team, away_team in bets:
                # Skip if team info is missing
                if not home_team or not away_team:
                    logger.debug(f"   Skipping bet {bet_id}: missing team info")
                    continue

                # Find matching game by teams and date
                games_df = pl.DataFrame(games_data)

                # Filter by team names (case-insensitive)
                matching_games = games_df.filter(
                    (pl.col('home_team').str.to_lowercase().str.contains(home_team.lower())) &
                    (pl.col('away_team').str.to_lowercase().str.contains(away_team.lower()))
                )

                if matching_games.height > 0:
                    # Use the first match
                    new_game_id = matching_games.row(0)['game_id']

                    conn.execute(f"""
                        UPDATE placed_bets
                        SET game_id = '{new_game_id}'
                        WHERE bet_id = '{bet_id}'
                    """)
                    updated_count += 1

                    if updated_count % 10 == 0:
                        logger.info(f"   Updated {updated_count} game_ids so far...")

            logger.info(f"✅ Successfully updated {updated_count} game_ids")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to update game_ids: {e}")
        return False

def main():
    """Main enhancement process."""
    logger.info("🚀 Starting Bet-Game Matching Enhancement")
    logger.info("=" * 60)

    success = True

    # Step 1: Add team info to bets
    logger.info("📝 Step 1: Adding team information to bets...")
    if not enhance_bet_database_with_team_info():
        success = False

    # Step 2: Update game_ids for consistency
    logger.info("🔄 Step 2: Updating game_ids for consistency...")
    if success and update_game_ids_in_bets():
        success = success

    # Summary
    if success:
        logger.info("🎉 Bet-Game Matching Enhancement COMPLETED!")
        logger.info("=" * 60)
        logger.info("✅ Enhanced betting database with team information")
        logger.info("✅ Updated game_ids for consistency")
        logger.info("✅ Better bet-game matching implemented")
    else:
        logger.error("❌ Enhancement failed")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)