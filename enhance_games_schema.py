#!/usr/bin/env python3
"""
Script per aggiornare lo schema dei file parquet dei games NBA.

Aggiunge le colonne mancanti necessarie per il funzionamento completo del
sistema di betting, mantenendo la compatibilità con i dati esistenti.
"""

import sys
import os
import polars as pl
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def backup_existing_files():
    """Create backup of existing parquet files before migration."""
    project_root = get_project_root()
    games_dir = project_root / "data" / "games"
    backup_dir = project_root / "data" / "games_backup"

    if not games_dir.exists():
        logger.warning(f"Games directory {games_dir} not found")
        return False

    # Create backup directory
    backup_dir.mkdir(exist_ok=True)

    # Copy all parquet files to backup
    backed_up_files = []
    for parquet_file in games_dir.glob("*.parquet"):
        backup_file = backup_dir / parquet_file.name
        logger.info(f"Backing up {parquet_file} to {backup_file}")

        # Read original and write to backup
        try:
            df = pl.read_parquet(parquet_file)
            df.write_parquet(backup_file)
            backed_up_files.append(parquet_file)
        except Exception as e:
            logger.error(f"Failed to backup {parquet_file}: {e}")
            return False

    logger.info(f"Successfully backed up {len(backed_up_files)} files")
    return True

def determine_game_status(game_date_str, home_score, away_score):
    """
    Determine game status based on scores and date.

    Args:
        game_date_str: Game date as string
        home_score: Home team score (int)
        away_score: Away team score (int)

    Returns:
        String status: 'Final', 'Scheduled', 'In Progress', or 'Postponed'
    """
    try:
        game_date = datetime.strptime(game_date_str, '%Y-%m-%d')
        today = datetime.now()

        # If both scores are greater than 0, game is likely Final
        if home_score > 0 and away_score > 0:
            return 'Final'

        # If game date is in the future, it's Scheduled
        if game_date > today:
            return 'Scheduled'

        # If game date is today or past but no scores, might be Scheduled or Postponed
        if game_date <= today:
            if home_score == 0 and away_score == 0:
                return 'Scheduled'  # Default to Scheduled for now
            else:
                return 'In Progress'  # Partial scores

        return 'Scheduled'
    except:
        return 'Scheduled'

def generate_game_id(home_team, away_team, game_date):
    """Generate a game_id if missing."""
    # Create simple game_id from teams and date
    home_abbr = home_team[:3].upper()
    away_abbr = away_team[:3].upper()
    date_str = game_date.replace('-', '')
    return f"{away_abbr}@{home_abbr}_{date_str}"

def enhance_games_schema():
    """
    Enhance existing games parquet files with missing columns.

    Ensures all files have the complete schema:
    - game_id: Unique game identifier
    - game_date: Game date
    - home_team: Home team name
    - away_team: Away team name
    - season: NBA season year
    - home_score: Home team score
    - away_score: Away team score
    - status: Game status ('Final', 'Scheduled', 'In Progress', 'Postponed')
    - time: Game time (string)
    - odds: Game odds (JSON string)
    - source: Data source identifier
    - updated_at: Last update timestamp
    """
    project_root = get_project_root()
    games_dir = project_root / "data" / "games"

    if not games_dir.exists():
        logger.error(f"Games directory {games_dir} not found")
        return False

    enhanced_files = []

    for parquet_file in games_dir.glob("*.parquet"):
        logger.info(f"Processing {parquet_file}")

        try:
            # Read existing data
            df = pl.read_parquet(parquet_file)
            logger.info(f"  Original shape: {df.shape}")
            logger.info(f"  Original columns: {df.columns}")

            # Create enhanced DataFrame with all required columns
            enhanced_data = []

            for row in df.iter_rows(named=True):
                # Handle date conversion
                game_date = row['game_date']
                if hasattr(game_date, 'strftime'):
                    game_date_str = game_date.strftime('%Y-%m-%d')
                else:
                    game_date_str = str(game_date)

                # Determine or get status
                if 'status' in row and row['status']:
                    status = row['status']
                else:
                    status = determine_game_status(
                        game_date_str,
                        row.get('home_score', 0),
                        row.get('away_score', 0)
                    )

                # Generate game_id if missing
                if 'game_id' in row and row['game_id']:
                    game_id = row['game_id']
                else:
                    game_id = generate_game_id(
                        row['home_team'],
                        row['away_team'],
                        game_date_str
                    )

                # Get time or default to empty
                time = row.get('time', '') if 'time' in row else ''

                # Get odds or default to empty JSON
                odds = row.get('odds', '{}') if 'odds' in row else '{}'

                # Get source or default
                source = row.get('source', 'Enhanced Data') if 'source' in row else 'Enhanced Data'

                # Get updated_at or use current time
                updated_at = row.get('updated_at') if 'updated_at' in row else datetime.now()

                # Enhanced row data with complete schema
                enhanced_row = {
                    'game_id': game_id,
                    'game_date': game_date_str,
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'season': row.get('season', 2025),
                    'home_score': row.get('home_score', 0),
                    'away_score': row.get('away_score', 0),
                    'status': status,
                    'time': time,
                    'odds': odds,
                    'source': source,
                    'updated_at': updated_at
                }

                enhanced_data.append(enhanced_row)

            # Create new DataFrame with enhanced schema
            enhanced_df = pl.DataFrame(enhanced_data)
            logger.info(f"  Enhanced shape: {enhanced_df.shape}")
            logger.info(f"  Enhanced columns: {enhanced_df.columns}")

            # Write back to same file
            enhanced_df.write_parquet(parquet_file)
            enhanced_files.append(parquet_file)

            logger.info(f"  ✅ Enhanced {parquet_file.name}")

        except Exception as e:
            logger.error(f"  ❌ Failed to enhance {parquet_file}: {e}")
            return False

    logger.info(f"Successfully enhanced {len(enhanced_files)} files")
    return True

def verify_enhancement():
    """Verify that the enhancement worked correctly."""
    project_root = get_project_root()
    games_dir = project_root / "data" / "games"

    sample_file = None
    for parquet_file in games_dir.glob("*.parquet"):
        sample_file = parquet_file
        break

    if not sample_file:
        logger.error("No parquet files found for verification")
        return False

    try:
        df = pl.read_parquet(sample_file)
        logger.info("✅ Verification successful!")
        logger.info(f"  Columns: {df.columns}")
        logger.info(f"  Shape: {df.shape}")

        # Check required columns exist
        required_columns = ['status', 'time', 'odds', 'source']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"  ❌ Missing columns: {missing_columns}")
            return False

        logger.info(f"  ✅ All required columns present: {required_columns}")

        # Show sample data
        logger.info("  Sample data:")
        for i, row in enumerate(df.head(3).iter_rows(named=True)):
            logger.info(f"    {i+1}. {row['home_team']} vs {row['away_team']} - {row['status']} ({row['game_date']})")

        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main():
    """Main migration process."""
    logger.info("🚀 Starting NBA Games Schema Enhancement")
    logger.info("=" * 60)

    # Step 1: Backup existing files
    logger.info("📦 Step 1: Creating backup of existing files...")
    if not backup_existing_files():
        logger.error("❌ Backup failed, aborting migration")
        return False
    logger.info("✅ Backup completed successfully")

    # Step 2: Enhance schema
    logger.info("🔧 Step 2: Enhancing games schema...")
    if not enhance_games_schema():
        logger.error("❌ Schema enhancement failed")
        return False
    logger.info("✅ Schema enhancement completed successfully")

    # Step 3: Verify enhancement
    logger.info("✅ Step 3: Verifying enhancement...")
    if not verify_enhancement():
        logger.error("❌ Verification failed")
        return False
    logger.info("✅ Verification completed successfully")

    # Summary
    logger.info("🎉 NBA Games Schema Enhancement COMPLETED!")
    logger.info("=" * 60)
    logger.info("✅ Files backed up to: data/games_backup/")
    logger.info("✅ Enhanced files: data/games/")
    logger.info("✅ New columns added: status, time, odds, source")
    logger.info("✅ Ready for betting workflow integration")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)