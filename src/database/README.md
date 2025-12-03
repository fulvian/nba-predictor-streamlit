# NBA Betting Database System

A robust, Context7-compliant database system for NBA betting operations, built with DuckDB for optimal performance and data integrity.

## 🚀 Features

### ✅ **Core Features**
- **ACID Transactions**: Full transaction support with rollback capabilities
- **Schema Validation**: Comprehensive constraint checking and data validation
- **Automatic Backups**: Scheduled and manual backup system with integrity verification
- **Data Migration**: Seamless migration from existing JSON files
- **Performance Optimized**: Columnar storage with efficient indexing
- **Type Safety**: Enum-based status management and data validation

### 🏗️ **Database Architecture**

#### Core Tables

1. **Games** (`games`)
   - NBA game information and metadata
   - Status tracking (scheduled, in_progress, completed, cancelled, postponed)
   - Team details and scoring information
   - Data completeness scoring

2. **Bets** (`bets`)
   - Complete betting transaction records
   - Odds, stakes, and probability calculations
   - Quality and confidence scores
   - Status tracking (pending, won, lost, cancelled, replaced)

3. **Bankroll** (`bankroll`)
   - Financial transaction tracking
   - Running balance calculations
   - Audit trail for all money movements
   - Transaction categorization

4. **Game Results** (`game_results`)
   - Actual game outcomes
   - Final scores and totals
   - Result verification data

5. **Betting Analysis** (`betting_analysis`)
   - Prediction data and analysis metrics
   - Model performance tracking
   - Confidence and quality indicators

#### Database Views

- **active_bets**: Current pending and placed bets
- **bankroll_summary**: Chronological financial transactions
- **betting_performance**: Aggregated betting statistics

## 📋 Installation and Setup

### Prerequisites

```bash
# Python 3.11+ required
python --version

# Install required packages
pip install duckdb pytest
```

### Quick Start

```python
from src.database import initialize_nba_database

# Initialize complete database system
results = initialize_nba_database(
    db_path="data/nba_betting.duckdb",
    data_dir="data",
    backup_dir="data/backups",
    run_migration=True
)

if results['success']:
    print("🎉 Database initialized successfully!")
    print(f"📊 Database statistics: {results['statistics']}")
else:
    print("❌ Initialization failed:")
    for error in results['errors']:
        print(f"  - {error}")
```

## 🗄️ Database Schema

### Tables Overview

#### Games Table
```sql
CREATE TABLE games (
    game_id VARCHAR PRIMARY KEY,
    api_game_id VARCHAR,
    home_team VARCHAR NOT NULL,
    home_team_abbr VARCHAR NOT NULL,
    away_team VARCHAR NOT NULL,
    away_team_abbr VARCHAR NOT NULL,
    game_date DATE NOT NULL,
    game_time VARCHAR,
    league VARCHAR DEFAULT 'NBA',
    season VARCHAR,
    status VARCHAR DEFAULT 'scheduled',
    home_score INTEGER,
    away_score INTEGER,
    total_score INTEGER,
    data_completeness_score INTEGER,
    data_sources_summary VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bets Table
```sql
CREATE TABLE bets (
    bet_id VARCHAR PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    bet_type VARCHAR NOT NULL CHECK (bet_type IN ('OVER', 'UNDER', 'SPREAD', 'MONEYLINE')),
    line DOUBLE,
    odds DOUBLE NOT NULL CHECK (odds > 0),
    stake DOUBLE NOT NULL CHECK (stake > 0),
    potential_payout DOUBLE GENERATED ALWAYS AS (stake * odds) VIRTUAL,
    probability DOUBLE CHECK (probability >= 0 AND probability <= 1),
    implied_probability DOUBLE CHECK (implied_probability >= 0 AND implied_probability <= 1),
    true_probability DOUBLE CHECK (true_probability >= 0 AND true_probability <= 1),
    edge DOUBLE,
    quality_score DOUBLE CHECK (quality_score >= 0 AND quality_score <= 1),
    confidence_score DOUBLE CHECK (confidence_score >= 0 AND confidence_score <= 1),
    risk_score DOUBLE CHECK (risk_score >= 0 AND risk_score <= 1),
    consistency_score DOUBLE CHECK (consistency_score >= 0 AND consistency_score <= 1),
    margin DOUBLE CHECK (margin >= 0),
    simulation_wins INTEGER CHECK (simulation_wins >= 0),
    total_simulations INTEGER CHECK (total_simulations > 0),
    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'cancelled', 'replaced')),
    is_value BOOLEAN DEFAULT false,
    original_bet_data VARCHAR,
    replaced_at TIMESTAMP,
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);
```

### Constraints and Validation

- **Foreign Key Constraints**: Ensures referential integrity
- **Check Constraints**: Validates data ranges and business rules
- **Generated Columns**: Automatic calculation of derived values
- **Cascade Deletes**: Maintains data consistency

## 🔄 Data Migration

### From JSON to Database

The system automatically migrates existing JSON data:

1. **pending_bets.json** → `bets` table
2. **bankroll.json** → `bankroll` table
3. **pending_results.json** → `games` and `betting_analysis` tables

### Migration Process

```python
from src.database.migration import DataMigrator

migrator = DataMigrator(
    db_path="data/nba_betting.duckdb",
    data_dir="data"
)

migration_report = migrator.run_full_migration()
print(f"Migration success: {migration_report['success']}")
print(f"Games migrated: {migration_report['migration_results']['games']['migrated']}")
print(f"Bets migrated: {migration_report['migration_results']['bets']['migrated']}")
```

## 💾 Backup System

### Automatic Backups

```python
from src.database.backup import DatabaseBackup

backup_system = DatabaseBackup(
    db_path="data/nba_betting.duckdb",
    backup_dir="data/backups"
)

# Create automatic daily backup
backup_path = backup_system.schedule_automatic_backup("daily")

# Create manual backup
backup_path = backup_system.create_full_backup("manual")
```

### Backup Features

- **Compression**: Gzip compression for space efficiency
- **Integrity Verification**: SHA-256 checksums for all data
- **Metadata Tracking**: Complete backup information and statistics
- **Automated Cleanup**: Old backup removal with configurable retention
- **Restore Verification**: Pre-restore integrity checks

### Restore Process

```python
# Restore from backup
success = backup_system.restore_from_backup(
    backup_path="data/backups/nba_betting_full_20240101_120000.duckdb.gz",
    verify_before_restore=True
)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest src/database/tests/ -v

# Run specific test categories
python -m pytest src/database/tests/test_database_integrity.py::TestDatabaseSchema -v
python -m pytest src/database/tests/test_database_integrity.py::TestDataMigration -v
python -m pytest src/database/tests/test_database_integrity.py::TestDatabaseBackup -v
```

### Test Coverage

- **Schema Validation**: Table creation, constraints, and relationships
- **Data Migration**: JSON to database conversion with integrity checks
- **Backup System**: Creation, verification, and restore functionality
- **Business Rules**: Bet status transitions, game status management
- **Data Integrity**: Foreign key constraints and cascade operations

## 📊 Performance Optimizations

### DuckDB Optimizations

1. **Columnar Storage**: Efficient compression and query performance
2. **Vectorized Execution**: Fast analytical queries
3. **Memory Management**: Configurable memory limits and optimization
4. **Parallel Processing**: Multi-threaded query execution

### Indexing Strategy

```sql
-- Performance indexes
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_bets_game_id ON bets(game_id);
CREATE INDEX idx_bets_status ON bets(status);
CREATE INDEX idx_bankroll_created_at ON bankroll(created_at);
```

## 🔧 Configuration

### Database Settings

```python
# Connection configuration
con.execute("PRAGMA enable_progress_bar=false;")
con.execute("PRAGMA preserve_insertion_order=true;")
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA memory_limit='1GB';")
```

### Backup Configuration

```python
# Backup retention settings
backup_system.cleanup_old_backups(
    keep_days=30,    # Keep backups for 30 days
    keep_count=10    # Always keep 10 most recent backups
)
```

## 📈 Monitoring and Statistics

### Database Statistics

```python
# Get comprehensive statistics
from src.database import _generate_database_statistics

stats = _generate_database_statistics(connection)
print(f"Total tables: {len(stats['tables'])}")
print(f"Total rows: {sum(t['row_count'] for t in stats['tables'].values())}")
```

### Performance Monitoring

```sql
-- View active bets
SELECT * FROM active_bets ORDER BY placed_at DESC;

-- View bankroll summary
SELECT * FROM bankroll_summary ORDER BY created_at DESC;

-- View betting performance
SELECT * FROM betting_performance;
```

## 🛡️ Security and Integrity

### Data Protection

- **ACID Transactions**: All operations are atomic and consistent
- **Constraint Validation**: Prevents invalid data entry
- **Backup Verification**: Ensures backup integrity before restore
- **Audit Trail**: Complete transaction history in bankroll table

### Error Handling

- **Transaction Rollback**: Automatic rollback on errors
- **Graceful Degradation**: System continues operating during partial failures
- **Comprehensive Logging**: Detailed error reporting and diagnostics
- **Recovery Procedures**: Clear steps for data recovery

## 📚 API Reference

### Core Classes

#### `NBADatabaseSchema`
```python
schema = NBADatabaseSchema(db_path="data/nba_betting.duckdb")
schema.connect()
schema.create_tables()
schema.create_views()
validation = schema.validate_schema()
schema.close()
```

#### `DataMigrator`
```python
migrator = DataMigrator(db_path="data/nba_betting.duckdb", data_dir="data")
migrator.connect()
report = migrator.run_full_migration()
```

#### `DatabaseBackup`
```python
backup = DatabaseBackup(db_path="data/nba_betting.duckdb", backup_dir="data/backups")
backup.connect()
backup_path = backup.create_full_backup("manual")
success = backup.restore_from_backup(backup_path)
```

### Enums

```python
from src.database import BetStatus, GameStatus, TransactionType

# Bet statuses
BetStatus.PENDING
BetStatus.WON
BetStatus.LOST
BetStatus.CANCELLED
BetStatus.REPLACED

# Game statuses
GameStatus.SCHEDULED
GameStatus.IN_PROGRESS
GameStatus.COMPLETED
GameStatus.CANCELLED
GameStatus.POSTPONED

# Transaction types
TransactionType.BET_PLACED
TransactionType.BET_WON
TransactionType.BET_LOST
TransactionType.DEPOSIT
TransactionType.WITHDRAWAL
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check file permissions
   - Verify directory exists
   - Ensure DuckDB is properly installed

2. **Migration Errors**
   - Validate JSON file format
   - Check for required fields
   - Verify data types and ranges

3. **Backup Failures**
   - Check disk space availability
   - Verify write permissions
   - Ensure database connection is valid

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger('src.database')
logger.setLevel(logging.DEBUG)
```

## 📄 License

This database system is part of the NBA Predictor Streamlit project and follows the same licensing terms.

## 🤝 Contributing

When contributing to the database system:

1. **Write Tests**: Ensure new functionality has comprehensive test coverage
2. **Update Documentation**: Keep this README and code comments current
3. **Follow Context7 Principles**: Maintain compliance with Context7 best practices
4. **Test Migrations**: Verify data migration works correctly with changes
5. **Check Constraints**: Ensure all database constraints remain valid

---

**Built with Context7 best practices for maximum reliability and performance** 🎯