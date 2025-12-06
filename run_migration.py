import os
import sys
import logging
from src.nba_predictor.bankroll.migration import BankrollMigrationService
from src.nba_predictor.bankroll.audit import AuditLogger
from src.nba_predictor.bankroll.engine import TransactionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("migration_runner")


def run_migration():
    # Paths
    project_root = os.getcwd()
    legacy_db_path = os.path.join(
        project_root, "data", "bankroll.json"
    )  # Primary trigger, reads others too
    new_db_path = os.path.join(project_root, "data", "nba_bankroll_v3.duckdb")

    logger.info("Starting Bankroll 3.0 Migration...")
    logger.info(f"Root: {project_root}")
    logger.info(f"Target: {new_db_path}")

    # Initialize Engine (Required for AuditLogger)
    engine = TransactionEngine(new_db_path)

    # Initialize Audit Logger (Optional but good for tracking)
    audit_logger = AuditLogger(
        engine
    )  # Will initialize tables if needed or migration service will

    # Initialize Service
    service = BankrollMigrationService(
        legacy_db_path=legacy_db_path,
        new_db_path=new_db_path,
        audit_logger=audit_logger,
    )

    try:
        # Execute Migration
        migration_id = service.start_migration()

        logger.info("-" * 50)
        logger.info(f"MIGRATION SUCCESSFUL! ID: {migration_id}")
        logger.info(f"New Database: {new_db_path}")
        logger.info("-" * 50)

        # Verify result
        if service.current_migration:
            stats = service.current_migration.to_dict()
            logger.info(f"Total Records: {stats['total_records']}")
            logger.info(f"Processed: {stats['processed_records']}")
            logger.info(f"Failed: {stats['failed_records']}")
            if stats["warnings"]:
                logger.warning(f"Warnings: {len(stats['warnings'])}")
                for w in stats["warnings"]:
                    logger.warning(f" - {w}")

    except Exception as e:
        logger.error(f"MIGRATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
