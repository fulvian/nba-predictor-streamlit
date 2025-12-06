#!/usr/bin/env python3
"""
NBA Bankroll 3.0 - Migration Script
Script eseguibile per migrare dal sistema attuale al nuovo Bankroll 3.0
Uso: python migrate_to_bankroll_v3.py [--legacy-db PATH] [--new-db PATH] [--backup]
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("migration.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def setup_argument_parser():
    """Configura parser argomenti linea comando"""
    parser = argparse.ArgumentParser(
        description="Migrate NBA betting system to Bankroll 3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python migrate_to_bankroll_v3.py --legacy-db data/nba_betting.duckdb --new-db data/nba_betting_v3.duckdb
  python migrate_to_bankroll_v3.py --legacy-db data/legacy.json --new-db data/nba_betting_v3.duckdb --backup
        """,
    )

    parser.add_argument(
        "--legacy-db", type=str, required=True, help="Path to legacy database file"
    )

    parser.add_argument(
        "--new-db",
        type=str,
        required=True,
        help="Path to new Bankroll 3.0 database file",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of existing new database before migration",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation only without actual migration",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def validate_paths(legacy_db_path: str, new_db_path: str) -> bool:
    """Valida percorsi database"""
    try:
        # Verifica esistenza database legacy
        legacy_path = Path(legacy_db_path)
        if not legacy_path.exists():
            logger.error(f"Legacy database not found: {legacy_db_path}")
            return False

        if not legacy_path.is_file():
            logger.error(f"Legacy database is not a file: {legacy_db_path}")
            return False

        # Verifica percorso nuovo database
        new_path = Path(new_db_path)
        new_dir = new_path.parent

        if not new_dir.exists():
            logger.info(f"Creating directory for new database: {new_dir}")
            new_dir.mkdir(parents=True, exist_ok=True)

        # Verifica permessi scrittura
        if not os.access(new_dir, os.W_OK):
            logger.error(f"No write permission for directory: {new_dir}")
            return False

        logger.info(f"Path validation successful")
        return True

    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        return False


def create_backup(new_db_path: str) -> str:
    """Crea backup del database esistente"""
    try:
        if os.path.exists(new_db_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{new_db_path}.backup_{timestamp}"

            import shutil

            shutil.copy2(new_db_path, backup_path)

            logger.info(f"Backup created: {backup_path}")
            return backup_path
        else:
            logger.info("No existing database to backup")
            return None

    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise


def run_migration(
    legacy_db_path: str,
    new_db_path: str,
    backup_path: Optional[str],
    batch_size: int,
    dry_run: bool,
) -> bool:
    """Esegue migrazione effettiva"""
    try:
        # Importa servizi Bankroll 3.0
        from src.nba_predictor.bankroll import BankrollMigrationService, MigrationStatus

        logger.info("Starting Bankroll 3.0 migration")
        logger.info(f"Legacy DB: {legacy_db_path}")
        logger.info(f"New DB: {new_db_path}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Dry run: {dry_run}")

        if dry_run:
            logger.info("DRY RUN MODE - Validation only")

        # Crea servizio migrazione
        migration_service = BankrollMigrationService(
            legacy_db_path=legacy_db_path, new_db_path=new_db_path
        )

        # Avvia migrazione
        migration_id = migration_service.start_migration(backup_path=backup_path)

        # Monitora progresso
        last_reported_phase = None

        while True:
            current_migration = migration_service.get_migration_status(migration_id)

            if not current_migration:
                logger.error("Migration lost - no status available")
                return False

            # Report progresso solo se cambia fase
            if current_migration.current_phase.value != last_reported_phase:
                logger.info(f"Migration phase: {current_migration.current_phase.value}")
                logger.info(
                    f"Progress: {current_migration.processed_records}/{current_migration.total_records} "
                    f"({(current_migration.processed_records / max(current_migration.total_records, 1)) * 100:.1f}%)"
                )
                last_reported_phase = current_migration.current_phase.value

            # Controlla completamento
            if current_migration.status in [
                MigrationStatus.COMPLETED,
                MigrationStatus.FAILED,
            ]:
                break

            # Attesa per evitare polling troppo frequente
            import time

            time.sleep(2)

        # Report finale
        final_migration = migration_service.get_migration_status(migration_id)

        if final_migration.status == MigrationStatus.COMPLETED:
            logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
            logger.info(f"Total records: {final_migration.total_records}")
            logger.info(f"Processed: {final_migration.processed_records}")
            logger.info(f"Failed: {final_migration.failed_records}")
            logger.info(
                f"Success rate: {(final_migration.processed_records / max(final_migration.total_records, 1)) * 100:.2f}%"
            )
            logger.info(
                f"Duration: {final_migration.end_time - final_migration.start_time}"
            )

            if final_migration.warnings:
                logger.warning(f"Warnings: {len(final_migration.warnings)}")
                for warning in final_migration.warnings:
                    logger.warning(f"  - {warning}")

            return True

        elif final_migration.status == MigrationStatus.FAILED:
            logger.error("❌ MIGRATION FAILED")
            logger.error(f"Failed in phase: {final_migration.current_phase.value}")
            logger.error(f"Errors: {len(final_migration.errors)}")
            for error in final_migration.errors:
                logger.error(f"  - {error}")

            return False

        else:
            logger.warning(
                f"Migration ended with status: {final_migration.status.value}"
            )
            return False

    except Exception as e:
        logger.error(f"Migration execution failed: {e}")
        return False


def print_migration_report(migration_service, migration_id: str):
    """Stampa report dettagliato migrazione"""
    try:
        report = migration_service.export_migration_report(migration_id)

        print("\n" + "=" * 60)
        print("MIGRATION REPORT")
        print("=" * 60)

        if "error" in report:
            print(f"❌ Error: {report['error']}")
            return

        migration_report = report["migration_report"]
        system_info = report["system_info"]

        print(f"Migration ID: {migration_report['migration_id']}")
        print(f"Status: {migration_report['status']}")
        print(f"Start Time: {migration_report['start_time']}")
        print(f"End Time: {migration_report.get('end_time', 'N/A')}")
        print(f"Current Phase: {migration_report['current_phase']}")

        print("\nSTATISTICS:")
        print(f"  Total Records: {migration_report['total_records']}")
        print(f"  Processed Records: {migration_report['processed_records']}")
        print(f"  Failed Records: {migration_report['failed_records']}")
        print(f"  Success Rate: {migration_report['success_rate']:.2f}%")

        print(f"\nWARNINGS: {len(migration_report['warnings'])}")
        for i, warning in enumerate(migration_report["warnings"], 1):
            print(f"  {i}. {warning}")

        print(f"\nERRORS: {len(migration_report['errors'])}")
        for i, error in enumerate(migration_report["errors"], 1):
            print(f"  {i}. {error}")

        print("\nSYSTEM INFORMATION:")
        print(f"  Legacy DB: {system_info['legacy_db_path']}")
        print(f"  New DB: {system_info['new_db_path']}")
        print(f"  Batch Size: {system_info['batch_size']}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Failed to generate migration report: {e}")
        print(f"❌ Error generating report: {e}")


def main():
    """Funzione principale"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configura livello logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    print("🏀 NBA Bankroll 3.0 Migration Tool")
    print("=" * 50)

    try:
        # Validazione percorsi
        if not validate_paths(args.legacy_db, args.new_db):
            logger.error("Path validation failed")
            sys.exit(1)

        # Creazione backup se richiesto
        backup_path = None
        if args.backup:
            try:
                backup_path = create_backup(args.new_db)
            except Exception as e:
                logger.error(f"Backup creation failed: {e}")
                sys.exit(1)

        # Esegui migrazione
        success = run_migration(
            legacy_db_path=args.legacy_db,
            new_db_path=args.new_db,
            backup_path=backup_path,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        if success:
            # Genera report finale
            from src.nba_predictor.bankroll import BankrollMigrationService

            migration_service = BankrollMigrationService(args.legacy_db, args.new_db)

            # Ottieni ID migrazione dall'ultima completata
            migration_history = migration_service.get_migration_history()
            if migration_history:
                last_migration = migration_history[-1]
                print_migration_report(migration_service, last_migration.migration_id)

            print("\n✅ Migration completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Migration failed!")
            print("Check migration.log for details")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        print("\n⚠️ Migration cancelled by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
