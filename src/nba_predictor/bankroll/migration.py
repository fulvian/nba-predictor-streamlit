"""
NBA Bankroll 3.0 - Migration Service
Servizio migrazione dati dal sistema legacy al nuovo ledger
Basato su best practice da enterprise data migration
"""

import hashlib
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from enum import Enum
import shutil

from .models import (
    Transaction,
    TransactionType,
    BetResult,
    BetRecord,
    BankrollState,
    create_transaction_id,
    create_bet_record,
)
from .exceptions import (
    MigrationError,
    DataCorruptionError,
    ValidationError,
    ChecksumMismatchError,
)
from .engine import TransactionEngine
from .audit import AuditLogger, AuditEventType, AuditSeverity


logger = logging.getLogger(__name__)


class MigrationPhase(Enum):
    """Fasi del processo di migrazione"""

    PREPARATION = "PREPARATION"
    DATA_EXTRACTION = "DATA_EXTRACTION"
    DATA_VALIDATION = "DATA_VALIDATION"
    DATA_TRANSFORMATION = "DATA_TRANSFORMATION"
    LEDGER_POPULATION = "LEDGER_POPULATION"
    VERIFICATION = "VERIFICATION"
    CLEANUP = "CLEANUP"
    COMPLETED = "COMPLETED"


class MigrationStatus(Enum):
    """Stati migrazione"""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class MigrationResult:
    """Risultato migrazione con statistiche complete"""

    migration_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: MigrationStatus
    current_phase: MigrationPhase
    total_records: int
    processed_records: int
    failed_records: int
    warnings: List[str]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    backup_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "current_phase": self.current_phase.value,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "failed_records": self.failed_records,
            "success_rate": (self.processed_records / max(self.total_records, 1)) * 100
            if self.total_records > 0
            else 0,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
            "backup_path": self.backup_path,
        }


@dataclass
class LegacyBetRecord:
    """Record scommessa legacy da migrare"""

    bet_id: str
    game_id: str
    bet_type: str
    selection: str
    odds: Decimal
    stake: Decimal
    result: Optional[str]
    payout: Optional[Decimal]
    profit_loss: Optional[Decimal]
    placed_timestamp: Optional[datetime]
    settled_timestamp: Optional[datetime]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": float(self.odds),
            "stake": float(self.stake),
            "result": self.result,
            "payout": float(self.payout) if self.payout else None,
            "profit_loss": float(self.profit_loss) if self.profit_loss else None,
            "placed_timestamp": self.placed_timestamp.isoformat()
            if self.placed_timestamp
            else None,
            "settled_timestamp": self.settled_timestamp.isoformat()
            if self.settled_timestamp
            else None,
            "metadata": self.metadata,
        }


class BankrollMigrationService:
    """
    Servizio migrazione enterprise-grade:
    - Migrazione sicura con rollback capability
    - Validazione completa dati
    - Transformazione dati legacy a nuovo formato
    - Progress tracking dettagliato
    - Error handling robusto con recovery
    """

    def __init__(
        self,
        legacy_db_path: str,
        new_db_path: str,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.legacy_db_path = legacy_db_path
        self.new_db_path = new_db_path
        self.audit_logger = audit_logger

        # Stato migrazione
        self.current_migration = None
        self.migration_history = []

        # Configurazione migrazione
        self.batch_size = 1000  # Records per batch
        self.max_retries = 3  # Tentativi per record
        self.timeout_seconds = 300  # Timeout per operazione

        logger.info(f"Migration service initialized: {legacy_db_path} -> {new_db_path}")

    def start_migration(self, backup_path: Optional[str] = None) -> str:
        """
        Avvia processo di migrazione completo
        Include backup, validazione, trasformazione, caricamento
        """
        import uuid

        migration_id = f"migration_{uuid.uuid4().hex[:16]}"

        try:
            # Crea risultato migrazione
            self.current_migration = MigrationResult(
                migration_id=migration_id,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                status=MigrationStatus.IN_PROGRESS,
                current_phase=MigrationPhase.PREPARATION,
                total_records=0,
                processed_records=0,
                failed_records=0,
                warnings=[],
                errors=[],
                metadata={},
                backup_path=backup_path,
            )

            # Log inizio migrazione
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Migration started: {migration_id}",
                    AuditSeverity.INFO,
                    additional_data={
                        "migration_id": migration_id,
                        "legacy_db": self.legacy_db_path,
                        "new_db": self.new_db_path,
                        "backup_path": backup_path,
                    },
                )

            # Esegui fasi migrazione
            self._execute_migration_phases()

            return migration_id

        except Exception as e:
            logger.error(f"Migration failed to start: {e}")
            if self.current_migration:
                self.current_migration.status = MigrationStatus.FAILED
                self.current_migration.errors.append(
                    {
                        "phase": "STARTUP",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            raise MigrationError(
                "Failed to start migration", "STARTUP_FAILED", 0, 0, str(e)
            )

    def _execute_migration_phases(self):
        """Esegui tutte le fasi della migrazione"""
        if not self.current_migration:
            raise MigrationError("No active migration", "NO_ACTIVE_MIGRATION")

        try:
            # Phase 1: Preparazione
            self._phase_preparation()

            # Phase 2: Estrazione dati legacy
            self._phase_data_extraction()

            # Phase 3: Validazione dati
            self._phase_data_validation()

            # Phase 4: Trasformazione dati
            self._phase_data_transformation()

            # Phase 5: Popolazione nuovo ledger
            self._phase_ledger_population()

            # Phase 6: Verifica
            self._phase_verification()

            # Phase 7: Cleanup
            self._phase_cleanup()

            # Completa migrazione
            self.current_migration.status = MigrationStatus.COMPLETED
            self.current_migration.current_phase = MigrationPhase.COMPLETED
            self.current_migration.end_time = datetime.now(timezone.utc)

            # Log completamento
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Migration completed: {self.current_migration.migration_id}",
                    AuditSeverity.INFO,
                    additional_data=self.current_migration.to_dict(),
                )

            # Aggiungi storia
            self.migration_history.append(self.current_migration)

            logger.info(
                f"Migration completed successfully: {self.current_migration.migration_id}"
            )

        except Exception as e:
            logger.error(
                f"Migration failed during phase {self.current_migration.current_phase}: {e}"
            )
            self.current_migration.status = MigrationStatus.FAILED
            self.current_migration.errors.append(
                {
                    "phase": self.current_migration.current_phase.value,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Tentativo rollback se errore critico
            if self._is_critical_error(e):
                logger.warning("Critical error detected, attempting rollback")
                self._rollback_migration()

            raise MigrationError(
                f"Migration failed in phase {self.current_migration.current_phase.value}",
                self.current_migration.current_phase.value,
                self.current_migration.processed_records,
                self.current_migration.failed_records + 1,
                str(e),
            )

    def _phase_preparation(self):
        """Phase 1: Preparazione ambiente migrazione"""
        self.current_migration.current_phase = MigrationPhase.PREPARATION

        try:
            # Verifica esistenza database legacy
            if not os.path.exists(self.legacy_db_path):
                raise FileNotFoundError(
                    f"Legacy database not found: {self.legacy_db_path}"
                )

            # Verifica accessibilità scrittura nuovo database
            new_dir = os.path.dirname(self.new_db_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir, exist_ok=True)

            # Backup automatico se non specificato
            if not self.current_migration.backup_path:
                backup_path = f"{self.new_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if os.path.exists(self.new_db_path):
                    shutil.copy2(self.new_db_path, backup_path)
                    self.current_migration.backup_path = backup_path
                    self.current_migration.warnings.append(
                        f"Automatic backup created: {backup_path}"
                    )

            # Inizializza nuovo database
            new_engine = TransactionEngine(self.new_db_path)

            # Verifica inizializzazione
            test_state = new_engine.get_transaction_count()

            self.current_migration.warnings.append(
                f"New database initialized successfully"
            )

            logger.info("Migration preparation completed")

        except Exception as e:
            raise MigrationError(
                "Preparation phase failed", "PREPARATION_FAILED", 0, 0, str(e)
            )

    def _phase_data_extraction(self):
        """Phase 2: Estrazione dati dal sistema legacy"""
        self.current_migration.current_phase = MigrationPhase.DATA_EXTRACTION

        try:
            # Estrai dati legacy
            legacy_data = self._extract_legacy_data()

            # Conta record estratti
            total_bets = len(legacy_data.get("bets", []))
            total_transactions = len(legacy_data.get("transactions", []))

            self.current_migration.total_records = total_bets + total_transactions

            # Salva dati temporanei per transformation
            self.current_migration.metadata["extracted_data"] = legacy_data

            self.current_migration.warnings.append(
                f"Extracted {total_bets} bets and {total_transactions} transactions"
            )

            logger.info(
                f"Data extraction completed: {self.current_migration.total_records} total records"
            )

        except Exception as e:
            raise MigrationError(
                "Data extraction phase failed", "EXTRACTION_FAILED", 0, 0, str(e)
            )

    def _extract_legacy_data(self) -> Dict[str, Any]:
        """
        Estrae dati dal sistema legacy
        Supporta bankroll_summary_report.json (Gold Standard) o fallback su file parziali
        """

        # 1. Prova a leggere bankroll_summary_report.json (Fonte più completa)
        # Check in 'data' folder (legacy_db_path dir)
        data_dir = os.path.dirname(self.legacy_db_path)
        report_path_data = os.path.join(data_dir, "bankroll_summary_report.json")

        # Check in project root (parent of 'data' folder)
        project_root = os.path.dirname(data_dir)
        report_path_root = os.path.join(project_root, "bankroll_summary_report.json")

        if os.path.exists(report_path_data):
            logger.info(f"Found Golden Record in data dir: {report_path_data}")
            return self._extract_from_summary_report(report_path_data)
        elif os.path.exists(report_path_root):
            logger.info(f"Found Golden Record in project root: {report_path_root}")
            return self._extract_from_summary_report(report_path_root)

        # Fallback logic (old implementation)
        logger.info(
            f"Summary report not found at {report_path_data} or {report_path_root}, falling back to partial files..."
        )
        return self._extract_legacy_partials()

    def _extract_from_summary_report(self, report_path: str) -> Dict[str, Any]:
        """Estrae dati dal report completo"""
        try:
            with open(report_path, "r") as f:
                report = json.load(f)

            legacy_data = {
                "bets": [],
                "transactions": [],
                "metadata": report.get("summary", {}),
            }

            # 1. Initial Deposit
            # Default to 84.75 if missing, as per report inspect
            initial_amount = Decimal(str(report.get("initial_deposit", 84.75)))

            # Sort bets to find start time
            bets_data = report.get("detailed_bets", [])
            bets_data.sort(key=lambda x: x.get("date", ""))

            first_bet_time = datetime.now(timezone.utc)
            if bets_data:
                first_time_str = bets_data[0].get("date")
                try:
                    # Handle format "YYYY-MM-DD HH:MM:SS"
                    first_bet_time = datetime.fromisoformat(
                        first_time_str.replace(" ", "T")
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    first_bet_time = datetime.now(timezone.utc)

            # Create Initial Deposit Transaction
            deposit_time = first_bet_time - timedelta(
                seconds=1
            )  # 1 sec before first bet
            deposit_txn = {
                "transaction_id": "legacy_initial_deposit",
                "timestamp": deposit_time.isoformat(),
                "type": "deposit",
                "amount": float(initial_amount),
                "balance": float(initial_amount),
                "description": "Migration: Initial Capital Import",
                "metadata": {"source": "bankroll_summary_report.json"},
            }
            legacy_data["transactions"].append(deposit_txn)

            # 2. Process Bets
            for bet_item in bets_data:
                # Generate Standard Bet Record
                try:
                    date_str = bet_item.get("date", "").replace(" ", "T")
                    bet_ts = datetime.fromisoformat(date_str).replace(
                        tzinfo=timezone.utc
                    )

                    # Bet Status Mapping
                    status_raw = bet_item.get("status", "PENDING").upper()

                    payout = Decimal(str(bet_item.get("payout", 0)))
                    stake = Decimal(str(bet_item.get("amount", 0)))
                    profit_loss = Decimal(str(bet_item.get("net_profit", 0)))

                    # Determine Result Logic
                    result = "PENDING"
                    if status_raw == "WON" or (
                        status_raw == "SETTLED" and profit_loss > 0
                    ):
                        result = "WON"
                    elif status_raw == "LOST" or (
                        status_raw == "SETTLED" and profit_loss <= 0
                    ):  # Assuming loss if not won
                        if payout == stake:
                            result = "PUSH"
                        else:
                            result = "LOST"
                    elif status_raw in ["PENDING", "OPEN"]:
                        result = "PENDING"

                    # Create Bet Record
                    bet_id = f"migrated_bet_{bet_item.get('index', 'unknown')}"

                    # Parse Match for teams
                    match_desc = bet_item.get("match", "Unknown vs Unknown")

                    legacy_bet = LegacyBetRecord(
                        bet_id=bet_id,
                        game_id=f"migrated_game_{bet_item.get('index', 'unknown')}",
                        bet_type="moneyline",  # Default
                        selection=match_desc,
                        odds=Decimal(str(bet_item.get("odds", 1.90))),
                        stake=stake,
                        result=result,
                        payout=payout if result in ["WON", "LOST", "PUSH"] else None,
                        profit_loss=profit_loss
                        if result in ["WON", "LOST", "PUSH"]
                        else None,
                        placed_timestamp=bet_ts,
                        settled_timestamp=bet_ts + timedelta(hours=2)
                        if result != "PENDING"
                        else None,
                        metadata={"original_index": bet_item.get("index")},
                    )
                    legacy_data["bets"].append(legacy_bet)

                except Exception as e:
                    logger.warning(f"Failed to parse report item {bet_item}: {e}")
                    # traceback.print_exc()

            return legacy_data
        except Exception as e:
            logger.error(f"CRITICAL ERROR parsing summary report: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _extract_legacy_partials(self) -> Dict[str, Any]:
        """Fallback method using partial files"""
        legacy_data = {"bets": [], "transactions": [], "metadata": {}}
        try:
            # 1. Leggi Bankroll Attuale (Cash)
            bankroll_path = os.path.join("data", "bankroll.json")
            current_cash = Decimal("0.00")
            if os.path.exists(bankroll_path):
                with open(bankroll_path, "r") as f:
                    data = json.load(f)
                    current_cash = Decimal(str(data.get("current_bankroll", 0)))
            else:
                logger.warning(
                    f"Bankroll file not found at {bankroll_path}, assuming 0."
                )

            # 2. Leggi Scommesse Pendenti
            bets_path = os.path.join("data", "pending_bets.json")
            pending_stake_total = Decimal("0.00")

            if os.path.exists(bets_path):
                with open(bets_path, "r") as f:
                    bets_data = json.load(f)
                    last_timestamp = None

                    for row in bets_data:
                        # Estrai dati flattening la struttura (bet_data nested)
                        bet_dict = self._flatten_legacy_bet_structure(row)

                        legacy_bet = self._convert_legacy_bet(bet_dict)
                        legacy_data["bets"].append(legacy_bet)

                        # Calcola totale stake per ricostruzione capitale iniziale
                        if legacy_bet.stake:
                            pending_stake_total += legacy_bet.stake

                        # Tieni traccia del timestamp più recente per il deposito
                        if legacy_bet.placed_timestamp:
                            if (
                                last_timestamp is None
                                or legacy_bet.placed_timestamp > last_timestamp
                            ):
                                last_timestamp = legacy_bet.placed_timestamp

            # 3. Sintetizza Transazione Deposito Iniziale (Capitale Totale = Cash + Pending Stakes)
            # Questo permette di ri-piazzare le scommesse (che scaleranno il balance)
            # e arrivare al 'current_cash' corretto.
            initial_capital = current_cash + pending_stake_total

            # Data deposito = un secondo prima della prima scommessa, o adesso se non c'è timestamp
            # Per semplicità usiamo un timestamp fittizio nel passato se non disponibile
            deposit_time = datetime.now(timezone.utc)
            if legacy_data["bets"]:
                # Trova la scommessa più vecchia
                oldest_bet = min(
                    legacy_data["bets"],
                    key=lambda b: b.placed_timestamp
                    or datetime.max.replace(tzinfo=timezone.utc),
                )
                if oldest_bet.placed_timestamp:
                    # 1 minuto prima della prima scommessa
                    deposit_time = oldest_bet.placed_timestamp.replace(
                        second=0, microsecond=0
                    )

            initial_deposit = {
                "transaction_id": "legacy_migration_initial_deposit",
                "timestamp": deposit_time.isoformat(),
                "type": "deposit",
                "amount": float(initial_capital),
                "balance": float(initial_capital),
                "description": "Migration: Legacy Capital Import (Cash + Pending Stakes)",
                "metadata": {
                    "imported_cash": float(current_cash),
                    "imported_pending_stakes": float(pending_stake_total),
                },
            }
            legacy_data["transactions"].append(initial_deposit)

            return legacy_data

        except Exception as e:
            logger.error(f"Failed to extract legacy data: {e}")
            raise

    def _flatten_legacy_bet_structure(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Appiattisce la struttura nidificata di pending_bets.json"""
        bet_data = row.get("bet_data", {})
        return {
            "id": row.get("bet_id"),
            "game_id": row.get("game_id"),
            "type": bet_data.get("type"),  # es. OVER/UNDER
            "selection": "Over"
            if "OVER" in str(row.get("bet_id"))
            else "Under",  # dedotto o da bet_data
            "odds": bet_data.get("odds"),
            "stake": bet_data.get("stake"),
            "result": "PENDING",  # pending_bets.json sono tutte pending
            "payout": 0,
            "profit_loss": 0,
            "placed_timestamp": row.get("timestamp"),
            "settled_timestamp": None,
            "metadata": {"original_data": row, "legacy_status": row.get("status")},
        }

    def _convert_legacy_bet(self, bet_data: Dict[str, Any]) -> LegacyBetRecord:
        """Converte dati bet legacy in formato standard"""
        # Gestione timestamp string -> datetime
        placed_ts = None
        if bet_data.get("placed_timestamp"):
            try:
                # Supporta sia ISO che formati custom se necessario
                placed_ts = datetime.fromisoformat(bet_data["placed_timestamp"])
                # Assicura UTC
                if placed_ts.tzinfo is None:
                    placed_ts = placed_ts.replace(tzinfo=timezone.utc)
            except ValueError:
                placed_ts = datetime.now(timezone.utc)

        return LegacyBetRecord(
            bet_id=bet_data.get("id", ""),
            game_id=bet_data.get("game_id", ""),
            bet_type=bet_data.get("type", "moneyline"),  # default se vuoto
            selection=bet_data.get("selection", ""),
            odds=Decimal(str(bet_data.get("odds", 0))),
            stake=Decimal(str(bet_data.get("stake", 0))),
            result=bet_data.get("result", "PENDING"),
            payout=Decimal(str(bet_data.get("payout", 0)))
            if bet_data.get("payout")
            else None,
            profit_loss=Decimal(str(bet_data.get("profit_loss", 0)))
            if bet_data.get("profit_loss")
            else None,
            placed_timestamp=placed_ts,
            settled_timestamp=None,
            metadata=bet_data.get("metadata", {}),
        )

    def _convert_legacy_transaction(self, txn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Converte dati transazione legacy (usato per il deposito sintetico)"""
        return txn_data

    # Rimuovi metodi inutilizzati _convert_legacy_bankroll e i _from_row per pulizia
    def _convert_legacy_bankroll(self, bankroll_data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _phase_data_validation(self):
        """Phase 3: Validazione dati estratti"""
        self.current_migration.current_phase = MigrationPhase.DATA_VALIDATION

        try:
            extracted_data = self.current_migration.metadata.get("extracted_data", {})

            # Validazione scommesse
            bets = extracted_data.get("bets", [])
            validated_bets = []

            for i, bet in enumerate(bets):
                try:
                    # Validazione campi richiesti
                    if not bet.bet_id or not bet.game_id:
                        raise ValidationError(
                            f"Invalid bet at index {i}: missing required fields"
                        )

                    # Validazione valori
                    if bet.stake <= 0 or bet.odds <= 0:
                        raise ValidationError(
                            f"Invalid bet values at index {i}: stake={bet.stake}, odds={bet.odds}"
                        )

                    # Validazione date
                    if bet.placed_timestamp and bet.placed_timestamp > datetime.now(
                        timezone.utc
                    ):
                        raise ValidationError(
                            f"Future timestamp at index {i}: {bet.placed_timestamp}"
                        )

                    validated_bets.append(bet)

                except ValidationError as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "bet_validation",
                            "index": i,
                            "bet_id": bet.bet_id,
                            "error": str(e),
                        }
                    )
                except Exception as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "bet_processing_error",
                            "index": i,
                            "bet_id": bet.bet_id,
                            "error": str(e),
                        }
                    )

            # Validazione transazioni
            transactions = extracted_data.get("transactions", [])
            validated_txns = []

            for i, txn in enumerate(transactions):
                try:
                    if not txn.get("transaction_id") or not txn.get("type"):
                        raise ValidationError(f"Invalid transaction at index {i}")

                    validated_txns.append(txn)

                except ValidationError as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "transaction_validation",
                            "index": i,
                            "transaction_id": txn.get("transaction_id", ""),
                            "error": str(e),
                        }
                    )
                except Exception as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "transaction_processing_error",
                            "index": i,
                            "transaction_id": txn.get("transaction_id", ""),
                            "error": str(e),
                        }
                    )

            # Aggiorna dati validati
            self.current_migration.metadata["validated_bets"] = validated_bets
            self.current_migration.metadata["validated_transactions"] = validated_txns

            validation_summary = {
                "total_bets": len(bets),
                "valid_bets": len(validated_bets),
                "invalid_bets": len(bets) - len(validated_bets),
                "total_transactions": len(transactions),
                "valid_transactions": len(validated_txns),
                "invalid_transactions": len(transactions) - len(validated_txns),
            }

            self.current_migration.warnings.append(
                f"Validation completed: {validation_summary}"
            )

            logger.info(f"Data validation completed: {validation_summary}")

        except Exception as e:
            raise MigrationError(
                "Data validation phase failed",
                "VALIDATION_FAILED",
                self.current_migration.processed_records,
                self.current_migration.failed_records,
                str(e),
            )

    def _phase_data_transformation(self):
        """Phase 4: Trasformazione dati nel nuovo formato"""
        self.current_migration.current_phase = MigrationPhase.DATA_TRANSFORMATION

        try:
            validated_bets = self.current_migration.metadata.get("validated_bets", [])
            validated_txns = self.current_migration.metadata.get(
                "validated_transactions", []
            )

            # Trasforma scommesse
            transformed_bets = []
            for i, legacy_bet in enumerate(validated_bets):
                try:
                    # Mappa risultato legacy a nuovo enum
                    result_mapping = {
                        "WON": BetResult.WON,
                        "LOST": BetResult.LOST,
                        "PUSH": BetResult.PUSH,
                        "PENDING": BetResult.PENDING,
                        "CANCELLED": BetResult.CANCELLED,
                        "VOID": BetResult.VOID,
                    }

                    new_result = result_mapping.get(
                        legacy_bet.result, BetResult.PENDING
                    )

                    # Crea nuovo record
                    new_bet = create_bet_record(
                        bet_id=legacy_bet.bet_id,
                        game_id=legacy_bet.game_id,
                        bet_type=legacy_bet.bet_type,
                        selection=legacy_bet.selection,
                        odds=legacy_bet.odds,
                        stake=legacy_bet.stake,
                        result=new_result,
                        payout=legacy_bet.payout or Decimal("0.00"),
                        profit_loss=legacy_bet.profit_loss or Decimal("0.00"),
                        placed_timestamp=legacy_bet.placed_timestamp
                        or datetime.now(timezone.utc),
                        settled_timestamp=legacy_bet.settled_timestamp,
                        metadata=legacy_bet.metadata,
                    )

                    transformed_bets.append(new_bet)
                    self.current_migration.processed_records += 1

                except Exception as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "bet_transformation",
                            "index": i,
                            "bet_id": legacy_bet.bet_id,
                            "error": str(e),
                        }
                    )

            # Trasforma transazioni
            transformed_txns = []

            # --- 1. Process Initial Deposit / Legacy Transactions ---
            # Trova il deposito iniziale (legacy import)
            running_balance = Decimal("0.00")
            initial_deposit_txn = None

            # Ordina transazioni legacy e convertile
            # (Nel nostro caso abbiamo solo il deposito sintetico o transazioni se estratte)
            sorted_legacy_txns = sorted(
                validated_txns,
                key=lambda x: datetime.fromisoformat(x.get("timestamp"))
                if x.get("timestamp")
                else datetime.min.replace(tzinfo=timezone.utc),
            )

            for i, legacy_txn in enumerate(sorted_legacy_txns):
                try:
                    # Mappa tipo transazione
                    type_mapping = {
                        "deposit": TransactionType.DEPOSIT,
                        "withdrawal": TransactionType.WITHDRAWAL,
                        "bet": TransactionType.BET_PLACEMENT,
                        "settlement": TransactionType.BET_SETTLEMENT,
                        "fee": TransactionType.FEE,
                    }

                    new_type = type_mapping.get(
                        legacy_txn.get("type", ""), TransactionType.ADJUSTMENT
                    )

                    amount = Decimal(str(legacy_txn.get("amount", 0)))

                    # Aggiorna running balance logicamente (se è il deposito iniziale)
                    # Nota: legacy_txn potrebbe avere 'balance' già settato, ma noi ricalcoliamo per coerenza
                    # con le transazioni di scommessa che aggiungeremo
                    running_balance += amount

                    # Crea nuova transazione
                    new_txn = Transaction(
                        transaction_id=legacy_txn.get(
                            "transaction_id", create_transaction_id()
                        ),
                        timestamp=datetime.fromisoformat(legacy_txn["timestamp"])
                        if legacy_txn.get("timestamp")
                        else datetime.now(timezone.utc),
                        transaction_type=new_type,
                        amount=amount,
                        balance_after=running_balance,
                        description=legacy_txn.get("description", ""),
                        metadata=legacy_txn.get("metadata", {}),
                    )

                    transformed_txns.append(new_txn)
                    self.current_migration.processed_records += 1

                except Exception as e:
                    self.current_migration.failed_records += 1
                    self.current_migration.errors.append(
                        {
                            "type": "transaction_transformation",
                            "index": i,
                            "transaction_id": legacy_txn.get("transaction_id", ""),
                            "error": str(e),
                        }
                    )

            # --- 2. Generate Placement Transactions for Bets ---
            # Ora aggiungiamo le transazioni per le scommesse migrate

            # Ordina bet trasformate per data
            transformed_bets.sort(key=lambda b: b.placed_timestamp)

            logger.info(f"DEBUG: Transaction class is {Transaction}")

            for bet in transformed_bets:
                # Per ogni bet, crea una transazione di piazzamento
                loss_amount = -abs(bet.stake)

                # 1. Placement Transaction
                try:
                    logger.info(
                        f"DEBUG: Creating PLACEMENT txn for bet {bet.bet_id}, amount={loss_amount}"
                    )
                    placement_txn = Transaction(
                        transaction_id=create_transaction_id(),
                        timestamp=bet.placed_timestamp,
                        transaction_type=TransactionType.BET_PLACEMENT,
                        amount=loss_amount,
                        balance_after=Decimal("0.00"),  # Will be recalculated
                        description=f"Migration: Place bet {bet.bet_id}",
                        bet_id=bet.bet_id,
                        game_id=bet.game_id,
                        metadata={"migration_generated": True},
                    )
                    transformed_txns.append(placement_txn)
                except Exception as e:
                    logger.error(f"DEBUG: Failed to create placement txn: {e}")
                    raise

                # 2. Settlement Transaction (if payout > 0)
                # Only WON/PUSH (or partial loss returning stake) generate a credit transaction
                if bet.payout and bet.payout > 0 and bet.settled_timestamp:
                    try:
                        logger.info(
                            f"DEBUG: Creating SETTLEMENT txn for bet {bet.bet_id}, amount={bet.payout}"
                        )
                        settlement_txn = Transaction(
                            transaction_id=create_transaction_id(),
                            timestamp=bet.settled_timestamp,
                            transaction_type=TransactionType.BET_SETTLEMENT,
                            amount=bet.payout,
                            balance_after=Decimal("0.00"),  # Will be recalculated
                            description=f"Migration: Payout for {bet.bet_id} ({bet.result})",
                            bet_id=bet.bet_id,
                            game_id=bet.game_id,
                            metadata={"migration_generated": True},
                        )
                        transformed_txns.append(settlement_txn)
                    except Exception as e:
                        logger.error(f"DEBUG: Failed to create settlement txn: {e}")
                        raise

            # Riordina tutte le transazioni per timestamp per sicurezza finale
            transformed_txns.sort(key=lambda t: t.timestamp)

            # Ricalcola balance_after sequenzialmente per garantire coerenza perfetta
            # (In caso l'ordinamento misto abbia sballato il running_balance calcolato in step separati)
            # Transaction è frozen, quindi dobbiamo usare replace
            from dataclasses import replace

            recalc_balance = Decimal("0.00")
            final_txns = []

            logger.info("DEBUG: Recalculating balances...")
            for txn in transformed_txns:
                recalc_balance += txn.amount
                try:
                    # Usa replace per oggetti frozen, ricalcola checksum automaticamente in post_init?
                    # Transaction.__post_init__ viene chiamato automaticamente da replace in Python 3.7+?
                    # No, replace non chiama __post_init__ automaticamente sempre.
                    # Ma nel nostro caso models.py __post_init__ calcola checksum.
                    # Dobbiamo assicurarci che il checksum sia aggiornato.
                    # dataclasses.replace chiama __init__ e poi __post_init__ se init=True.
                    # Transaction ha parametri standard, quindi dovrebbe funzionare.

                    new_txn = replace(txn, balance_after=recalc_balance)

                    # Force calling post_init if replace doesn't?
                    # Standard replace calls __init__ on a generated new object, triggering post_init.
                    # Verifichiamo.
                    if hasattr(new_txn, "__post_init__"):
                        new_txn.__post_init__()

                    final_txns.append(new_txn)
                except Exception as e:
                    logger.error(
                        f"DEBUG: Failed to replace txn {txn.transaction_id}: {e}"
                    )
                    raise

            # Salva dati trasformati
            self.current_migration.metadata["transformed_bets"] = transformed_bets
            self.current_migration.metadata["transformed_transactions"] = final_txns

            transformation_summary = {
                "transformed_bets": len(transformed_bets),
                "transformed_transactions": len(final_txns),
                "transformation_errors": len(transformed_bets)
                + len(final_txns)
                - self.current_migration.processed_records,
            }

            self.current_migration.warnings.append(
                f"Data transformation completed: {transformation_summary}"
            )

            logger.info(f"Data transformation completed: {transformation_summary}")

        except Exception as e:
            raise MigrationError(
                "Data transformation phase failed",
                "TRANSFORMATION_FAILED",
                self.current_migration.processed_records,
                self.current_migration.failed_records,
                str(e),
            )

    def _phase_ledger_population(self):
        """Phase 5: Popolazione nuovo ledger con dati trasformati"""
        self.current_migration.current_phase = MigrationPhase.LEDGER_POPULATION

        try:
            new_engine = TransactionEngine(self.new_db_path)
            transformed_bets = self.current_migration.metadata.get(
                "transformed_bets", []
            )
            transformed_txns = self.current_migration.metadata.get(
                "transformed_transactions", []
            )

            # Popola scommesse in batch
            self._populate_bets_in_batches(new_engine, transformed_bets)

            # Popola transazioni in batch
            self._populate_transactions_in_batches(new_engine, transformed_txns)

            population_summary = {
                "bets_populated": len(transformed_bets),
                "transactions_populated": len(transformed_txns),
                "total_batches": (len(transformed_bets) + len(transformed_txns))
                // self.batch_size
                + 1,
            }

            self.current_migration.warnings.append(
                f"Ledger population completed: {population_summary}"
            )

            logger.info(f"Ledger population completed: {population_summary}")

        except Exception as e:
            raise MigrationError(
                "Ledger population phase failed",
                "POPULATION_FAILED",
                self.current_migration.processed_records,
                self.current_migration.failed_records,
                str(e),
            )

    def _populate_bets_in_batches(
        self, engine: TransactionEngine, bets: List[BetRecord]
    ):
        """Popola scommesse in batch per performance"""
        total_bets = len(bets)

        for i in range(0, total_bets, self.batch_size):
            batch = bets[i : i + self.batch_size]

            try:
                for bet in batch:
                    engine._save_bet_record(bet)

                logger.debug(
                    f"Populated bet batch {i // self.batch_size + 1}: {len(batch)} bets"
                )

            except Exception as e:
                logger.error(
                    f"Failed to populate bet batch {i // self.batch_size + 1}: {e}"
                )
                self.current_migration.failed_records += len(batch)
                self.current_migration.errors.append(
                    {
                        "type": "bet_population_batch",
                        "batch": i // self.batch_size + 1,
                        "error": str(e),
                    }
                )

    def _populate_transactions_in_batches(
        self, engine: TransactionEngine, transactions: List[Transaction]
    ):
        """Popola transazioni in batch per performance"""
        total_txns = len(transactions)

        for i in range(0, total_txns, self.batch_size):
            batch = transactions[i : i + self.batch_size]

            try:
                for txn in batch:
                    engine._execute_transaction(txn)

                logger.debug(
                    f"Populated transaction batch {i // self.batch_size + 1}: {len(batch)} transactions"
                )

            except Exception as e:
                logger.error(
                    f"Failed to populate transaction batch {i // self.batch_size + 1}: {e}"
                )
                self.current_migration.failed_records += len(batch)
                self.current_migration.errors.append(
                    {
                        "type": "transaction_population_batch",
                        "batch": i // self.batch_size + 1,
                        "error": str(e),
                    }
                )

    def _phase_verification(self):
        """Phase 6: Verifica integrità dati migrati"""
        self.current_migration.current_phase = MigrationPhase.VERIFICATION

        try:
            new_engine = TransactionEngine(self.new_db_path)

            # Verifica conteggio record
            expected_bets = len(
                self.current_migration.metadata.get("transformed_bets", [])
            )
            expected_txns = len(
                self.current_migration.metadata.get("transformed_transactions", [])
            )

            actual_bets = len(new_engine.get_bet_records(limit=100000))
            actual_txns = new_engine.get_transaction_count()

            verification_results = {
                "expected_bets": expected_bets,
                "actual_bets": actual_bets,
                "expected_transactions": expected_txns,
                "actual_transactions": actual_txns,
                "bet_mismatch": expected_bets - actual_bets,
                "transaction_mismatch": expected_txns - actual_txns,
            }

            if (
                verification_results["bet_mismatch"] != 0
                or verification_results["transaction_mismatch"] != 0
            ):
                self.current_migration.warnings.append(
                    f"Record count mismatch detected: {verification_results}"
                )

            # Verifica coerenza balance
            try:
                current_state = self.state_calculator.get_current_state()
                calculated_balance = self._calculate_balance_from_migrated_data()

                balance_diff = abs(current_state.current_balance - calculated_balance)
                if balance_diff > Decimal("0.01"):  # Tolleranza 1 cent
                    self.current_migration.warnings.append(
                        f"Balance mismatch: {balance_diff}"
                    )

            except Exception as e:
                self.current_migration.warnings.append(f"Could not verify balance: {e}")

            self.current_migration.warnings.append(
                f"Verification completed: {verification_results}"
            )

            logger.info(f"Verification completed: {verification_results}")

        except Exception as e:
            raise MigrationError(
                "Verification phase failed",
                "VERIFICATION_FAILED",
                self.current_migration.processed_records,
                self.current_migration.failed_records,
                str(e),
            )

    def _calculate_balance_from_migrated_data(self) -> Decimal:
        """Calcola balance atteso da dati migrati"""
        transformed_txns = self.current_migration.metadata.get(
            "transformed_transactions", []
        )

        balance = Decimal("0.00")
        for txn in transformed_txns:
            balance += txn.amount

        return balance

    def _phase_cleanup(self):
        """Phase 7: Cleanup temporaneo e finalizzazione"""
        self.current_migration.current_phase = MigrationPhase.CLEANUP

        try:
            # Pulizia dati temporanei
            cleanup_keys = [
                "extracted_data",
                "validated_bets",
                "validated_transactions",
                "transformed_bets",
                "transformed_transactions",
            ]

            for key in cleanup_keys:
                if key in self.current_migration.metadata:
                    del self.current_migration.metadata[key]

            # Calcola statistiche finali
            success_rate = (
                (
                    self.current_migration.processed_records
                    / max(self.current_migration.total_records, 1)
                    * 100
                )
                if self.current_migration.total_records > 0
                else 0
            )

            final_summary = {
                "total_records": self.current_migration.total_records,
                "processed_records": self.current_migration.processed_records,
                "failed_records": self.current_migration.failed_records,
                "success_rate": success_rate,
                "total_warnings": len(self.current_migration.warnings),
                "total_errors": len(self.current_migration.errors),
                "migration_duration_seconds": (
                    datetime.now(timezone.utc) - self.current_migration.start_time
                ).total_seconds(),
            }

            self.current_migration.metadata["final_summary"] = final_summary
            self.current_migration.warnings.append(
                f"Cleanup completed: {final_summary}"
            )

            logger.info(f"Migration cleanup completed: {final_summary}")

        except Exception as e:
            raise MigrationError(
                "Cleanup phase failed",
                "CLEANUP_FAILED",
                self.current_migration.processed_records,
                self.current_migration.failed_records,
                str(e),
            )

    def _is_critical_error(self, error: Exception) -> bool:
        """Determina se errore richiede rollback"""
        critical_errors = [
            DataCorruptionError,
            ChecksumMismatchError,
            ConnectionError,
            PermissionError,
        ]

        return any(isinstance(error, error_type) for error_type in critical_errors)

    def _rollback_migration(self):
        """Rollback migrazione in caso di errore critico"""
        try:
            if self.current_migration.backup_path and os.path.exists(
                self.current_migration.backup_path
            ):
                # Ripristina backup
                if os.path.exists(self.new_db_path):
                    os.remove(self.new_db_path)

                shutil.copy2(self.current_migration.backup_path, self.new_db_path)

                self.current_migration.warnings.append(
                    f"Rollback completed from backup: {self.current_migration.backup_path}"
                )

                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        f"Migration rollback completed: {self.current_migration.migration_id}",
                        AuditSeverity.WARNING,
                        additional_data={
                            "migration_id": self.current_migration.migration_id,
                            "backup_path": self.current_migration.backup_path,
                        },
                    )

                logger.warning("Migration rollback completed successfully")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.current_migration.errors.append(
                {"type": "rollback_failed", "error": str(e)}
            )

    def get_migration_status(
        self, migration_id: Optional[str] = None
    ) -> Optional[MigrationResult]:
        """Restituisce stato migrazione corrente o specifica"""
        if migration_id:
            # Cerca migrazione specifica
            for migration in self.migration_history:
                if migration.migration_id == migration_id:
                    return migration
            return None

        # Restituisce migrazione corrente
        return self.current_migration

    def get_migration_history(self) -> List[MigrationResult]:
        """Restituisce storia completa migrazioni"""
        return self.migration_history.copy()

    def cancel_migration(self, reason: str) -> bool:
        """Cancella migrazione corrente"""
        if not self.current_migration:
            return False

        try:
            self.current_migration.status = MigrationStatus.PAUSED
            self.current_migration.end_time = datetime.now(timezone.utc)

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Migration cancelled: {self.current_migration.migration_id}",
                    AuditSeverity.WARNING,
                    additional_data={
                        "migration_id": self.current_migration.migration_id,
                        "cancellation_reason": reason,
                        "progress": self.current_migration.to_dict(),
                    },
                )

            logger.warning(f"Migration cancelled: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel migration: {e}")
            return False

    def export_migration_report(
        self, migration_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Esporta report dettagliato migrazione"""
        migration = self.get_migration_status(migration_id)

        if not migration:
            return {"error": "Migration not found"}

        return {
            "migration_report": migration.to_dict(),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "system_info": {
                "legacy_db_path": self.legacy_db_path,
                "new_db_path": self.new_db_path,
                "batch_size": self.batch_size,
                "max_retries": self.max_retries,
            },
        }
