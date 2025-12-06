"""
NBA Bankroll 3.0 - Audit Logger
Sistema di audit trail completo con logging strutturato
Basato su best practice da sistemi finanziari enterprise
"""

import logging
import json
import hashlib
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict

from .models import Transaction, BetRecord, BankrollState, TransactionType, BetResult
from .exceptions import AuditError, ChecksumMismatchError, IntegrityError
from .engine import TransactionEngine


logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Tipi di eventi audit"""

    TRANSACTION_CREATED = "TRANSACTION_CREATED"
    TRANSACTION_VERIFIED = "TRANSACTION_VERIFIED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    BET_PLACED = "BET_PLACED"
    BET_SETTLED = "BET_SETTLED"
    BET_CANCELLED = "BET_CANCELLED"
    STATE_CALCULATED = "STATE_CALCULATED"
    STATE_VERIFIED = "STATE_VERIFIED"
    INTEGRITY_CHECK = "INTEGRITY_CHECK"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    RISK_VALIDATION = "RISK_VALIDATION"
    SECURITY_EVENT = "SECURITY_EVENT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DATA_MIGRATION = "DATA_MIGRATION"
    RECONCILIATION = "RECONCILIATION"


class AuditSeverity(Enum):
    """Livelli severità eventi audit"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEvent:
    """Evento audit immutabile con checksum"""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    description: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    entity_type: Optional[str] = None  # 'transaction', 'bet', 'state'
    entity_id: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    checksum: str = None

    def __post_init__(self):
        """Calcola checksum dopo creazione"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calcola checksum SHA-256 per integrità evento"""
        # Ensure timestamp is consistent
        ts_str = (
            self.timestamp.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
        )

        event_data = {
            "event_id": self.event_id,
            "timestamp": ts_str,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "metadata": self.metadata or {},
        }
        # Use simple string conversion for values to avoid JSON decimal issues if present
        # But for stability, we rely on input being JSON-serializable types already
        event_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def verify_checksum(self) -> bool:
        """Verifica integrità checksum evento"""
        return self._calculate_checksum() == self.checksum
        calculated = self._calculate_checksum()
        return calculated == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per storage"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "metadata": self.metadata or {},
            "checksum": self.checksum,
        }


class AuditLogger:
    """
    Logger audit con caratteristiche enterprise:
    - Logging strutturato con checksum
    - Ricerca avanzata con filtri
    - Archiviazione automatica
    - Integrità garantita
    - Performance ottimizzata
    """

    def __init__(self, engine: TransactionEngine):
        self.engine = engine
        self._initialize_audit_tables()
        self._audit_stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "last_cleanup": datetime.now(timezone.utc),
        }

    def _initialize_audit_tables(self):
        """Inizializza tabelle audit nel database"""
        try:
            with self.engine._get_connection() as conn:
                # Tabella eventi audit
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id VARCHAR PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        event_type VARCHAR NOT NULL,
                        severity VARCHAR NOT NULL,
                        description VARCHAR NOT NULL,
                        user_id VARCHAR,
                        session_id VARCHAR,
                        ip_address VARCHAR,
                        user_agent VARCHAR,
                        entity_type VARCHAR,
                        entity_id VARCHAR,
                        old_values JSON,
                        new_values JSON,
                        metadata JSON,
                        checksum VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Crea indici separatamente per DuckDB
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entity ON audit_events(entity_type, entity_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)"
                )

                # Tabella statistiche audit
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_statistics (
                        stat_date DATE PRIMARY KEY,
                        total_events INTEGER NOT NULL,
                        events_by_type JSON NOT NULL,
                        events_by_severity JSON NOT NULL,
                        top_users JSON,
                        top_entities JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                logger.info("Audit tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audit tables: {e}")
            raise AuditError(f"Audit table initialization failed: {e}")

    def log_transaction(
        self,
        transaction: Transaction,
        event_type: AuditEventType = AuditEventType.TRANSACTION_CREATED,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """Log transazione con contesto completo"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=self._determine_severity(event_type),
            description=f"Transaction {transaction.transaction_type.value}: {transaction.description}",
            user_id=user_context.get("user_id") if user_context else None,
            session_id=user_context.get("session_id") if user_context else None,
            ip_address=user_context.get("ip_address") if user_context else None,
            user_agent=user_context.get("user_agent") if user_context else None,
            entity_type="transaction",
            entity_id=transaction.transaction_id,
            new_values=transaction.to_dict(),
            metadata={
                "amount": float(transaction.amount),
                "balance_after": float(transaction.balance_after),
            },
        )

        self._log_event(event)

    def log_bet(
        self,
        bet_record: BetRecord,
        event_type: AuditEventType = AuditEventType.BET_PLACED,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """Log scommessa con dettagli completi"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=self._determine_severity(event_type),
            description=f"Bet {event_type.value}: {bet_record.bet_type} {bet_record.selection}",
            user_id=user_context.get("user_id") if user_context else None,
            session_id=user_context.get("session_id") if user_context else None,
            ip_address=user_context.get("ip_address") if user_context else None,
            entity_type="bet",
            entity_id=bet_record.bet_id,
            new_values=bet_record.to_dict(),
            metadata={
                "game_id": bet_record.game_id,
                "stake": float(bet_record.stake),
                "odds": float(bet_record.odds),
                "result": bet_record.result.value,
            },
        )

        self._log_event(event)

    def log_state_change(
        self,
        old_state: Optional[BankrollState],
        new_state: BankrollState,
        event_type: AuditEventType = AuditEventType.STATE_CALCULATED,
    ):
        """Log cambio stato bankroll"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=self._determine_severity(event_type),
            description=f"Bankroll state {event_type.value}",
            entity_type="state",
            entity_id="current",
            old_values=old_state.to_dict() if old_state else None,
            new_values=new_state.to_dict(),
            metadata={
                "balance_change": float(new_state.current_balance)
                - float(old_state.current_balance)
                if old_state
                else 0,
                "total_transactions": self.engine.get_transaction_count(),
            },
        )

        self._log_event(event)

    def log_integrity_check(
        self, check_type: str, result: bool, details: Optional[Dict[str, Any]] = None
    ):
        """Log check integrità"""
        severity = AuditSeverity.ERROR if not result else AuditSeverity.INFO

        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.INTEGRITY_CHECK,
            severity=severity,
            description=f"Integrity check: {check_type} - {'PASSED' if result else 'FAILED'}",
            entity_type="system",
            entity_id="integrity",
            metadata={
                "check_type": check_type,
                "result": result,
                "details": details or {},
            },
        )

        self._log_event(event)

    def log_security_event(
        self,
        event_description: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        user_context: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log evento sicurezza"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            description=f"Security: {event_description}",
            user_id=user_context.get("user_id") if user_context else None,
            session_id=user_context.get("session_id") if user_context else None,
            ip_address=user_context.get("ip_address") if user_context else None,
            user_agent=user_context.get("user_agent") if user_context else None,
            entity_type="security",
            entity_id="system",
            metadata=additional_data or {},
        )

        self._log_event(event)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log errore sistema con contesto"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SYSTEM_ERROR,
            severity=AuditSeverity.ERROR,
            description=f"System error: {str(error)}",
            entity_type="system",
            entity_id="error",
            metadata={
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "context": context or {},
            },
        )

        self._log_event(event)

    def _log_event(self, event: AuditEvent):
        """Salva evento audit nel database"""
        try:
            with self.engine._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events 
                    (event_id, timestamp, event_type, severity, description,
                     user_id, session_id, ip_address, user_agent, entity_type,
                     entity_id, old_values, new_values, metadata, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        event.event_id,
                        event.timestamp,
                        event.event_type.value,
                        event.severity.value,
                        event.description,
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.entity_type,
                        event.entity_id,
                        json.dumps(event.old_values) if event.old_values else None,
                        json.dumps(event.new_values) if event.new_values else None,
                        json.dumps(event.metadata or {}),
                        event.checksum,
                    ],
                )

            # Aggiorna statistiche
            self._update_statistics(event)

            logger.debug(
                f"Audit event logged: {event.event_type.value} - {event.event_id}"
            )

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise AuditError(f"Audit logging failed: {e}")

    def _generate_event_id(self) -> str:
        """Genera ID evento univoco"""
        import uuid

        return f"audit_{uuid.uuid4().hex[:16]}_{int(datetime.now(timezone.utc).timestamp())}"

    def _determine_severity(self, event_type: AuditEventType) -> AuditSeverity:
        """Determina severità basandosi su tipo evento"""
        severity_mapping = {
            AuditEventType.TRANSACTION_CREATED: AuditSeverity.INFO,
            AuditEventType.TRANSACTION_VERIFIED: AuditSeverity.INFO,
            AuditEventType.TRANSACTION_FAILED: AuditSeverity.ERROR,
            AuditEventType.BET_PLACED: AuditSeverity.INFO,
            AuditEventType.BET_SETTLED: AuditSeverity.INFO,
            AuditEventType.BET_CANCELLED: AuditSeverity.WARNING,
            AuditEventType.STATE_CALCULATED: AuditSeverity.INFO,
            AuditEventType.STATE_VERIFIED: AuditSeverity.INFO,
            AuditEventType.INTEGRITY_CHECK: AuditSeverity.INFO,
            AuditEventType.CHECKSUM_MISMATCH: AuditSeverity.CRITICAL,
            AuditEventType.RISK_VALIDATION: AuditSeverity.WARNING,
            AuditEventType.SECURITY_EVENT: AuditSeverity.WARNING,
            AuditEventType.SYSTEM_ERROR: AuditSeverity.ERROR,
            AuditEventType.DATA_MIGRATION: AuditSeverity.INFO,
            AuditEventType.RECONCILIATION: AuditSeverity.INFO,
        }

        return severity_mapping.get(event_type, AuditSeverity.INFO)

    def _update_statistics(self, event: AuditEvent):
        """Aggiorna statistiche audit"""
        self._audit_stats["total_events"] += 1

        # Per tipo
        event_type = event.event_type.value
        self._audit_stats["events_by_type"][event_type] = (
            self._audit_stats["events_by_type"].get(event_type, 0) + 1
        )

        # Per severità
        severity = event.severity.value
        self._audit_stats["events_by_severity"][severity] = (
            self._audit_stats["events_by_severity"].get(severity, 0) + 1
        )

    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Ricerca avanzata eventi audit con filtri multipli
        Supporta ricerca combinata per investigazioni
        """
        try:
            with self.engine._get_connection() as conn:
                # Costruisci query dinamica
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)

                if severity:
                    query += " AND severity = ?"
                    params.append(severity.value)

                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type)

                if entity_id:
                    query += " AND entity_id = ?"
                    params.append(entity_id)

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                results = conn.execute(query, params).fetchall()
                return [self._row_to_audit_event(row) for row in results]

        except Exception as e:
            logger.error(f"Audit search failed: {e}")
            raise AuditError(f"Audit search failed: {e}")

    def _row_to_audit_event(self, row) -> AuditEvent:
        """Converte row database in AuditEvent"""
        return AuditEvent(
            event_id=row[0],
            timestamp=datetime.fromisoformat(str(row[1])),
            event_type=AuditEventType(row[2]),
            severity=AuditSeverity(row[3]),
            description=row[4],
            user_id=row[5],
            session_id=row[6],
            ip_address=row[7],
            user_agent=row[8],
            entity_type=row[9],
            entity_id=row[10],
            old_values=json.loads(row[11]) if row[11] else None,
            new_values=json.loads(row[12]) if row[12] else None,
            metadata=json.loads(row[13]) if row[13] else None,
            checksum=row[14],
        )

    def get_audit_trail(self, entity_id: str, entity_type: str) -> List[AuditEvent]:
        """
        Recupera audit trail completo per entità
        Utile per tracciamento completo modifiche
        """
        return self.search_events(
            entity_type=entity_type, entity_id=entity_id, limit=1000
        )

    def verify_audit_integrity(self) -> Dict[str, Any]:
        """
        Verifica integrità completa sistema audit
        Include checksum verification e anomaly detection
        """
        try:
            integrity_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events_checked": 0,
                "checksum_mismatches": 0,
                "orphaned_events": 0,
                "sequence_gaps": [],
                "overall_status": "PASSED",
            }

            # Recupera tutti eventi per verifica
            all_events = self.search_events(limit=10000)
            integrity_results["total_events_checked"] = len(all_events)

            # Verifica checksum eventi
            for event in all_events:
                if not event.verify_checksum():
                    integrity_results["checksum_mismatches"] += 1
                    self.log_checksum_mismatch(event.event_id, event.checksum)

            # Verifica sequenza temporale
            timestamps = [event.timestamp for event in all_events]
            timestamps.sort()

            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i - 1]:
                    integrity_results["sequence_gaps"].append(
                        {
                            "position": i,
                            "timestamp": timestamps[i].isoformat(),
                            "previous_timestamp": timestamps[i - 1].isoformat(),
                        }
                    )

            # Determina stato complessivo
            if (
                integrity_results["checksum_mismatches"] > 0
                or len(integrity_results["sequence_gaps"]) > 0
            ):
                integrity_results["overall_status"] = "FAILED"

            # Log risultato verifica
            self.log_integrity_check(
                "audit_integrity_verification",
                integrity_results["overall_status"] == "PASSED",
                integrity_results,
            )

            return integrity_results

        except Exception as e:
            logger.error(f"Audit integrity verification failed: {e}")
            raise AuditError(f"Integrity verification failed: {e}")

    def log_checksum_mismatch(self, event_id: str, expected_checksum: str):
        """Log specifico per checksum mismatch"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.CHECKSUM_MISMATCH,
            severity=AuditSeverity.CRITICAL,
            description=f"Checksum mismatch detected for event {event_id}",
            entity_type="audit_event",
            entity_id=event_id,
            metadata={
                "expected_checksum": expected_checksum,
                "issue_type": "checksum_corruption",
            },
        )

        self._log_event(event)

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche complete audit"""
        return {
            "total_events": self._audit_stats["total_events"],
            "events_by_type": self._audit_stats["events_by_type"],
            "events_by_severity": self._audit_stats["events_by_severity"],
            "last_cleanup": self._audit_stats["last_cleanup"].isoformat(),
            "database_size": self._get_database_size(),
        }

    def _get_database_size(self) -> Dict[str, Any]:
        """Calcola dimensione database audit"""
        try:
            with self.engine._get_connection() as conn:
                # Conta eventi
                count_result = conn.execute("""
                    SELECT COUNT(*) FROM audit_events
                """).fetchone()

                # Dimensione tabella (stimata)
                size_result = conn.execute("""
                    SELECT COUNT(*) * AVG(LENGTH(event_id || timestamp || event_type || 
                                   severity || description || COALESCE(user_id, '') || 
                                   COALESCE(metadata, '') || checksum)) as estimated_size
                    FROM audit_events
                """).fetchone()

                return {
                    "total_events": count_result[0],
                    "estimated_size_bytes": int(size_result[0])
                    if size_result[0]
                    else 0,
                    "estimated_size_mb": int(size_result[0]) / (1024 * 1024)
                    if size_result[0]
                    else 0,
                }

        except Exception as e:
            logger.warning(f"Failed to calculate database size: {e}")
            return {"error": str(e)}

    def cleanup_old_events(self, days_to_keep: int = 90):
        """
        Cleanup eventi vecchi per mantenere performance
        Mantiene eventi recenti per compliance
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            with self.engine._get_connection() as conn:
                # Conta eventi da cancellare
                count_result = conn.execute(
                    """
                    SELECT COUNT(*) FROM audit_events 
                    WHERE timestamp < ?
                """,
                    [cutoff_date],
                ).fetchone()

                events_to_delete = count_result[0]

                if events_to_delete > 0:
                    # Esegui cancellazione
                    conn.execute(
                        """
                        DELETE FROM audit_events 
                        WHERE timestamp < ?
                    """,
                        [cutoff_date],
                    )

                    # Aggiorna statistiche
                    self._audit_stats["last_cleanup"] = datetime.now(timezone.utc)

                    logger.info(
                        f"Cleaned up {events_to_delete} old audit events (older than {days_to_keep} days)"
                    )

                    # Log cleanup
                    self.log_integrity_check(
                        "audit_cleanup",
                        True,
                        {
                            "events_deleted": events_to_delete,
                            "cutoff_date": cutoff_date.isoformat(),
                            "days_kept": days_to_keep,
                        },
                    )

                return {
                    "events_deleted": events_to_delete,
                    "cutoff_date": cutoff_date.isoformat(),
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Audit cleanup failed: {e}")
            return {"events_deleted": 0, "error": str(e), "success": False}

    def export_audit_trail(
        self, output_format: str = "json", filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Esporta audit trail in vari formati
        Supporta JSON, CSV per compliance e reporting
        """
        try:
            # Applica filtri se forniti
            events = self.search_events(
                event_type=filters.get("event_type") if filters else None,
                severity=filters.get("severity") if filters else None,
                start_time=filters.get("start_time") if filters else None,
                end_time=filters.get("end_time") if filters else None,
                limit=filters.get("limit", 1000) if filters else 1000,
            )

            if output_format.lower() == "json":
                return json.dumps(
                    [event.to_dict() for event in events], indent=2, default=str
                )

            elif output_format.lower() == "csv":
                import csv
                import io

                output = io.StringIO()
                if events:
                    fieldnames = events[0].to_dict().keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())

                return output.getvalue()

            else:
                raise ValueError(f"Unsupported export format: {output_format}")

        except Exception as e:
            logger.error(f"Audit export failed: {e}")
            raise AuditError(f"Export failed: {e}")
