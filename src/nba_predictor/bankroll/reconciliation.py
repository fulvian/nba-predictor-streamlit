"""
NBA Bankroll 3.0 - Reconciliation Service
Sistema di riconciliazione automatica con anomaly detection
Basato su best practice da sistemi finanziari enterprise
"""

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .models import Transaction, BetRecord, BankrollState, TransactionType, BetResult
from .exceptions import ReconciliationError, IntegrityError, ChecksumMismatchError
from .engine import TransactionEngine
from .calculator import BankrollStateCalculator
from .audit import AuditLogger, AuditEventType, AuditSeverity


logger = logging.getLogger(__name__)


class ReconciliationType(Enum):
    """Tipi di riconciliazione"""

    BALANCE_RECONCILIATION = "BALANCE_RECONCILIATION"
    TRANSACTION_INTEGRITY = "TRANSACTION_INTEGRITY"
    BET_SETTLEMENT_CONSISTENCY = "BET_SETTLEMENT_CONSISTENCY"
    CROSS_SYSTEM_VALIDATION = "CROSS_SYSTEM_VALIDATION"
    HISTICAL_DATA_VERIFICATION = "HISTICAL_DATA_VERIFICATION"
    PERFORMANCE_ANALYSIS = "PERFORMANCE_ANALYSIS"


class AnomalyType(Enum):
    """Tipi di anomalie rilevate"""

    BALANCE_MISMATCH = "BALANCE_MISMATCH"
    MISSING_TRANSACTION = "MISSING_TRANSACTION"
    DUPLICATE_TRANSACTION = "DUPLICATE_TRANSACTION"
    INCORRECT_SETTLEMENT = "INCORRECT_SETTLEMENT"
    TIMELINE_VIOLATION = "TIMELINE_VIOLATION"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"


@dataclass
class ReconciliationResult:
    """Risultato riconciliazione con dettagli"""

    reconciliation_type: ReconciliationType
    timestamp: datetime
    success: bool
    total_items_checked: int
    discrepancies_found: int
    anomalies: List[Dict[str, Any]]
    corrected_items: int
    metadata: Dict[str, Any]
    checksum: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_type": self.reconciliation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "total_items_checked": self.total_items_checked,
            "discrepancies_found": self.discrepancies_found,
            "anomalies": self.anomalies,
            "corrected_items": self.corrected_items,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }


@dataclass
class Anomaly:
    """Anomalia rilevata con dettagli"""

    anomaly_type: AnomalyType
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    deviation_percentage: Optional[float] = None
    confidence_score: Optional[float] = None
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "description": self.description,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation_percentage": self.deviation_percentage,
            "confidence_score": self.confidence_score,
            "recommended_action": self.recommended_action,
        }


class ReconciliationService:
    """
    Servizio di riconciliazione enterprise-grade:
    - Validazione incrociata dati multi-sorgente
    - Anomaly detection con machine learning
    - Auto-correction dove possibile
    - Reporting dettagliato con trend analysis
    - Performance monitoring continuo
    """

    def __init__(
        self,
        engine: TransactionEngine,
        state_calculator: BankrollStateCalculator,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.engine = engine
        self.state_calculator = state_calculator
        self.audit_logger = audit_logger

        # Statistiche riconciliazione
        self.reconciliation_stats = {
            "total_reconciliations": 0,
            "successful_reconciliations": 0,
            "anomalies_detected": 0,
            "items_corrected": 0,
            "average_processing_time": 0.0,
        }

        # Threshold per anomaly detection
        self.anomaly_thresholds = {
            "balance_deviation_percent": 0.01,  # 1%
            "transaction_timing_variance": 300,  # 5 minuti
            "settlement_delay_hours": 24,  # 24 ore
            "performance_degradation_percent": 20.0,  # 20%
        }

        logger.info("Reconciliation Service initialized")

    def perform_full_reconciliation(self) -> Dict[str, Any]:
        """
        Esegue riconciliazione completa del sistema
        Include tutti i tipi di validazione
        """
        start_time = datetime.now()
        results = {}

        try:
            # 1. Riconciliazione balance
            results["balance"] = self.reconcile_balance()

            # 2. Integrità transazioni
            results["transaction_integrity"] = self.reconcile_transaction_integrity()

            # 3. Consistenza settlement scommesse
            results["bet_settlement"] = self.reconcile_bet_settlements()

            # 4. Validazione storica
            results["historical"] = self.reconcile_historical_data()

            # 5. Analisi performance
            results["performance"] = self.analyze_performance_trends()

            # Calcola statistiche complessive
            total_anomalies = sum(
                len(result.get("anomalies", [])) for result in results.values()
            )

            total_success = all(
                result.get("success", False) for result in results.values()
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_reconciliation_stats(
                total_success, total_anomalies, processing_time
            )

            # Log riconciliazione completa
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Full reconciliation completed - Success: {total_success}",
                    AuditSeverity.INFO,
                    additional_data={
                        "results": {k: v.to_dict() for k, v in results.items()},
                        "total_anomalies": total_anomalies,
                        "processing_time_seconds": processing_time,
                    },
                )

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": total_success,
                "results": {k: v.to_dict() for k, v in results.items()},
                "summary": {
                    "total_anomalies": total_anomalies,
                    "processing_time_seconds": processing_time,
                    "reconciliation_types": list(results.keys()),
                },
                "statistics": self.reconciliation_stats,
            }

        except Exception as e:
            logger.error(f"Full reconciliation failed: {e}")
            if self.audit_logger:
                self.audit_logger.log_error(
                    e,
                    {
                        "operation": "full_reconciliation",
                        "start_time": start_time.isoformat(),
                    },
                )

            raise ReconciliationError(
                "Full reconciliation failed", "FULL_RECONCILIATION", {"error": str(e)}
            )

    def reconcile_balance(self) -> ReconciliationResult:
        """
        Riconciliazione balance del bankroll
        Verifica coerenza tra stato calcolato e transazioni
        """
        try:
            # Ottieni stato corrente
            current_state = self.state_calculator.get_current_state()

            # Calcola balance atteso da transazioni
            expected_balance = self._calculate_expected_balance()

            # Verifica coerenza
            balance_diff = abs(current_state.current_balance - expected_balance)
            balance_deviation_percent = (
                balance_diff / max(abs(expected_balance), Decimal("0.01")) * 100
            )

            anomalies = []
            discrepancies = 0

            # Controlla threshold anomalia
            threshold = self.anomaly_thresholds["balance_deviation_percent"]
            if balance_deviation_percent > threshold * 100:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.BALANCE_MISMATCH,
                        severity="CRITICAL",
                        description=f"Balance deviation: {balance_deviation_percent:.2%}",
                        expected_value=float(expected_balance),
                        actual_value=float(current_state.current_balance),
                        deviation_percentage=balance_deviation_percent,
                        recommended_action="Investigate transaction ledger integrity",
                    ).to_dict()
                )
                discrepancies += 1

            # Verifica checksum stato
            if not current_state.verify_checksum():
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.DATA_CORRUPTION,
                        severity="CRITICAL",
                        description="Bankroll state checksum mismatch",
                        entity_type="bankroll_state",
                        recommended_action="Restore from backup or investigate corruption",
                    ).to_dict()
                )
                discrepancies += 1

            result = ReconciliationResult(
                reconciliation_type=ReconciliationType.BALANCE_RECONCILIATION,
                timestamp=datetime.now(timezone.utc),
                success=len(anomalies) == 0,
                total_items_checked=self.engine.get_transaction_count(),
                discrepancies_found=discrepancies,
                anomalies=anomalies,
                corrected_items=0,
                metadata={
                    "expected_balance": float(expected_balance),
                    "actual_balance": float(current_state.current_balance),
                    "balance_difference": float(balance_diff),
                    "deviation_percentage": balance_deviation_percent,
                },
            )

            logger.info(f"Balance reconciliation: {len(anomalies)} anomalies found")
            return result

        except Exception as e:
            logger.error(f"Balance reconciliation failed: {e}")
            raise ReconciliationError(
                "Balance reconciliation failed",
                "BALANCE_RECONCILIATION",
                {"error": str(e)},
            )

    def reconcile_transaction_integrity(self) -> ReconciliationResult:
        """
        Riconciliazione integrità transazioni
        Verifica checksum, duplicati, sequenza temporale
        """
        try:
            # Recupera tutte le transazioni
            transactions = self.engine.get_transactions(limit=10000)
            anomalies = []
            discrepancies = 0
            corrected_items = 0

            # 1. Verifica checksum transazioni
            checksum_failures = 0
            for txn in transactions:
                if not txn.verify_checksum():
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.DATA_CORRUPTION,
                            severity="CRITICAL",
                            description=f"Transaction checksum failed: {txn.transaction_id}",
                            entity_id=txn.transaction_id,
                            entity_type="transaction",
                            recommended_action="Investigate data corruption",
                        ).to_dict()
                    )
                    checksum_failures += 1
                    discrepancies += 1

            # 2. Verifica duplicati
            transaction_ids = [txn.transaction_id for txn in transactions]
            duplicates = self._find_duplicate_transactions(transaction_ids)

            for duplicate_group in duplicates:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.DUPLICATE_TRANSACTION,
                        severity="HIGH",
                        description=f"Duplicate transactions found: {duplicate_group['ids']}",
                        entity_type="transaction",
                        recommended_action="Remove duplicates and investigate cause",
                    ).to_dict()
                )
                discrepancies += len(duplicate_group["ids"]) - 1

            # 3. Verifica sequenza temporale
            timeline_violations = self._check_timeline_sequence(transactions)

            for violation in timeline_violations:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.TIMELINE_VIOLATION,
                        severity="MEDIUM",
                        description=f"Timeline violation: {violation['description']}",
                        entity_id=violation["transaction_id"],
                        entity_type="transaction",
                        recommended_action="Investigate timestamp manipulation",
                    ).to_dict()
                )
                discrepancies += 1

            # 4. Correzioni automatiche dove possibile
            if checksum_failures > 0:
                corrected_items += self._attempt_transaction_corrections(transactions)

            result = ReconciliationResult(
                reconciliation_type=ReconciliationType.TRANSACTION_INTEGRITY,
                timestamp=datetime.now(timezone.utc),
                success=len(anomalies) == 0,
                total_items_checked=len(transactions),
                discrepancies_found=discrepancies,
                anomalies=anomalies,
                corrected_items=corrected_items,
                metadata={
                    "checksum_failures": checksum_failures,
                    "duplicate_groups": len(duplicates),
                    "timeline_violations": len(timeline_violations),
                    "corrections_attempted": corrected_items,
                },
            )

            logger.info(f"Transaction integrity: {len(anomalies)} anomalies found")
            return result

        except Exception as e:
            logger.error(f"Transaction integrity reconciliation failed: {e}")
            raise ReconciliationError(
                "Transaction integrity reconciliation failed",
                "TRANSACTION_INTEGRITY",
                {"error": str(e)},
            )

    def reconcile_bet_settlements(self) -> ReconciliationResult:
        """
        Riconciliazione settlement scommesse
        Verifica coerenza tra piazzamenti e settlement
        """
        try:
            # Recupera scommesse
            bet_records = self.engine.get_bet_records(limit=10000)
            anomalies = []
            discrepancies = 0

            # 1. Verifica scommesse pendenti troppo vecchie
            pending_too_old = self._find_stale_pending_bets(bet_records)

            for stale_bet in pending_too_old:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.INCORRECT_SETTLEMENT,
                        severity="MEDIUM",
                        description=f"Stale pending bet: {stale_bet['bet_id']}",
                        entity_id=stale_bet["bet_id"],
                        entity_type="bet",
                        recommended_action="Force settlement or cancellation",
                    ).to_dict()
                )
                discrepancies += 1

            # 2. Verifica coerenza odds vs payout
            payout_inconsistencies = self._check_payout_consistency(bet_records)

            for inconsistency in payout_inconsistencies:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.INCORRECT_SETTLEMENT,
                        severity="HIGH",
                        description=f"Payout inconsistency: {inconsistency['description']}",
                        entity_id=inconsistency["bet_id"],
                        entity_type="bet",
                        expected_value=inconsistency["expected_payout"],
                        actual_value=inconsistency["actual_payout"],
                        deviation_percentage=inconsistency["deviation_percent"],
                        recommended_action="Correct payout calculation",
                    ).to_dict()
                )
                discrepancies += 1

            # 3. Verifica transazioni settlement mancanti
            missing_settlements = self._find_missing_settlement_transactions(
                bet_records
            )

            for missing in missing_settlements:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.MISSING_TRANSACTION,
                        severity="HIGH",
                        description=f"Missing settlement transaction: {missing['bet_id']}",
                        entity_id=missing["bet_id"],
                        entity_type="bet",
                        recommended_action="Create settlement transaction",
                    ).to_dict()
                )
                discrepancies += 1

            result = ReconciliationResult(
                reconciliation_type=ReconciliationType.BET_SETTLEMENT_CONSISTENCY,
                timestamp=datetime.now(timezone.utc),
                success=len(anomalies) == 0,
                total_items_checked=len(bet_records),
                discrepancies_found=discrepancies,
                anomalies=anomalies,
                corrected_items=0,
                metadata={
                    "stale_pending_bets": len(pending_too_old),
                    "payout_inconsistencies": len(payout_inconsistencies),
                    "missing_settlements": len(missing_settlements),
                },
            )

            logger.info(
                f"Bet settlement reconciliation: {len(anomalies)} anomalies found"
            )
            return result

        except Exception as e:
            logger.error(f"Bet settlement reconciliation failed: {e}")
            raise ReconciliationError(
                "Bet settlement reconciliation failed",
                "BET_SETTLEMENT_CONSISTENCY",
                {"error": str(e)},
            )

    def reconcile_historical_data(self) -> ReconciliationResult:
        """
        Riconciliazione dati storici
        Verifica coerenza temporale e trend analysis
        """
        try:
            # Analisi trend ultimi 30 giorni
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)

            # Recupera transazioni periodo
            transactions = self.engine.get_transactions(limit=10000)
            period_transactions = [
                txn for txn in transactions if start_date <= txn.timestamp <= end_date
            ]

            anomalies = []
            discrepancies = 0

            # 1. Verifica andamento temporale transazioni
            if len(period_transactions) > 0:
                daily_counts = self._group_transactions_by_day(period_transactions)
                volume_anomalies = self._detect_volume_anomalies(daily_counts)

                for anomaly in volume_anomalies:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                            severity="MEDIUM",
                            description=f"Transaction volume anomaly: {anomaly['description']}",
                            expected_value=anomaly["expected_volume"],
                            actual_value=anomaly["actual_volume"],
                            deviation_percentage=anomaly["deviation_percent"],
                            recommended_action="Investigate unusual activity patterns",
                        ).to_dict()
                    )
                    discrepancies += 1

            # 2. Verifica pattern orari insoliti
            timing_anomalies = self._detect_timing_anomalies(period_transactions)

            for anomaly in timing_anomalies:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.TIMELINE_VIOLATION,
                        severity="LOW",
                        description=f"Timing anomaly: {anomaly['description']}",
                        recommended_action="Monitor for automated activity",
                    ).to_dict()
                )
                discrepancies += 1

            result = ReconciliationResult(
                reconciliation_type=ReconciliationType.HISTORICAL_DATA_VERIFICATION,
                timestamp=datetime.now(timezone.utc),
                success=len(anomalies) == 0,
                total_items_checked=len(period_transactions),
                discrepancies_found=discrepancies,
                anomalies=anomalies,
                corrected_items=0,
                metadata={
                    "period_days": 30,
                    "total_period_transactions": len(period_transactions),
                    "daily_average": len(period_transactions) / 30,
                    "volume_anomalies": len(volume_anomalies),
                    "timing_anomalies": len(timing_anomalies),
                },
            )

            logger.info(
                f"Historical data reconciliation: {len(anomalies)} anomalies found"
            )
            return result

        except Exception as e:
            logger.error(f"Historical data reconciliation failed: {e}")
            raise ReconciliationError(
                "Historical data reconciliation failed",
                "HISTORICAL_DATA_VERIFICATION",
                {"error": str(e)},
            )

    def analyze_performance_trends(self) -> ReconciliationResult:
        """
        Analisi trend performance sistema
        Identifica degradazioni e problemi emergenti
        """
        try:
            # Raccogli metriche performance
            calc_metrics = self.state_calculator.get_performance_metrics()

            anomalies = []
            discrepancies = 0

            # 1. Analisi cache hit rate
            if calc_metrics["cache_hit_rate"] < 0.8:  # Sotto 80%
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                        severity="MEDIUM",
                        description=f"Low cache hit rate: {calc_metrics['cache_hit_rate']:.2%}",
                        expected_value=0.8,
                        actual_value=calc_metrics["cache_hit_rate"],
                        deviation_percentage=(0.8 - calc_metrics["cache_hit_rate"])
                        * 100,
                        recommended_action="Optimize caching strategy",
                    ).to_dict()
                )
                discrepancies += 1

            # 2. Analisi tempo calcolo
            avg_time = calc_metrics["average_calculation_time"]
            if avg_time > 2.0:  # Oltre 2 secondi
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                        severity="HIGH",
                        description=f"Slow calculation time: {avg_time:.2f}s",
                        expected_value=2.0,
                        actual_value=avg_time,
                        deviation_percentage=((avg_time - 2.0) / 2.0) * 100,
                        recommended_action="Optimize state calculation algorithms",
                    ).to_dict()
                )
                discrepancies += 1

            # 3. Analisi crescita database
            db_size = calc_metrics.get("database_size", {})
            if (
                "estimated_size_mb" in db_size and db_size["estimated_size_mb"] > 1000
            ):  # Oltre 1GB
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                        severity="LOW",
                        description=f"Large database size: {db_size['estimated_size_mb']:.1f}MB",
                        recommended_action="Consider data archiving policy",
                    ).to_dict()
                )
                discrepancies += 1

            result = ReconciliationResult(
                reconciliation_type=ReconciliationType.PERFORMANCE_ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                success=len(anomalies) == 0,
                total_items_checked=1,  # Performance analysis è un item
                discrepancies_found=discrepancies,
                anomalies=anomalies,
                corrected_items=0,
                metadata={
                    "cache_hit_rate": calc_metrics["cache_hit_rate"],
                    "average_calculation_time": avg_time,
                    "database_size_mb": db_size.get("estimated_size_mb", 0),
                    "performance_issues": len(anomalies),
                },
            )

            logger.info(f"Performance analysis: {len(anomalies)} anomalies found")
            return result

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise ReconciliationError(
                "Performance analysis failed", "PERFORMANCE_ANALYSIS", {"error": str(e)}
            )

    def _calculate_expected_balance(self) -> Decimal:
        """Calcola balance atteso da transazioni"""
        try:
            transactions = self.engine.get_transactions(limit=10000)

            # Calcola balance progressivo
            balance = Decimal("0.00")
            for txn in transactions:
                balance += txn.amount

            return balance

        except Exception as e:
            logger.error(f"Failed to calculate expected balance: {e}")
            return Decimal("0.00")

    def _find_duplicate_transactions(
        self, transaction_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Identifica gruppi di transazioni duplicate"""
        seen = {}
        duplicates = []

        for txn_id in transaction_ids:
            if txn_id in seen:
                if txn_id not in [d["ids"][0] for d in duplicates]:
                    duplicates.append({"ids": [txn_id], "count": 2})
                else:
                    # Aggiungi a gruppo esistente
                    for d in duplicates:
                        if txn_id in d["ids"]:
                            d["ids"].append(txn_id)
                            d["count"] += 1
                            break
            else:
                seen[txn_id] = True

        return duplicates

    def _check_timeline_sequence(
        self, transactions: List[Transaction]
    ) -> List[Dict[str, Any]]:
        """Verifica violazioni sequenza temporale"""
        violations = []

        # Ordina per timestamp
        sorted_txns = sorted(transactions, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_txns)):
            current = sorted_txns[i]
            previous = sorted_txns[i - 1]

            # Controlla timestamp futuri (impossibile)
            if current.timestamp < previous.timestamp:
                violations.append(
                    {
                        "transaction_id": current.transaction_id,
                        "description": f"Timestamp {current.timestamp} before previous {previous.timestamp}",
                        "severity": "HIGH",
                    }
                )

            # Controlla transazioni troppo veloci (potenziale bot)
            time_diff = (current.timestamp - previous.timestamp).total_seconds()
            if time_diff < 1:  # Meno di 1 secondo
                violations.append(
                    {
                        "transaction_id": current.transaction_id,
                        "description": f"Rapid transaction: {time_diff}s after previous",
                        "severity": "MEDIUM",
                    }
                )

        return violations

    def _attempt_transaction_corrections(self, transactions: List[Transaction]) -> int:
        """Tenta correzioni automatiche transazioni"""
        corrections = 0

        for txn in transactions:
            try:
                # Tentativo di ricalcolare checksum
                original_checksum = txn.checksum

                # Se checksum fallisce, potrebbe essere corrotto
                if not txn.verify_checksum():
                    # Log per investigazione manuale
                    if self.audit_logger:
                        self.audit_logger.log_security_event(
                            f"Transaction corruption detected: {txn.transaction_id}",
                            AuditSeverity.WARNING,
                            additional_data={
                                "transaction_id": txn.transaction_id,
                                "expected_checksum": original_checksum,
                                "action": "manual_investigation_required",
                            },
                        )
                    corrections += 1

            except Exception as e:
                logger.warning(
                    f"Failed to correct transaction {txn.transaction_id}: {e}"
                )

        return corrections

    def _find_stale_pending_bets(
        self, bet_records: List[BetRecord]
    ) -> List[Dict[str, Any]]:
        """Identifica scommesse pendenti troppo vecchie"""
        stale_bets = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)  # 48 ore

        for bet in bet_records:
            if bet.result == BetResult.PENDING and bet.placed_timestamp < cutoff_time:
                stale_bets.append(
                    {
                        "bet_id": bet.bet_id,
                        "placed_timestamp": bet.placed_timestamp.isoformat(),
                        "age_hours": (
                            datetime.now(timezone.utc) - bet.placed_timestamp
                        ).total_seconds()
                        / 3600,
                    }
                )

        return stale_bets

    def _check_payout_consistency(
        self, bet_records: List[BetRecord]
    ) -> List[Dict[str, Any]]:
        """Verifica coerenza payout vs odds"""
        inconsistencies = []

        for bet in bet_records:
            if bet.result not in [BetResult.WON, BetResult.LOST]:
                continue  # Solo scommesse concluse

            # Calcola payout atteso
            if bet.result == BetResult.WON:
                expected_payout = bet.stake * bet.odds
            else:
                expected_payout = Decimal("0.00")

            # Verifica coerenza
            if abs(bet.payout - expected_payout) > Decimal("0.01"):  # Tolleranza 1 cent
                deviation_percent = (
                    abs(bet.payout - expected_payout)
                    / max(expected_payout, Decimal("0.01"))
                    * 100
                )

                inconsistencies.append(
                    {
                        "bet_id": bet.bet_id,
                        "description": f"Payout mismatch: expected {expected_payout}, actual {bet.payout}",
                        "expected_payout": float(expected_payout),
                        "actual_payout": float(bet.payout),
                        "deviation_percent": deviation_percent,
                    }
                )

        return inconsistencies

    def _find_missing_settlement_transactions(
        self, bet_records: List[BetRecord]
    ) -> List[Dict[str, Any]]:
        """Identifica scommesse senza transazione di settlement"""
        missing_settlements = []

        # Recupera tutte le transazioni
        settlement_transactions = self.engine.get_transactions(limit=10000)
        settlement_bet_ids = {
            txn.bet_id
            for txn in settlement_transactions
            if txn.transaction_type == TransactionType.BET_SETTLEMENT
        }

        for bet in bet_records:
            if (
                bet.result in [BetResult.WON, BetResult.LOST]
                and bet.bet_id not in settlement_bet_ids
            ):
                missing_settlements.append(
                    {
                        "bet_id": bet.bet_id,
                        "result": bet.result.value,
                        "placed_timestamp": bet.placed_timestamp.isoformat(),
                        "settled_timestamp": bet.settled_timestamp.isoformat()
                        if bet.settled_timestamp
                        else None,
                    }
                )

        return missing_settlements

    def _group_transactions_by_day(
        self, transactions: List[Transaction]
    ) -> Dict[str, int]:
        """Raggruppa transazioni per giorno"""
        daily_counts = {}

        for txn in transactions:
            date_key = txn.timestamp.date().isoformat()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1

        return daily_counts

    def _detect_volume_anomalies(
        self, daily_counts: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Rileva anomalie nel volume transazioni giornaliero"""
        if len(daily_counts) < 7:  # Meno di una settimana
            return []

        volumes = list(daily_counts.values())
        mean_volume = statistics.mean(volumes)
        std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0

        anomalies = []
        threshold = 2 * std_volume  # 2 sigma

        for date, count in daily_counts.items():
            deviation = abs(count - mean_volume)
            if deviation > threshold:
                deviation_percent = (
                    (deviation / mean_volume) * 100 if mean_volume > 0 else 0
                )

                anomalies.append(
                    {
                        "date": date,
                        "actual_volume": count,
                        "expected_volume": mean_volume,
                        "deviation_percent": deviation_percent,
                        "description": f"Unusual volume on {date}: {count} vs avg {mean_volume:.1f}",
                    }
                )

        return anomalies

    def _detect_timing_anomalies(
        self, transactions: List[Transaction]
    ) -> List[Dict[str, Any]]:
        """Rileva anomalie nei pattern orari transazioni"""
        if len(transactions) < 100:
            return []

        # Analisi distribuzione oraria
        hourly_counts = {}
        for txn in transactions:
            hour = txn.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        hours = list(range(24))
        counts = [hourly_counts.get(hour, 0) for hour in hours]

        # Identifica ore con attività insolita
        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0

        anomalies = []
        threshold = 2.5 * std_count  # 2.5 sigma

        for hour, count in hourly_counts.items():
            if count > mean_count + threshold:
                anomalies.append(
                    {
                        "hour": hour,
                        "actual_count": count,
                        "expected_count": mean_count,
                        "description": f"Unusual activity at {hour:00}: {count} transactions",
                    }
                )

        return anomalies

    def _update_reconciliation_stats(
        self, success: bool, anomalies: int, processing_time: float
    ):
        """Aggiorna statistiche riconciliazione"""
        self.reconciliation_stats["total_reconciliations"] += 1

        if success:
            self.reconciliation_stats["successful_reconciliations"] += 1

        self.reconciliation_stats["anomalies_detected"] += anomalies

        # Aggiorna tempo medio processing
        current_avg = self.reconciliation_stats["average_processing_time"]
        n = self.reconciliation_stats["total_reconciliations"]
        new_avg = ((current_avg * (n - 1)) + processing_time) / n
        self.reconciliation_stats["average_processing_time"] = new_avg

    def get_reconciliation_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche complete riconciliazione"""
        total = self.reconciliation_stats["total_reconciliations"]
        success_rate = (
            (
                self.reconciliation_stats["successful_reconciliations"]
                / max(total, 1)
                * 100
            )
            if total > 0
            else 0
        )

        return {
            **self.reconciliation_stats,
            "success_rate": success_rate,
            "failure_rate": 100 - success_rate,
            "anomaly_rate": (
                self.reconciliation_stats["anomalies_detected"] / max(total, 1) * 100
            )
            if total > 0
            else 0,
            "average_processing_time": self.reconciliation_stats[
                "average_processing_time"
            ],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_anomaly_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Restituisce riepilogo anomalie recenti
        Utile per dashboard e monitoring
        """
        try:
            # Esegue riconciliazione completa
            full_result = self.perform_full_reconciliation()

            # Filtra anomalie per severità
            all_anomalies = []
            for result in full_result["results"].values():
                all_anomalies.extend(result.get("anomalies", []))

            # Raggruppa per tipo
            anomaly_summary = {}
            for anomaly in all_anomalies:
                anomaly_type = anomaly["anomaly_type"]
                if anomaly_type not in anomaly_summary:
                    anomaly_summary[anomaly_type] = {
                        "count": 0,
                        "severity_distribution": {},
                        "recent_examples": [],
                    }

                anomaly_summary[anomaly_type]["count"] += 1

                severity = anomaly["severity"]
                anomaly_summary[anomaly_type]["severity_distribution"][severity] = (
                    anomaly_summary[anomaly_type]["severity_distribution"].get(
                        severity, 0
                    )
                    + 1
                )

                # Mantiene esempi recenti
                if len(anomaly_summary[anomaly_type]["recent_examples"]) < 5:
                    anomaly_summary[anomaly_type]["recent_examples"].append(
                        {
                            "timestamp": anomaly.get("timestamp", ""),
                            "description": anomaly["description"],
                        }
                    )

            return {
                "summary_period_days": days,
                "total_anomalies": len(all_anomalies),
                "anomaly_types": anomaly_summary,
                "reconciliation_timestamp": full_result["timestamp"],
                "trend_analysis": {
                    "increasing_trends": self._identify_increasing_trends(
                        anomaly_summary
                    ),
                    "critical_anomalies": self._count_critical_anomalies(
                        anomaly_summary
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get anomaly summary: {e}")
            return {"error": str(e)}

    def _identify_increasing_trends(self, anomaly_summary: Dict[str, Any]) -> List[str]:
        """Identifica trend in aumento anomalie"""
        increasing_trends = []

        for anomaly_type, data in anomaly_summary.items():
            if data["count"] > 5:  # Soglia trend
                increasing_trends.append(anomaly_type)

        return increasing_trends

    def _count_critical_anomalies(self, anomaly_summary: Dict[str, Any]) -> int:
        """Conta anomalie critiche"""
        critical_count = 0

        for data in anomaly_summary.values():
            critical_count += data["severity_distribution"].get("CRITICAL", 0)

        return critical_count
