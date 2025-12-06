"""
NBA Bankroll 3.0 - Bankroll Manager
Interfaccia principale per gestione bankroll enterprise-grade
Coordina tutti i componenti del sistema Bankroll 3.0
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .models import (
    Transaction,
    TransactionType,
    BetResult,
    BetRecord,
    BankrollState,
    BetPlacementRequest,
    ValidationResult,
    create_bet_record,
    create_transaction_id,
)
from .exceptions import (
    BankrollError,
    TransactionError,
    InsufficientFundsError,
    RiskLimitExceededError,
    ValidationError,
)
from .engine import TransactionEngine
from .calculator import BankrollStateCalculator
from .risk import RiskValidator
from .audit import AuditLogger, AuditEventType
from .security import SecurityManager


logger = logging.getLogger(__name__)


@dataclass
class BetPlacementResult:
    """Risultato piazzamento scommessa con dettagli"""

    success: bool
    bet_id: Optional[str] = None
    transaction_id: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "bet_id": self.bet_id,
            "transaction_id": self.transaction_id,
            "validation_result": self.validation_result.to_dict()
            if self.validation_result
            else None,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


@dataclass
class BankrollSummary:
    """Riepilogo bankroll con metriche derivate"""

    current_balance: Decimal
    available_balance: Decimal
    total_deposits: Decimal
    total_withdrawals: Decimal
    active_bets_count: int
    pending_bets_count: int
    total_profit: Decimal
    roi_percentage: Decimal
    win_rate_percentage: Decimal
    average_bet_size: Decimal
    last_transaction: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_balance": float(self.current_balance),
            "available_balance": float(self.available_balance),
            "total_deposits": float(self.total_deposits),
            "total_withdrawals": float(self.total_withdrawals),
            "active_bets_count": self.active_bets_count,
            "pending_bets_count": self.pending_bets_count,
            "total_profit": float(self.total_profit),
            "roi_percentage": float(self.roi_percentage),
            "win_rate_percentage": float(self.win_rate_percentage),
            "average_bet_size": float(self.average_bet_size),
            "last_transaction": self.last_transaction.isoformat()
            if self.last_transaction
            else None,
        }


class BankrollManager:
    """
    Bankroll Manager enterprise-grade:
    - Interfaccia unificata per tutte le operazioni
    - Coordinazione tra engine, calculator, risk, audit, security
    - Validazione completa e logging automatico
    - Performance ottimizzata con caching
    - Error handling robusto con retry logic
    """

    def __init__(
        self,
        engine: TransactionEngine,
        state_calculator: BankrollStateCalculator,
        risk_validator: RiskValidator,
        audit_logger: AuditLogger,
        security_manager: Optional[SecurityManager] = None,
    ):
        self.engine = engine
        self.state_calculator = state_calculator
        self.risk_validator = risk_validator
        self.audit_logger = audit_logger
        self.security_manager = security_manager

        # Statistiche operazioni
        self.operation_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "risk_blocks": 0,
            "average_response_time": 0.0,
        }

        logger.info("Bankroll Manager initialized")

    def place_bet(
        self,
        request: BetPlacementRequest,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> BetPlacementResult:
        """
        Piazzamento scommessa completo con validazioni
        Include risk validation, transaction logging, audit trail
        """
        start_time = datetime.now()
        operation_id = create_transaction_id()

        try:
            # Security check
            if self.security_manager:
                security_result = self.security_manager.validate_operation(
                    "place_bet", user_context or {}
                )
                if not security_result["allowed"]:
                    self.operation_stats["risk_blocks"] += 1
                    return BetPlacementResult(
                        success=False,
                        error_message=f"Security check failed: {security_result['reason']}",
                    )

            # Risk validation
            validation_result = self.risk_validator.validate_bet_placement(request)

            if not validation_result.is_valid:
                self.operation_stats["risk_blocks"] += 1

                # Log risk validation failure
                self.audit_logger.log_security_event(
                    f"Bet blocked by risk validation: {validation_result.error_message}",
                    user_context=user_context,
                    additional_data={
                        "bet_id": request.bet_id,
                        "validation_result": validation_result.to_dict(),
                    },
                )

                return BetPlacementResult(
                    success=False,
                    validation_result=validation_result,
                    error_message=validation_result.error_message,
                )

            # Crea record scommessa
            bet_record = create_bet_record(
                bet_id=request.bet_id,
                game_id=request.game_id,
                bet_type=request.bet_type,
                selection=request.selection,
                odds=request.odds,
                stake=request.stake,
                result=BetResult.PENDING,
                payout=Decimal("0.00"),
                profit_loss=Decimal("0.00"),
                placed_timestamp=datetime.now(timezone.utc),
                settled_timestamp=None,
                metadata=request.metadata,
            )

            # Esegui transazione
            transaction_id = self.engine.place_bet(bet_record)

            # Log audit
            self.audit_logger.log_bet(bet_record, user_context=user_context)
            self.audit_logger.log_transaction(
                self.engine.get_transaction_by_id(transaction_id),
                user_context=user_context,
            )

            # Aggiorna statistiche e invalida cache
            self._update_operation_stats(True, start_time)
            self.state_calculator.invalidate_cache()

            logger.info(f"Bet placed successfully: {request.bet_id}")

            return BetPlacementResult(
                success=True,
                bet_id=request.bet_id,
                transaction_id=transaction_id,
                validation_result=validation_result,
                warnings=validation_result.risk_warnings,
            )

        except InsufficientFundsError as e:
            self.operation_stats["failed_operations"] += 1
            self._update_operation_stats(False, start_time)

            # Log specific error
            self.audit_logger.log_error(
                e,
                {
                    "operation": "place_bet",
                    "bet_id": request.bet_id,
                    "requested_amount": float(request.stake),
                },
            )

            return BetPlacementResult(
                success=False, error_message=f"Insufficient funds: {e.message}"
            )

        except Exception as e:
            self.operation_stats["failed_operations"] += 1
            self._update_operation_stats(False, start_time)

            # Log unexpected error
            self.audit_logger.log_error(
                e,
                {
                    "operation": "place_bet",
                    "bet_id": request.bet_id,
                    "operation_id": operation_id,
                },
            )

            logger.error(f"Bet placement failed: {e}")
            return BetPlacementResult(
                success=False, error_message=f"System error: {str(e)}"
            )

    def settle_bet(
        self,
        bet_id: str,
        result: BetResult,
        payout: Decimal,
        profit_loss: Decimal,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Settlement scommessa con validazioni complete
        Aggiorna stato bankroll e audit trail
        """
        start_time = datetime.now()

        try:
            # Security check
            if self.security_manager:
                security_result = self.security_manager.validate_operation(
                    "settle_bet", user_context or {}
                )
                if not security_result["allowed"]:
                    self.audit_logger.log_security_event(
                        f"Bet settlement blocked: {security_result['reason']}",
                        user_context=user_context,
                        additional_data={"bet_id": bet_id},
                    )
                    return False

            # Verifica esistenza scommessa
            bet_record = self.engine.get_bet_record(bet_id)
            if not bet_record:
                raise TransactionError(f"Bet not found: {bet_id}")

            # Esegui settlement
            transaction_id = self.engine.settle_bet(bet_id, result, payout, profit_loss)

            # Recupera record aggiornato
            updated_record = self.engine.get_bet_record(bet_id)

            # Log audit
            self.audit_logger.log_bet(
                updated_record,
                event_type=AuditEventType.BET_SETTLED,
                user_context=user_context,
            )

            self.audit_logger.log_transaction(
                self.engine.get_transaction_by_id(transaction_id),
                event_type=AuditEventType.BET_SETTLED,
                user_context=user_context,
            )

            # Aggiorna statistiche e invalida cache
            self._update_operation_stats(True, start_time)
            self.state_calculator.invalidate_cache()

            logger.info(f"Bet settled: {bet_id} - {result.value} - P&L: {profit_loss}")
            # Transazione delegata a engine.settle_bet

            logger.info(f"Bet settled: {bet_id} - {result.value} - P&L: {profit_loss}")
            return True

        except Exception as e:
            self.operation_stats["failed_operations"] += 1
            self._update_operation_stats(False, start_time)

            # Log error
            self.audit_logger.log_error(
                e,
                {
                    "operation": "settle_bet",
                    "bet_id": bet_id,
                    "result": result.value,
                    "payout": float(payout),
                    "profit_loss": float(profit_loss),
                },
            )

            logger.error(f"Bet settlement failed: {e}")
            return False

    def add_deposit(
        self,
        amount: Decimal,
        description: str = "Deposit",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Aggiunge deposito con validazioni e audit
        """
        start_time = datetime.now()

        try:
            if amount <= 0:
                raise ValidationError("Deposit amount must be positive")

            # Security check
            if self.security_manager:
                security_result = self.security_manager.validate_operation(
                    "add_deposit", user_context or {}
                )
                if not security_result["allowed"]:
                    self.audit_logger.log_security_event(
                        f"Deposit blocked: {security_result['reason']}",
                        user_context=user_context,
                        additional_data={"amount": float(amount)},
                    )
                    raise ValidationError(security_result["reason"])

            # Esegui deposito
            transaction_id = self.engine.add_deposit(amount, description)

            # Log audit
            transaction = self.engine.get_transaction_by_id(transaction_id)
            self.audit_logger.log_transaction(transaction, user_context=user_context)

            # Aggiorna statistiche e invalida cache
            self._update_operation_stats(True, start_time)
            self.state_calculator.invalidate_cache()

            logger.info(f"Deposit added: {amount} - TXN: {transaction_id}")
            return transaction_id

        except Exception as e:
            self.operation_stats["failed_operations"] += 1
            self._update_operation_stats(False, start_time)

            # Log error
            self.audit_logger.log_error(
                e,
                {
                    "operation": "add_deposit",
                    "amount": float(amount),
                    "description": description,
                },
            )

            logger.error(f"Deposit failed: {e}")
            raise

    def add_withdrawal(
        self,
        amount: Decimal,
        description: str = "Withdrawal",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Esegue prelievo con validazioni complete
        """
        start_time = datetime.now()

        try:
            if amount <= 0:
                raise ValidationError("Withdrawal amount must be positive")

            # Security check
            if self.security_manager:
                security_result = self.security_manager.validate_operation(
                    "add_withdrawal", user_context or {}
                )
                if not security_result["allowed"]:
                    self.audit_logger.log_security_event(
                        f"Withdrawal blocked: {security_result['reason']}",
                        user_context=user_context,
                        additional_data={"amount": float(amount)},
                    )
                    raise ValidationError(security_result["reason"])

            # Esegui prelievo
            transaction_id = self.engine.add_withdrawal(amount, description)

            # Log audit
            transaction = self.engine.get_transaction_by_id(transaction_id)
            self.audit_logger.log_transaction(transaction, user_context=user_context)

            # Aggiorna statistiche e invalida cache
            self._update_operation_stats(True, start_time)
            self.state_calculator.invalidate_cache()

            logger.info(f"Withdrawal processed: {amount} - TXN: {transaction_id}")
            return transaction_id

        except Exception as e:
            self.operation_stats["failed_operations"] += 1
            self._update_operation_stats(False, start_time)

            # Log error
            self.audit_logger.log_error(
                e,
                {
                    "operation": "add_withdrawal",
                    "amount": float(amount),
                    "description": description,
                },
            )

            logger.error(f"Withdrawal failed: {e}")
            raise

    def get_current_state(self, force_recalculate: bool = False) -> BankrollState:
        """
        Restituisce stato corrente bankroll
        Usa caching per performance
        """
        try:
            state = self.state_calculator.get_current_state(force_recalculate)

            # Log stato calcolato
            self.audit_logger.log_state_change(None, state)

            return state

        except Exception as e:
            self.audit_logger.log_error(e, {"operation": "get_current_state"})
            logger.error(f"Failed to get bankroll state: {e}")
            raise

    def get_bankroll_summary(self) -> BankrollSummary:
        """
        Restituisce riepilogo bankroll con metriche derivate
        Include KPI e statistiche performance
        """
        try:
            state = self.get_current_state()

            # Calcola metriche derivate
            total_bets_returned = state.total_bets_won + state.total_bets_lost
            roi = (
                (
                    (state.total_bets_won - state.total_bets_lost)
                    / max(state.total_bets_placed, Decimal("0.01"))
                    * 100
                )
                if state.total_bets_placed > 0
                else Decimal("0.00")
            )

            win_rate = (
                (state.total_bets_won / max(total_bets_returned, Decimal("0.01")) * 100)
                if total_bets_returned > 0
                else Decimal("0.00")
            )

            avg_bet_size = (
                (state.total_bets_placed / max(total_bets_returned, 1))
                if total_bets_returned > 0
                else Decimal("0.00")
            )

            # Calcola balance disponibile (esclusi stake attivi)
            active_bets_value = self._calculate_active_bets_value()
            available_balance = state.current_balance - active_bets_value

            return BankrollSummary(
                current_balance=state.current_balance,
                available_balance=available_balance,
                total_deposits=state.total_deposits,
                total_withdrawals=state.total_withdrawals,
                active_bets_count=state.active_bets_count,
                pending_bets_count=state.pending_bets_count,
                total_profit=state.total_bets_won - state.total_bets_lost,
                roi_percentage=roi,
                win_rate_percentage=win_rate,
                average_bet_size=avg_bet_size,
                last_transaction=state.last_transaction_timestamp,
            )

        except Exception as e:
            self.audit_logger.log_error(e, {"operation": "get_bankroll_summary"})
            logger.error(f"Failed to get bankroll summary: {e}")
            raise

    def get_transaction_history(
        self,
        limit: int = 100,
        transaction_type: Optional[TransactionType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Transaction]:
        """
        Recupera storia transazioni con filtri
        Supporta paginazione e ricerca avanzata
        """
        try:
            transactions = self.engine.get_transactions(limit, transaction_type)

            # Filtra per date se specificate
            if start_date or end_date:
                filtered_transactions = []
                for txn in transactions:
                    if start_date and txn.timestamp < start_date:
                        continue
                    if end_date and txn.timestamp > end_date:
                        continue
                    filtered_transactions.append(txn)
                return filtered_transactions

            return transactions

        except Exception as e:
            self.audit_logger.log_error(e, {"operation": "get_transaction_history"})
            logger.error(f"Failed to get transaction history: {e}")
            raise

    def get_bet_history(
        self,
        limit: int = 100,
        result_filter: Optional[BetResult] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[BetRecord]:
        """
        Recupera storia scommesse con filtri
        Include metriche e statistiche
        """
        try:
            bet_records = self.engine.get_bet_records(limit, result_filter)

            # Filtra per date se specificate
            if start_date or end_date:
                filtered_bets = []
                for bet in bet_records:
                    if start_date and bet.placed_timestamp < start_date:
                        continue
                    if end_date and bet.placed_timestamp > end_date:
                        continue
                    filtered_bets.append(bet)
                return filtered_bets

            return bet_records

        except Exception as e:
            self.audit_logger.log_error(e, {"operation": "get_bet_history"})
            logger.error(f"Failed to get bet history: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Restituisce metriche performance complete
        Include statistiche operazioni e componenti
        """
        try:
            # Metriche calculator
            calc_metrics = self.state_calculator.get_performance_metrics()

            # Metriche risk validator
            risk_metrics = self.risk_validator.get_risk_metrics()

            # Metriche audit
            audit_metrics = self.audit_logger.get_audit_statistics()

            # Metriche security
            security_metrics = {}
            if self.security_manager:
                security_metrics = self.security_manager.get_security_metrics()

            # Statistiche operazioni manager
            total_ops = self.operation_stats["total_operations"]
            success_rate = (
                (
                    self.operation_stats["successful_operations"]
                    / max(total_ops, 1)
                    * 100
                )
                if total_ops > 0
                else 0
            )

            return {
                "manager_stats": {
                    **self.operation_stats,
                    "success_rate": success_rate,
                    "failure_rate": 100 - success_rate,
                },
                "calculator_metrics": calc_metrics,
                "risk_metrics": risk_metrics,
                "audit_metrics": audit_metrics,
                "security_metrics": security_metrics,
                "system_health": {
                    "overall_status": "HEALTHY" if success_rate >= 95 else "DEGRADED",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            self.audit_logger.log_error(e, {"operation": "get_performance_metrics"})
            logger.error(f"Failed to get performance metrics: {e}")
            raise

    def _calculate_active_bets_value(self) -> Decimal:
        """Calcola valore totale scommesse attive"""
        try:
            active_bets = self.engine.get_bet_records(
                limit=1000, result_filter=BetResult.PENDING
            )

            total_value = Decimal("0.00")
            for bet in active_bets:
                total_value += bet.stake

            return total_value

        except Exception as e:
            logger.warning(f"Failed to calculate active bets value: {e}")
            return Decimal("0.00")

    def _update_operation_stats(self, success: bool, start_time: datetime):
        """Aggiorna statistiche operazioni"""
        self.operation_stats["total_operations"] += 1

        if success:
            self.operation_stats["successful_operations"] += 1
        else:
            self.operation_stats["failed_operations"] += 1

        # Aggiorna tempo medio risposta
        response_time = (datetime.now() - start_time).total_seconds()
        current_avg = self.operation_stats["average_response_time"]
        n = self.operation_stats["total_operations"]
        new_avg = ((current_avg * (n - 1)) + response_time) / n
        self.operation_stats["average_response_time"] = new_avg

    def validate_operation_permissions(
        self, operation: str, user_context: Dict[str, Any]
    ) -> bool:
        """
        Validazione permessi operazione
        Integra con security manager se disponibile
        """
        if self.security_manager:
            result = self.security_manager.validate_operation(operation, user_context)
            return result["allowed"]

        # Default: consenti tutte le operazioni
        return True

    def get_risk_limits(self) -> Dict[str, Any]:
        """Restituisce configurazione limiti rischio"""
        return self.risk_validator.get_risk_limits()

    def update_risk_limit(self, rule_name: str, threshold: Decimal):
        """Aggiorna limite rischio specifico"""
        from .risk import RiskRule

        try:
            rule = RiskRule(rule_name)
            from .risk import RiskLimit

            limit = RiskLimit(
                rule=rule,
                threshold=threshold,
                description=f"Updated limit for {rule_name}",
            )

            self.risk_validator.update_risk_limit(rule, limit)

            # Log modifica
            self.audit_logger.log_security_event(
                f"Risk limit updated: {rule_name} -> {threshold}",
                additional_data={"rule": rule_name, "new_threshold": float(threshold)},
            )

            logger.info(f"Risk limit updated: {rule_name} = {threshold}")
            return True

        except ValueError:
            logger.error(f"Invalid risk rule: {rule_name}")
            return False
        except Exception as e:
            self.audit_logger.log_error(
                e,
                {
                    "operation": "update_risk_limit",
                    "rule_name": rule_name,
                    "threshold": float(threshold),
                },
            )
            return False

    def reset_performance_metrics(self):
        """Resetta metriche performance"""
        self.operation_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "risk_blocks": 0,
            "average_response_time": 0.0,
        }

        # Resetta metriche componenti
        self.state_calculator.clear_cache()
        self.risk_validator.reset_metrics()

        # Log reset
        self.audit_logger.log_security_event(
            "Performance metrics reset", additional_data={"initiated_by": "system"}
        )

        logger.info("Performance metrics reset")

    def close(self):
        """Chiudi manager e tutte le risorse"""
        try:
            # Log chiusura
            self.audit_logger.log_security_event(
                "Bankroll manager shutdown",
                additional_data={"final_stats": self.operation_stats},
            )

            # Chiudi componenti
            if self.engine:
                self.engine.close()

            logger.info("Bankroll manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing bankroll manager: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
