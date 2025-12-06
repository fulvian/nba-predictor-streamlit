"""
NBA Bankroll 3.0 - Exception Hierarchy
Gerarchia di eccezioni specifiche per il sistema bankroll
Basato su best practice per error handling in sistemi finanziari
"""

from typing import Optional, Dict, Any


class BankrollError(Exception):
    """Base exception per tutti gli errori del bankroll system"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per logging/API"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class TransactionError(BankrollError):
    """Errori durante operazioni sulle transazioni"""

    pass


class InsufficientFundsError(TransactionError):
    """Fondi insufficienti per operazione"""

    def __init__(
        self,
        message: str,
        requested_amount: Optional[float] = None,
        available_balance: Optional[float] = None,
    ):
        super().__init__(message, "INSUFFICIENT_FUNDS")
        self.requested_amount = requested_amount
        self.available_balance = available_balance
        self.context.update(
            {
                "requested_amount": requested_amount,
                "available_balance": available_balance,
            }
        )


class InvalidTransactionError(TransactionError):
    """Transazione non valida per motivi di business logic"""

    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        validation_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, "INVALID_TRANSACTION")
        self.transaction_id = transaction_id
        self.validation_details = validation_details or {}
        self.context.update(
            {"transaction_id": transaction_id, "validation_details": validation_details}
        )


class DuplicateTransactionError(TransactionError):
    """Tentativo di transazione duplicata"""

    def __init__(
        self,
        message: str,
        transaction_id: str,
        original_timestamp: Optional[str] = None,
    ):
        super().__init__(message, "DUPLICATE_TRANSACTION")
        self.transaction_id = transaction_id
        self.original_timestamp = original_timestamp
        self.context.update(
            {"transaction_id": transaction_id, "original_timestamp": original_timestamp}
        )


class ValidationError(BankrollError):
    """Errori durante validazione dati"""

    pass


class RiskLimitExceededError(ValidationError):
    """Superamento limiti di rischio"""

    def __init__(
        self,
        message: str,
        risk_type: str,
        current_value: float,
        limit_value: float,
        recommendation: Optional[str] = None,
    ):
        super().__init__(message, "RISK_LIMIT_EXCEEDED")
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.recommendation = recommendation
        self.context.update(
            {
                "risk_type": risk_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "recommendation": recommendation,
            }
        )


class BetValidationError(ValidationError):
    """Validazione fallita per scommessa"""

    def __init__(
        self,
        message: str,
        bet_id: Optional[str] = None,
        validation_errors: Optional[list] = None,
    ):
        super().__init__(message, "BET_VALIDATION_FAILED")
        self.bet_id = bet_id
        self.validation_errors = validation_errors or []
        self.context.update({"bet_id": bet_id, "validation_errors": validation_errors})


class AuditError(BankrollError):
    """Errori nel sistema di audit"""

    pass


class ChecksumMismatchError(AuditError):
    """Checksum non corrispondente - possibile corruzione dati"""

    def __init__(
        self,
        message: str,
        expected_checksum: str,
        actual_checksum: str,
        entity_type: str,
        entity_id: Optional[str] = None,
    ):
        super().__init__(message, "CHECKSUM_MISMATCH")
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.context.update(
            {
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        )


class IntegrityError(AuditError):
    """Violazione integrità dati"""

    def __init__(
        self, message: str, integrity_check: str, expected_value: Any, actual_value: Any
    ):
        super().__init__(message, "INTEGRITY_VIOLATION")
        self.integrity_check = integrity_check
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.context.update(
            {
                "integrity_check": integrity_check,
                "expected_value": expected_value,
                "actual_value": actual_value,
            }
        )


class ReconciliationError(AuditError):
    """Errori durante reconciliation"""

    def __init__(
        self,
        message: str,
        reconciliation_type: str,
        discrepancies: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, "RECONCILIATION_FAILED")
        self.reconciliation_type = reconciliation_type
        self.discrepancies = discrepancies or {}
        self.context.update(
            {"reconciliation_type": reconciliation_type, "discrepancies": discrepancies}
        )


class DataError(BankrollError):
    """Errori relativi a dati storage/recupero"""

    pass


class DatabaseConnectionError(DataError):
    """Errore connessione database"""

    def __init__(
        self,
        message: str,
        database_path: Optional[str] = None,
        connection_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, "DB_CONNECTION_ERROR")
        self.database_path = database_path
        self.connection_details = connection_details or {}
        self.context.update(
            {"database_path": database_path, "connection_details": connection_details}
        )


class DataCorruptionError(DataError):
    """Dati corrotti rilevati"""

    def __init__(
        self,
        message: str,
        corrupted_data_id: str,
        corruption_type: str,
        recovery_suggestion: Optional[str] = None,
    ):
        super().__init__(message, "DATA_CORRUPTION")
        self.corrupted_data_id = corrupted_data_id
        self.corruption_type = corruption_type
        self.recovery_suggestion = recovery_suggestion
        self.context.update(
            {
                "corrupted_data_id": corrupted_data_id,
                "corruption_type": corruption_type,
                "recovery_suggestion": recovery_suggestion,
            }
        )


class MigrationError(BankrollError):
    """Errori durante migrazione dati"""

    def __init__(
        self,
        message: str,
        migration_step: str,
        failed_records: Optional[int] = None,
        total_records: Optional[int] = None,
    ):
        super().__init__(message, "MIGRATION_FAILED")
        self.migration_step = migration_step
        self.failed_records = failed_records
        self.total_records = total_records
        self.context.update(
            {
                "migration_step": migration_step,
                "failed_records": failed_records,
                "total_records": total_records,
            }
        )


class SecurityError(BankrollError):
    """Errori relativi a sicurezza"""

    pass


class UnauthorizedAccessError(SecurityError):
    """Accesso non autorizzato"""

    def __init__(
        self,
        message: str,
        resource: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        super().__init__(message, "UNAUTHORIZED_ACCESS")
        self.resource = resource
        self.user_id = user_id
        self.ip_address = ip_address
        self.context.update(
            {"resource": resource, "user_id": user_id, "ip_address": ip_address}
        )


class RateLimitExceededError(SecurityError):
    """Rate limit superato"""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_count: int,
        limit_threshold: int,
        reset_time: Optional[str] = None,
    ):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")
        self.limit_type = limit_type
        self.current_count = current_count
        self.limit_threshold = limit_threshold
        self.reset_time = reset_time
        self.context.update(
            {
                "limit_type": limit_type,
                "current_count": current_count,
                "limit_threshold": limit_threshold,
                "reset_time": reset_time,
            }
        )


class ConfigurationError(BankrollError):
    """Errori di configurazione sistema"""

    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any,
        valid_options: Optional[list] = None,
    ):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.config_value = config_value
        self.valid_options = valid_options or []
        self.context.update(
            {
                "config_key": config_key,
                "config_value": config_value,
                "valid_options": valid_options,
            }
        )


# Exception handler utilities
def format_exception_for_api(exception: BankrollError) -> Dict[str, Any]:
    """Formatta eccezione per response API"""
    return {
        "success": False,
        "error": {
            "type": exception.__class__.__name__,
            "code": exception.error_code,
            "message": exception.message,
            "context": exception.context,
        },
    }


def format_exception_for_logging(exception: BankrollError) -> str:
    """Formatta eccezione per logging"""
    context_str = ", ".join([f"{k}={v}" for k, v in exception.context.items()])
    return f"{exception.__class__.__name__}: {exception.message} | {context_str}"


def is_retriable_error(exception: BankrollError) -> bool:
    """Determina se l'errore è ritentabile"""
    retriable_errors = (
        DatabaseConnectionError,
        # Aggiungere altri errori ritentabili se necessario
    )
    return isinstance(exception, retriable_errors)


def is_critical_error(exception: BankrollError) -> bool:
    """Determina se l'errore è critico (richiede intervento immediato)"""
    critical_errors = (
        DataCorruptionError,
        ChecksumMismatchError,
        IntegrityError,
        UnauthorizedAccessError,
    )
    return isinstance(exception, critical_errors)
