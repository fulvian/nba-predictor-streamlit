"""
NBA Bankroll 3.0 - Immutable Ledger System
Sistema di gestione bankroll enterprise-grade basato su:
- Immutable Ledger con event sourcing
- Transaction Engine atomico con ACID compliance
- Precise decimal arithmetic per calcoli finanziari
- Audit trail completo con checksum SHA-256
- Risk management con validazione in tempo reale
- Reconciliation automatica con anomaly detection
- Security enterprise-grade con rate limiting e device fingerprinting
"""

from .engine import TransactionEngine
from .manager import BankrollManager
from .models import (
    Transaction,
    TransactionType,
    BetResult,
    BetPlacementRequest,
    ValidationResult,
    BankrollState,
    RiskLevel,
    create_bet_record,
    create_bet_placement_request,
    create_transaction_id,
)
from .exceptions import (
    BankrollError,
    TransactionError,
    InsufficientFundsError,
    RiskLimitExceededError,
    AuditError,
    ValidationError,
)
from .calculator import BankrollStateCalculator
from .risk import RiskValidator
from .audit import AuditLogger
from .reconciliation import ReconciliationService
from .security import SecurityManager
from .migration import BankrollMigrationService

__version__ = "3.0.0"
__author__ = "NBA Predictor Development Team"
__description__ = "Enterprise-grade bankroll management system"

# Export main classes for easy import
__all__ = [
    "TransactionEngine",
    "BankrollManager",
    "BankrollStateCalculator",
    "RiskValidator",
    "AuditLogger",
    "ReconciliationService",
    "SecurityManager",
    "BankrollMigrationService",
    # Models
    "Transaction",
    "TransactionType",
    "BetResult",
    "BetPlacementRequest",
    "ValidationResult",
    "BankrollState",
    "RiskLevel",
    "create_bet_record",
    "create_bet_placement_request",
    "create_transaction_id",
    # Exceptions
    "BankrollError",
    "TransactionError",
    "InsufficientFundsError",
    "RiskLimitExceededError",
    "AuditError",
    "ValidationError",
]


# Factory functions for easy instantiation
def create_bankroll_manager(
    db_path: str = "data/nba_betting_v3.duckdb",
) -> BankrollManager:
    """Factory function per creare BankrollManager con dipendenze iniettate"""
    engine = TransactionEngine(db_path)
    state_calculator = BankrollStateCalculator(engine)
    risk_validator = RiskValidator(state_calculator)
    audit_logger = AuditLogger(engine)

    return BankrollManager(
        engine=engine,
        state_calculator=state_calculator,
        risk_validator=risk_validator,
        audit_logger=audit_logger,
    )


def create_migration_service(
    old_db_path: str, new_db_path: str
) -> BankrollMigrationService:
    """Factory function per creare migration service"""
    return BankrollMigrationService(old_db_path, new_db_path)


def create_reconciliation_service(engine: TransactionEngine) -> ReconciliationService:
    """Factory function per creare reconciliation service"""
    return ReconciliationService(engine)


def create_security_manager(engine: TransactionEngine) -> SecurityManager:
    """Factory function per creare security manager"""
    return SecurityManager(engine)


# Legacy compatibility
def get_bankroll_manager():
    """Compatibility function for existing code"""
    return create_bankroll_manager()
