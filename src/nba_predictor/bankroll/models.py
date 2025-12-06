"""
NBA Bankroll 3.0 - Immutable Data Models
Modelli di dati immutabili con checksum SHA-256 per integrità garantita
Basato su best practice da Flumine (betcode-org/flumine) e Beancount
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
import hashlib
import json
import uuid


class TransactionType(Enum):
    """Tipi di transazioni supportate dal sistema"""

    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    BET_PLACEMENT = "BET_PLACEMENT"
    BET_SETTLEMENT = "BET_SETTLEMENT"
    ADJUSTMENT = "ADJUSTMENT"
    FEE = "FEE"
    REFUND = "REFUND"


class BetResult(Enum):
    """Risultati possibili di una scommessa"""

    PENDING = "PENDING"
    WON = "WON"
    LOST = "LOST"
    PUSH = "PUSH"
    VOID = "VOID"
    CANCELLED = "CANCELLED"
    PARTIAL_WIN = "PARTIAL_WIN"
    PARTIAL_LOSS = "PARTIAL_LOSS"


class RiskLevel(Enum):
    """Livelli di rischio per validazione"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class BetPlacementRequest:
    """Request immutabile per piazzare scommessa"""

    bet_id: str
    game_id: str
    bet_type: str  # 'moneyline', 'spread', 'total'
    selection: str  # 'home', 'away', 'over', 'under'
    odds: Decimal
    stake: Decimal
    expected_value: Optional[Decimal] = None
    confidence_score: Optional[Decimal] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validazione automatica dopo creazione"""
        if self.stake <= 0:
            raise ValueError("Stake must be positive")
        if self.odds <= 0:
            raise ValueError("Odds must be positive")
        if not self.bet_id:
            raise ValueError("Bet ID is required")

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per storage"""
        return {
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": float(self.odds),
            "stake": float(self.stake),
            "expected_value": float(self.expected_value)
            if self.expected_value
            else None,
            "confidence_score": float(self.confidence_score)
            if self.confidence_score
            else None,
            "risk_level": self.risk_level.value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Risultato della validazione con dettagli"""

    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    risk_warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "risk_warnings": self.risk_warnings,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Transaction:
    """Transazione immutabile con checksum SHA-256"""

    transaction_id: str
    timestamp: datetime
    transaction_type: TransactionType
    amount: Decimal
    balance_after: Decimal
    description: str
    bet_id: Optional[str] = None
    game_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = field(init=False)

    def __post_init__(self):
        """Calcolo checksum SHA-256 per integrità"""
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            object.__setattr__(
                self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc)
            )

        # Calculate checksum
        # Use fixed-point formatting for decimals to avoid precision mismatches
        checksum_data = {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "transaction_type": self.transaction_type.value,
            "amount": f"{self.amount:.2f}",
            "balance_after": f"{self.balance_after:.2f}",
            "description": self.description,
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "metadata": self.metadata,
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_str.encode()).hexdigest()
        object.__setattr__(self, "checksum", checksum)

    def verify_checksum(self) -> bool:
        """Verifica integrità della transazione"""
        # Recalculate checksum and compare
        temp_checksum = self.checksum
        object.__setattr__(self, "checksum", "")

        checksum_data = {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "transaction_type": self.transaction_type.value,
            "amount": f"{self.amount:.2f}",
            "balance_after": f"{self.balance_after:.2f}",
            "description": self.description,
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "metadata": self.metadata,
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        calculated_checksum = hashlib.sha256(checksum_str.encode()).hexdigest()

        object.__setattr__(self, "checksum", temp_checksum)
        return calculated_checksum == temp_checksum

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per storage"""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_type": self.transaction_type.value,
            "amount": float(self.amount),
            "balance_after": float(self.balance_after),
            "description": self.description,
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        """Creazione da dizionario (da storage)"""
        return cls(
            transaction_id=data["transaction_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            transaction_type=TransactionType(data["transaction_type"]),
            amount=Decimal(str(data["amount"])),
            balance_after=Decimal(str(data["balance_after"])),
            description=data["description"],
            bet_id=data.get("bet_id"),
            game_id=data.get("game_id"),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


@dataclass(frozen=True)
class BankrollState:
    """Stato del bankroll calcolato dal ledger"""

    current_balance: Decimal
    total_deposits: Decimal
    total_withdrawals: Decimal
    total_bets_placed: Decimal
    total_bets_won: Decimal
    total_bets_lost: Decimal
    total_fees: Decimal
    active_bets_count: int
    pending_bets_count: int
    last_transaction_timestamp: Optional[datetime]
    last_updated: datetime
    checksum: str = field(init=False)

    def __post_init__(self):
        """Calcolo checksum SHA-256 per integrità stato"""
        # Ensure timestamps are timezone-aware
        if (
            self.last_transaction_timestamp
            and self.last_transaction_timestamp.tzinfo is None
        ):
            object.__setattr__(
                self,
                "last_transaction_timestamp",
                self.last_transaction_timestamp.replace(tzinfo=timezone.utc),
            )
        if self.last_updated.tzinfo is None:
            object.__setattr__(
                self, "last_updated", self.last_updated.replace(tzinfo=timezone.utc)
            )

        # Calculate checksum
        # Use fixed-point formatting for decimals to avoid precision mismatches
        checksum_data = {
            "current_balance": f"{self.current_balance:.2f}",
            "total_deposits": f"{self.total_deposits:.2f}",
            "total_withdrawals": f"{self.total_withdrawals:.2f}",
            "total_bets_placed": f"{self.total_bets_placed:.2f}",
            "total_bets_won": f"{self.total_bets_won:.2f}",
            "total_bets_lost": f"{self.total_bets_lost:.2f}",
            "total_fees": f"{self.total_fees:.2f}",
            "active_bets_count": self.active_bets_count,
            "pending_bets_count": self.pending_bets_count,
            "last_transaction_timestamp": self.last_transaction_timestamp.astimezone(
                timezone.utc
            )
            .replace(tzinfo=None)
            .isoformat()
            if self.last_transaction_timestamp
            else None,
            "last_updated": self.last_updated.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_str.encode()).hexdigest()
        object.__setattr__(self, "checksum", checksum)

    def verify_checksum(self) -> bool:
        """Verifica integrità dello stato"""
        temp_checksum = self.checksum
        object.__setattr__(self, "checksum", "")

        # Use fixed-point formatting for decimals to avoid precision mismatches
        checksum_data = {
            "current_balance": f"{self.current_balance:.2f}",
            "total_deposits": f"{self.total_deposits:.2f}",
            "total_withdrawals": f"{self.total_withdrawals:.2f}",
            "total_bets_placed": f"{self.total_bets_placed:.2f}",
            "total_bets_won": f"{self.total_bets_won:.2f}",
            "total_bets_lost": f"{self.total_bets_lost:.2f}",
            "total_fees": f"{self.total_fees:.2f}",
            "active_bets_count": self.active_bets_count,
            "pending_bets_count": self.pending_bets_count,
            "last_transaction_timestamp": self.last_transaction_timestamp.astimezone(
                timezone.utc
            )
            .replace(tzinfo=None)
            .isoformat()
            if self.last_transaction_timestamp
            else None,
            "last_updated": self.last_updated.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        calculated_checksum = hashlib.sha256(checksum_str.encode()).hexdigest()

        object.__setattr__(self, "checksum", temp_checksum)
        return calculated_checksum == temp_checksum

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per storage"""
        return {
            "current_balance": float(self.current_balance),
            "total_deposits": float(self.total_deposits),
            "total_withdrawals": float(self.total_withdrawals),
            "total_bets_placed": float(self.total_bets_placed),
            "total_bets_won": float(self.total_bets_won),
            "total_bets_lost": float(self.total_bets_lost),
            "total_fees": float(self.total_fees),
            "active_bets_count": self.active_bets_count,
            "pending_bets_count": self.pending_bets_count,
            "last_transaction_timestamp": self.last_transaction_timestamp.isoformat()
            if self.last_transaction_timestamp
            else None,
            "last_updated": self.last_updated.isoformat(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BankrollState":
        """Creazione da dizionario (da storage)"""
        return cls(
            current_balance=Decimal(str(data["current_balance"])),
            total_deposits=Decimal(str(data["total_deposits"])),
            total_withdrawals=Decimal(str(data["total_withdrawals"])),
            total_bets_placed=Decimal(str(data["total_bets_placed"])),
            total_bets_won=Decimal(str(data["total_bets_won"])),
            total_bets_lost=Decimal(str(data["total_bets_lost"])),
            total_fees=Decimal(str(data["total_fees"])),
            active_bets_count=data["active_bets_count"],
            pending_bets_count=data["pending_bets_count"],
            last_transaction_timestamp=datetime.fromisoformat(
                data["last_transaction_timestamp"]
            )
            if data.get("last_transaction_timestamp")
            else None,
            last_updated=datetime.fromisoformat(data["last_updated"]),
            checksum=data.get("checksum", ""),
        )


@dataclass(frozen=True)
class BetRecord:
    """Record immutabile di scommessa"""

    bet_id: str
    game_id: str
    bet_type: str
    selection: str
    odds: Decimal
    stake: Decimal
    result: BetResult = BetResult.PENDING
    payout: Decimal = Decimal("0.00")
    profit_loss: Decimal = Decimal("0.00")
    placed_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    settled_timestamp: Optional[datetime] = None
    user_id: str = "test_user_001"
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = field(init=False)

    def __post_init__(self):
        """Calcolo checksum SHA-256 per integrità"""
        # Ensure timestamps are timezone-aware
        if self.placed_timestamp.tzinfo is None:
            object.__setattr__(
                self,
                "placed_timestamp",
                self.placed_timestamp.replace(tzinfo=timezone.utc),
            )
        if self.settled_timestamp and self.settled_timestamp.tzinfo is None:
            object.__setattr__(
                self,
                "settled_timestamp",
                self.settled_timestamp.replace(tzinfo=timezone.utc),
            )

        # Calculate checksum
        # Use fixed-point formatting for decimals to avoid precision mismatches
        checksum_data = {
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": f"{self.odds:.4f}",
            "stake": f"{self.stake:.2f}",
            "result": self.result.value,
            "payout": f"{self.payout:.2f}",
            "profit_loss": f"{self.profit_loss:.2f}",
            "placed_timestamp": self.placed_timestamp.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "settled_timestamp": self.settled_timestamp.astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
            if self.settled_timestamp
            else None,
            "metadata": self.metadata,
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_str.encode()).hexdigest()
        object.__setattr__(self, "checksum", checksum)

    def to_dict(self) -> Dict[str, Any]:
        """Conversione a dizionario per storage"""
        return {
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": float(self.odds),
            "stake": float(self.stake),
            "result": self.result.value,
            "payout": float(self.payout),
            "profit_loss": float(self.profit_loss),
            "placed_timestamp": self.placed_timestamp.isoformat(),
            "settled_timestamp": self.settled_timestamp.isoformat()
            if self.settled_timestamp
            else None,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BetRecord":
        """Creazione da dizionario (da storage)"""
        return cls(
            bet_id=data["bet_id"],
            game_id=data["game_id"],
            bet_type=data["bet_type"],
            selection=data["selection"],
            odds=Decimal(str(data["odds"])),
            stake=Decimal(str(data["stake"])),
            result=BetResult(data["result"]),
            payout=Decimal(str(data["payout"])),
            profit_loss=Decimal(str(data["profit_loss"])),
            placed_timestamp=datetime.fromisoformat(data["placed_timestamp"]),
            settled_timestamp=datetime.fromisoformat(data["settled_timestamp"])
            if data.get("settled_timestamp")
            else None,
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


# Factory functions per creare istanze con ID univoci
def create_transaction_id() -> str:
    """Genera ID transazione univoco"""
    return f"txn_{uuid.uuid4().hex[:16]}_{int(datetime.now(timezone.utc).timestamp())}"


def create_bet_id() -> str:
    """Genera ID scommessa univoco"""
    return f"bet_{uuid.uuid4().hex[:16]}_{int(datetime.now(timezone.utc).timestamp())}"


def create_bet_record(
    bet_id: str,
    game_id: str,
    bet_type: str,
    selection: str,
    odds: Decimal,
    stake: Decimal,
    result: BetResult,
    payout: Decimal,
    profit_loss: Decimal,
    placed_timestamp: datetime,
    settled_timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> BetRecord:
    """Factory function per creare BetRecord con timestamp automatici"""
    return BetRecord(
        bet_id=bet_id,
        game_id=game_id,
        bet_type=bet_type,
        selection=selection,
        odds=odds,
        stake=stake,
        result=result,
        payout=payout,
        profit_loss=profit_loss,
        placed_timestamp=placed_timestamp,
        settled_timestamp=settled_timestamp,
        metadata=metadata or {},
    )


def create_bet_placement_request(
    game_id: str,
    bet_type: str,
    selection: str,
    odds: Decimal,
    stake: Decimal,
    expected_value: Optional[Decimal] = None,
    confidence_score: Optional[Decimal] = None,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    metadata: Optional[Dict[str, Any]] = None,
) -> BetPlacementRequest:
    """Factory function per BetPlacementRequest con ID generato"""
    return BetPlacementRequest(
        bet_id=create_bet_id(),
        game_id=game_id,
        bet_type=bet_type,
        selection=selection,
        odds=odds,
        stake=stake,
        expected_value=expected_value,
        confidence_score=confidence_score,
        risk_level=risk_level,
        metadata=metadata or {},
    )
