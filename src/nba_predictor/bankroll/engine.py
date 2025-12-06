"""
NBA Bankroll 3.0 - Transaction Engine
Motore transazionale ACID-compliant con ledger immutabile
Basato su best practice da Flumine (betcode-org/flumine)
"""

import duckdb
import logging
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import threading
import time
from dataclasses import replace

from .models import (
    Transaction,
    TransactionType,
    BetResult,
    BetRecord,
    BankrollState,
    create_transaction_id,
)
from .exceptions import (
    TransactionError,
    InsufficientFundsError,
    DuplicateTransactionError,
    DatabaseConnectionError,
    ChecksumMismatchError,
    IntegrityError,
)


logger = logging.getLogger(__name__)


class TransactionEngine:
    """
    Transaction Engine con supporto ACID:
    - Atomicity: Transazioni atomiche con rollback
    - Consistency: Stato sempre consistente
    - Isolation: Transazioni isolate tra loro
    - Durability: Dati persistenti su disk
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.RLock()
        self._initialize_database()

    def _initialize_database(self):
        """Inizializza schema database con tabelle ottimizzate"""
        try:
            with self._get_connection() as conn:
                # Tabella transazioni immutabile
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id VARCHAR PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        transaction_type VARCHAR NOT NULL,
                        amount DECIMAL(15,2) NOT NULL,
                        balance_after DECIMAL(15,2) NOT NULL,
                        description VARCHAR NOT NULL,
                        bet_id VARCHAR,
                        game_id VARCHAR,
                        metadata JSON,
                        checksum VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Crea indici separatamente per DuckDB
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_transaction_type ON transactions(transaction_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bet_id ON transactions(bet_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_game_id ON transactions(game_id)"
                )

                # Tabella scommesse immutabile
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bet_records (
                        bet_id VARCHAR PRIMARY KEY,
                        game_id VARCHAR NOT NULL,
                        bet_type VARCHAR NOT NULL,
                        selection VARCHAR NOT NULL,
                        odds DECIMAL(10,4) NOT NULL,
                        stake DECIMAL(15,2) NOT NULL,
                        result VARCHAR NOT NULL,
                        payout DECIMAL(15,2) NOT NULL,
                        profit_loss DECIMAL(15,2) NOT NULL,
                        placed_timestamp TIMESTAMP NOT NULL,
                        settled_timestamp TIMESTAMP,
                        metadata JSON,
                        checksum VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Crea indici separatamente per DuckDB
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_game_id ON bet_records(game_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_result ON bet_records(result)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_placed_timestamp ON bet_records(placed_timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_settled_timestamp ON bet_records(settled_timestamp)"
                )

                # Tabella snapshots stato per performance
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bankroll_snapshots (
                        snapshot_id VARCHAR PRIMARY KEY,
                        current_balance DECIMAL(15,2) NOT NULL,
                        total_deposits DECIMAL(15,2) NOT NULL,
                        total_withdrawals DECIMAL(15,2) NOT NULL,
                        total_bets_placed DECIMAL(15,2) NOT NULL,
                        total_bets_won DECIMAL(15,2) NOT NULL,
                        total_bets_lost DECIMAL(15,2) NOT NULL,
                        total_fees DECIMAL(15,2) NOT NULL,
                        active_bets_count INTEGER NOT NULL,
                        pending_bets_count INTEGER NOT NULL,
                        last_transaction_timestamp TIMESTAMP,
                        last_updated TIMESTAMP NOT NULL,
                        checksum VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Crea indice separatamente per DuckDB
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_last_updated ON bankroll_snapshots(last_updated)"
                )

                # DuckDB non supporta trigger complessi, usa vista per snapshot più recente
                conn.execute("""
                    CREATE OR REPLACE VIEW latest_bankroll_snapshot AS
                    SELECT * FROM bankroll_snapshots
                    WHERE created_at = (
                        SELECT MAX(created_at) FROM bankroll_snapshots
                    )
                """)

                logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseConnectionError(
                f"Database initialization failed: {e}", database_path=self.db_path
            )

    @contextmanager
    def _get_connection(self):
        """Context manager per connessione thread-safe"""
        with self._lock:
            try:
                if self._connection is None:
                    self._connection = duckdb.connect(self.db_path)

                # Imposta configurazioni per performance e consistenza
                # DuckDB non supporta PRAGMA specifici di SQLite, usa configurazioni predefinite
                # Le configurazioni DuckDB sono gestite automaticamente

                yield self._connection

            except Exception as e:
                logger.error(f"Database connection error: {e}")
                raise DatabaseConnectionError(
                    f"Connection failed: {e}", database_path=self.db_path
                )

    def _verify_transaction_integrity(self, transaction: Transaction) -> bool:
        """Verifica integrità transazione con checksum"""
        try:
            return transaction.verify_checksum()
        except Exception as e:
            logger.error(f"Transaction integrity check failed: {e}")
            return False

    def _check_duplicate_transaction(self, transaction_id: str) -> bool:
        """Verifica se transazione esiste già"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE transaction_id = ?",
                [transaction_id],
            ).fetchone()
            return result[0] > 0

    def get_current_balance(self) -> Decimal:
        """Restituisce il balance corrente (Public 3.0 API)"""
        return self._get_current_balance()

    def _get_current_balance(self) -> Decimal:
        """Ottiene balance corrente da snapshot più recente"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT current_balance FROM latest_bankroll_snapshot LIMIT 1"
            ).fetchone()

            if result:
                return Decimal(str(result[0]))

            # Fallback: calcola da transazioni
            result = conn.execute(
                "SELECT balance_after FROM transactions ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            return Decimal(str(result[0])) if result else Decimal("0.00")

    def _execute_transaction(self, transaction: Transaction) -> bool:
        """Esegue transazione singola con validazioni"""
        # Verifica integrità
        if not self._verify_transaction_integrity(transaction):
            raise ChecksumMismatchError(
                "Transaction checksum verification failed",
                transaction.checksum,
                "CORRUPTED",
                "transaction",
                transaction.transaction_id,
            )

        # Verifica duplicati
        if self._check_duplicate_transaction(transaction.transaction_id):
            raise DuplicateTransactionError(
                "Transaction already exists", transaction.transaction_id
            )

        with self._get_connection() as conn:
            try:
                # Inizia transazione
                conn.execute("BEGIN TRANSACTION")

                # Verifica fondi sufficienti per operazioni di debito
                current_balance = self._get_current_balance()
                if transaction.amount < 0:
                    if current_balance + transaction.amount < 0:
                        raise InsufficientFundsError(
                            "Insufficient funds for transaction",
                            float(abs(transaction.amount)),
                            float(current_balance),
                        )

                # Inserisci transazione
                conn.execute(
                    """INSERT INTO transactions
                       (transaction_id, timestamp, transaction_type, amount,
                        balance_after, description, bet_id, game_id, metadata, checksum)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        transaction.transaction_id,
                        transaction.timestamp.astimezone(timezone.utc).replace(
                            tzinfo=None
                        ),
                        transaction.transaction_type.value,
                        float(transaction.amount),
                        float(transaction.balance_after),
                        transaction.description,
                        transaction.bet_id,
                        transaction.game_id,
                        json.dumps(transaction.metadata),
                        transaction.checksum,
                    ],
                )

                # Commit transazione
                conn.execute("COMMIT")

                logger.info(f"Transaction executed: {transaction.transaction_id}")
                return True

            except Exception as e:
                # Rollback in caso di errore
                conn.execute("ROLLBACK")
                logger.error(f"Transaction failed: {e}")
                raise TransactionError(f"Transaction execution failed: {e}")

    def place_bet(self, bet_record: BetRecord) -> str:
        """Piazza scommessa con transazioni atomiche"""
        with self._lock:
            # Genera ID transazione
            placement_txn_id = create_transaction_id()

            # Transazione di piazzamento scommessa
            placement_txn = Transaction(
                transaction_id=placement_txn_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type=TransactionType.BET_PLACEMENT,
                amount=-bet_record.stake,
                balance_after=self._get_current_balance() - bet_record.stake,
                description=f"Bet placed: {bet_record.bet_type} {bet_record.selection}",
                bet_id=bet_record.bet_id,
                game_id=bet_record.game_id,
                metadata={"bet_details": bet_record.to_dict()},
            )

            # Esegui transazione
            self._execute_transaction(placement_txn)

            # Salva record scommessa
            self._save_bet_record(bet_record)

            return placement_txn_id

    def settle_bet(
        self, bet_id: str, result: BetResult, payout: Decimal, profit_loss: Decimal
    ) -> str:
        """
        Settle an existing bet, creating a settlement transaction.
        """
        with self._lock:
            # Recupera record scommessa
            bet_record = self.get_bet_record(bet_id)
            if not bet_record:
                raise TransactionError(f"Bet record not found: {bet_id}")

            # Genera ID transazione
            settlement_txn_id = create_transaction_id()

            # Transazione di settlement
            settlement_txn = Transaction(
                transaction_id=settlement_txn_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type=TransactionType.BET_SETTLEMENT,
                amount=payout,
                balance_after=self._get_current_balance() + payout,
                description=f"Bet settled: {result.value} P&L: {profit_loss}",
                bet_id=bet_id,
                game_id=bet_record.game_id,
                metadata={
                    "result": result.value,
                    "payout": float(payout),
                    "profit_loss": float(profit_loss),
                },
            )

            # Esegui transazione
            self._execute_transaction(settlement_txn)

            # Aggiorna record scommessa
            self._update_bet_record(bet_id, result, payout, profit_loss)

            return settlement_txn_id

    def add_deposit(self, amount: Decimal, description: str = "Deposit") -> str:
        """Aggiunge deposito al bankroll"""
        with self._lock:
            if amount <= 0:
                raise TransactionError("Deposit amount must be positive")

            txn_id = create_transaction_id()
            current_balance = self._get_current_balance()

            txn = Transaction(
                transaction_id=txn_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type=TransactionType.DEPOSIT,
                amount=amount,
                balance_after=current_balance + amount,
                description=description,
                metadata={"deposit_source": "manual"},
            )

            self._execute_transaction(txn)
            return txn_id

    def add_withdrawal(self, amount: Decimal, description: str = "Withdrawal") -> str:
        """Esegue prelievo dal bankroll"""
        with self._lock:
            if amount <= 0:
                raise TransactionError("Withdrawal amount must be positive")

            current_balance = self._get_current_balance()
            if current_balance < amount:
                raise InsufficientFundsError(
                    "Insufficient funds for withdrawal",
                    float(amount),
                    float(current_balance),
                )

            txn_id = create_transaction_id()

            txn = Transaction(
                transaction_id=txn_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type=TransactionType.WITHDRAWAL,
                amount=-amount,
                balance_after=current_balance - amount,
                description=description,
                metadata={"withdrawal_type": "manual"},
            )

            self._execute_transaction(txn)
            return txn_id

    def _save_bet_record(self, bet_record: BetRecord):
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO bet_records
                       (bet_id, game_id, bet_type, selection, odds, stake, result,
                        payout, profit_loss, placed_timestamp, settled_timestamp,
                        metadata, checksum)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    bet_record.bet_id,
                    bet_record.game_id,
                    bet_record.bet_type,
                    bet_record.selection,
                    float(bet_record.odds),
                    float(bet_record.stake),
                    bet_record.result.value,
                    float(bet_record.payout),
                    float(bet_record.profit_loss),
                    bet_record.placed_timestamp.astimezone(timezone.utc).replace(
                        tzinfo=None
                    ),
                    bet_record.settled_timestamp.astimezone(timezone.utc).replace(
                        tzinfo=None
                    )
                    if bet_record.settled_timestamp
                    else None,
                    json.dumps(bet_record.metadata),
                    bet_record.checksum,
                ],
            )

    def _update_bet_record(
        self, bet_id: str, result: BetResult, payout: Decimal, profit_loss: Decimal
    ):
        """Aggiorna record scommessa (solo settlement) incl. checksum"""
        # Recupera record attuale per ricalcolo checksum
        current_record = self.get_bet_record(bet_id)
        if not current_record:
            raise ValueError(f"Cannot update non-existent bet: {bet_id}")

        settled_ts = datetime.now(timezone.utc)

        # Crea nuovo record con valori aggiornati (triggera ricalcolo checksum)
        updated_record = replace(
            current_record,
            result=result,
            payout=payout,
            profit_loss=profit_loss,
            settled_timestamp=settled_ts,
        )

        with self._get_connection() as conn:
            conn.execute(
                """UPDATE bet_records
                   SET result = ?, payout = ?, profit_loss = ?,
                       settled_timestamp = ?, checksum = ?
                   WHERE bet_id = ?""",
                [
                    result.value,
                    float(payout),
                    float(profit_loss),
                    settled_ts.replace(tzinfo=None),
                    updated_record.checksum,
                    bet_id,
                ],
            )

    def get_bet_record(self, bet_id: str) -> Optional[BetRecord]:
        """Recupera record scommessa per ID"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM bet_records WHERE bet_id = ?",
                [bet_id],
            ).fetchone()

            if result:
                return self._row_to_bet_record(result)
            return None

    def _row_to_bet_record(self, row) -> BetRecord:
        """Converte row database in BetRecord"""
        placed_ts = row[9]
        if isinstance(placed_ts, str):
            placed_ts = datetime.fromisoformat(placed_ts)
        if placed_ts.tzinfo is None:
            placed_ts = placed_ts.replace(tzinfo=timezone.utc)

        settled_ts = row[10]
        if settled_ts:
            if isinstance(settled_ts, str):
                settled_ts = datetime.fromisoformat(settled_ts)
            if settled_ts.tzinfo is None:
                settled_ts = settled_ts.replace(tzinfo=timezone.utc)

        metadata = row[11]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = eval(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        bet_record = BetRecord(
            bet_id=row[0],
            game_id=row[1],
            bet_type=row[2],
            selection=row[3],
            odds=Decimal(str(row[4])),
            stake=Decimal(str(row[5])),
            result=BetResult(row[6]),
            payout=Decimal(str(row[7])),
            profit_loss=Decimal(str(row[8])),
            placed_timestamp=placed_ts,
            settled_timestamp=settled_ts,
            metadata=metadata,
        )

        # Verify checksum matches stored value
        if bet_record.checksum != row[12]:
            raise ChecksumMismatchError(
                f"Checksum mismatch for bet {row[0]}: expected {bet_record.checksum}, found {row[12]}",
                expected_checksum=bet_record.checksum,
                actual_checksum=row[12],
                entity_type="bet_record",
                entity_id=row[0],
            )

        return bet_record

        # Verifica integrità record
        recalc_checksum = bet_record._calculate_checksum()
        if recalc_checksum != row[12]:
            msg = f"Checksum mismatch for bet {row[0]}: expected {recalc_checksum}, found {row[12]}"
            logger.critical(msg)
            raise ChecksumMismatchError(
                msg,
                expected_checksum=recalc_checksum,
                actual_checksum=row[12],
                entity_type="bet_record",
                entity_id=row[0],
            )

        return bet_record

    def get_transactions(
        self, limit: int = 100, transaction_type: Optional[TransactionType] = None
    ) -> List[Transaction]:
        """Recupera transazioni con filtri"""
        with self._get_connection() as conn:
            query = "SELECT * FROM transactions"
            params = []

            if transaction_type:
                query += " WHERE transaction_type = ?"
                params.append(transaction_type.value)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            results = conn.execute(query, params).fetchall()
            return [self._row_to_transaction(row) for row in results]

    def _row_to_transaction(self, row) -> Transaction:
        """Converte row database in Transaction"""
        ts = row[1]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        metadata = row[8]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = eval(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        txn = Transaction(
            transaction_id=row[0],
            timestamp=ts,
            transaction_type=TransactionType(row[2]),
            amount=Decimal(str(row[3])),
            balance_after=Decimal(str(row[4])),
            description=row[5],
            bet_id=row[6],
            game_id=row[7],
            metadata=metadata,
        )
        # Verifica che il checksum corrisponda
        if txn.checksum != row[9]:
            msg = f"Checksum mismatch for transaction {row[0]}: expected {txn.checksum}, found {row[9]}"
            logger.critical(msg)
            raise ChecksumMismatchError(
                msg,
                expected_checksum=txn.checksum,
                actual_checksum=row[9],
                entity_type="transaction",
                entity_id=row[0],
            )
        return txn

    def get_transaction_count(self) -> int:
        """Conta totale transazioni"""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
            return result[0]

    def get_bet_records(
        self, limit: int = 100, result_filter: Optional[BetResult] = None
    ) -> List[BetRecord]:
        """Recupera record scommesse con filtri"""
        with self._get_connection() as conn:
            query = "SELECT * FROM bet_records"
            params = []

            if result_filter:
                query += " WHERE result = ?"
                params.append(result_filter.value)

            query += " ORDER BY placed_timestamp DESC LIMIT ?"
            params.append(limit)

            results = conn.execute(query, params).fetchall()
            return [self._row_to_bet_record(row) for row in results]

    def close(self):
        """Chiudi connessione database"""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
                logger.info("Database connection closed")

    def __enter__(self):
        return self

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Transaction]:
        """Recupera transazione per ID"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM transactions WHERE transaction_id = ?",
                [transaction_id],
            ).fetchone()

            if result:
                return self._row_to_transaction(result)
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
