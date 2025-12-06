"""
NBA Bankroll 3.0 - Bankroll State Calculator
Calcolatore di stato basato su event sourcing dal ledger immutabile
Implementa pattern Event Sourcing per ricostruzione stato garantita
"""

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from .models import Transaction, TransactionType, BetResult, BetRecord, BankrollState
from .exceptions import IntegrityError, ChecksumMismatchError
from .engine import TransactionEngine


logger = logging.getLogger(__name__)


class BankrollStateCalculator:
    """
    Calcolatore stato bankroll con event sourcing:
    - Ricostruisce stato completo da ledger immutabile
    - Calcoli precisi con Decimal arithmetic
    - Performance ottimizzata con caching intelligente
    - Validazione integrità continua
    """

    def __init__(self, engine: TransactionEngine):
        self.engine = engine
        self._state_cache = {}
        self._cache_ttl = timedelta(minutes=5)  # Cache di 5 minuti
        self._performance_stats = {
            "calculations": 0,
            "cache_hits": 0,
            "avg_calculation_time": 0.0,
        }

    def invalidate_cache(self):
        """Invalida cache stato corrente"""
        self._state_cache.clear()
        logger.debug("Bankroll state cache invalidated")

    def get_current_state(self, force_recalculate: bool = False) -> BankrollState:
        """
        Calcola stato corrente del bankroll dal ledger
        Usa cache intelligente per performance
        """
        start_time = datetime.now()

        # Verifica cache
        cache_key = "current_state"
        if not force_recalculate and self._is_cache_valid(cache_key):
            self._performance_stats["cache_hits"] += 1
            logger.debug("Using cached bankroll state")
            return self._state_cache[cache_key]["state"]

        try:
            # Ricostruisci stato da zero (event sourcing)
            state = self._rebuild_state_from_ledger()

            # Validazione integrità stato
            self._validate_state_integrity(state)

            # Cache risultato
            self._cache_state(cache_key, state)

            # Update performance stats
            calc_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(calc_time)

            logger.info(f"Bankroll state calculated in {calc_time:.3f}s")
            return state

        except Exception as e:
            logger.error(f"Failed to calculate bankroll state: {e}")
            raise IntegrityError(
                "State calculation failed", "bankroll_state", "valid_state", str(e)
            )

    def _rebuild_state_from_ledger(self) -> BankrollState:
        """
        Ricostruisce stato completo da ledger transazioni
        Implementa pattern Event Sourcing puro
        """
        # Inizializza aggregati
        totals = defaultdict(Decimal)
        totals.update(
            {
                "current_balance": Decimal("0.00"),
                "total_deposits": Decimal("0.00"),
                "total_withdrawals": Decimal("0.00"),
                "total_bets_placed": Decimal("0.00"),
                "total_bets_won": Decimal("0.00"),
                "total_bets_lost": Decimal("0.00"),
                "total_fees": Decimal("0.00"),
                "active_bets_count": 0,
                "pending_bets_count": 0,
            }
        )

        # Processa tutte le transazioni in ordine cronologico
        transactions = self.engine.get_transactions(limit=10000)  # Large batch
        transactions.reverse()  # DuckDB returns DESC (newest first), we need ASC (oldest first)

        last_timestamp = None
        for txn in transactions:
            # Aggiorna aggregati basandosi sul tipo
            if txn.transaction_type == TransactionType.DEPOSIT:
                totals["total_deposits"] += txn.amount
                totals["current_balance"] += txn.amount

            elif txn.transaction_type == TransactionType.WITHDRAWAL:
                totals["total_withdrawals"] += abs(txn.amount)
                totals["current_balance"] += txn.amount  # amount è negativo

            elif txn.transaction_type == TransactionType.BET_PLACEMENT:
                totals["total_bets_placed"] += abs(txn.amount)
                totals["current_balance"] += txn.amount  # amount è negativo

            elif txn.transaction_type == TransactionType.BET_SETTLEMENT:
                if txn.amount > 0:
                    totals["total_bets_won"] += txn.amount
                else:
                    totals["total_bets_lost"] += abs(txn.amount)
                totals["current_balance"] += txn.amount

            elif txn.transaction_type == TransactionType.FEE:
                totals["total_fees"] += abs(txn.amount)
                totals["current_balance"] += txn.amount  # amount è negativo

            last_timestamp = txn.timestamp

        # Calcola conteggi scommesse
        bet_counts = self._calculate_bet_counts()
        totals["active_bets_count"] = bet_counts["active"]
        totals["pending_bets_count"] = bet_counts["pending"]

        # Crea stato finale
        state = BankrollState(
            current_balance=totals["current_balance"],
            total_deposits=totals["total_deposits"],
            total_withdrawals=totals["total_withdrawals"],
            total_bets_placed=totals["total_bets_placed"],
            total_bets_won=totals["total_bets_won"],
            total_bets_lost=totals["total_bets_lost"],
            total_fees=totals["total_fees"],
            active_bets_count=totals["active_bets_count"],
            pending_bets_count=totals["pending_bets_count"],
            last_transaction_timestamp=last_timestamp,
            last_updated=datetime.now(timezone.utc),
        )

        return state

    def _calculate_bet_counts(self) -> Dict[str, int]:
        """Calcola conteggi scommesse per stato"""
        counts = {"active": 0, "pending": 0}

        try:
            # Recupera tutte le scommesse
            bet_records = self.engine.get_bet_records(limit=10000)

            for bet in bet_records:
                if bet.result in [BetResult.PENDING]:
                    counts["pending"] += 1
                    counts["active"] += 1  # PENDING counts as ACTIVE
                elif bet.result in [BetResult.WON, BetResult.LOST, BetResult.PUSH]:
                    # Scommesse concluse
                    pass
                else:
                    # Altri stati considerati attivi
                    counts["active"] += 1

        except Exception as e:
            logger.warning(f"Failed to calculate bet counts: {e}")

        return counts

    def _validate_state_integrity(self, state: BankrollState):
        """
        Validazione integrità stato con multiple checksum
        Verifica consistenza matematica e checksum SHA-256
        """
        try:
            # Verifica checksum SHA-256
            if not state.verify_checksum():
                raise ChecksumMismatchError(
                    "Bankroll state checksum verification failed",
                    state.checksum,
                    "CALCULATED",
                    "bankroll_state",
                )

            # Validazione matematica fondamentale
            expected_balance = (
                state.total_deposits
                - state.total_withdrawals
                - state.total_bets_placed
                + state.total_bets_won
                - state.total_bets_lost
                - state.total_fees
            )

            if abs(state.current_balance - expected_balance) > Decimal("0.01"):
                raise IntegrityError(
                    "Balance calculation mismatch",
                    "balance_equation",
                    expected_balance,
                    state.current_balance,
                )

            # Validazioni business logic
            if state.total_deposits < 0:
                raise IntegrityError(
                    "Total deposits cannot be negative",
                    "deposits_positive",
                    ">= 0",
                    state.total_deposits,
                )

            if state.total_withdrawals < 0:
                raise IntegrityError(
                    "Total withdrawals cannot be negative",
                    "withdrawals_positive",
                    ">= 0",
                    state.total_withdrawals,
                )

            if state.active_bets_count < 0 or state.pending_bets_count < 0:
                raise IntegrityError(
                    "Bet counts cannot be negative",
                    "bet_counts_positive",
                    ">= 0",
                    f"active={state.active_bets_count}, pending={state.pending_bets_count}",
                )

            logger.debug("Bankroll state integrity validation passed")

        except (ChecksumMismatchError, IntegrityError):
            raise  # Rilancia eccezioni di integrità
        except Exception as e:
            raise IntegrityError(
                "State integrity validation failed",
                "comprehensive_check",
                "valid",
                str(e),
            )

    def get_state_at_timestamp(self, timestamp: datetime) -> BankrollState:
        """
        Ricostruisce stato a timestamp specifico (time travel)
        Utile per audit e analisi storiche
        """
        try:
            # Filtra transazioni fino a timestamp
            transactions = self.engine.get_transactions(limit=10000)
            filtered_txns = [t for t in transactions if t.timestamp <= timestamp]

            # Ricostruisci stato parziale
            totals = defaultdict(Decimal)
            totals.update(
                {
                    "current_balance": Decimal("0.00"),
                    "total_deposits": Decimal("0.00"),
                    "total_withdrawals": Decimal("0.00"),
                    "total_bets_placed": Decimal("0.00"),
                    "total_bets_won": Decimal("0.00"),
                    "total_bets_lost": Decimal("0.00"),
                    "total_fees": Decimal("0.00"),
                    "active_bets_count": 0,
                    "pending_bets_count": 0,
                }
            )

            last_txn_timestamp = None
            for txn in filtered_txns:
                # Stessa logica di _rebuild_state_from_ledger
                if txn.transaction_type == TransactionType.DEPOSIT:
                    totals["total_deposits"] += txn.amount
                    totals["current_balance"] += txn.amount
                elif txn.transaction_type == TransactionType.WITHDRAWAL:
                    totals["total_withdrawals"] += abs(txn.amount)
                    totals["current_balance"] += txn.amount
                elif txn.transaction_type == TransactionType.BET_PLACEMENT:
                    totals["total_bets_placed"] += abs(txn.amount)
                    totals["current_balance"] += txn.amount
                elif txn.transaction_type == TransactionType.BET_SETTLEMENT:
                    if txn.amount > 0:
                        totals["total_bets_won"] += txn.amount
                    else:
                        totals["total_bets_lost"] += abs(txn.amount)
                    totals["current_balance"] += txn.amount
                elif txn.transaction_type == TransactionType.FEE:
                    totals["total_fees"] += abs(txn.amount)
                    totals["current_balance"] += txn.amount

                last_txn_timestamp = txn.timestamp

            # Calcola conteggi scommesse al timestamp
            bet_counts = self._calculate_historical_bet_counts(timestamp)

            # Crea stato storico
            historical_state = BankrollState(
                current_balance=totals["current_balance"],
                total_deposits=totals["total_deposits"],
                total_withdrawals=totals["total_withdrawals"],
                total_bets_placed=totals["total_bets_placed"],
                total_bets_won=totals["total_bets_won"],
                total_bets_lost=totals["total_bets_lost"],
                total_fees=totals["total_fees"],
                active_bets_count=bet_counts["active"],
                pending_bets_count=bet_counts["pending"],
                last_transaction_timestamp=last_txn_timestamp,
                last_updated=datetime.now(timezone.utc),
            )

            return historical_state

        except Exception as e:
            logger.error(f"Failed to reconstruct state at timestamp {timestamp}: {e}")
            raise IntegrityError(
                "Historical state reconstruction failed",
                "time_travel",
                "valid_state",
                str(e),
            )

    def _calculate_historical_bet_counts(self, timestamp: datetime) -> Dict[str, int]:
        """Calcola conteggi scommesse storiche"""
        counts = {"active": 0, "pending": 0}

        try:
            bet_records = self.engine.get_bet_records(limit=10000)

            for bet in bet_records:
                if bet.placed_timestamp <= timestamp:
                    if bet.result in [BetResult.PENDING]:
                        counts["pending"] += 1
                    elif bet.result in [BetResult.WON, BetResult.LOST, BetResult.PUSH]:
                        pass
                    else:
                        counts["active"] += 1

        except Exception as e:
            logger.warning(f"Failed to calculate historical bet counts: {e}")

        return counts

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche performance del calcolatore"""
        return {
            "calculations_performed": self._performance_stats["calculations"],
            "cache_hit_rate": (
                self._performance_stats["cache_hits"]
                / max(1, self._performance_stats["calculations"])
            )
            if self._performance_stats["calculations"] > 0
            else 0,
            "average_calculation_time": self._performance_stats["avg_calculation_time"],
            "cache_size": len(self._state_cache),
            "cache_ttl_minutes": self._cache_ttl.total_seconds() / 60,
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica se cache entry è ancora valida"""
        if cache_key not in self._state_cache:
            return False

        cache_entry = self._state_cache[cache_key]
        return datetime.now(timezone.utc) - cache_entry["timestamp"] < self._cache_ttl

    def _cache_state(self, cache_key: str, state: BankrollState):
        """Salva stato in cache"""
        self._state_cache[cache_key] = {
            "state": state,
            "timestamp": datetime.now(timezone.utc),
        }

        # Cleanup cache entries vecchi
        self._cleanup_cache()

    def _cleanup_cache(self):
        """Rimuove entry cache scadute"""
        current_time = datetime.now(timezone.utc)
        expired_keys = [
            key
            for key, entry in self._state_cache.items()
            if current_time - entry["timestamp"] > self._cache_ttl
        ]

        for key in expired_keys:
            del self._state_cache[key]

    def _update_performance_stats(self, calculation_time: float):
        """Aggiorna statistiche performance"""
        self._performance_stats["calculations"] += 1

        # Calcola media mobile
        current_avg = self._performance_stats["avg_calculation_time"]
        n = self._performance_stats["calculations"]
        new_avg = ((current_avg * (n - 1)) + calculation_time) / n
        self._performance_stats["avg_calculation_time"] = new_avg

    def clear_cache(self):
        """Svuota cache completamente"""
        self._state_cache.clear()
        logger.info("Bankroll state cache cleared")

    def invalidate_cache(self, cache_key: Optional[str] = None):
        """Invalida cache entry specifico o tutto"""
        if cache_key and cache_key in self._state_cache:
            del self._state_cache[cache_key]
            logger.info(f"Cache entry {cache_key} invalidated")
        else:
            self.clear_cache()

    def get_state_summary(
        self, state: Optional[BankrollState] = None
    ) -> Dict[str, Any]:
        """
        Restituisce riepilogo stato leggibile
        Include metriche derivate e KPI
        """
        if state is None:
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

        return {
            "balance": float(state.current_balance),
            "total_deposited": float(state.total_deposits),
            "total_withdrawn": float(state.total_withdrawals),
            "total_bet_amount": float(state.total_bets_placed),
            "total_winnings": float(state.total_bets_won),
            "total_losses": float(state.total_bets_lost),
            "total_fees": float(state.total_fees),
            "net_profit": float(state.total_bets_won - state.total_bets_lost),
            "roi_percentage": float(roi),
            "win_rate_percentage": float(win_rate),
            "average_bet_size": float(avg_bet_size),
            "active_bets": state.active_bets_count,
            "pending_bets": state.pending_bets_count,
            "last_transaction": state.last_transaction_timestamp.isoformat()
            if state.last_transaction_timestamp
            else None,
            "last_updated": state.last_updated.isoformat(),
        }
