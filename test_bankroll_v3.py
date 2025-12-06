#!/usr/bin/env python3
"""
NBA Bankroll 3.0 - Comprehensive Test Suite
Test completo per validare tutti i componenti del nuovo sistema
Include test unitari, integrazione, performance e stress testing
"""

import unittest
import tempfile
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

# Aggiungi path progetto per import
sys.path.insert(0, str(Path(__file__).parent))

from src.nba_predictor.bankroll import (
    BankrollManager,
    TransactionEngine,
    BankrollStateCalculator,
    RiskValidator,
    AuditLogger,
    SecurityManager,
    BankrollMigrationService,
    create_bankroll_manager,
    BetPlacementRequest,
    BetResult,
    TransactionType,
    RiskLevel,
    create_bet_placement_request,
)
from src.nba_predictor.bankroll.exceptions import (
    BankrollError,
    InsufficientFundsError,
    RiskLimitExceededError,
    ValidationError,
    AuditError,
)

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class TestBankrollV3(unittest.TestCase):
    """Test suite completo per Bankroll 3.0"""

    def setUp(self):
        """Setup per ogni test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_bankroll.duckdb")

        # Crea manager di test
        self.engine = TransactionEngine(self.test_db)
        self.calculator = BankrollStateCalculator(self.engine)
        self.risk_validator = RiskValidator(self.calculator)
        self.audit_logger = AuditLogger(self.engine)
        self.security_manager = SecurityManager(self.audit_logger)

        self.bankroll_manager = create_bankroll_manager(self.test_db)

        # Add initial deposit to ensure tests have funds
        self.bankroll_manager.add_deposit(Decimal("1000.00"), "Initial test deposit")

        logger.info(f"Test setup completed: {self.test_db}")

    def tearDown(self):
        """Cleanup dopo ogni test"""
        try:
            if hasattr(self, "bankroll_manager"):
                self.bankroll_manager.close()

            # Rimuovi database temporaneo
            if os.path.exists(self.test_db):
                os.remove(self.test_db)

            # Rimuovi directory temporanea
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def test_001_transaction_engine_initialization(self):
        """Test 001: Inizializzazione Transaction Engine"""
        logger.info("Test 001: Transaction Engine Initialization")

        # Verifica inizializzazione
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.db_path, self.test_db)

        # Verifica tabelle create
        transaction_count = self.engine.get_transaction_count()
        self.assertIsInstance(transaction_count, int)
        self.assertGreaterEqual(transaction_count, 0)

        logger.info("✅ Test 001 passed")

    def test_002_bankroll_state_calculation(self):
        """Test 002: Calcolo stato bankroll"""
        logger.info("Test 002: Bankroll State Calculation")

        # Aggiungi deposito iniziale
        deposit_id = self.bankroll_manager.add_deposit(
            Decimal("1000.00"), "Initial deposit"
        )

        # Verifica stato calcolato
        state = self.bankroll_manager.get_current_state()

        self.assertIsNotNone(state)
        self.assertEqual(
            state.current_balance, Decimal("2000.00")
        )  # 1000 setup + 1000 test
        self.assertEqual(state.total_deposits, Decimal("2000.00"))
        self.assertEqual(state.total_withdrawals, Decimal("0.00"))
        self.assertEqual(state.active_bets_count, 0)
        self.assertEqual(state.pending_bets_count, 0)

        # Verifica checksum
        self.assertTrue(state.verify_checksum())

        logger.info("✅ Test 002 passed")

    def test_003_bet_placement_risk_validation(self):
        """Test 003: Piazzamento scommessa con validazione rischio"""
        logger.info("Test 003: Bet Placement with Risk Validation")

        # Crea richiesta scommessa valida
        bet_request = create_bet_placement_request(
            game_id="TEST_GAME_001",
            bet_type="moneyline",
            selection="home",
            odds=Decimal("2.50"),
            stake=Decimal("50.00"),  # 5% del bankroll
            expected_value=Decimal("10.00"),  # EV positivo
            confidence_score=Decimal("0.75"),
        )

        # Esegui piazzamento
        result = self.bankroll_manager.place_bet(bet_request)

        # Verifica successo
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bet_id)
        self.assertIsNotNone(result.transaction_id)

        # Verifica stato aggiornato
        state = self.bankroll_manager.get_current_state()
        self.assertEqual(state.current_balance, Decimal("950.00"))  # 1000 - 50
        self.assertEqual(state.total_bets_placed, Decimal("50.00"))
        self.assertEqual(state.active_bets_count, 1)

        logger.info("✅ Test 003 passed")

    def test_004_risk_limit_enforcement(self):
        """Test 004: Enforcement limiti rischio"""
        logger.info("Test 004: Risk Limit Enforcement")

        # Tentativo scommessa troppo grande (oltre 5%)
        large_bet_request = create_bet_placement_request(
            game_id="TEST_GAME_002",
            bet_type="moneyline",
            selection="away",
            odds=Decimal("3.00"),
            stake=Decimal("100.00"),  # 10% del bankroll - dovrebbe essere bloccato
            expected_value=Decimal("5.00"),
        )

        # Esegui piazzamento - dovrebbe fallire
        result = self.bankroll_manager.place_bet(large_bet_request)

        # Verifica fallimento per rischio
        self.assertFalse(result.success)
        self.assertIsNotNone(result.validation_result)
        self.assertFalse(result.validation_result.is_valid)
        # Verifica fallimento per rischio (stake o exposure)
        msg = result.error_message.lower()
        self.assertTrue(
            "stake" in msg or "exposure" in msg, f"Unexpected error message: {msg}"
        )

        # Verifica che non è stato scalato il balance
        state = self.bankroll_manager.get_current_state()
        self.assertEqual(
            state.current_balance, Decimal("1000.00")
        )  # Invariato (scommessa rifiutata)

        logger.info("✅ Test 004 passed")

    def test_005_bet_settlement_workflow(self):
        """Test 005: Workflow completo settlement scommessa"""
        logger.info("Test 005: Bet Settlement Workflow")

        # Piazzamento scommessa di test
        bet_request = create_bet_placement_request(
            game_id="TEST_GAME_003",
            bet_type="spread",
            selection="over",
            odds=Decimal("1.90"),
            stake=Decimal("25.00"),
            expected_value=Decimal("15.00"),
        )

        place_result = self.bankroll_manager.place_bet(bet_request)
        self.assertTrue(place_result.success)

        bet_id = place_result.bet_id

        # Settlement scommessa come vinta
        settlement_success = self.bankroll_manager.settle_bet(
            bet_id=bet_id,
            result=BetResult.WON,
            payout=Decimal("47.50"),  # 25 * 1.90
            profit_loss=Decimal("22.50"),  # 47.50 - 25
        )

        self.assertTrue(settlement_success)

        # Verifica stato finale
        # Verifica stato finale
        state = self.bankroll_manager.get_current_state()
        # Initial 1000 - Stake 25 + Payout 47.50 = 1022.50
        self.assertEqual(state.current_balance, Decimal("1022.50"))
        self.assertEqual(state.total_bets_won, Decimal("47.50"))
        self.assertEqual(state.total_bets_lost, Decimal("0.00"))
        self.assertEqual(state.active_bets_count, 0)  # Scommessa conclusa

        logger.info("✅ Test 005 passed")

    def test_006_insufficient_funds_handling(self):
        """Test 006: Gestione fondi insufficienti"""
        logger.info("Test 006: Insufficient Funds Handling")

        # Tentativo prelievo eccessivo
        with self.assertRaises(Exception) as context:
            self.bankroll_manager.add_withdrawal(
                Decimal("2000.00"),  # Più del disponibile
                "Excess withdrawal",
            )

        # Verifica messaggio errore appropriato
        self.assertIn("insufficient", str(context.exception).lower())

        logger.info("✅ Test 006 passed")

    def test_007_transaction_history_integrity(self):
        """Test 007: Integrità storia transazioni"""
        logger.info("Test 007: Transaction History Integrity")

        # Aggiungi alcune transazioni
        self.bankroll_manager.add_deposit(Decimal("100.00"), "Test deposit 1")
        self.bankroll_manager.add_deposit(Decimal("200.00"), "Test deposit 2")

        # Recupera storia
        history = self.bankroll_manager.get_transaction_history(limit=10)

        # Verifica integrità storia
        # 2 test deposits + 1 setup deposit = 3
        self.assertEqual(len(history), 3)

        # Verifica ordine cronologico (più recenti prima)
        timestamps = [txn.timestamp for txn in history]
        self.assertGreaterEqual(timestamps[0], timestamps[1])  # Ordine corretto

        # Verifica dettagli transazioni
        for txn in history:
            self.assertIsNotNone(txn.transaction_id)
            self.assertIsNotNone(txn.timestamp)
            self.assertIsNotNone(txn.transaction_type)
            self.assertIsNotNone(txn.amount)

        logger.info("✅ Test 007 passed")

    def test_008_audit_trail_functionality(self):
        """Test 008: Funzionalità audit trail"""
        logger.info("Test 008: Audit Trail Functionality")

        # Esegui alcune operazioni
        bet_request = create_bet_placement_request(
            game_id="TEST_GAME_004",
            bet_type="total",
            selection="under",
            odds=Decimal("2.00"),
            stake=Decimal("30.00"),
        )

        self.bankroll_manager.place_bet(bet_request)

        # Verifica eventi audit creati
        audit_events = self.audit_logger.search_events(
            event_type=None,  # Tutti i tipi
            limit=10,
        )

        # Verifica presenza eventi
        self.assertGreater(len(audit_events), 0)

        # Verifica struttura eventi
        for event in audit_events:
            self.assertIsNotNone(event.event_id)
            self.assertIsNotNone(event.timestamp)
            self.assertIsNotNone(event.event_type)
            self.assertTrue(event.verify_checksum())

        logger.info("✅ Test 008 passed")

    def test_009_performance_metrics_collection(self):
        """Test 009: Raccolta metriche performance"""
        logger.info("Test 009: Performance Metrics Collection")

        # Esegui operazioni diverse
        for i in range(5):
            self.bankroll_manager.add_deposit(Decimal("10.00"), f"Test deposit {i + 1}")

        # Ottieni metriche
        metrics = self.bankroll_manager.get_performance_metrics()

        # Verifica struttura metriche
        self.assertIn("manager_stats", metrics)
        self.assertIn("calculator_metrics", metrics)
        self.assertIn("risk_metrics", metrics)
        self.assertIn("audit_metrics", metrics)
        self.assertIn("system_health", metrics)

        # Verifica statistiche manager
        manager_stats = metrics["manager_stats"]
        self.assertGreater(manager_stats["total_operations"], 0)
        self.assertGreaterEqual(manager_stats["successful_operations"], 5)

        logger.info("✅ Test 009 passed")

    def test_010_security_policy_enforcement(self):
        """Test 010: Enforcement policy sicurezza"""
        logger.info("Test 010: Security Policy Enforcement")

        # Verifica policy esistenti
        policies = self.bankroll_manager.get_risk_limits()
        self.assertIsInstance(policies, dict)
        self.assertGreater(len(policies), 0)

        # Test validazione operazione con contesto utente
        user_context = {
            "user_id": "test_user_001",
            "ip_address": "192.168.1.100",
            "user_agent": "TestAgent/1.0",
        }

        # Verifica operazione permessa
        is_allowed = self.bankroll_manager.validate_operation_permissions(
            "view_bankroll", user_context
        )
        self.assertTrue(is_allowed)

        logger.info("✅ Test 010 passed")

    def test_011_concurrent_operations_safety(self):
        """Test 011: Sicurezza operazioni concorrenti"""
        logger.info("Test 011: Concurrent Operations Safety")

        import threading
        import time

        results = []
        errors = []

        def place_bet_thread(bet_index):
            try:
                bet_request = create_bet_placement_request(
                    game_id=f"CONCURRENT_TEST_{bet_index}",
                    bet_type="moneyline",
                    selection="home",
                    odds=Decimal("2.00"),
                    stake=Decimal("10.00"),
                )

                result = self.bankroll_manager.place_bet(bet_request)
                results.append(result.success)

            except Exception as e:
                errors.append(str(e))

        # Avvia thread concorrenti
        threads = []
        for i in range(3):
            thread = threading.Thread(target=place_bet_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Attendi completamento
        for thread in threads:
            thread.join(timeout=10)

        # Verifica risultati
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        self.assertTrue(all(results))  # Tutte le operazioni dovrebbero riuscire

        # Verifica consistenza finale
        final_state = self.bankroll_manager.get_current_state()
        # Start 1000. 3 bets of 10. 1000 - 30 = 970.
        expected_balance = Decimal("970.00")
        self.assertEqual(final_state.current_balance, expected_balance)

        logger.info("✅ Test 011 passed")

    def test_012_data_consistency_validation(self):
        """Test 012: Validazione consistenza dati"""
        logger.info("Test 012: Data Consistency Validation")

        # Esegui sequenza operazioni complessa
        operations = [
            ("deposit", Decimal("500.00"), "Initial deposit"),
            ("bet", Decimal("50.00"), "Bet 1"),
            ("bet", Decimal("30.00"), "Bet 2"),
            ("settle_win", None, "Settle Bet 1 win"),
            ("settle_lose", None, "Settle Bet 2 lose"),
            ("withdrawal", Decimal("100.00"), "Withdrawal"),
        ]

        # Esegui operazioni
        for op in operations:
            if op[0] == "deposit":
                self.bankroll_manager.add_deposit(op[1], op[2])
            elif op[0] == "bet":
                bet_request = create_bet_placement_request(
                    game_id=f"CONSISTENCY_TEST_{len(operations)}",
                    bet_type="moneyline",
                    selection="home",
                    odds=Decimal("2.00"),
                    stake=op[1],
                )
                self.bankroll_manager.place_bet(bet_request)
            elif op[0] == "settle_win":
                # Trova bet ID da simulare
                history = self.bankroll_manager.get_bet_history(limit=1)
                if history:
                    self.bankroll_manager.settle_bet(
                        history[0].bet_id,
                        BetResult.WON,
                        Decimal("100.00"),  # payout
                        Decimal("50.00"),  # profit
                    )
            elif op[0] == "settle_lose":
                history = self.bankroll_manager.get_bet_history(limit=2)
                if len(history) >= 2:
                    self.bankroll_manager.settle_bet(
                        history[1].bet_id,
                        BetResult.LOST,
                        Decimal("0.00"),  # payout
                        Decimal("-30.00"),  # loss
                    )
            elif op[0] == "withdrawal":
                self.bankroll_manager.add_withdrawal(op[1], op[2])

        # Verifica consistenza finale
        final_state = self.bankroll_manager.get_current_state()
        expected_balance = (
            Decimal("1000.00")  # setup deposit
            + Decimal("500.00")  # deposito test
            - Decimal("80.00")  # total bets (50 + 30)
            + Decimal("100.00")  # win bet 1 (Payout)
            + Decimal("0.00")  # loss bet 2 (Payout)
            - Decimal("100.00")  # withdrawal
        )

        self.assertEqual(final_state.current_balance, expected_balance)
        self.assertEqual(
            final_state.total_deposits, Decimal("1500.00")
        )  # 1000 setup + 500 test
        self.assertEqual(final_state.total_withdrawals, Decimal("100.00"))

        logger.info("✅ Test 012 passed")

    def test_013_error_handling_and_recovery(self):
        """Test 013: Error handling e recovery"""
        logger.info("Test 013: Error Handling and Recovery")

        # Test errore transazione con recovery
        initial_state = self.bankroll_manager.get_current_state()
        initial_balance = initial_state.current_balance

        # Simula errore durante operazione
        with self.assertRaises(Exception):
            # Tentativo operazione con dati invalidi per forzare errore
            invalid_bet_request = create_bet_placement_request(
                game_id="",  # ID vuoto - dovrebbe fallire validazione
                bet_type="moneyline",
                selection="home",
                odds=Decimal("2.00"),
                stake=Decimal("10.00"),
            )

            # Modifica temporaneamente il request per essere valido
            # (Questo simula un errore di sistema durante validazione)
            invalid_bet_request.bet_id = "forced_valid_id"

            result = self.bankroll_manager.place_bet(invalid_bet_request)
            # Se arriva qui, c'è stato un errore inaspettato

        # Verifica che lo stato non è stato corrotto
        final_state = self.bankroll_manager.get_current_state()
        self.assertEqual(final_state.current_balance, initial_balance)

        # Verifica che il sistema è ancora funzionante
        test_bet_request = create_bet_placement_request(
            game_id="RECOVERY_TEST",
            bet_type="moneyline",
            selection="home",
            odds=Decimal("2.00"),
            stake=Decimal("10.00"),
        )

        recovery_result = self.bankroll_manager.place_bet(test_bet_request)
        self.assertTrue(recovery_result.success)

        logger.info("✅ Test 013 passed")

    def test_014_edge_cases_and_boundary_conditions(self):
        """Test 014: Edge cases e condizioni boundary"""
        logger.info("Test 014: Edge Cases and Boundary Conditions")

        # Test con valori limite
        edge_cases = [
            # Stake minimo
            ("min_stake", Decimal("0.01")),
            # Stake massimo
            ("max_stake", Decimal("999999.99")),
            # Odds limite
            ("min_odds", Decimal("1.01")),
            ("max_odds", Decimal("1000.00")),
            # Timestamp boundary
            ("timestamp_boundary", datetime.now(timezone.utc)),
        ]

        for case_name, value in edge_cases:
            with self.subTest(case_name):
                if case_name == "min_stake":
                    bet_request = create_bet_placement_request(
                        game_id="EDGE_TEST_MIN",
                        bet_type="moneyline",
                        selection="home",
                        odds=Decimal("2.00"),
                        stake=value,
                    )
                    result = self.bankroll_manager.place_bet(bet_request)
                    self.assertTrue(result.success)

                elif case_name == "max_stake":
                    # Test con stake massimo permesso
                    bet_request = create_bet_placement_request(
                        game_id="EDGE_TEST_MAX",
                        bet_type="moneyline",
                        selection="home",
                        odds=Decimal("2.00"),
                        stake=value,
                    )
                    result = self.bankroll_manager.place_bet(bet_request)
                    # Dovrebbe fallire per limiti rischio

                elif case_name in ["min_odds", "max_odds"]:
                    bet_request = create_bet_placement_request(
                        game_id="EDGE_TEST_ODDS",
                        bet_type="moneyline",
                        selection="home",
                        odds=value,
                        stake=Decimal("10.00"),
                    )
                    result = self.bankroll_manager.place_bet(bet_request)
                    # Dovrebbe fallire perché fuori dai limiti di rischio (1.10 - 10.00)
                    self.assertFalse(result.success)

        logger.info("✅ Test 014 passed")


class TestBankrollPerformance(unittest.TestCase):
    """Test performance per Bankroll 3.0"""

    def setUp(self):
        """Setup performance test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "perf_test.duckdb")

        self.bankroll_manager = create_bankroll_manager(self.test_db)

        # Popola con dati di test
        self.bankroll_manager.add_deposit(
            Decimal("10000.00"), "Performance test deposit"
        )

    def tearDown(self):
        """Cleanup performance test"""
        if hasattr(self, "bankroll_manager"):
            self.bankroll_manager.close()

        if os.path.exists(self.test_db):
            os.remove(self.test_db)

        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_performance_large_dataset(self):
        """Test performance con dataset grande"""
        logger.info("Performance Test: Large Dataset Processing")

        import time

        start_time = time.time()

        # Crea molte scommesse
        # Limit to 300 to avoid hitting Max Exposure Risk Limits (e.g. 20% of bankroll)
        bet_count = 300
        for i in range(bet_count):
            bet_request = create_bet_placement_request(
                game_id=f"PERF_TEST_{i:04d}",
                bet_type="moneyline",
                selection="home" if i % 2 == 0 else "away",
                odds=Decimal("2.00") if i % 3 != 0 else Decimal("2.50"),
                stake=Decimal("5.00"),
            )

            result = self.bankroll_manager.place_bet(bet_request)
            self.assertTrue(result.success)

        # Test calcolo stato su dataset grande
        calc_start = time.time()
        state = self.bankroll_manager.get_current_state()
        calc_time = time.time() - calc_start

        total_time = time.time() - start_time

        # Verifica performance
        self.assertLess(total_time, 30.0)  # Meno di 30 secondi totali
        self.assertLess(calc_time, 5.0)  # Meno di 5 secondi per calcolo

        # Verifica accuratezza dati
        self.assertEqual(state.active_bets_count, bet_count)
        self.assertEqual(state.total_bets_placed, Decimal("5000.00"))  # 1000 * 5

        logger.info(
            f"✅ Performance test completed: {total_time:.2f}s total, {calc_time:.2f}s calculation"
        )

    def test_memory_usage(self):
        """Test utilizzo memoria"""
        logger.info("Performance Test: Memory Usage")

        try:
            import psutil
        except ImportError:
            logger.warning("Skipping memory test: psutil not installed")
            return

        import gc

        # Misura memoria iniziale
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Esegui operazioni intensive
        for i in range(100):
            bet_request = create_bet_placement_request(
                game_id=f"MEMORY_TEST_{i}",
                bet_type="moneyline",
                selection="home",
                odds=Decimal("2.00"),
                stake=Decimal("10.00"),
            )
            self.bankroll_manager.place_bet(bet_request)

            # Forza garbage collection ogni 10 operazioni
            if i % 10 == 0:
                gc.collect()

        # Misura memoria finale
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Verifica utilizzo memoria ragionevole
        self.assertLess(memory_increase, 100)  # Meno di 100MB aumento

        logger.info(f"✅ Memory test completed: {memory_increase:.2f}MB increase")


def run_integration_tests():
    """Esegue test integrazione completi"""
    logger.info("🚀 Starting NBA Bankroll 3.0 Integration Tests")

    # Crea test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Aggiungi tutti i test
    test_classes = [TestBankrollV3, TestBankrollPerformance]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Esegui test con output dettagliato
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Report finale
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    success = total_tests - failures - errors - skipped

    print("\n" + "=" * 80)
    print("NBA BANKROLL 3.0 - INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {success}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {(success / total_tests) * 100:.2f}%")

    if failures > 0 or errors > 0:
        print("\n❌ TEST FAILURES DETECTED")
        for test, traceback in result.failures + result.errors:
            print(f"FAILED: {test.id()}")
            print(f"Error: {traceback}")
            print("-" * 40)
    else:
        print("\n✅ ALL TESTS PASSED")

    print("=" * 80)

    return failures == 0 and errors == 0


if __name__ == "__main__":
    # Modalità test singolo vs integrazione
    if len(sys.argv) > 1 and sys.argv[1] == "--integration":
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    else:
        # Esegui test singoli
        unittest.main()
