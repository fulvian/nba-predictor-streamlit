"""
Test suite for Enhanced Bet Validation Engine
Phase 4 Day 13 - Task 4.1.1: Comprehensive Bet Validation Rules
"""

import pytest
import time
import tempfile
import shutil
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import duckdb

from src.nba_predictor.validation.enhanced_bet_validation_engine import (
    EnhancedBetValidationEngine, ValidationLevel, ValidationCategory, BetType,
    BetValidationRequest, BetValidationResponse, ValidationRule,
    UserBehaviorProfile, create_validation_engine, validate_bet_request
)


class TestEnhancedBetValidationEngine:
    """Test enhanced bet validation engine"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = str(Path(temp_dir) / "test_betting.duckdb")
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def validation_engine(self, temp_db_path):
        """Create validation engine instance for testing"""
        return EnhancedBetValidationEngine(temp_db_path)

    @pytest.fixture
    def sample_bet_request(self):
        """Create sample bet validation request"""
        return BetValidationRequest(
            user_id="test_user_123",
            game_id="game_456",
            bet_type=BetType.MONEYLINE,
            amount=100.0,
            odds=150.0,
            selection={"team": "Lakers", "type": "moneyline"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 (Test Browser)"
        )

    def test_validation_engine_initialization(self, validation_engine):
        """Test validation engine initialization"""
        assert validation_engine.db_path is not None
        assert len(validation_engine.validation_rules) > 0
        assert validation_engine.validation_stats['total_validations'] == 0
        assert isinstance(validation_engine.user_profiles, dict)
        assert isinstance(validation_engine.validation_cache, dict)

    def test_validation_rules_loading(self, validation_engine):
        """Test validation rules are loaded correctly"""
        # Check for expected default rules
        expected_rules = [
            'BET_AMOUNT_POSITIVE',
            'BET_ODDS_RANGE',
            'SELECTION_FORMAT',
            'GAME_STATUS_VALID',
            'BETTING_DEADLINE',
            'MAX_BET_AMOUNT',
            'USER_BET_FREQUENCY',
            'AMOUNT_DEVIATION',
            'PARLAY_RISK',
            'MULTIPLE_ACCOUNTS',
            'SUSPICIOUS_PATTERN',
            'RAPID_FIRE_BETTING',
            'AGE_VERIFICATION',
            'GEOLOCATION_CHECK'
        ]

        for rule_id in expected_rules:
            assert rule_id in validation_engine.validation_rules
            rule = validation_engine.validation_rules[rule_id]
            assert isinstance(rule, ValidationRule)
            assert rule.enabled == True

    def test_valid_bet_request(self, validation_engine, sample_bet_request):
        """Test validation of a valid bet request"""
        response = validation_engine.validate_bet(sample_bet_request)

        assert isinstance(response, BetValidationResponse)
        assert response.is_valid == True  # Should be valid
        assert response.validation_level in [ValidationLevel.INFO, ValidationLevel.WARNING]
        assert len(response.results) > 0
        assert response.risk_score >= 0.0 and response.risk_score <= 1.0
        assert response.processing_time_ms > 0
        assert 'overall' in response.context7_compliance

    def test_invalid_bet_amount_negative(self, validation_engine, sample_bet_request):
        """Test validation with negative bet amount"""
        sample_bet_request.amount = -50.0

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

        # Check for specific validation error
        amount_errors = [r for r in response.results if r.field == "amount" and r.level == ValidationLevel.ERROR]
        assert len(amount_errors) > 0
        assert "must be positive" in amount_errors[0].message.lower()

    def test_invalid_bet_odds_out_of_range(self, validation_engine, sample_bet_request):
        """Test validation with odds outside valid range"""
        sample_bet_request.odds = 50000.0  # Way outside range

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

        # Check for specific validation error
        odds_errors = [r for r in response.results if r.field == "odds" and r.level == ValidationLevel.ERROR]
        assert len(odds_errors) > 0
        assert "outside valid range" in odds_errors[0].message.lower()

    def test_invalid_selection_format(self, validation_engine, sample_bet_request):
        """Test validation with invalid selection format"""
        sample_bet_request.selection = {"invalid": "format"}  # Missing required fields

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

        # Check for specific validation error
        selection_errors = [r for r in response.results if r.field == "selection"]
        assert len(selection_errors) > 0
        assert "missing" in selection_errors[0].message.lower()

    def test_spread_bet_validation(self, validation_engine, sample_bet_request):
        """Test spread bet specific validation"""
        sample_bet_request.bet_type = BetType.SPREAD
        sample_bet_request.selection = {"team": "Lakers", "type": "spread", "spread": -5.5}

        response = validation_engine.validate_bet(sample_bet_request)

        # Should be valid with spread information
        assert response.is_valid == True
        assert response.validation_level in [ValidationLevel.INFO, ValidationLevel.WARNING]

        # Check selection format validation passed
        selection_results = [r for r in response.results if r.field == "selection"]
        assert len(selection_results) > 0
        assert any("valid" in result.message.lower() for result in selection_results)

    def test_spread_bet_missing_spread(self, validation_engine, sample_bet_request):
        """Test spread bet without spread value"""
        sample_bet_request.bet_type = BetType.SPREAD
        sample_bet_request.selection = {"team": "Lakers", "type": "spread"}  # Missing spread

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

        # Check for specific validation error
        selection_errors = [r for r in response.results if r.field == "selection" and r.level == ValidationLevel.ERROR]
        assert len(selection_errors) > 0
        assert "spread" in selection_errors[0].message.lower()

    def test_total_bet_validation(self, validation_engine, sample_bet_request):
        """Test total bet specific validation"""
        sample_bet_request.bet_type = BetType.TOTAL
        sample_bet_request.selection = {"team": "over", "type": "total", "total": 220.5, "over_under": "over"}

        response = validation_engine.validate_bet(sample_bet_request)

        # Should be valid with total information
        assert response.is_valid == True
        assert response.validation_level in [ValidationLevel.INFO, ValidationLevel.WARNING]

    def test_total_bet_missing_fields(self, validation_engine, sample_bet_request):
        """Test total bet without required fields"""
        sample_bet_request.bet_type = BetType.TOTAL
        sample_bet_request.selection = {"team": "over", "type": "total"}  # Missing total and over_under

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

    def test_parlay_bet_validation(self, validation_engine, sample_bet_request):
        """Test parlay bet validation"""
        legs = [
            {"team": "Lakers", "type": "moneyline", "odds": 150},
            {"team": "Celtics", "type": "spread", "spread": -2.5, "odds": -110},
            {"team": "Nets", "type": "total", "total": 210.5, "over_under": "over", "odds": -105}
        ]

        sample_bet_request.bet_type = BetType.PARLAY
        sample_bet_request.selection = {"legs": legs}

        response = validation_engine.validate_bet(sample_bet_request)

        # Should be valid with reasonable parlay
        assert response.is_valid == True
        assert response.validation_level in [ValidationLevel.INFO, ValidationLevel.WARNING]

        # Check parlay risk validation
        parlay_results = [r for r in response.results if r.rule_id == "PARLAY_RISK"]
        assert len(parlay_results) > 0

    def test_parlay_too_many_legs(self, validation_engine, sample_bet_request):
        """Test parlay bet with too many legs"""
        # Create parlay with too many legs
        legs = [{"team": f"Team_{i}", "type": "moneyline", "odds": 110} for i in range(15)]

        sample_bet_request.bet_type = BetType.PARLAY
        sample_bet_request.selection = {"legs": legs}

        response = validation_engine.validate_bet(sample_bet_request)

        # Should generate warning for too many legs
        parlay_results = [r for r in response.results if r.rule_id == "PARLAY_RISK" and r.level == ValidationLevel.WARNING]
        assert len(parlay_results) > 0
        assert "exceeding maximum" in parlay_results[0].message.lower()

    def test_bet_amount_minimum(self, validation_engine, sample_bet_request):
        """Test bet amount minimum validation"""
        sample_bet_request.amount = 0.50  # Below minimum

        response = validation_engine.validate_bet(sample_bet_request)

        assert response.is_valid == False
        assert response.validation_level == ValidationLevel.ERROR

        # Check for minimum amount error
        amount_errors = [r for r in response.results if r.field == "amount" and r.level == ValidationLevel.ERROR]
        assert len(amount_errors) > 0
        assert "below minimum" in amount_errors[0].message.lower()

    def test_high_bet_amount_warning(self, validation_engine, sample_bet_request):
        """Test warning for high bet amounts"""
        sample_bet_request.amount = 15000.0  # High amount

        response = validation_engine.validate_bet(sample_bet_request)

        # Should still be valid but with warnings
        assert response.is_valid == True
        assert response.validation_level == ValidationLevel.WARNING

        # Check for high amount warning
        amount_warnings = [r for r in response.results if r.field == "amount" and r.level == ValidationLevel.WARNING]
        assert len(amount_warnings) > 0
        assert "above recommended" in amount_warnings[0].message.lower()

    def test_context7_compliance_tracking(self, validation_engine, sample_bet_request):
        """Test Context7 compliance tracking"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have Context7 compliance scores
        assert 'overall' in response.context7_compliance
        assert 0 <= response.context7_compliance['overall'] <= 1

        # Should have individual pattern scores
        context7_patterns = ["responsive_design_system", "accessibility_features",
                          "adaptive_ui_layouts", "pwa_features", "real_time_updates",
                          "intelligent_cache", "advanced_ml_operations"]

        for pattern in context7_patterns:
            if pattern in response.context7_compliance:
                assert 0 <= response.context7_compliance[pattern] <= 1

    def test_user_behavior_profile_creation(self, validation_engine, sample_bet_request):
        """Test user behavior profile creation and tracking"""
        # First bet should create profile
        response = validation_engine.validate_bet(sample_bet_request)
        assert response.is_valid == True

        # Check profile was created
        profile = validation_engine._get_user_profile(sample_bet_request.user_id)
        assert profile.user_id == sample_bet_request.user_id
        assert profile.total_bets == 1
        assert profile.total_amount == 100.0
        assert profile.average_bet_amount == 100.0
        assert profile.max_single_bet == 100.0

    def test_user_behavior_profile_updates(self, validation_engine, sample_bet_request):
        """Test user behavior profile updates with multiple bets"""
        # First bet
        response1 = validation_engine.validate_bet(sample_bet_request)
        assert response1.is_valid == True

        # Second bet with different amount
        sample_bet_request.amount = 250.0
        sample_bet_request.bet_id = str(uuid.uuid4())  # New bet ID
        response2 = validation_engine.validate_bet(sample_bet_request)
        assert response2.is_valid == True

        # Check profile updates
        profile = validation_engine._get_user_profile(sample_bet_request.user_id)
        assert profile.total_bets == 2
        assert profile.total_amount == 350.0
        assert profile.average_bet_amount == 175.0
        assert profile.max_single_bet == 250.0

    def test_validation_caching(self, validation_engine, sample_bet_request):
        """Test validation result caching"""
        # First validation
        start_time = time.time()
        response1 = validation_engine.validate_bet(sample_bet_request)
        first_time = time.time() - start_time

        # Second validation (should use cache)
        start_time = time.time()
        response2 = validation_engine.validate_bet(sample_bet_request)
        second_time = time.time() - start_time

        # Should be identical responses
        assert response1.bet_id == response2.bet_id
        assert response1.is_valid == response2.is_valid
        assert response1.validation_level == response2.validation_level

        # Second should be faster due to caching
        assert second_time < first_time

        # Should have cache hit
        assert validation_engine.validation_stats['cache_hits'] > 0

    def test_validation_statistics(self, validation_engine, sample_bet_request):
        """Test validation statistics tracking"""
        # Initial stats
        initial_stats = validation_engine.get_validation_statistics()
        assert initial_stats['total_validations'] == 0

        # Perform validation
        response = validation_engine.validate_bet(sample_bet_request)

        # Updated stats
        updated_stats = validation_engine.get_validation_statistics()
        assert updated_stats['total_validations'] == 1
        assert updated_stats['valid_bets'] == 1
        assert updated_stats['invalid_bets'] == 0
        assert updated_stats['avg_processing_time'] > 0
        assert updated_stats['total_rules'] > 0

    def test_add_custom_validation_rule(self, validation_engine):
        """Test adding custom validation rules"""
        custom_rule = ValidationRule(
            rule_id="CUSTOM_RULE",
            name="Custom Test Rule",
            description="Test rule for custom validation",
            category=ValidationCategory.BUSINESS,
            level=ValidationLevel.WARNING,
            context7_pattern="responsive_design_system",
            parameters={"test_param": "test_value"}
        )

        validation_engine.add_validation_rule(custom_rule)

        # Rule should be added
        assert "CUSTOM_RULE" in validation_engine.validation_rules
        assert validation_engine.validation_rules["CUSTOM_RULE"].enabled == True

    def test_disable_validation_rule(self, validation_engine):
        """Test disabling validation rules"""
        # Disable a rule
        validation_engine.disable_validation_rule("BET_AMOUNT_POSITIVE")

        # Rule should be disabled
        assert validation_engine.validation_rules["BET_AMOUNT_POSITIVE"].enabled == False

    def test_age_verification_validation(self, validation_engine, sample_bet_request):
        """Test age verification validation"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have age verification result
        age_results = [r for r in response.results if r.rule_id == "AGE_VERIFICATION"]
        assert len(age_results) > 0
        assert age_results[0].level == ValidationLevel.INFO  # Mock user passes age verification

    def test_geolocation_validation(self, validation_engine, sample_bet_request):
        """Test geolocation validation"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have geolocation result
        geo_results = [r for r in response.results if r.rule_id == "GEOLOCATION_CHECK"]
        assert len(geo_results) > 0
        assert geo_results[0].level == ValidationLevel.INFO  # Mock user in allowed location

    def test_fraud_detection_patterns(self, validation_engine, sample_bet_request):
        """Test fraud detection pattern validation"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have fraud detection results
        fraud_results = [r for r in response.results if r.category == ValidationCategory.FRAUD]
        assert len(fraud_results) > 0

        # Should check for multiple accounts and suspicious patterns
        rule_ids = [r.rule_id for r in fraud_results]
        assert "MULTIPLE_ACCOUNTS" in rule_ids
        assert "SUSPICIOUS_PATTERN" in rule_ids

    def test_risk_assessment_calculation(self, validation_engine, sample_bet_request):
        """Test risk assessment calculation"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have risk score
        assert 0 <= response.risk_score <= 1

        # Normal bet should have low risk
        assert response.risk_score < 0.5

    def test_validation_recommendations(self, validation_engine, sample_bet_request):
        """Test validation recommendations generation"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should have recommendations
        assert len(response.recommendations) > 0

        # Should include Context7 compliance recommendation
        assert any("context7" in rec.lower() for rec in response.recommendations)

    def test_convenience_function(self, temp_db_path):
        """Test convenience function for validation"""
        response = validate_bet_request(
            user_id="test_user_456",
            game_id="game_789",
            bet_type="moneyline",
            amount=75.0,
            odds=120.0,
            selection={"team": "Warriors", "type": "moneyline"},
            db_path=temp_db_path
        )

        assert isinstance(response, BetValidationResponse)
        assert response.is_valid == True

    def test_processing_time_tracking(self, validation_engine, sample_bet_request):
        """Test processing time tracking"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Should track processing time
        assert response.processing_time_ms > 0
        assert response.processing_time_ms < 1000  # Should be under 1 second

    def test_database_persistence(self, validation_engine, sample_bet_request):
        """Test that validation results are persisted to database"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Verify data was written to database
        conn = duckdb.connect(validation_engine.db_path)
        validation_count = conn.execute("SELECT COUNT(*) FROM validation_results").fetchone()[0]
        conn.close()

        assert validation_count > 0

    def test_user_profile_persistence(self, validation_engine, sample_bet_request):
        """Test that user profiles are persisted to database"""
        response = validation_engine.validate_bet(sample_bet_request)

        # Verify user profile was written to database
        conn = duckdb.connect(validation_engine.db_path)
        profile_count = conn.execute("SELECT COUNT(*) FROM user_behavior_profiles WHERE user_id = ?",
                                  (sample_bet_request.user_id,)).fetchone()[0]
        conn.close()

        assert profile_count == 1

    def test_error_handling_invalid_request(self, validation_engine):
        """Test error handling with invalid request data"""
        # Create invalid request
        invalid_request = BetValidationRequest(
            user_id="",  # Empty user ID
            game_id="",  # Empty game ID
            bet_type=BetType.MONEYLINE,
            amount="invalid_amount",  # Invalid amount
            odds="invalid_odds",  # Invalid odds
            selection={}  # Empty selection
        )

        response = validation_engine.validate_bet(invalid_request)

        # Should handle gracefully
        assert isinstance(response, BetValidationResponse)
        assert response.is_valid == False
        assert response.validation_level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]

    def test_context7_pattern_mapping(self, validation_engine):
        """Test that Context7 patterns are properly mapped to rules"""
        context7_rules = [rule for rule in validation_engine.validation_rules.values()
                         if rule.context7_pattern is not None]

        assert len(context7_rules) > 0

        # Check for expected Context7 patterns
        expected_patterns = [
            "responsive_design_system",
            "accessibility_features",
            "adaptive_ui_layouts",
            "pwa_features",
            "real_time_updates",
            "intelligent_cache",
            "advanced_ml_operations"
        ]

        mapped_patterns = set(rule.context7_pattern for rule in context7_rules)
        for pattern in expected_patterns:
            assert pattern in mapped_patterns, f"Pattern {pattern} not mapped to any rule"


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])