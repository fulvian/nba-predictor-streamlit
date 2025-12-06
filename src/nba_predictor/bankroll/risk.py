"""
NBA Bankroll 3.0 - Risk Validator
Sistema di validazione rischio con controlli multi-livello
Basato su best practice da sistemi betting professionali
"""

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import (
    BetPlacementRequest,
    ValidationResult,
    RiskLevel,
    BetResult,
    BankrollState,
    TransactionType,
)
from .exceptions import RiskLimitExceededError, BetValidationError
from .calculator import BankrollStateCalculator


logger = logging.getLogger(__name__)


class RiskRule(Enum):
    """Tipi di regole di rischio"""

    MAX_STAKE_PERCENTAGE = "MAX_STAKE_PERCENTAGE"
    MAX_DAILY_EXPOSURE = "MAX_DAILY_EXPOSURE"
    MAX_CONCURRENT_BETS = "MAX_CONCURRENT_BETS"
    MIN_ODDS_THRESHOLD = "MIN_ODDS_THRESHOLD"
    MAX_ODDS_THRESHOLD = "MAX_ODDS_THRESHOLD"
    KELLY_CRITERION = "KELLY_CRITERION"
    MAX_LOSING_STREAK = "MAX_LOSING_STREAK"
    BANKROLL_PROTECTION = "BANKROLL_PROTECTION"
    CONFIDENCE_THRESHOLD = "CONFIDENCE_THRESHOLD"
    GAME_CONCENTRATION = "GAME_CONCENTRATION"


@dataclass
class RiskLimit:
    """Limite di rischio configurabile"""

    rule: RiskRule
    threshold: Decimal
    enabled: bool = True
    severity: RiskLevel = RiskLevel.MEDIUM
    description: str = ""
    cooldown_period: Optional[timedelta] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule.value,
            "threshold": float(self.threshold),
            "enabled": self.enabled,
            "severity": self.severity.value,
            "description": self.description,
            "cooldown_hours": self.cooldown_period.total_seconds() / 3600
            if self.cooldown_period
            else None,
        }


class RiskValidator:
    """
    Validatore rischio con regole configurabili:
    - Controlli pre-bet multi-livello
    - Limiti dinamici basati su stato bankroll
    - Kelly criterion per sizing ottimale
    - Protezione contro streak negativi
    - Concentration limits
    """

    def __init__(self, state_calculator: BankrollStateCalculator):
        self.state_calculator = state_calculator
        self.risk_limits = self._initialize_default_limits()
        self.validation_history = []
        self.risk_metrics = {
            "total_validations": 0,
            "blocked_bets": 0,
            "risk_warnings": 0,
            "average_risk_score": 0.0,
        }

    def _initialize_default_limits(self) -> Dict[RiskRule, RiskLimit]:
        """Inizializza limiti rischio default"""
        return {
            # Limiti basati su percentage del bankroll
            RiskRule.MAX_STAKE_PERCENTAGE: RiskLimit(
                rule=RiskRule.MAX_STAKE_PERCENTAGE,
                threshold=Decimal("0.05"),  # 5% max per bet
                severity=RiskLevel.HIGH,
                description="Maximum stake as percentage of bankroll",
            ),
            # Limiti esposizione giornaliera
            RiskRule.MAX_DAILY_EXPOSURE: RiskLimit(
                rule=RiskRule.MAX_DAILY_EXPOSURE,
                threshold=Decimal("0.25"),  # 25% max daily exposure
                severity=RiskLevel.HIGH,
                description="Maximum daily exposure as percentage of bankroll",
            ),
            # Limite scommesse concorrenti
            RiskRule.MAX_CONCURRENT_BETS: RiskLimit(
                rule=RiskRule.MAX_CONCURRENT_BETS,
                threshold=Decimal("10"),  # Max 10 active bets
                severity=RiskLevel.MEDIUM,
                description="Maximum number of concurrent bets",
            ),
            # Limiti odds
            RiskRule.MIN_ODDS_THRESHOLD: RiskLimit(
                rule=RiskRule.MIN_ODDS_THRESHOLD,
                threshold=Decimal("1.10"),  # Odds minimi
                severity=RiskLevel.LOW,
                description="Minimum acceptable odds",
            ),
            RiskRule.MAX_ODDS_THRESHOLD: RiskLimit(
                rule=RiskRule.MAX_ODDS_THRESHOLD,
                threshold=Decimal("10.00"),  # Odds massimi
                severity=RiskLevel.MEDIUM,
                description="Maximum acceptable odds",
            ),
            # Kelly criterion (frazione conservativa)
            RiskRule.KELLY_CRITERION: RiskLimit(
                rule=RiskRule.KELLY_CRITERION,
                threshold=Decimal("0.25"),  # 25% Kelly fraction
                severity=RiskLevel.HIGH,
                description="Kelly criterion fraction limit",
            ),
            # Protezione bankroll
            RiskRule.BANKROLL_PROTECTION: RiskLimit(
                rule=RiskRule.BANKROLL_PROTECTION,
                threshold=Decimal("0.10"),  # 10% minimum reserve
                severity=RiskLevel.CRITICAL,
                description="Minimum bankroll reserve percentage",
            ),
            # Confidence threshold
            RiskRule.CONFIDENCE_THRESHOLD: RiskLimit(
                rule=RiskRule.CONFIDENCE_THRESHOLD,
                threshold=Decimal("0.60"),  # 60% min confidence
                severity=RiskLevel.MEDIUM,
                description="Minimum confidence score for betting",
            ),
            # Game concentration
            RiskRule.GAME_CONCENTRATION: RiskLimit(
                rule=RiskRule.GAME_CONCENTRATION,
                threshold=Decimal("0.05"),  # 5% max per game
                severity=RiskLevel.MEDIUM,
                description="Maximum exposure per single game",
            ),
        }

    def validate_bet_placement(self, request: BetPlacementRequest) -> ValidationResult:
        """
        Validazione completa scommessa con tutte le regole
        Restituisce risultato dettagliato con raccomandazioni
        """
        try:
            # Ottieni stato corrente
            current_state = self.state_calculator.get_current_state()

            # Inizializza risultato
            # Accumulate results locally first
            is_valid = True
            risk_warnings = []
            recommendations = []
            total_risk_score = 0.0
            error_code = None
            error_message = None
            applied_rules = []

            # Applica tutte le regole abilitate
            for rule, limit in self.risk_limits.items():
                if not limit.enabled:
                    continue

                try:
                    rule_result = self._apply_rule(rule, limit, request, current_state)

                    if not rule_result.get("is_valid", True):
                        is_valid = False
                        error_code = f"RISK_RULE_VIOLATION_{rule.value}"
                        error_message = rule_result.get("message", "Rule violation")

                    if rule_result.get("warning"):
                        risk_warnings.append(rule_result["warning"])

                    if rule_result.get("recommendation"):
                        recommendations.append(rule_result["recommendation"])

                    total_risk_score += rule_result.get("risk_score", 0.0)
                    applied_rules.append(
                        {
                            "rule": rule.value,
                            "passed": rule_result.get("is_valid", True),
                            "risk_score": rule_result.get("risk_score", 0.0),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error applying risk rule {rule.value}: {e}")
                    risk_warnings.append(f"Rule {rule.value} failed to apply")

            # Create the immutable result object at the end
            result = ValidationResult(
                is_valid=is_valid,
                risk_warnings=risk_warnings,
                recommendations=recommendations,
                error_code=error_code,
                error_message=error_message,
                metadata={
                    "risk_score": total_risk_score,
                    "applied_rules": applied_rules,
                },
            )

            # Aggiorna statistiche
            self._update_validation_metrics(result, total_risk_score)

            # Log validazione
            self._log_validation(request, result)

            return result

        except Exception as e:
            logger.error(f"Bet validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_code="VALIDATION_ERROR",
                error_message=f"Validation system error: {e}",
                metadata={"error_details": str(e)},
                risk_warnings=[],
                recommendations=[],
            )

    def _apply_rule(
        self,
        rule: RiskRule,
        limit: RiskLimit,
        request: BetPlacementRequest,
        state: BankrollState,
    ) -> Dict[str, Any]:
        """Applica singola regola di rischio"""

        if rule == RiskRule.MAX_STAKE_PERCENTAGE:
            return self._validate_max_stake_percentage(limit, request, state)

        elif rule == RiskRule.MAX_DAILY_EXPOSURE:
            return self._validate_daily_exposure(limit, request, state)

        elif rule == RiskRule.MAX_CONCURRENT_BETS:
            return self._validate_concurrent_bets(limit, request, state)

        elif rule == RiskRule.MIN_ODDS_THRESHOLD:
            return self._validate_min_odds(limit, request)

        elif rule == RiskRule.MAX_ODDS_THRESHOLD:
            return self._validate_max_odds(limit, request)

        elif rule == RiskRule.KELLY_CRITERION:
            return self._validate_kelly_criterion(limit, request, state)

        elif rule == RiskRule.BANKROLL_PROTECTION:
            return self._validate_bankroll_protection(limit, request, state)

        elif rule == RiskRule.CONFIDENCE_THRESHOLD:
            return self._validate_confidence_threshold(limit, request)

        elif rule == RiskRule.GAME_CONCENTRATION:
            return self._validate_game_concentration(limit, request, state)

        else:
            return {
                "is_valid": True,
                "risk_score": 0.0,
                "message": "",
                "warning": "",
                "recommendation": "",
            }

    def _validate_max_stake_percentage(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida stake massimo come percentuale del bankroll"""
        stake_percentage = (
            request.stake / state.current_balance
            if state.current_balance > 0
            else Decimal("0")
        )
        max_allowed = state.current_balance * limit.threshold

        risk_score = (
            float(stake_percentage / limit.threshold) if stake_percentage > 0 else 0.0
        )

        if stake_percentage > limit.threshold:
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Stake {stake_percentage:.2%} exceeds maximum {limit.threshold:.2%} of bankroll",
                "warning": "",
                "recommendation": f"Reduce stake to {max_allowed:.2f} or less",
            }

        if stake_percentage > limit.threshold * Decimal("0.8"):
            return {
                "is_valid": True,
                "risk_score": risk_score,
                "message": "",
                "warning": f"Stake is close to maximum limit ({stake_percentage:.2%} vs {limit.threshold:.2%})",
                "recommendation": "Consider smaller stake for better risk management",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score,
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_daily_exposure(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida esposizione giornaliera"""
        today = datetime.now(timezone.utc).date()
        daily_exposure = self._calculate_daily_exposure(today)
        new_exposure = daily_exposure + request.stake
        max_daily = (
            state.current_balance * limit.threshold
            if state.current_balance > 0
            else Decimal("0")
        )

        exposure_percentage = (
            new_exposure / state.current_balance
            if state.current_balance > 0
            else Decimal("0")
        )
        risk_score = float(exposure_percentage / limit.threshold)

        if new_exposure > max_daily:
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Daily exposure {new_exposure:.2f} exceeds maximum {max_daily:.2f}",
                "warning": "",
                "recommendation": f"Wait until tomorrow or reduce stake by {new_exposure - max_daily:.2f}",
            }

        if exposure_percentage > limit.threshold * Decimal("0.8"):
            return {
                "is_valid": True,
                "risk_score": risk_score,
                "message": "",
                "warning": f"Daily exposure approaching limit ({exposure_percentage:.2%} vs {limit.threshold:.2%})",
                "recommendation": "Monitor daily exposure carefully",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score * 0.5,  # Lower weight
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_concurrent_bets(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida numero scommesse concorrenti"""
        current_concurrent = state.active_bets_count + state.pending_bets_count
        new_concurrent = current_concurrent + 1

        risk_score = (
            float(new_concurrent / float(limit.threshold))
            if limit.threshold > 0
            else 0.0
        )

        if new_concurrent > int(limit.threshold):
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Too many concurrent bets: {new_concurrent} vs max {int(limit.threshold)}",
                "warning": "",
                "recommendation": "Wait for some bets to settle before placing new ones",
            }

        if new_concurrent > int(limit.threshold) * 0.8:
            return {
                "is_valid": True,
                "risk_score": risk_score * 0.3,  # Lower weight
                "message": "",
                "warning": f"High number of concurrent bets: {new_concurrent}",
                "recommendation": "Consider reducing bet frequency",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score * 0.2,  # Low weight
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_min_odds(
        self, limit: RiskLimit, request: BetPlacementRequest
    ) -> Dict[str, Any]:
        """Valida odds minimi"""
        if request.odds < limit.threshold:
            return {
                "is_valid": False,
                "risk_score": 1.0,
                "message": f"Odds {request.odds} below minimum {limit.threshold}",
                "warning": "",
                "recommendation": f"Find better odds or skip this bet",
            }

        return {
            "is_valid": True,
            "risk_score": 0.1,  # Very low risk
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_max_odds(
        self, limit: RiskLimit, request: BetPlacementRequest
    ) -> Dict[str, Any]:
        """Valida odds massimi"""
        if request.odds > limit.threshold:
            return {
                "is_valid": False,
                "risk_score": 0.8,
                "message": f"Odds {request.odds} above maximum {limit.threshold}",
                "warning": "",
                "recommendation": f"Consider lower odds or reduce stake significantly",
            }

        if request.odds > limit.threshold * Decimal("0.7"):
            return {
                "is_valid": True,
                "risk_score": 0.4,
                "message": "",
                "warning": f"High odds detected: {request.odds}",
                "recommendation": "Consider reduced stake for high odds",
            }

        return {
            "is_valid": True,
            "risk_score": 0.2,
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_kelly_criterion(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida sizing usando Kelly criterion"""
        if not request.expected_value or request.expected_value <= 0:
            return {
                "is_valid": True,
                "risk_score": 0.3,
                "message": "",
                "warning": "No expected value provided for Kelly calculation",
                "recommendation": "Provide EV for optimal stake sizing",
            }

        # Kelly formula: f* = (bp - q) / b
        # dove: b = odds - 1, p = probabilità win, q = probabilità lose
        implied_prob = Decimal("1") / request.odds
        edge = request.expected_value

        # Kelly fraction semplificata basata su EV
        kelly_fraction = edge / (request.odds - Decimal("1"))
        kelly_fraction = max(kelly_fraction, Decimal("0"))  # No negative Kelly

        # Applica limite conservativo
        max_kelly_stake = state.current_balance * kelly_fraction * limit.threshold

        risk_score = (
            float(request.stake / max_kelly_stake) if max_kelly_stake > 0 else 1.0
        )

        if request.stake > max_kelly_stake and max_kelly_stake > 0:
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Stake exceeds Kelly recommendation: {max_kelly_stake:.2f}",
                "warning": "",
                "recommendation": f"Use Kelly-optimal stake: {max_kelly_stake:.2f}",
            }

        if risk_score > 0.8:
            return {
                "is_valid": True,
                "risk_score": risk_score,
                "message": "",
                "warning": f"Stake significantly above Kelly optimal",
                "recommendation": f"Consider Kelly stake: {max_kelly_stake:.2f}",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score * 0.6,  # Medium weight
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_bankroll_protection(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida protezione bankroll minimo"""
        remaining_balance = state.current_balance - request.stake
        reserve_percentage = (
            remaining_balance / state.current_balance
            if state.current_balance > 0
            else Decimal("0")
        )

        risk_score = 1.0 - float(reserve_percentage / limit.threshold)

        if reserve_percentage < limit.threshold:
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Bet would leave insufficient reserve: {reserve_percentage:.2%} vs min {limit.threshold:.2%}",
                "warning": "",
                "recommendation": f"Reduce stake to maintain {limit.threshold:.2%} reserve",
            }

        if reserve_percentage < limit.threshold * Decimal("1.5"):
            return {
                "is_valid": True,
                "risk_score": risk_score * 0.7,
                "message": "",
                "warning": f"Low reserve after bet: {reserve_percentage:.2%}",
                "recommendation": "Consider maintaining larger reserve",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score * 0.3,
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_confidence_threshold(
        self, limit: RiskLimit, request: BetPlacementRequest
    ) -> Dict[str, Any]:
        """Valida confidence score minimo"""
        if not request.confidence_score:
            return {
                "is_valid": True,
                "risk_score": 0.4,
                "message": "",
                "warning": "No confidence score provided",
                "recommendation": "Include confidence scores for better risk assessment",
            }

        if request.confidence_score < limit.threshold:
            return {
                "is_valid": False,
                "risk_score": 0.9,
                "message": f"Confidence score {request.confidence_score} below minimum {limit.threshold}",
                "warning": "",
                "recommendation": f"Only bet with confidence >= {limit.threshold}",
            }

        return {
            "is_valid": True,
            "risk_score": 0.2,
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _validate_game_concentration(
        self, limit: RiskLimit, request: BetPlacementRequest, state: BankrollState
    ) -> Dict[str, Any]:
        """Valida concentrazione per singolo gioco"""
        game_exposure = self._calculate_game_exposure(request.game_id)
        new_exposure = game_exposure + request.stake
        max_game_exposure = state.current_balance * limit.threshold

        concentration_percentage = (
            new_exposure / state.current_balance
            if state.current_balance > 0
            else Decimal("0")
        )
        risk_score = float(concentration_percentage / limit.threshold)

        if new_exposure > max_game_exposure:
            return {
                "is_valid": False,
                "risk_score": risk_score,
                "message": f"Game exposure {new_exposure:.2f} exceeds maximum {max_game_exposure:.2f}",
                "warning": "",
                "recommendation": f"Reduce stake or choose different game",
            }

        if concentration_percentage > limit.threshold * Decimal("0.8"):
            return {
                "is_valid": True,
                "risk_score": risk_score * 0.5,
                "message": "",
                "warning": f"High concentration on single game: {concentration_percentage:.2%}",
                "recommendation": "Consider diversifying across games",
            }

        return {
            "is_valid": True,
            "risk_score": risk_score * 0.3,
            "message": "",
            "warning": "",
            "recommendation": "",
        }

    def _calculate_daily_exposure(self, date: datetime.date) -> Decimal:
        """Calcola esposizione totale per data"""
        # Implementazione semplificata - da integrare con engine
        return Decimal("0.00")  # TODO: Implementare con dati reali

    def _calculate_game_exposure(self, game_id: str) -> Decimal:
        """Calcola esposizione totale per gioco"""
        # Implementazione semplificata - da integrare con engine
        return Decimal("0.00")  # TODO: Implementare con dati reali

    def _update_validation_metrics(self, result: ValidationResult, risk_score: float):
        """Aggiorna metriche validazioni"""
        self.risk_metrics["total_validations"] += 1

        if not result.is_valid:
            self.risk_metrics["blocked_bets"] += 1

        if result.risk_warnings:
            self.risk_metrics["risk_warnings"] += 1

        # Aggiorna media risk score
        current_avg = self.risk_metrics["average_risk_score"]
        n = self.risk_metrics["total_validations"]
        new_avg = ((current_avg * (n - 1)) + risk_score) / n
        self.risk_metrics["average_risk_score"] = new_avg

    def _log_validation(self, request: BetPlacementRequest, result: ValidationResult):
        """Log validazione per audit trail"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bet_id": request.bet_id,
            "game_id": request.game_id,
            "stake": float(request.stake),
            "odds": float(request.odds),
            "is_valid": result.is_valid,
            "risk_score": result.metadata.get("risk_score", 0.0),
            "warnings": result.risk_warnings,
            "error_code": result.error_code,
            "error_message": result.error_message,
        }

        self.validation_history.append(log_entry)

        # Mantiene solo ultimi 1000 log per performance
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]

        logger.info(
            f"Bet validation: {request.bet_id} - Valid: {result.is_valid} - Risk Score: {result.metadata.get('risk_score', 0.0):.2f}"
        )

    def update_risk_limit(self, rule: RiskRule, limit: RiskLimit):
        """Aggiorna limite rischio specifico"""
        self.risk_limits[rule] = limit
        logger.info(f"Updated risk limit: {rule.value} -> {limit.threshold}")

    def get_risk_limits(self) -> Dict[str, Dict[str, Any]]:
        """Restituisce tutti i limiti rischio configurati"""
        return {rule.value: limit.to_dict() for rule, limit in self.risk_limits.items()}

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche rischio"""
        total = self.risk_metrics["total_validations"]
        return {
            **self.risk_metrics,
            "validation_rate": 1.0
            if total == 0
            else (total - self.risk_metrics["blocked_bets"]) / total,
            "warning_rate": 0.0
            if total == 0
            else self.risk_metrics["risk_warnings"] / total,
            "recent_validations": len(self.validation_history),
        }

    def enable_rule(self, rule: RiskRule):
        """Abilita regola rischio"""
        if rule in self.risk_limits:
            self.risk_limits[rule].enabled = True
            logger.info(f"Enabled risk rule: {rule.value}")

    def disable_rule(self, rule: RiskRule):
        """Disabilita regola rischio"""
        if rule in self.risk_limits:
            self.risk_limits[rule].enabled = False
            logger.info(f"Disabled risk rule: {rule.value}")

    def reset_metrics(self):
        """Resetta metriche rischio"""
        self.risk_metrics = {
            "total_validations": 0,
            "blocked_bets": 0,
            "risk_warnings": 0,
            "average_risk_score": 0.0,
        }
        self.validation_history.clear()
        logger.info("Risk validator metrics reset")
