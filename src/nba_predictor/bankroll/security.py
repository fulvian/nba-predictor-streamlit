"""
NBA Bankroll 3.0 - Security Manager
Sistema di sicurezza enterprise-grade con protezioni multi-livello
Basato su best practice da sistemi finanziari e betting
"""

import logging
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import ipaddress
import re


from .exceptions import (
    SecurityError,
    UnauthorizedAccessError,
    RateLimitExceededError,
    ConfigurationError,
)
from .audit import AuditLogger, AuditEventType, AuditSeverity


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Livelli di sicurezza"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OperationType(Enum):
    """Tipi di operazioni protette"""

    PLACE_BET = "place_bet"
    SETTLE_BET = "settle_bet"
    ADD_DEPOSIT = "add_deposit"
    ADD_WITHDRAWAL = "add_withdrawal"
    VIEW_TRANSACTIONS = "view_transactions"
    VIEW_BANKROLL = "view_bankroll"
    MODIFY_SETTINGS = "modify_settings"
    EXPORT_DATA = "export_data"


@dataclass
class SecurityPolicy:
    """Policy di sicurezza configurabile"""

    operation: OperationType
    requires_authentication: bool = True
    requires_authorization: bool = False
    max_attempts_per_hour: int = 100
    session_timeout_minutes: int = 30
    ip_whitelist: Optional[List[str]] = None
    ip_blacklist: Optional[List[str]] = None
    user_agent_whitelist: Optional[List[str]] = None
    min_security_level: SecurityLevel = SecurityLevel.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "requires_authentication": self.requires_authentication,
            "requires_authorization": self.requires_authorization,
            "max_attempts_per_hour": self.max_attempts_per_hour,
            "session_timeout_minutes": self.session_timeout_minutes,
            "ip_whitelist": self.ip_whitelist,
            "ip_blacklist": self.ip_blacklist,
            "user_agent_whitelist": self.user_agent_whitelist,
            "min_security_level": self.min_security_level.value,
        }


@dataclass
class SecurityContext:
    """Contesto sicurezza per operazione"""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[datetime] = None
    device_fingerprint: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "device_fingerprint": self.device_fingerprint,
            "security_level": self.security_level.value,
        }


class SecurityManager:
    """
    Security Manager enterprise-grade:
    - Rate limiting per IP e utente
    - Device fingerprinting
    - IP whitelist/blacklist
    - Session management sicuro
    - Audit trail completo
    - Anomaly detection
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger

        # Policy di sicurezza
        self.security_policies = self._initialize_default_policies()

        # Rate limiting
        self.rate_limits = {}  # {key: {attempts: count, reset_time: datetime}}

        # Sessioni attive
        self.active_sessions = {}  # {session_id: {user_id, created_at, last_activity}}

        # Device fingerprints
        self.device_fingerprints = {}  # {fingerprint: {user_id, first_seen, last_seen}}

        # Statistiche sicurezza
        self.security_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "suspicious_activities": 0,
            "active_sessions": 0,
            "rate_limit_blocks": 0,
        }

        logger.info("Security Manager initialized")

    def _initialize_default_policies(self) -> Dict[OperationType, SecurityPolicy]:
        """Inizializza policy di sicurezza default"""
        return {
            # Operazioni betting
            OperationType.PLACE_BET: SecurityPolicy(
                operation=OperationType.PLACE_BET,
                requires_authentication=True,
                requires_authorization=True,
                max_attempts_per_hour=50,
                session_timeout_minutes=15,
                min_security_level=SecurityLevel.HIGH,
            ),
            OperationType.SETTLE_BET: SecurityPolicy(
                operation=OperationType.SETTLE_BET,
                requires_authentication=True,
                requires_authorization=True,
                max_attempts_per_hour=100,
                session_timeout_minutes=30,
                min_security_level=SecurityLevel.HIGH,
            ),
            # Operazioni finanziarie
            OperationType.ADD_DEPOSIT: SecurityPolicy(
                operation=OperationType.ADD_DEPOSIT,
                requires_authentication=True,
                requires_authorization=False,
                max_attempts_per_hour=20,
                session_timeout_minutes=10,
                min_security_level=SecurityLevel.MEDIUM,
            ),
            OperationType.ADD_WITHDRAWAL: SecurityPolicy(
                operation=OperationType.ADD_WITHDRAWAL,
                requires_authentication=True,
                requires_authorization=True,
                max_attempts_per_hour=10,
                session_timeout_minutes=10,
                min_security_level=SecurityLevel.CRITICAL,
            ),
            # Operazioni view
            OperationType.VIEW_TRANSACTIONS: SecurityPolicy(
                operation=OperationType.VIEW_TRANSACTIONS,
                requires_authentication=True,
                requires_authorization=False,
                max_attempts_per_hour=200,
                session_timeout_minutes=60,
                min_security_level=SecurityLevel.LOW,
            ),
            OperationType.VIEW_BANKROLL: SecurityPolicy(
                operation=OperationType.VIEW_BANKROLL,
                requires_authentication=True,
                requires_authorization=False,
                max_attempts_per_hour=200,
                session_timeout_minutes=60,
                min_security_level=SecurityLevel.LOW,
            ),
            # Operazioni admin
            OperationType.MODIFY_SETTINGS: SecurityPolicy(
                operation=OperationType.MODIFY_SETTINGS,
                requires_authentication=True,
                requires_authorization=True,
                max_attempts_per_hour=5,
                session_timeout_minutes=5,
                min_security_level=SecurityLevel.CRITICAL,
            ),
            OperationType.EXPORT_DATA: SecurityPolicy(
                operation=OperationType.EXPORT_DATA,
                requires_authentication=True,
                requires_authorization=True,
                max_attempts_per_hour=3,
                session_timeout_minutes=5,
                min_security_level=SecurityLevel.CRITICAL,
            ),
        }

    def validate_operation(
        self, operation: str, user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validazione completa operazione con tutti i controlli sicurezza
        Restituisce risultato con dettagli
        """
        try:
            # Converti string a OperationType
            try:
                op_type = OperationType(operation)
            except ValueError:
                return {
                    "allowed": False,
                    "reason": f"Invalid operation: {operation}",
                    "error_code": "INVALID_OPERATION",
                }

            # Crea contesto sicurezza
            security_context = self._create_security_context(user_context)

            # Aggiorna statistiche
            self.security_stats["total_requests"] += 1

            # Recupera policy
            policy = self.security_policies.get(op_type)
            if not policy:
                return {
                    "allowed": False,
                    "reason": f"No security policy for operation: {operation}",
                    "error_code": "NO_POLICY",
                }

            # Esegui validazioni in ordine
            validation_result = {
                "allowed": True,
                "reason": "",
                "error_code": None,
                "warnings": [],
                "security_context": security_context.to_dict(),
                "policy_applied": policy.to_dict(),
            }

            # 1. Validazione IP
            ip_result = self._validate_ip_address(security_context.ip_address, policy)
            if not ip_result["allowed"]:
                return ip_result

            validation_result["warnings"].extend(ip_result.get("warnings", []))

            # 2. Validazione User Agent
            ua_result = self._validate_user_agent(security_context.user_agent, policy)
            if not ua_result["allowed"]:
                return ua_result

            validation_result["warnings"].extend(ua_result.get("warnings", []))

            # 3. Rate limiting
            rate_result = self._validate_rate_limit(security_context, policy)
            if not rate_result["allowed"]:
                self.security_stats["rate_limit_blocks"] += 1
                return rate_result

            validation_result["warnings"].extend(rate_result.get("warnings", []))

            # 4. Validazione sessione
            session_result = self._validate_session(security_context, policy)
            if not session_result["allowed"]:
                return session_result

            validation_result["warnings"].extend(session_result.get("warnings", []))

            # 5. Validazione device fingerprint
            device_result = self._validate_device_fingerprint(security_context, policy)
            if not device_result["allowed"]:
                return device_result

            validation_result["warnings"].extend(device_result.get("warnings", []))

            # 6. Validazione livello sicurezza
            level_result = self._validate_security_level(security_context, policy)
            if not level_result["allowed"]:
                return level_result

            validation_result["warnings"].extend(level_result.get("warnings", []))

            # Log successo
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Operation allowed: {operation}",
                    AuditSeverity.INFO,
                    user_context=user_context,
                    additional_data={
                        "operation": operation,
                        "security_context": security_context.to_dict(),
                        "warnings": validation_result["warnings"],
                    },
                )

            return validation_result

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Security validation error: {str(e)}",
                    AuditSeverity.ERROR,
                    user_context=user_context,
                    additional_data={"operation": operation, "error": str(e)},
                )

            return {
                "allowed": False,
                "reason": f"Security validation error: {str(e)}",
                "error_code": "VALIDATION_ERROR",
            }

    def _create_security_context(self, user_context: Dict[str, Any]) -> SecurityContext:
        """Crea contesto sicurezza da request"""
        return SecurityContext(
            user_id=user_context.get("user_id"),
            session_id=user_context.get("session_id"),
            ip_address=user_context.get("ip_address"),
            user_agent=user_context.get("user_agent"),
            timestamp=datetime.now(timezone.utc),
            device_fingerprint=self._generate_device_fingerprint(user_context),
            security_level=self._determine_security_level(user_context),
        )

    def _validate_ip_address(
        self, ip_address: Optional[str], policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione indirizzo IP"""
        if not ip_address:
            return {"allowed": True, "warnings": ["No IP address provided"]}

        try:
            # Parsing IP
            ip_obj = ipaddress.ip_address(ip_address)

            # Check blacklist
            if policy.ip_blacklist and ip_address in policy.ip_blacklist:
                self.security_stats["blocked_requests"] += 1

                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        f"IP address blocked: {ip_address}",
                        AuditSeverity.WARNING,
                        additional_data={
                            "ip_address": ip_address,
                            "reason": "blacklisted",
                        },
                    )

                return {
                    "allowed": False,
                    "reason": f"IP address {ip_address} is blacklisted",
                    "error_code": "IP_BLACKLISTED",
                }

            # Check whitelist
            if policy.ip_whitelist and ip_address not in policy.ip_whitelist:
                return {
                    "allowed": False,
                    "reason": f"IP address {ip_address} not whitelisted",
                    "error_code": "IP_NOT_WHITELISTED",
                }

            # Check range private (warning)
            if ip_obj.is_private:
                return {
                    "allowed": True,
                    "warnings": [f"Private IP address: {ip_address}"],
                }

            return {"allowed": True, "warnings": []}

        except ValueError:
            return {
                "allowed": False,
                "reason": f"Invalid IP address: {ip_address}",
                "error_code": "INVALID_IP",
            }

    def _validate_user_agent(
        self, user_agent: Optional[str], policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione User Agent"""
        if not user_agent:
            return {"allowed": True, "warnings": ["No User-Agent provided"]}

        # Check whitelist
        if policy.user_agent_whitelist:
            if not any(
                pattern in user_agent for pattern in policy.user_agent_whitelist
            ):
                return {
                    "allowed": False,
                    "reason": "User-Agent not whitelisted",
                    "error_code": "UA_NOT_WHITELISTED",
                }

        # Check suspicious patterns
        suspicious_patterns = [
            r"bot",
            r"crawler",
            r"spider",
            r"scanner",
            r"curl",
            r"wget",
        ]

        warnings = []
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                warnings.append(f"Suspicious User-Agent pattern: {pattern}")

        return {"allowed": True, "warnings": warnings}

    def _validate_rate_limit(
        self, context: SecurityContext, policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione rate limiting"""
        # Chiavi per rate limiting
        ip_key = f"ip:{context.ip_address}" if context.ip_address else None
        user_key = f"user:{context.user_id}" if context.user_id else None
        session_key = f"session:{context.session_id}" if context.session_id else None

        current_time = datetime.now(timezone.utc)

        for key in [ip_key, user_key, session_key]:
            if not key:
                continue

            # Recupera/crea record rate limit
            if key not in self.rate_limits:
                self.rate_limits[key] = {
                    "attempts": 0,
                    "reset_time": current_time + timedelta(hours=1),
                }

            rate_record = self.rate_limits[key]

            # Reset se scaduto
            if current_time > rate_record["reset_time"]:
                rate_record["attempts"] = 0
                rate_record["reset_time"] = current_time + timedelta(hours=1)

            # Incrementa contatore
            rate_record["attempts"] += 1

            # Check limite
            if rate_record["attempts"] > policy.max_attempts_per_hour:
                return {
                    "allowed": False,
                    "reason": f"Rate limit exceeded for {key}: {rate_record['attempts']}/{policy.max_attempts_per_hour}",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": rate_record["reset_time"].isoformat(),
                }

            # Warning se vicino limite
            if rate_record["attempts"] > policy.max_attempts_per_hour * 0.8:
                return {
                    "allowed": True,
                    "warnings": [
                        f"Rate limit warning: {rate_record['attempts']}/{policy.max_attempts_per_hour}"
                    ],
                }

        return {"allowed": True, "warnings": []}

    def _validate_session(
        self, context: SecurityContext, policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione sessione"""
        if not policy.requires_authentication:
            return {"allowed": True, "warnings": []}

        if not context.session_id:
            return {
                "allowed": False,
                "reason": "Authentication required but no session provided",
                "error_code": "NO_SESSION",
            }

        # Check session esiste
        if context.session_id not in self.active_sessions:
            return {
                "allowed": False,
                "reason": f"Invalid session: {context.session_id}",
                "error_code": "INVALID_SESSION",
            }

        session = self.active_sessions[context.session_id]
        current_time = datetime.now(timezone.utc)

        # Check timeout
        session_age = current_time - session["last_activity"]
        if session_age > timedelta(minutes=policy.session_timeout_minutes):
            # Rimuovi sessione scaduta
            del self.active_sessions[context.session_id]

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Session expired: {context.session_id}",
                    AuditSeverity.INFO,
                    additional_data={
                        "session_id": context.session_id,
                        "user_id": session["user_id"],
                        "age_minutes": session_age.total_seconds() / 60,
                    },
                )

            return {
                "allowed": False,
                "reason": f"Session expired: {context.session_id}",
                "error_code": "SESSION_EXPIRED",
            }

        # Aggiorna last activity
        session["last_activity"] = current_time

        return {"allowed": True, "warnings": []}

    def _validate_device_fingerprint(
        self, context: SecurityContext, policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione device fingerprint"""
        if not context.device_fingerprint:
            return {"allowed": True, "warnings": ["No device fingerprint"]}

        # Check se dispositivo conosciuto
        if context.device_fingerprint in self.device_fingerprints:
            device_info = self.device_fingerprints[context.device_fingerprint]

            # Check se associato a stesso utente
            if device_info["user_id"] != context.user_id:
                self.security_stats["suspicious_activities"] += 1

                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        f"Suspicious device access: {context.device_fingerprint}",
                        AuditSeverity.WARNING,
                        additional_data={
                            "device_fingerprint": context.device_fingerprint,
                            "expected_user": device_info["user_id"],
                            "actual_user": context.user_id,
                        },
                    )

                return {
                    "allowed": False,
                    "reason": "Device fingerprint associated with different user",
                    "error_code": "DEVICE_MISMATCH",
                }

            # Aggiorna last seen
            device_info["last_seen"] = context.timestamp

        else:
            # Nuovo dispositivo - registra
            self.device_fingerprints[context.device_fingerprint] = {
                "user_id": context.user_id,
                "first_seen": context.timestamp,
                "last_seen": context.timestamp,
            }

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"New device registered: {context.device_fingerprint}",
                    AuditSeverity.INFO,
                    additional_data={
                        "device_fingerprint": context.device_fingerprint,
                        "user_id": context.user_id,
                    },
                )

        return {"allowed": True, "warnings": []}

    def _validate_security_level(
        self, context: SecurityContext, policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Validazione livello sicurezza richiesto"""
        required_level = policy.min_security_level

        # Mapping livelli a valori numerici
        level_values = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.HIGH: 3,
            SecurityLevel.CRITICAL: 4,
        }

        current_value = level_values.get(context.security_level, 0)
        required_value = level_values.get(required_level, 2)

        if current_value < required_value:
            return {
                "allowed": False,
                "reason": f"Insufficient security level: {context.security_level.value} < {required_level.value}",
                "error_code": "INSUFFICIENT_SECURITY_LEVEL",
            }

        return {"allowed": True, "warnings": []}

    def _generate_device_fingerprint(self, user_context: Dict[str, Any]) -> str:
        """Genera fingerprint dispositivo"""
        components = [
            user_context.get("user_agent", ""),
            user_context.get("ip_address", ""),
            user_context.get("accept_language", ""),
            user_context.get("platform", ""),
            user_context.get("screen_resolution", ""),
        ]

        fingerprint_data = "|".join(filter(None, components))
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def _determine_security_level(self, user_context: Dict[str, Any]) -> SecurityLevel:
        """Determina livello sicurezza da contesto"""
        # Logica semplificata - da implementare più complessa
        if user_context.get("is_admin", False):
            return SecurityLevel.CRITICAL

        if user_context.get("has_2fa", False):
            return SecurityLevel.HIGH

        if user_context.get("is_authenticated", False):
            return SecurityLevel.MEDIUM

        return SecurityLevel.LOW

    def create_session(self, user_id: str, user_context: Dict[str, Any]) -> str:
        """Crea nuova sessione sicura"""
        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "ip_address": user_context.get("ip_address"),
            "user_agent": user_context.get("user_agent"),
        }

        self.security_stats["active_sessions"] = len(self.active_sessions)

        if self.audit_logger:
            self.audit_logger.log_security_event(
                f"Session created: {session_id}",
                AuditSeverity.INFO,
                user_context=user_context,
                additional_data={"session_id": session_id, "user_id": user_id},
            )

        return session_id

    def invalidate_session(self, session_id: str):
        """Invalida sessione specifica"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            del self.active_sessions[session_id]

            self.security_stats["active_sessions"] = len(self.active_sessions)

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Session invalidated: {session_id}",
                    AuditSeverity.INFO,
                    additional_data={
                        "session_id": session_id,
                        "user_id": session["user_id"],
                    },
                )

            return True

        return False

    def cleanup_expired_sessions(self):
        """Cleanup sessioni scadute"""
        current_time = datetime.now(timezone.utc)
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            age = current_time - session["last_activity"]
            if age > timedelta(hours=24):  # 24 ore timeout
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.invalidate_session(session_id)

        return len(expired_sessions)

    def get_security_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche sicurezza"""
        return {
            **self.security_stats,
            "active_sessions": len(self.active_sessions),
            "known_devices": len(self.device_fingerprints),
            "rate_limit_entries": len(self.rate_limits),
            "security_policies_count": len(self.security_policies),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def update_security_policy(self, operation: str, policy_updates: Dict[str, Any]):
        """Aggiorna policy sicurezza specifica"""
        try:
            op_type = OperationType(operation)
            current_policy = self.security_policies.get(op_type)

            if not current_policy:
                raise ConfigurationError(
                    f"Policy not found: {operation}",
                    operation,
                    None,
                    list(OperationType),
                )

            # Aggiorna policy con nuovi valori
            for key, value in policy_updates.items():
                if hasattr(current_policy, key):
                    setattr(current_policy, key, value)

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Security policy updated: {operation}",
                    AuditSeverity.INFO,
                    additional_data={
                        "operation": operation,
                        "updates": policy_updates,
                        "new_policy": current_policy.to_dict(),
                    },
                )

            logger.info(f"Security policy updated: {operation}")
            return True

        except ValueError:
            raise ConfigurationError(
                f"Invalid operation: {operation}", operation, None, list(OperationType)
            )
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Policy update failed: {str(e)}",
                    AuditSeverity.ERROR,
                    additional_data={"operation": operation, "error": str(e)},
                )
            raise

    def get_all_policies(self) -> Dict[str, Dict[str, Any]]:
        """Restituisce tutte le policy sicurezza"""
        return {
            op.value: policy.to_dict() for op, policy in self.security_policies.items()
        }
