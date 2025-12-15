"""
Bayesian Confidence Validator Module

Implements Kill-Switch logic using Bayesian Credible Intervals.
Prevents betting on confidence buckets with insufficient statistical evidence.

Consensus requirement: Activate only if n_bucket ≥ 50 AND ECE_bucket < 0.15
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy.stats import beta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BucketValidationResult:
    """Result of bucket validation check."""

    is_valid: bool
    reason: str
    ci_lower: float
    ci_upper: float
    win_rate: float
    n_samples: int


class BayesianConfidenceValidator:
    """
    Validates confidence buckets using Bayesian statistics.

    Prevents betting when credible interval excludes break-even point (50%).
    Implements Consensus safety requirements (n≥50, ECE<0.15).
    """

    def __init__(
        self,
        min_samples: int = 50,  # Consensus requirement
        min_winrate: float = 0.50,
        max_bucket_ece: float = 0.15,  # Consensus requirement
    ):
        """
        Initialize validator.

        Args:
            min_samples: Minimum samples required to trust bucket stats
            min_winrate: Minimum acceptable win rate (break-even point)
            max_bucket_ece: Maximum ECE for bucket to be considered calibrated
        """
        self.min_samples = min_samples
        self.min_winrate = min_winrate
        self.max_bucket_ece = max_bucket_ece

    def validate_bucket(
        self, wins: int, losses: int, bucket_ece: Optional[float] = None
    ) -> BucketValidationResult:
        """
        Validate if a confidence bucket is trustworthy for betting.

        Args:
            wins: Number of wins in bucket
            losses: Number of losses in bucket
            bucket_ece: Expected Calibration Error for this bucket (optional)

        Returns:
            BucketValidationResult with validation decision and reasoning
        """
        n_samples = wins + losses

        # Check 1: Minimum sample size (Consensus: n ≥ 50)
        if n_samples < self.min_samples:
            return BucketValidationResult(
                is_valid=False,
                reason=(
                    f"Insufficient samples: N={n_samples} < {self.min_samples}. "
                    "Bucket statistics unreliable (Consensus requirement)."
                ),
                ci_lower=0.0,
                ci_upper=1.0,
                win_rate=0.0 if n_samples == 0 else wins / n_samples,
                n_samples=n_samples,
            )

        # Check 2: ECE threshold (if provided)
        if bucket_ece is not None and bucket_ece > self.max_bucket_ece:
            return BucketValidationResult(
                is_valid=False,
                reason=(
                    f"Poor calibration: ECE={bucket_ece:.3f} > {self.max_bucket_ece}. "
                    "Bucket predictions unreliable (Consensus requirement)."
                ),
                ci_lower=0.0,
                ci_upper=1.0,
                win_rate=wins / n_samples,
                n_samples=n_samples,
            )

        # Compute Bayesian Credible Interval
        # Beta(α, β) with α = wins+1, β = losses+1 (Jeffreys prior)
        alpha, beta_param = wins + 1, losses + 1
        ci_lower = beta.ppf(0.025, alpha, beta_param)  # 2.5th percentile
        ci_upper = beta.ppf(0.975, alpha, beta_param)  # 97.5th percentile
        win_rate = wins / n_samples

        # Check 3: Credible interval excludes break-even (Kill-Switch logic)
        if ci_upper < self.min_winrate:
            return BucketValidationResult(
                is_valid=False,
                reason=(
                    f"Credible Interval [{ci_lower:.2f}, {ci_upper:.2f}] "
                    f"excludes break-even ({self.min_winrate:.2f}). "
                    "Statistical evidence of negative edge. KILL-SWITCH ACTIVATED."
                ),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                win_rate=win_rate,
                n_samples=n_samples,
            )

        # Bucket is valid
        return BucketValidationResult(
            is_valid=True,
            reason=(
                f"Bucket validated: N={n_samples}, WR={win_rate:.1%}, "
                f"CI=[{ci_lower:.2f}, {ci_upper:.2f}]"
            ),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            win_rate=win_rate,
            n_samples=n_samples,
        )

    def should_allow_bet(
        self, confidence: float, bucket_stats: dict
    ) -> Tuple[bool, str]:
        """
        Pre-bet validation check.

        Args:
            confidence: Model's calibrated confidence
            bucket_stats: Dict with keys: wins, losses, ece (optional)

        Returns:
            (allow_bet, reason)
        """
        wins = bucket_stats.get("wins", 0)
        losses = bucket_stats.get("losses", 0)
        bucket_ece = bucket_stats.get("ece", None)

        result = self.validate_bucket(wins, losses, bucket_ece)

        if not result.is_valid:
            logger.error(
                f"[KILL-SWITCH] Bet VETOED. Confidence={confidence:.2f}. "
                f"{result.reason}"
            )
            return False, result.reason

        logger.info(
            f"[VALIDATION] Bet APPROVED. Confidence={confidence:.2f}. {result.reason}"
        )
        return True, result.reason

    def get_bucket_range(
        self, confidence: float, bucket_width: float = 0.1
    ) -> Tuple[float, float]:
        """
        Get bucket boundaries for a given confidence.

        Default: 0.1 width buckets (0-0.1, 0.1-0.2, ..., 0.9-1.0)
        """
        bucket_idx = int(confidence // bucket_width)
        lower = bucket_idx * bucket_width
        upper = min((bucket_idx + 1) * bucket_width, 1.0)
        return lower, upper
