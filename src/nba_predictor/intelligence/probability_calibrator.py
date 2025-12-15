"""
Probability Calibrator Module

Implements Platt Scaling for post-hoc probability calibration.
Fixes overconfidence issue where model predicts 95% confidence but achieves 48% win rate.

Research basis:
- Platt Scaling best for N<100 (low variance, 2 parameters)
- Kurz et al. 2022: Comparison of calibration methods
"""

import numpy as np
import logging
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from scipy.stats import beta
import json

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """
    Post-hoc probability calibration using Platt Scaling.

    Maps raw model confidence -> calibrated probability aligned with true win rate.
    Uses logistic regression on logit-transformed confidences.

    Attributes:
        calibrator: LogisticRegression model (a, b parameters)
        is_fitted: Whether calibrator has been trained
        calibration_data: Historical (confidence, outcome) pairs
    """

    def __init__(self, regularization_strength: float = 0.01):  # Conservative for N=59
        """
        Initialize Platt Calibrator with EXTREME regularization for small samples.

        Args:
            regularization_strength: Inverse of regularization (C parameter).
                                   Default 0.01 = 100x stronger than standard (1.0).
                                   Per Consensus: Conservative calibration for N<100.
        """
        self.C = regularization_strength
        # Platt Scaling: fit logistic regression on raw confidences
        self.platt_model = LogisticRegression(
            penalty="l2",
            C=self.C,
            solver="lbfgs",
            class_weight="balanced",  # Handle imbalanced wins/losses (Consensus rec)
            random_state=42,
            max_iter=1000,
        )
        self.is_fitted = False
        self.calibration_data = []

    def fit(self, raw_confidences: np.ndarray, outcomes: np.ndarray):
        """
        Fit calibrator on historical data.

        Args:
            raw_confidences: Array of model confidences [0,1]
            outcomes: Array of binary outcomes (0=loss, 1=win)

        Note: For time-series data (betting), use time-ordered split, not random CV.
        """
        if len(raw_confidences) < 10:
            logger.warning(
                f"[PLATT CALIBRATION] Only {len(raw_confidences)} samples. "
                "Calibration may be unreliable. Consider increasing C (regularization)."
            )

        # Store for analysis
        self.calibration_data = list(zip(raw_confidences, outcomes))

        # Clip to avoid log(0) or log(1)
        clipped_conf = np.clip(raw_confidences, 0.01, 0.99)

        # Transform to logits
        logits = logit(clipped_conf).reshape(-1, 1)

        # Fit logistic regression
        self.platt_model.fit(logits, outcomes)
        self.is_fitted = True

        # Log parameters
        a = self.platt_model.intercept_[0]
        b = self.platt_model.coef_[0][0]

        logger.info(
            f"[PLATT CALIBRATION] Fitted on N={len(raw_confidences)}. "
            f"Parameters: a={a:.3f}, b={b:.3f}"
        )

        # Compute calibration metrics
        calibrated_probs = self.calibrate_batch(raw_confidences)
        ece = self._compute_ece(calibrated_probs, outcomes, n_bins=5)
        brier_before = np.mean((raw_confidences - outcomes) ** 2)
        brier_after = np.mean((calibrated_probs - outcomes) ** 2)

        logger.info(
            f"[PLATT CALIBRATION] ECE={ece:.3f}, "
            f"Brier: {brier_before:.3f} -> {brier_after:.3f} "
            f"(Improvement: {(1 - brier_after / brier_before) * 100:.1f}%)"
        )

    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a single confidence value.

        Args:
            raw_confidence: Model output confidence [0,1]

        Returns:
            Calibrated probability aligned with true win rate
        """
        if not self.is_fitted:
            logger.warning(
                "[PLATT CALIBRATION] Calibrator not fitted. Returning raw confidence."
            )
            return raw_confidence

        # Clip and transform
        clipped = np.clip(raw_confidence, 0.01, 0.99)
        logit_raw = logit(clipped)

        # Predict calibrated probability
        calibrated_prob = self.platt_model.predict_proba([[logit_raw]])[0][1]

        correction = calibrated_prob - raw_confidence

        logger.debug(
            f"[CALIBRATION] Raw={raw_confidence:.3f} -> "
            f"Calibrated={calibrated_prob:.3f} "
            f"(Δ={correction:+.3f})"
        )

        return float(calibrated_prob)

    def calibrate_batch(self, raw_confidences: np.ndarray) -> np.ndarray:
        """Calibrate multiple confidences at once."""
        if not self.is_fitted:
            return raw_confidences

        clipped = np.clip(raw_confidences, 0.01, 0.99)
        logits = logit(clipped).reshape(-1, 1)
        calibrated = self.platt_model.predict_proba(logits)[:, 1]

        return calibrated

    def _compute_ece(
        self, confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 5
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures calibration quality. Lower is better.
        Target: ECE < 0.10 (Consensus requirement)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1]
            )
            if bin_mask.sum() > 0:
                bin_conf = confidences[bin_mask].mean()
                bin_acc = outcomes[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(confidences)
                ece += bin_weight * abs(bin_conf - bin_acc)

        return ece

    def get_bucket_stats(self, confidence: float, window: float = 0.1) -> dict:
        """
        Get statistics for confidence bucket (for Bayesian Kill-Switch).

        Args:
            confidence: Target confidence value
            window: Bucket width (e.g., 0.1 = ±5% around confidence)

        Returns:
            Dict with wins, losses, win_rate, credible_interval
        """
        if not self.calibration_data:
            return {
                "n": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 1.0,
            }

        # Find samples in bucket
        bucket_data = [
            (conf, outcome)
            for conf, outcome in self.calibration_data
            if abs(conf - confidence) <= window / 2
        ]

        if not bucket_data:
            return {
                "n": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 1.0,
            }

        outcomes = [outcome for _, outcome in bucket_data]
        wins = sum(outcomes)
        losses = len(outcomes) - wins

        # Bayesian Credible Interval (Beta distribution)
        alpha, beta_param = wins + 1, losses + 1
        ci_lower = beta.ppf(0.025, alpha, beta_param)
        ci_upper = beta.ppf(0.975, alpha, beta_param)

        return {
            "n": len(outcomes),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(outcomes) if outcomes else 0.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def save(self, path: str):
        """Save calibrator state to file."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "calibrator": self.platt_model,
                    "is_fitted": self.is_fitted,
                    "calibration_data": self.calibration_data,
                },
                f,
            )
        logger.info(f"[PLATT CALIBRATION] Saved to {path}")

    def load(self, path: str):
        """Load calibrator state from file."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        self.platt_model = state["calibrator"]
        self.is_fitted = state["is_fitted"]
        self.calibration_data = state["calibration_data"]
        logger.info(f"[PLATT CALIBRATION] Loaded from {path}")
