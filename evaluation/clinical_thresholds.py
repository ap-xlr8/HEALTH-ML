"""Clinical Acceptance Criteria & Hard Approval Gates for Health OS ML Models."""

from __future__ import annotations

from typing import Dict, Any, Tuple, List


class ClinicalThresholdViolation(Exception):
    """Raised when a trained model violates clinical safety thresholds."""
    pass


class ClinicalThresholdValidator:
    """Enforces strict medical standards before promoting models to Model Registry."""

    # Official Health OS Clinical Approval Thresholds
    THRESHOLDS = {
        "min_recall": 0.95,          # Sensitivity > 95% (minimize false negatives)
        "max_fpr": 0.05,             # False Positive Rate < 5% (prevent alarm fatigue)
        "min_precision": 0.80,       # Precision > 80%
        "max_latency_p95_ms": 5.0,   # On-device latency < 5ms
        "max_size_mb": 2.0,          # Mobile on-device payload < 2MB
    }

    @classmethod
    def evaluate_model_approval(
        cls,
        metrics: Dict[str, Any],
        is_on_device: bool = True,
        strict_exception: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Validate metrics against clinical approval gates."""
        rejection_reasons: List[str] = []

        recall = metrics.get("recall", metrics.get("sensitivity", 0.0))
        if recall < cls.THRESHOLDS["min_recall"]:
            rejection_reasons.append(
                f"Recall ({recall:.3f}) is below minimum clinical threshold of {cls.THRESHOLDS['min_recall']:.2f}."
            )

        fpr = metrics.get("false_positive_rate", metrics.get("fpr", 0.0))
        if fpr > cls.THRESHOLDS["max_fpr"]:
            rejection_reasons.append(
                f"False Positive Rate ({fpr:.3f}) exceeds maximum allowed clinical threshold of {cls.THRESHOLDS['max_fpr']:.2f}."
            )

        precision = metrics.get("precision", 0.0)
        if precision < cls.THRESHOLDS["min_precision"]:
            rejection_reasons.append(
                f"Precision ({precision:.3f}) is below minimum clinical threshold of {cls.THRESHOLDS['min_precision']:.2f}."
            )

        if is_on_device:
            latency_p95 = metrics.get("latency_p95_ms", 0.0)
            if latency_p95 > cls.THRESHOLDS["max_latency_p95_ms"]:
                rejection_reasons.append(
                    f"Inference Latency P95 ({latency_p95:.2f}ms) exceeds on-device budget of {cls.THRESHOLDS['max_latency_p95_ms']:.1f}ms."
                )

            size_mb = metrics.get("size_mb", 0.0)
            if size_mb > cls.THRESHOLDS["max_size_mb"]:
                rejection_reasons.append(
                    f"Model binary size ({size_mb:.2f}MB) exceeds mobile limit of {cls.THRESHOLDS['max_size_mb']:.1f}MB."
                )

        approved = len(rejection_reasons) == 0

        if not approved and strict_exception:
            raise ClinicalThresholdViolation(
                f"Model rejected due to clinical safety violations:\n - " + "\n - ".join(rejection_reasons)
            )

        return approved, rejection_reasons
