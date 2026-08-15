"""Tests for Clinical Safety Thresholds and Approval Gates."""

import pytest
from evaluation.clinical_thresholds import ClinicalThresholdValidator, ClinicalThresholdViolation


def test_clinical_validator_approval_pass():
    good_metrics = {
        "recall": 0.965,
        "false_positive_rate": 0.032,
        "precision": 0.85,
        "latency_p95_ms": 3.1,
        "size_mb": 0.8,
    }
    approved, reasons = ClinicalThresholdValidator.evaluate_model_approval(good_metrics, is_on_device=True)
    assert approved is True
    assert len(reasons) == 0


def test_clinical_validator_rejection_low_recall():
    low_recall_metrics = {
        "recall": 0.91,  # Below 0.95 minimum
        "false_positive_rate": 0.02,
        "precision": 0.88,
        "latency_p95_ms": 2.5,
        "size_mb": 0.5,
    }
    approved, reasons = ClinicalThresholdValidator.evaluate_model_approval(low_recall_metrics, is_on_device=True)
    assert approved is False
    assert any("Recall" in r for r in reasons)


def test_clinical_validator_rejection_high_fpr():
    high_fpr_metrics = {
        "recall": 0.98,
        "false_positive_rate": 0.082,  # Above 0.05 max (alarm fatigue risk)
        "precision": 0.88,
        "latency_p95_ms": 2.5,
        "size_mb": 0.5,
    }
    approved, reasons = ClinicalThresholdValidator.evaluate_model_approval(high_fpr_metrics, is_on_device=True)
    assert approved is False
    assert any("False Positive Rate" in r for r in reasons)


def test_clinical_validator_strict_exception():
    bad_metrics = {"recall": 0.70, "false_positive_rate": 0.20, "precision": 0.50}
    with pytest.raises(ClinicalThresholdViolation):
        ClinicalThresholdValidator.evaluate_model_approval(bad_metrics, strict_exception=True)
