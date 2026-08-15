"""Tests for Synthetic Data Generation and Clinical Expectations."""

import pytest
import pandas as pd
from datasets.synthetic_generator import generate_synthetic_patient_data
from datasets.expectations.suite import ClinicalDataValidator, validate_heart_rate_data


def test_synthetic_data_structure():
    df = generate_synthetic_patient_data(n_patients=5, records_per_patient=20)
    assert len(df) == 100
    assert df["patient_id"].nunique() == 5
    assert "hr_bpm" in df.columns
    assert "hrv_ms" in df.columns
    assert "spo2_percent" in df.columns
    assert "is_hr_anomaly" in df.columns
    assert "consent_for_ml_training" in df.columns


def test_clinical_expectations_validator_valid():
    df = generate_synthetic_patient_data(n_patients=5, records_per_patient=20)
    # Ensure consenting only for clinical validation
    df["consent_for_ml_training"] = True
    result = ClinicalDataValidator.validate_dataframe(df)
    assert result["success"] is True
    assert len(result["errors"]) == 0


def test_clinical_expectations_validator_out_of_bounds():
    df = generate_synthetic_patient_data(n_patients=2, records_per_patient=10)
    df.loc[0, "hr_bpm"] = 350.0  # Clinically impossible HR
    result = ClinicalDataValidator.validate_dataframe(df)
    assert result["success"] is False
    assert any("above clinical maximum" in err for err in result["errors"])
