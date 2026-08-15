"""Tests for Anonymization, Consent Verification, and Patient Splitting."""

import pytest
import pandas as pd
from datasets.synthetic_generator import generate_synthetic_patient_data
from preprocessing.anonymizer import process_and_anonymize, split_by_patient


def test_consent_enforcement_missing_flag():
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "hr_bpm": [70, 80]})
    with pytest.raises(ValueError, match="consent_for_ml_training"):
        process_and_anonymize(df)


def test_consent_enforcement_zero_consented():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "hr_bpm": [70, 80],
        "consent_for_ml_training": [False, False],
    })
    with pytest.raises(ValueError, match="Zero records with active consent"):
        process_and_anonymize(df)


def test_pii_removal_and_tokenization():
    df = generate_synthetic_patient_data(n_patients=3, records_per_patient=10, include_pii=True)
    df["consent_for_ml_training"] = True

    anonymized = process_and_anonymize(df)
    for pii_col in ["name", "email", "ip_address", "device_mac"]:
        assert pii_col not in anonymized.columns

    # Verify patient IDs are tokenized
    for pid in anonymized["patient_id"]:
        assert pid.startswith("anon-")
        assert not pid.startswith("PAT-")


def test_patient_isolated_splits_no_data_leakage():
    df = generate_synthetic_patient_data(n_patients=10, records_per_patient=20)
    df["consent_for_ml_training"] = True
    anonymized = process_and_anonymize(df)

    train_df, val_df, test_df = split_by_patient(anonymized, ratios=(0.70, 0.15, 0.15))

    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())

    # Ensure absolute 0 patient overlap (Zero Data Leakage)
    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0
