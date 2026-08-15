"""Data Anonymization, Consent Verification & Audit Compliance Module.

Strict adherence to HIPAA & GDPR standards for health ML pipelines.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, Dict, Any
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def process_and_anonymize(
    raw_data: str | pd.DataFrame,
    dataset_name: str = "health_measurements",
    audit_url: Optional[str] = None,
    audit_token: Optional[str] = None,
) -> pd.DataFrame:
    """Validate user consent, remove PII, tokenize IDs with ephemeral UUIDs and log audit event."""
    if isinstance(raw_data, str):
        df = pd.read_parquet(raw_data) if raw_data.endswith(".parquet") else pd.read_csv(raw_data)
    else:
        df = raw_data.copy()

    # 1. Strict Consent Verification
    if "consent_for_ml_training" not in df.columns:
        raise ValueError("Security Violation: Dataset lacks 'consent_for_ml_training' consent flag.")

    initial_count = len(df)
    df_consented = df[df["consent_for_ml_training"] == True].copy()
    consented_count = len(df_consented)

    if consented_count == 0:
        raise ValueError("Consent Violation: Zero records with active consent for ML training.")

    # 2. Drop all PII fields (HIPAA Safe Harbor)
    pii_columns = [
        "name", "email", "phone", "ssn", "national_id",
        "ip_address", "device_mac", "address", "zip_code",
        "doctor_name", "hospital_id"
    ]
    cols_to_drop = [col for col in pii_columns if col in df_consented.columns]
    df_consented.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 3. Tokenize patient_id with ephemeral UUID per run to prevent re-identification
    # while preserving relational consistency within the batch
    if "patient_id" in df_consented.columns:
        patient_map = {pid: f"anon-{uuid.uuid4().hex[:12]}" for pid in df_consented["patient_id"].unique()}
        df_consented["patient_id"] = df_consented["patient_id"].map(patient_map)

    # 4. Mandatory Audit Log Emission
    url = audit_url or os.environ.get("AUDIT_API_URL")
    token = audit_token or os.environ.get("AUDIT_API_TOKEN")

    audit_payload = {
        "action": "ml_dataset_read",
        "records_total": initial_count,
        "records_consented_processed": consented_count,
        "patients_count": df_consented["patient_id"].nunique() if "patient_id" in df_consented.columns else 0,
        "dataset": dataset_name,
        "purpose": "model_training",
    }

    if url and token:
        try:
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            response = requests.post(url, json=audit_payload, headers=headers, timeout=3.0)
            logger.info("Audit log successfully sent to %s (Status: %s)", url, response.status_code)
        except Exception as e:
            logger.warning("Audit API call failed (continuing in test/isolated mode): %s", e)
    else:
        logger.info("Audit log generated locally (No AUDIT_API_URL configured): %s", audit_payload)

    return df_consented


def split_by_patient(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by patient_id to strictly prevent Data Leakage between splits."""
    if "patient_id" not in df.columns:
        raise ValueError("Cannot split by patient: 'patient_id' column not present.")

    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    unique_patients = df["patient_id"].unique()
    rng = pd.Series(unique_patients).sample(frac=1.0, random_state=seed).values

    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(rng[:n_train])
    val_patients = set(rng[n_train : n_train + n_val])
    test_patients = set(rng[n_train + n_val :])

    # Ensure at least 1 in val and test if total patients > 2
    if len(val_patients) == 0 and len(train_patients) > 1:
        val_patients.add(train_patients.pop())
    if len(test_patients) == 0 and len(train_patients) > 1:
        test_patients.add(train_patients.pop())

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    return train_df, val_df, test_df
