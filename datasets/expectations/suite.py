"""Data Quality & Clinical Expectations Suite for Health OS ML Module."""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, List


class ClinicalDataValidator:
    """Validates physiological boundaries and structural consistency of biometric data."""

    # Physiological hard boundary rules
    RULES = {
        "hr_bpm": {"min": 30.0, "max": 240.0, "required": True},
        "hrv_ms": {"min": 0.0, "max": 300.0, "required": True},
        "spo2_percent": {"min": 60.0, "max": 100.0, "required": True},
        "resp_rate": {"min": 4.0, "max": 60.0, "required": False},
        "temp_c": {"min": 30.0, "max": 45.0, "required": False},
        "bp_sys": {"min": 50.0, "max": 260.0, "required": False},
        "bp_dia": {"min": 30.0, "max": 160.0, "required": False},
        "activity_intensity": {"allowed": [0, 1, 2, 3], "required": True},
    }

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input DataFrame against clinical constraints and structure."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {
            "total_records": len(df),
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
        }

        # Check required patient_id
        if "patient_id" not in df.columns:
            errors.append("Missing required column: 'patient_id'")
        elif df["patient_id"].isnull().any():
            errors.append("Null values found in 'patient_id'")

        # Validate range constraints
        for col, rule in cls.RULES.items():
            if col not in df.columns:
                if rule.get("required", False):
                    errors.append(f"Required clinical column missing: '{col}'")
                continue

            # Null check
            null_count = df[col].isnull().sum()
            if null_count > 0:
                warnings.append(f"Column '{col}' has {null_count} null entries.")

            # Numeric range checks
            valid_series = df[col].dropna()
            if "min" in rule and (valid_series < rule["min"]).any():
                violations = int((valid_series < rule["min"]).sum())
                errors.append(f"Column '{col}' has {violations} values below clinical minimum {rule['min']}")

            if "max" in rule and (valid_series > rule["max"]).any():
                violations = int((valid_series > rule["max"]).sum())
                errors.append(f"Column '{col}' has {violations} values above clinical maximum {rule['max']}")

            if "allowed" in rule and not set(valid_series.unique()).issubset(set(rule["allowed"])):
                invalid = set(valid_series.unique()) - set(rule["allowed"])
                errors.append(f"Column '{col}' contains invalid categorical values: {invalid}")

        success = len(errors) == 0
        return {
            "success": success,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }


def validate_heart_rate_data(file_or_df: str | pd.DataFrame) -> Dict[str, Any]:
    """Helper function to validate dataset before training."""
    if isinstance(file_or_df, str):
        if file_or_df.endswith(".parquet"):
            df = pd.read_parquet(file_or_df)
        elif file_or_df.endswith(".csv"):
            df = pd.read_csv(file_or_df)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
    else:
        df = file_or_df

    result = ClinicalDataValidator.validate_dataframe(df)
    if not result["success"]:
        raise ValueError(f"Data Quality Validation Failed: {result['errors']}")
    return result
