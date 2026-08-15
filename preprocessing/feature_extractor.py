"""Biometric Feature Extraction Module for Health OS ML Pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class FeatureExtractor:
    """Transforms raw health measurements into structured features for ML models."""

    SCHEMA_VERSION = "1.0.0"

    @staticmethod
    def extract_heart_rate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for Isolation Forest heart rate anomaly model.
        Features: hr_bpm, hrv_ms, activity_intensity
        """
        required = ["hr_bpm", "hrv_ms", "activity_intensity"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing required feature column: {col}")
        
        feats = df[required].copy()
        feats["hr_bpm"] = feats["hr_bpm"].astype(np.float32)
        feats["hrv_ms"] = feats["hrv_ms"].astype(np.float32)
        feats["activity_intensity"] = feats["activity_intensity"].astype(np.float32)
        return feats

    @staticmethod
    def extract_spo2_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for SpO2 critical drop model.
        Features: spo2_percent, elevation_meters, is_sleeping
        """
        required = ["spo2_percent", "elevation_meters", "is_sleeping"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing required feature column: {col}")

        feats = df[required].copy()
        feats["spo2_percent"] = feats["spo2_percent"].astype(np.float32)
        feats["elevation_meters"] = feats["elevation_meters"].astype(np.float32)
        feats["is_sleeping"] = feats["is_sleeping"].astype(np.float32)
        return feats

    @staticmethod
    def extract_combined_vitals_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for multivariate vitals AutoEncoder model.
        Features: hr_bpm, spo2_percent, resp_rate, temp_c, bp_sys, bp_dia
        """
        required = ["hr_bpm", "spo2_percent", "resp_rate", "temp_c", "bp_sys", "bp_dia"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing required feature column: {col}")

        feats = df[required].copy()
        for col in required:
            feats[col] = feats[col].astype(np.float32)
        return feats

    @staticmethod
    def extract_sleep_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for Sleep Quality model.
        Features: movement_variance, hr_bpm, hrv_ms, duration_mins (simulated/aggregated)
        """
        feats = pd.DataFrame()
        feats["movement_variance"] = df["movement_variance"].astype(np.float32) if "movement_variance" in df else np.float32(0.02)
        feats["hr_avg"] = df["hr_bpm"].astype(np.float32)
        feats["hrv_avg"] = df["hrv_ms"].astype(np.float32)
        feats["duration_mins"] = np.float32(480.0) # default standard sleep window
        return feats

    @staticmethod
    def extract_activity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for Activity Recognition (Accelerometer & Gyroscope).
        Features: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, accel_magnitude
        """
        cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        feats = pd.DataFrame()
        for c in cols:
            feats[c] = df[c].astype(np.float32) if c in df else np.float32(0.0)
        
        # Derived acceleration magnitude
        feats["accel_magnitude"] = np.sqrt(
            feats["accel_x"] ** 2 + feats["accel_y"] ** 2 + feats["accel_z"] ** 2
        ).astype(np.float32)
        return feats

    @staticmethod
    def extract_risk_score_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract aggregated 30-day historical features for Cloud Risk Score."""
        required = ["hr_bpm", "hrv_ms", "spo2_percent", "resp_rate", "temp_c", "bp_sys", "bp_dia", "activity_intensity"]
        feats = df[required].copy()
        for c in required:
            feats[c] = feats[c].astype(np.float32)
        return feats
