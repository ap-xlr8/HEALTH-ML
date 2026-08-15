"""Time Series Synchronization and Wearable Clock Drift Compensation Module."""

from __future__ import annotations

import pandas as pd
import numpy as np


class TimeSyncProcessor:
    """Aligns asynchronous biometric wearable streams and corrects clock drift."""

    @staticmethod
    def align_and_resample(
        df: pd.DataFrame,
        time_col: str = "timestamp",
        freq: str = "1min",
        method: str = "linear",
    ) -> pd.DataFrame:
        """Resample irregularly sampled biometric streams to regular intervals."""
        if time_col not in df.columns:
            raise KeyError(f"Timestamp column '{time_col}' missing.")

        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.sort_values(by=time_col)

        # Resample numeric columns
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
        resampled = (
            df_sorted.set_index(time_col)[numeric_cols]
            .resample(freq)
            .mean()
            .interpolate(method=method)
            .bfill()
            .ffill()
            .reset_index()
        )
        return resampled

    @staticmethod
    def detect_clock_drift(
        device_timestamps: list[float] | np.ndarray,
        server_timestamps: list[float] | np.ndarray,
        max_drift_seconds: float = 5.0,
    ) -> dict[str, float | bool]:
        """Detect clock drift between wearable device time and server NTP time."""
        dev = np.array(device_timestamps)
        srv = np.array(server_timestamps)
        differences = np.abs(dev - srv)
        mean_drift = float(np.mean(differences))
        max_observed = float(np.max(differences))
        is_drift_critical = max_observed > max_drift_seconds

        return {
            "mean_drift_sec": mean_drift,
            "max_drift_sec": max_observed,
            "is_drift_critical": is_drift_critical,
        }
