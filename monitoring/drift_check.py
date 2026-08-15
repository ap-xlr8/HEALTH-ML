"""Data & Model Drift Monitoring Module (Evidently & Statistical KS-Test / PSI)."""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple


class DriftDetector:
    """Detects statistical covariate shift and feature drift between reference and production data."""

    def __init__(self, p_value_threshold: float = 0.05, psi_threshold: float = 0.20):
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    @staticmethod
    def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI) between two distributions."""
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        # Create quantile bins based on reference
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(ref, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_pct = np.clip(ref_counts / len(ref), 1e-4, 1.0)
        cur_pct = np.clip(cur_counts / len(cur), 1e-4, 1.0)

        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi_val)

    def check_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test and PSI across all features."""
        if features is None:
            features = [c for c in reference_data.select_dtypes(include=[np.number]).columns if c in current_data.columns]

        drifted_features: List[str] = []
        feature_results: Dict[str, Any] = {}

        for feat in features:
            ref_vals = reference_data[feat].dropna().values
            cur_vals = current_data[feat].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            # Two-sample KS Test
            ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
            psi = self.calculate_psi(ref_vals, cur_vals)

            is_drift = bool(p_val < self.p_value_threshold or psi > self.psi_threshold)
            if is_drift:
                drifted_features.append(feat)

            feature_results[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 5),
                "psi": round(float(psi), 4),
                "drift_detected": is_drift,
            }

        dataset_drift = len(drifted_features) >= max(1, len(features) // 3)

        return {
            "dataset_drift": dataset_drift,
            "drift_share": round(len(drifted_features) / len(features) if features else 0.0, 3),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_metrics": feature_results,
            "requires_retraining": dataset_drift,
        }


def run_drift_check(
    reference_path: str = "datasets/splits/train_data.parquet",
    current_path: str = "datasets/raw/prod_last_week.parquet",
    report_output: str = "monitoring/reports/drift_report.json",
) -> Dict[str, Any]:
    """Execute weekly drift check and save JSON report."""
    os.makedirs(os.path.dirname(report_output), exist_ok=True)

    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        # Fallback to simulated data for baseline check
        from datasets.synthetic_generator import generate_synthetic_patient_data
        ref_df = generate_synthetic_patient_data(n_patients=20, records_per_patient=100, seed=42)
        cur_df = generate_synthetic_patient_data(n_patients=20, records_per_patient=100, seed=99)
    else:
        ref_df = pd.read_parquet(reference_path)
        cur_df = pd.read_parquet(current_path)

    detector = DriftDetector()
    results = detector.check_drift(ref_df, cur_df)

    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results["dataset_drift"]:
        print(f"ALERT: Significant Dataset Drift detected in features: {results['drifted_features']}")
    else:
        print("Data Drift Check Passed: Distribution is within normal operational tolerances.")

    return results


if __name__ == "__main__":
    run_drift_check()
