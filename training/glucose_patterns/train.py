"""Glucose Patterns Estimation Training Pipeline.

Detects physiological patterns correlated with glycemic excursions from indirect wearable metrics.
DISCLAIMER: Non-diagnostic helper; detects physiological correlates, not direct blood glucose.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import calculate_classification_metrics, benchmark_inference_latency, get_model_size_info
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class GlucosePatternDetector:
    """Detects physiological proxy patterns correlating with autonomic glycemic responses."""

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        self.clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GlucosePatternDetector:
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


def train_glucose_patterns_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/glucose_patterns_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: glucose_patterns (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=50, records_per_patient=250)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    anonymized_df = process_and_anonymize(raw_df, dataset_name="glucose_patterns_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    feature_cols = ["hr_bpm", "hrv_ms", "movement_variance", "temp_c"]
    X_train_raw = train_df[feature_cols].copy()
    X_val_raw = val_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    normalizer = PhysiologicalNormalizer(method="robust")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_train = train_df["glucose_pattern_flag"].values
    y_val = val_df["glucose_pattern_flag"].values
    y_test = test_df["glucose_pattern_flag"].values

    model = GlucosePatternDetector(n_estimators=50)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    scores_val = model.predict_proba(X_val)[:, 1]

    metrics = calculate_classification_metrics(y_val, y_pred_val, scores_val)
    metrics.update(benchmark_inference_latency(model, X_test, iterations=100))

    os.makedirs(output_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, "model.pkl")
    normalizer_path = os.path.join(output_dir, "normalizer.pkl")
    joblib.dump(model, model_pkl_path)
    joblib.dump(normalizer, normalizer_path)

    metrics.update(get_model_size_info(model_pkl_path))

    approved = metrics["recall"] >= 0.85 and metrics["false_positive_rate"] <= 0.10
    reasons = [] if approved else ["Glucose pattern sensitivity threshold not met"]

    print(f"\n--- Glucose Pattern Results ---")
    print(f"Recall: {metrics['recall'] * 100:.1f}% | FPR: {metrics['false_positive_rate'] * 100:.1f}% | Precision: {metrics['precision'] * 100:.1f}%")

    report_path = save_evaluation_report(
        model_name="glucose_patterns",
        version=version,
        metrics=metrics,
        feature_schema="1.0.0",
        dataset_hash="glucose-v1-synthetic",
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="glucose_patterns",
        version=version,
        model_type="pattern_correlation_classifier",
        target="glucose_pattern_flag",
        deployed_to=["android", "backend"],
        algorithm="GradientBoosting_Indirect_Correlate",
        hyperparameters={"n_estimators": 50, "features": feature_cols},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version="1.0.0",
        changelog="Indirect autonomic correlation model for metabolic excursion risk",
        patient_count=anonymized_df["patient_id"].nunique(),
        sample_count=len(anonymized_df),
    )

    return {
        "model": model,
        "normalizer": normalizer,
        "metrics": metrics,
        "approved": approved,
        "artifacts_dir": output_dir,
    }


if __name__ == "__main__":
    train_glucose_patterns_pipeline()
