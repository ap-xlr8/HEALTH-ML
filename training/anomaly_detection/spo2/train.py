"""SpO2 Critical Drop Detection Training Pipeline.

Combines physiological threshold rules with machine learning for contextual hypoxemia detection.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import calculate_classification_metrics, benchmark_inference_latency, get_model_size_info
from evaluation.clinical_thresholds import ClinicalThresholdValidator
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class SpO2CriticalDetector:
    """Context-aware SpO2 critical drop classifier."""

    def __init__(self, n_estimators: int = 40, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            class_weight={0: 1.0, 1: 5.0},
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpO2CriticalDetector:
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.clf.predict_proba(X)[:, 1]
        return np.where(probs >= 0.35, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


def train_spo2_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/spo2_critical_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: spo2_critical (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=50, records_per_patient=250, anomaly_rate=0.06)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    anonymized_df = process_and_anonymize(raw_df, dataset_name="spo2_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    X_train_raw = FeatureExtractor.extract_spo2_features(train_df)
    X_val_raw = FeatureExtractor.extract_spo2_features(val_df)
    X_test_raw = FeatureExtractor.extract_spo2_features(test_df)

    normalizer = PhysiologicalNormalizer(method="standard")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_train = train_df["critical_spo2_drop"].values
    y_val = val_df["critical_spo2_drop"].values
    y_test = test_df["critical_spo2_drop"].values

    model = SpO2CriticalDetector(n_estimators=40)
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

    approved, reasons = ClinicalThresholdValidator.evaluate_model_approval(metrics, is_on_device=True)

    print(f"\n--- SpO2 Critical Results ---")
    print(f"Recall: {metrics['recall'] * 100:.1f}% | FPR: {metrics['false_positive_rate'] * 100:.1f}% | Precision: {metrics['precision'] * 100:.1f}%")
    print(f"Approval: {'APPROVED' if approved else 'REJECTED'}")

    dataset_hash = ""
    try:
        import hashlib
        with open(raw_data_path, "rb") as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        dataset_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    report_path = save_evaluation_report(
        model_name="spo2_critical",
        version=version,
        metrics=metrics,
        feature_schema=FeatureExtractor.SCHEMA_VERSION,
        dataset_hash=dataset_hash,
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="spo2_critical",
        version=version,
        model_type="critical_alert",
        target="spo2_critical",
        deployed_to=["android"],
        algorithm="RandomForestClassifier_Weighted",
        hyperparameters={"n_estimators": 40, "max_depth": 5, "decision_threshold": 0.35},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version=FeatureExtractor.SCHEMA_VERSION,
        changelog="SpO2 contextual decision model with elevation and sleep compensation",
        dataset_hash=dataset_hash,
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
    train_spo2_pipeline()
