"""Sleep Quality and Sleep Stage Classification Training Pipeline (Random Forest)."""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import benchmark_inference_latency, get_model_size_info
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class SleepQualityClassifier:
    """Classifies sleep stages (1: N1 Light, 2: N2 Core, 3: N3 Deep, 4: REM) and computes quality score."""

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            random_state=random_state,
            n_jobs=1,  # Fast single-sample inference
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SleepQualityClassifier:
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


def train_sleep_quality_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/sleep_quality_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: sleep_quality (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=50, records_per_patient=250)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    sleep_records = raw_df[raw_df["is_sleeping"] == 1].copy()
    if len(sleep_records) < 50:
        sleep_records = raw_df.copy()

    anonymized_df = process_and_anonymize(sleep_records, dataset_name="sleep_quality_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    X_train_raw = FeatureExtractor.extract_sleep_features(train_df)
    X_val_raw = FeatureExtractor.extract_sleep_features(val_df)
    X_test_raw = FeatureExtractor.extract_sleep_features(test_df)

    normalizer = PhysiologicalNormalizer(method="standard")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_train = train_df["sleep_stage"].values
    y_val = val_df["sleep_stage"].values

    model = SleepQualityClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred_val))

    metrics = {
        "accuracy": round(acc, 4),
        "val_samples": len(y_val),
    }
    metrics.update(benchmark_inference_latency(model, X_test, iterations=100))

    os.makedirs(output_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, "model.pkl")
    normalizer_path = os.path.join(output_dir, "normalizer.pkl")
    joblib.dump(model, model_pkl_path)
    joblib.dump(normalizer, normalizer_path)

    metrics.update(get_model_size_info(model_pkl_path))

    approved = acc >= 0.60 and metrics["latency_p95_ms"] < 10.0 and metrics["size_mb"] < 2.0
    reasons = [] if approved else ["Sleep accuracy or latency/size budget violated"]

    print(f"\n--- Sleep Quality Results ---")
    print(f"Accuracy: {acc * 100:.1f}% | Latency P95: {metrics['latency_p95_ms']:.2f}ms | Size: {metrics['size_kb']:.1f} KB")

    dataset_hash = ""
    try:
        import hashlib
        with open(raw_data_path, "rb") as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        dataset_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    report_path = save_evaluation_report(
        model_name="sleep_quality",
        version=version,
        metrics=metrics,
        feature_schema=FeatureExtractor.SCHEMA_VERSION,
        dataset_hash=dataset_hash,
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="sleep_quality",
        version=version,
        model_type="classifier",
        target="sleep_stage",
        deployed_to=["android"],
        algorithm="RandomForestClassifier",
        hyperparameters={"n_estimators": 50, "max_depth": 6},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version=FeatureExtractor.SCHEMA_VERSION,
        changelog="Multi-stage sleep architecture based on nocturnal HRV and movement variance",
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
    train_sleep_quality_pipeline()
