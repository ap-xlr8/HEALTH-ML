"""Physical Activity Recognition Training Pipeline (1D CNN / Multi-Layer Perceptron)."""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import benchmark_inference_latency, get_model_size_info
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class ActivityClassifier:
    """Classifies movement signals into physical activity categories for mobile edge inference."""

    CLASSES = ["REST", "WALKING", "RUNNING", "HIGH_INTENSITY"]

    def __init__(self, hidden_layer_sizes: tuple = (32, 16), max_iter: int = 200, random_state: int = 42):
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> ActivityClassifier:
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


def train_activity_recognition_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/activity_recognition_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: activity_recognition (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=50, records_per_patient=250)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    anonymized_df = process_and_anonymize(raw_df, dataset_name="activity_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    X_train_raw = FeatureExtractor.extract_activity_features(train_df)
    X_val_raw = FeatureExtractor.extract_activity_features(val_df)
    X_test_raw = FeatureExtractor.extract_activity_features(test_df)

    normalizer = PhysiologicalNormalizer(method="standard")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_train = train_df["activity_class"].values
    y_val = val_df["activity_class"].values
    y_test = test_df["activity_class"].values

    model = ActivityClassifier(hidden_layer_sizes=(32, 16), max_iter=250)
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

    approved = acc >= 0.85 and metrics["latency_p95_ms"] < 5.0 and metrics["size_mb"] < 2.0
    reasons = [] if approved else ["Activity accuracy, latency or size threshold violated"]

    print(f"\n--- Activity Recognition Results ---")
    print(f"Accuracy: {acc * 100:.1f}% | Latency P95: {metrics['latency_p95_ms']:.2f}ms | Size: {metrics['size_kb']:.1f} KB")

    report_path = save_evaluation_report(
        model_name="activity_recognition",
        version=version,
        metrics=metrics,
        feature_schema=FeatureExtractor.SCHEMA_VERSION,
        dataset_hash="activity-v1-synthetic",
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="activity_recognition",
        version=version,
        model_type="neural_classifier",
        target="activity_class",
        deployed_to=["android"],
        algorithm="MLP_Activity_Classifier",
        hyperparameters={"hidden_layer_sizes": [32, 16], "classes": ActivityClassifier.CLASSES},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version=FeatureExtractor.SCHEMA_VERSION,
        changelog="6-axis IMU + magnitude neural activity classifier",
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
    train_activity_recognition_pipeline()
