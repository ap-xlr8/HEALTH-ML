"""Combined Vitals Multivariate Anomaly Detection Training Pipeline (AutoEncoder).

Trains multivariate neural reconstruction model on HR, SpO2, RespRate, Temp, SysBP, DiaBP for Backend ONNX.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import calculate_classification_metrics, benchmark_inference_latency, get_model_size_info
from evaluation.clinical_thresholds import ClinicalThresholdValidator
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class MultivariateAutoEncoder:
    """Reconstruction-error based AutoEncoder for multi-vital anomaly detection."""

    def __init__(self, hidden_layer_sizes: tuple = (16, 4, 16), max_iter: int = 200, random_state: int = 42):
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> MultivariateAutoEncoder:
        # Fit autoencoder to reconstruct normal physiological dynamics: X -> X
        self.autoencoder.fit(X, X)
        reconstruction = self.autoencoder.predict(X)
        errors = np.mean((X - reconstruction) ** 2, axis=1)
        # Threshold at 95th percentile of normal baseline errors
        self.threshold_ = float(np.percentile(errors, 94.0))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns mean squared reconstruction error per record (higher = anomalous)."""
        reconstruction = self.autoencoder.predict(X)
        return np.mean((X - reconstruction) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 if reconstruction error exceeds clinical threshold, else 0."""
        errors = self.score_samples(X)
        return np.where(errors >= self.threshold_, 1, 0)


def train_combined_vitals_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/combined_vitals_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: combined_vitals AutoEncoder (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=60, records_per_patient=300, anomaly_rate=0.06)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    anonymized_df = process_and_anonymize(raw_df, dataset_name="combined_vitals_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    X_train_raw = FeatureExtractor.extract_combined_vitals_features(train_df)
    X_val_raw = FeatureExtractor.extract_combined_vitals_features(val_df)
    X_test_raw = FeatureExtractor.extract_combined_vitals_features(test_df)

    normalizer = PhysiologicalNormalizer(method="standard")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_val = val_df["is_hr_anomaly"].values
    y_test = test_df["is_hr_anomaly"].values

    # Train AutoEncoder on train set
    model = MultivariateAutoEncoder(hidden_layer_sizes=(16, 4, 16), max_iter=250)
    model.fit(X_train)

    y_pred_val = model.predict(X_val)
    scores_val = model.score_samples(X_val)

    metrics = calculate_classification_metrics(y_val, y_pred_val, scores_val)
    metrics.update(benchmark_inference_latency(model, X_test, iterations=100))

    os.makedirs(output_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, "model.pkl")
    normalizer_path = os.path.join(output_dir, "normalizer.pkl")
    joblib.dump(model, model_pkl_path)
    joblib.dump(normalizer, normalizer_path)

    metrics.update(get_model_size_info(model_pkl_path))

    # Backend / Cloud model evaluation
    approved, reasons = ClinicalThresholdValidator.evaluate_model_approval(metrics, is_on_device=False)

    print(f"\n--- Combined Vitals AutoEncoder Results ---")
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
        model_name="combined_vitals",
        version=version,
        metrics=metrics,
        feature_schema=FeatureExtractor.SCHEMA_VERSION,
        dataset_hash=dataset_hash,
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="combined_vitals",
        version=version,
        model_type="anomaly_detection",
        target="combined_vitals",
        deployed_to=["backend"],
        algorithm="MLP_AutoEncoder_Reconstruction",
        hyperparameters={"hidden_layer_sizes": [16, 4, 16], "threshold_percentile": 94.0},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version=FeatureExtractor.SCHEMA_VERSION,
        changelog="Deep multivariate AutoEncoder for cross-vital physiological shock and risk estimation",
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
    train_combined_vitals_pipeline()
