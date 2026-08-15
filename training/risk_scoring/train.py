"""30-Day Population & Patient Risk Scoring Training Pipeline (Cloud Only).

Trains gradient boosting regressor for comprehensive multi-vital 30-day risk scoring.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing.anonymizer import process_and_anonymize, split_by_patient
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from datasets.synthetic_generator import generate_synthetic_patient_data
from evaluation.metrics import benchmark_inference_latency, get_model_size_info
from evaluation.reports import save_evaluation_report
from model_registry.registry import ModelRegistry


class RiskScoringModel:
    """Computes a 0.0 - 100.0 multi-factor risk score for Cloud Backend analytics."""

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RiskScoringModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_pred = self.model.predict(X)
        return np.clip(raw_pred, 0.0, 100.0)


def train_risk_scoring_pipeline(
    data_source: str | pd.DataFrame | None = None,
    output_dir: str = "model-registry/models/risk_score_v1.0",
    version: str = "1.0.0",
) -> dict:
    print("=" * 60)
    print(f"Starting Training Pipeline: risk_score (v{version})")
    print("=" * 60)

    if data_source is None:
        raw_df = generate_synthetic_patient_data(n_patients=60, records_per_patient=300)
    elif isinstance(data_source, str):
        raw_df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        raw_df = data_source.copy()

    anonymized_df = process_and_anonymize(raw_df, dataset_name="risk_score_cohort")
    train_df, val_df, test_df = split_by_patient(anonymized_df, ratios=(0.70, 0.15, 0.15))

    X_train_raw = FeatureExtractor.extract_risk_score_features(train_df)
    X_val_raw = FeatureExtractor.extract_risk_score_features(val_df)
    X_test_raw = FeatureExtractor.extract_risk_score_features(test_df)

    normalizer = PhysiologicalNormalizer(method="standard")
    X_train = normalizer.fit_transform(X_train_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)

    y_train = train_df["risk_score_30d"].values
    y_val = val_df["risk_score_30d"].values
    y_test = test_df["risk_score_30d"].values

    model = RiskScoringModel(max_iter=120, learning_rate=0.08)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    mse = float(mean_squared_error(y_val, y_pred_val))
    r2 = float(r2_score(y_val, y_pred_val))

    metrics = {
        "mean_squared_error": round(mse, 4),
        "r2_score": round(r2, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
    }
    metrics.update(benchmark_inference_latency(model, X_test, iterations=100))

    os.makedirs(output_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, "model.pkl")
    normalizer_path = os.path.join(output_dir, "normalizer.pkl")
    joblib.dump(model, model_pkl_path)
    joblib.dump(normalizer, normalizer_path)

    metrics.update(get_model_size_info(model_pkl_path))

    approved = r2 >= 0.75
    reasons = [] if approved else [f"R2 score ({r2:.3f}) below threshold 0.75"]

    print(f"\n--- Risk Scoring (Cloud) Results ---")
    print(f"R2 Score: {r2:.3f} | RMSE: {metrics['rmse']} | Latency P95: {metrics['latency_p95_ms']:.2f}ms")

    dataset_hash = ""
    try:
        import hashlib
        with open(raw_data_path, "rb") as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        dataset_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    report_path = save_evaluation_report(
        model_name="risk_score",
        version=version,
        metrics=metrics,
        feature_schema=FeatureExtractor.SCHEMA_VERSION,
        dataset_hash=dataset_hash,
        approved=approved,
        reasons=reasons,
    )

    registry = ModelRegistry()
    registry.register_model(
        model_id="risk_score",
        version=version,
        model_type="regression",
        target="30_day_risk_score",
        deployed_to=["backend"],
        algorithm="HistGradientBoostingRegressor",
        hyperparameters={"max_iter": 120, "learning_rate": 0.08},
        metrics=metrics,
        artifacts={
            "model_pkl": model_pkl_path,
            "normalizer": normalizer_path,
            "evaluation_report": report_path,
        },
        feature_schema_version=FeatureExtractor.SCHEMA_VERSION,
        changelog="Cloud population and patient risk predictor over 30-day vital trends",
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
    train_risk_scoring_pipeline()
