"""Health OS ML Complete Catalog Training & Export Pipeline Runner."""

from __future__ import annotations

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.synthetic_generator import generate_and_save_datasets
from training.anomaly_detection.heart_rate.train import train_heart_rate_pipeline
from training.anomaly_detection.spo2.train import train_spo2_pipeline
from training.anomaly_detection.combined.train import train_combined_vitals_pipeline
from training.sleep_quality.train import train_sleep_quality_pipeline
from training.activity_recognition.train import train_activity_recognition_pipeline
from training.glucose_patterns.train import train_glucose_patterns_pipeline
from training.risk_scoring.train import train_risk_scoring_pipeline
from export.to_tflite import TFLiteExporter
from export.to_onnx import ONNXExporter
from export.quantize import ModelQuantizer
from monitoring.drift_check import run_drift_check
from model_registry.registry import ModelRegistry


def main():
    start_time = time.time()
    print("=" * 70)
    print("HEALTH OS - FULL MACHINE LEARNING PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Datasets Ingestion & Validation
    print("\n>>> 1. Datasets Ingestion & Validation...")
    raw_data_path = generate_and_save_datasets()

    # 2. Train all models
    print("\n>>> 2. Training Models in Catalog...")

    # Model 1: Heart Rate Anomaly
    hr_res = train_heart_rate_pipeline(raw_data_path)
    TFLiteExporter.export_model(
        hr_res["model"],
        os.path.join(hr_res["artifacts_dir"], "model.tflite"),
        input_shape=(1, 3),
    )
    ONNXExporter.export_model(
        hr_res["model"],
        os.path.join(hr_res["artifacts_dir"], "model.onnx"),
        n_features=3,
    )
    ModelQuantizer.compress_model(
        os.path.join(hr_res["artifacts_dir"], "model.pkl"),
        os.path.join(hr_res["artifacts_dir"], "model.quant.pkl"),
    )

    # Model 2: SpO2 Critical
    spo2_res = train_spo2_pipeline(raw_data_path)
    TFLiteExporter.export_model(
        spo2_res["model"],
        os.path.join(spo2_res["artifacts_dir"], "model.tflite"),
        input_shape=(1, 3),
    )

    # Model 3: Combined Vitals (AutoEncoder)
    vitals_res = train_combined_vitals_pipeline(raw_data_path)
    ONNXExporter.export_model(
        vitals_res["model"],
        os.path.join(vitals_res["artifacts_dir"], "model.onnx"),
        n_features=6,
    )

    # Model 4: Sleep Quality
    sleep_res = train_sleep_quality_pipeline(raw_data_path)
    TFLiteExporter.export_model(
        sleep_res["model"],
        os.path.join(sleep_res["artifacts_dir"], "model.tflite"),
        input_shape=(1, 4),
    )

    # Model 5: Activity Recognition
    act_res = train_activity_recognition_pipeline(raw_data_path)
    TFLiteExporter.export_model(
        act_res["model"],
        os.path.join(act_res["artifacts_dir"], "model.tflite"),
        input_shape=(1, 7),
    )

    # Model 6: Glucose Patterns
    glu_res = train_glucose_patterns_pipeline(raw_data_path)
    TFLiteExporter.export_model(
        glu_res["model"],
        os.path.join(glu_res["artifacts_dir"], "model.tflite"),
        input_shape=(1, 4),
    )
    ONNXExporter.export_model(
        glu_res["model"],
        os.path.join(glu_res["artifacts_dir"], "model.onnx"),
        n_features=4,
    )

    # Model 7: Risk Scoring (Cloud)
    risk_res = train_risk_scoring_pipeline(raw_data_path)
    ONNXExporter.export_model(
        risk_res["model"],
        os.path.join(risk_res["artifacts_dir"], "model.onnx"),
        n_features=8,
    )

    # 3. Drift Monitoring Check
    print("\n>>> 3. Running Drift Baseline Check...")
    run_drift_check()

    # 4. Summary from Model Registry
    print("\n>>> 4. Model Registry Status:")
    registry = ModelRegistry()
    for m_id, versions in registry.data.get("models", {}).items():
        latest_ver = sorted(list(versions.keys()))[-1]
        entry = versions[latest_ver]
        alg = entry.get("training", {}).get("algorithm", "N/A")
        print(f"  - [{m_id}] v{latest_ver} ({alg}) -> Targets: {entry['deployedTo']}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
