"""Tests for Model Export (TFLite, ONNX, Quantization) and Monitoring."""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from export.to_tflite import TFLiteExporter
from export.to_onnx import ONNXExporter
from export.quantize import ModelQuantizer
from monitoring.drift_check import DriftDetector
from datasets.synthetic_generator import generate_synthetic_patient_data


def test_tflite_and_onnx_export(tmp_path):
    # Dummy fitted model
    X = np.random.normal(70, 10, size=(100, 3))
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(X)

    tflite_path = str(tmp_path / "model.tflite")
    onnx_path = str(tmp_path / "model.onnx")

    tf_res = TFLiteExporter.export_model(model, tflite_path, input_shape=(1, 3))
    assert os.path.exists(tflite_path)
    assert tf_res["is_within_budget"] is True
    assert os.path.exists(str(tmp_path / "metadata.json"))

    onnx_res = ONNXExporter.export_model(model, onnx_path, n_features=3)
    assert os.path.exists(onnx_path)
    assert onnx_res["is_converted"] is True


def test_quantizer_compression(tmp_path):
    X = np.random.normal(0, 1, size=(200, 5))
    model = IsolationForest(n_estimators=50, random_state=42)
    model.fit(X)

    orig_path = str(tmp_path / "orig.pkl")
    comp_path = str(tmp_path / "comp.pkl")

    import joblib
    joblib.dump(model, orig_path)

    res = ModelQuantizer.compress_model(orig_path, comp_path, compress_level=5)
    assert os.path.exists(comp_path)
    assert res["compressed_size_kb"] > 0


def test_drift_detector():
    ref_df = generate_synthetic_patient_data(n_patients=10, records_per_patient=50, seed=42)
    # Drifted distribution (elevated HR by 40 bpm)
    cur_df = generate_synthetic_patient_data(n_patients=10, records_per_patient=50, seed=99)
    cur_df["hr_bpm"] += 45.0

    detector = DriftDetector()
    results = detector.check_drift(ref_df, cur_df, features=["hr_bpm", "spo2_percent"])
    assert "hr_bpm" in results["drifted_features"]
    assert results["feature_metrics"]["hr_bpm"]["drift_detected"] is True
