"""Integration Tests for All Model Training Pipelines."""

import pytest
import tempfile
from datasets.synthetic_generator import generate_synthetic_patient_data
from training.anomaly_detection.heart_rate.train import train_heart_rate_pipeline
from training.anomaly_detection.spo2.train import train_spo2_pipeline
from training.anomaly_detection.combined.train import train_combined_vitals_pipeline
from training.sleep_quality.train import train_sleep_quality_pipeline
from training.activity_recognition.train import train_activity_recognition_pipeline
from training.glucose_patterns.train import train_glucose_patterns_pipeline
from training.risk_scoring.train import train_risk_scoring_pipeline


@pytest.fixture(scope="module")
def shared_synthetic_data():
    return generate_synthetic_patient_data(n_patients=30, records_per_patient=120, seed=42)


def test_heart_rate_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "hr_model")
    result = train_heart_rate_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert result["metrics"]["recall"] > 0.0
    assert result["metrics"]["latency_mean_ms"] < 25.0


def test_spo2_critical_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "spo2_model")
    result = train_spo2_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert "precision" in result["metrics"]


def test_combined_vitals_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "vitals_model")
    result = train_combined_vitals_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert "recall" in result["metrics"]


def test_sleep_quality_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "sleep_model")
    result = train_sleep_quality_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert result["metrics"]["accuracy"] >= 0.40


def test_activity_recognition_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "activity_model")
    result = train_activity_recognition_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert result["metrics"]["accuracy"] > 0.60


def test_glucose_patterns_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "glucose_model")
    result = train_glucose_patterns_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert "recall" in result["metrics"]


def test_risk_scoring_training_pipeline(shared_synthetic_data, tmp_path):
    out_dir = str(tmp_path / "risk_model")
    result = train_risk_scoring_pipeline(data_source=shared_synthetic_data, output_dir=out_dir)
    assert result["model"] is not None
    assert "r2_score" in result["metrics"]
