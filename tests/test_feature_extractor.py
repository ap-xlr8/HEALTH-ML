"""Tests for Feature Extractor, Normalizer, and Time Sync Modules."""

import pytest
import numpy as np
import pandas as pd
from datasets.synthetic_generator import generate_synthetic_patient_data
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.normalizer import PhysiologicalNormalizer
from preprocessing.time_sync import TimeSyncProcessor


def test_feature_extractor_all_schemas():
    df = generate_synthetic_patient_data(n_patients=2, records_per_patient=10)

    hr_feats = FeatureExtractor.extract_heart_rate_features(df)
    assert list(hr_feats.columns) == ["hr_bpm", "hrv_ms", "activity_intensity"]

    spo2_feats = FeatureExtractor.extract_spo2_features(df)
    assert list(spo2_feats.columns) == ["spo2_percent", "elevation_meters", "is_sleeping"]

    combined_feats = FeatureExtractor.extract_combined_vitals_features(df)
    assert len(combined_feats.columns) == 6

    sleep_feats = FeatureExtractor.extract_sleep_features(df)
    assert "movement_variance" in sleep_feats.columns

    act_feats = FeatureExtractor.extract_activity_features(df)
    assert "accel_magnitude" in act_feats.columns


def test_physiological_normalizer_methods(tmp_path):
    X = np.array([[10.0, 20.0], [15.0, 25.0], [100.0, 200.0]])
    norm = PhysiologicalNormalizer(method="robust")
    X_trans = norm.fit_transform(X)
    assert X_trans.shape == X.shape

    # Test serialization
    save_path = str(tmp_path / "norm.pkl")
    norm.save(save_path)
    loaded = PhysiologicalNormalizer.load(save_path)
    assert np.allclose(loaded.transform(X), X_trans)


def test_time_sync_processor():
    timestamps = pd.date_range("2026-08-14 10:00", periods=5, freq="73s")
    df = pd.DataFrame({"timestamp": timestamps, "hr_bpm": [70.0, 72.0, 75.0, 73.0, 71.0]})

    resampled = TimeSyncProcessor.align_and_resample(df, freq="1min")
    assert len(resampled) > 0

    drift_res = TimeSyncProcessor.detect_clock_drift(
        device_timestamps=[100.0, 101.0, 102.0],
        server_timestamps=[100.1, 101.2, 102.1],
    )
    assert drift_res["is_drift_critical"] is False
