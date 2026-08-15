"""Synthetic Dataset Generator for Health OS ML Module.

Generates realistic physiological datasets conforming to Health OS schemas with
controlled anomalies and ground truth annotations for training & validation.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def generate_synthetic_patient_data(
    n_patients: int = 40,
    records_per_patient: int = 250,
    anomaly_rate: float = 0.05,
    seed: int = 42,
    include_pii: bool = True,
) -> pd.DataFrame:
    """Generate comprehensive biometric time-series data for multiple patients."""
    np.random.seed(seed)
    records = []
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for p_idx in range(n_patients):
        patient_id = f"PAT-{p_idx + 1000:04d}"
        p_name = f"Patient {p_idx + 1}"
        p_email = f"patient_{p_idx + 1}@healthos-demo.internal"
        p_ip = f"192.168.1.{10 + p_idx % 200}"
        p_mac = f"00:1A:2B:3C:{p_idx % 99:02X}:{p_idx % 255:02X}"
        consent = True

        # Patient baseline characteristics
        base_hr = np.random.normal(72, 5)
        base_hrv = np.random.normal(45, 6)
        base_spo2 = np.random.normal(98, 0.5)
        base_temp = np.random.normal(36.7, 0.15)
        base_sys = np.random.normal(120, 6)
        base_dia = np.random.normal(80, 4)
        base_resp = np.random.normal(15, 1.5)
        elevation = np.random.uniform(0, 1200)

        for r_idx in range(records_per_patient):
            timestamp = base_time + timedelta(minutes=r_idx * 15 + p_idx * 3)
            hour = timestamp.hour
            is_sleeping = int(23 <= hour or hour <= 6)
            
            if is_sleeping:
                activity_intensity = 0
                activity_class = 0  # Rest / Sleep
            else:
                activity_intensity = int(np.random.choice([0, 1, 2, 3], p=[0.50, 0.30, 0.15, 0.05]))
                activity_class = activity_intensity

            is_anomaly = np.random.random() < anomaly_rate

            if is_anomaly:
                anomaly_type = np.random.choice(["tachycardia", "bradycardia", "hypoxemia", "hypertensive_crisis"])
                if anomaly_type == "tachycardia":
                    hr = base_hr + np.random.uniform(55, 85)
                    hrv = max(5.0, base_hrv - np.random.uniform(25, 35))
                    spo2 = base_spo2 - np.random.uniform(1.0, 3.0)
                    sys_bp = base_sys + np.random.uniform(20, 35)
                    dia_bp = base_dia + np.random.uniform(12, 22)
                    resp = base_resp + np.random.uniform(8, 14)
                    temp = base_temp + np.random.uniform(0.5, 1.8)
                    crit_spo2 = False
                elif anomaly_type == "bradycardia":
                    hr = max(34.0, base_hr - np.random.uniform(25, 35))
                    hrv = base_hrv + np.random.uniform(15, 35)
                    spo2 = base_spo2 - np.random.uniform(0.5, 2.0)
                    sys_bp = max(75.0, base_sys - np.random.uniform(25, 40))
                    dia_bp = max(45.0, base_dia - np.random.uniform(15, 25))
                    resp = max(8.0, base_resp - np.random.uniform(4, 7))
                    temp = base_temp
                    crit_spo2 = False
                elif anomaly_type == "hypoxemia":
                    hr = base_hr + np.random.uniform(25, 45)
                    hrv = max(8.0, base_hrv - 18)
                    spo2 = np.random.uniform(75.0, 88.0)
                    sys_bp = base_sys
                    dia_bp = base_dia
                    resp = base_resp + np.random.uniform(9, 16)
                    temp = base_temp
                    crit_spo2 = True
                else:  # Hypertensive crisis
                    hr = base_hr + np.random.uniform(20, 35)
                    hrv = max(10.0, base_hrv - 20)
                    spo2 = base_spo2
                    sys_bp = np.random.uniform(185.0, 225.0)
                    dia_bp = np.random.uniform(120.0, 145.0)
                    resp = base_resp + 5
                    temp = base_temp
                    crit_spo2 = False
            else:
                activity_hr_boost = activity_intensity * 20.0
                hr = max(55.0, min(175.0, base_hr + activity_hr_boost + np.random.normal(0, 2.5)))
                hrv = max(25.0, min(110.0, base_hrv - (activity_intensity * 6.0) + np.random.normal(0, 3)))
                spo2 = max(94.0, min(100.0, base_spo2 - (elevation / 5000.0) + np.random.normal(0, 0.3)))
                temp = max(36.2, min(37.4, base_temp + np.random.normal(0, 0.1)))
                sys_bp = max(95.0, min(145.0, base_sys + (activity_intensity * 10.0) + np.random.normal(0, 3)))
                dia_bp = max(65.0, min(90.0, base_dia + (activity_intensity * 5.0) + np.random.normal(0, 2)))
                resp = max(12.0, min(22.0, base_resp + (activity_intensity * 3.0) + np.random.normal(0, 1.0)))
                crit_spo2 = spo2 < 90.0

            if is_sleeping:
                stage_rnd = np.random.random()
                if stage_rnd < 0.15:
                    sleep_stage = 1
                    movement_variance = max(0.02, min(0.08, np.random.normal(0.04, 0.01)))
                elif stage_rnd < 0.60:
                    sleep_stage = 2
                    movement_variance = max(0.005, min(0.02, np.random.normal(0.01, 0.003)))
                elif stage_rnd < 0.80:
                    sleep_stage = 3
                    movement_variance = max(0.0005, min(0.005, np.random.normal(0.002, 0.001)))
                    hr = max(34.0, hr * 0.92)
                else:
                    sleep_stage = 4
                    movement_variance = max(0.002, min(0.015, np.random.normal(0.008, 0.002)))
                    hrv = hrv * 1.20
            else:
                movement_variance = max(0.1, np.random.exponential(0.5 * (activity_intensity + 1)))
                sleep_stage = 0

            # Enforce hard physiological bounds
            hr = float(np.clip(hr, 32.0, 220.0))
            hrv = float(np.clip(hrv, 2.0, 250.0))
            spo2 = float(np.clip(spo2, 65.0, 100.0))
            resp = float(np.clip(resp, 6.0, 50.0))
            temp = float(np.clip(temp, 34.0, 42.0))
            sys_bp = float(np.clip(sys_bp, 70.0, 240.0))
            dia_bp = float(np.clip(dia_bp, 40.0, 150.0))

            accel_x = np.random.normal(0.0, 0.05) + (0.4 * activity_intensity)
            accel_y = np.random.normal(0.98, 0.05)
            accel_z = np.random.normal(0.0, 0.05)
            gyro_x = np.random.normal(0.0, 0.15 * (activity_intensity + 0.1))
            gyro_y = np.random.normal(0.0, 0.15 * (activity_intensity + 0.1))
            gyro_z = np.random.normal(0.0, 0.15 * (activity_intensity + 0.1))

            risk_base = (
                (max(0, hr - 100) * 0.45) +
                (max(0, 95 - spo2) * 2.8) +
                (max(0, sys_bp - 140) * 0.35) +
                (max(0, dia_bp - 90) * 0.45) +
                (max(0, temp - 37.5) * 8.0)
            )
            risk_score = float(np.clip(risk_base + np.random.normal(5, 1.5), 0.0, 100.0))
            glucose_pattern_flag = int((hrv < 22.0 and hr > 88 and activity_intensity == 0) or is_anomaly)

            row = {
                "patient_id": patient_id,
                "timestamp": timestamp.isoformat(),
                "hr_bpm": round(float(hr), 2),
                "hrv_ms": round(float(hrv), 2),
                "spo2_percent": round(float(spo2), 2),
                "elevation_meters": round(float(elevation), 1),
                "is_sleeping": is_sleeping,
                "activity_intensity": activity_intensity,
                "resp_rate": round(float(resp), 2),
                "temp_c": round(float(temp), 2),
                "bp_sys": round(float(sys_bp), 2),
                "bp_dia": round(float(dia_bp), 2),
                "movement_variance": round(float(movement_variance), 5),
                "accel_x": round(float(accel_x), 3),
                "accel_y": round(float(accel_y), 3),
                "accel_z": round(float(accel_z), 3),
                "gyro_x": round(float(gyro_x), 3),
                "gyro_y": round(float(gyro_y), 3),
                "gyro_z": round(float(gyro_z), 3),
                "sleep_stage": sleep_stage,
                "activity_class": activity_class,
                "is_hr_anomaly": int(is_anomaly),
                "critical_spo2_drop": int(crit_spo2),
                "glucose_pattern_flag": glucose_pattern_flag,
                "risk_score_30d": round(risk_score, 2),
                "consent_for_ml_training": consent,
            }

            if include_pii:
                row["name"] = p_name
                row["email"] = p_email
                row["ip_address"] = p_ip
                row["device_mac"] = p_mac

            records.append(row)

    df = pd.DataFrame(records)
    return df


def generate_and_save_datasets(output_dir: str = "datasets/raw") -> str:
    """Generate synthetic dataset and save to parquet."""
    os.makedirs(output_dir, exist_ok=True)
    df = generate_synthetic_patient_data()
    file_path = os.path.join(output_dir, "synthetic_health_measurements.parquet")
    df.to_parquet(file_path, index=False)
    print(f"Generated {len(df)} synthetic records across {df['patient_id'].nunique()} patients in {file_path}")
    return file_path


if __name__ == "__main__":
    generate_and_save_datasets()
