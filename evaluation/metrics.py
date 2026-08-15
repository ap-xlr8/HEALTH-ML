"""Clinical & Operational Evaluation Metrics for Health OS ML Models."""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> Dict[str, float]:
    """Calculate clinically essential classification metrics."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    if y_scores is not None and len(np.unique(y_t)) > 1:
        try:
            auc = float(roc_auc_score(y_t, y_scores))
            metrics["auc_roc"] = round(auc, 4)
        except Exception:
            metrics["auc_roc"] = 0.0

    return metrics


def benchmark_inference_latency(
    model: Any,
    sample_input: np.ndarray,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Measure inference latency per record in milliseconds (mean, p95, p99)."""
    # Warmup
    for _ in range(warmup):
        _ = model.predict(sample_input[:1])

    latencies = []
    single_record = sample_input[:1]

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(single_record)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_arr = np.array(latencies)
    return {
        "latency_mean_ms": round(float(np.mean(latencies_arr)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "latency_min_ms": round(float(np.min(latencies_arr)), 3),
        "latency_max_ms": round(float(np.max(latencies_arr)), 3),
    }


def get_model_size_info(filepath: str) -> Dict[str, float]:
    """Calculate file size in KB and MB."""
    if not os.path.exists(filepath):
        return {"size_kb": 0.0, "size_mb": 0.0}
    size_bytes = os.path.getsize(filepath)
    return {
        "size_kb": round(size_bytes / 1024.0, 2),
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 3),
    }
