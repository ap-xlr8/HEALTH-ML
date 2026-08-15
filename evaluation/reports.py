"""Evaluation Report Generator for Health OS ML Module."""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any


def save_evaluation_report(
    model_name: str,
    version: str,
    metrics: Dict[str, Any],
    feature_schema: str,
    dataset_hash: str,
    approved: bool,
    reasons: list[str],
    output_dir: str = "evaluation/reports",
) -> str:
    """Persist structured clinical evaluation report to disk."""
    os.makedirs(output_dir, exist_ok=True)
    report_data = {
        "model_name": model_name,
        "version": version,
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "approved_for_export" if approved else "rejected",
        "clinical_approval": approved,
        "rejection_reasons": reasons,
        "feature_schema_version": feature_schema,
        "dataset_hash": dataset_hash,
        "metrics": metrics,
    }

    report_path = os.path.join(output_dir, f"{model_name}_{version}_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_path
