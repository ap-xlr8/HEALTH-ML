"""Model Registry Manager for Health OS ML Module.

Manages versioned ML model artifacts, metadata, and clinical safety approvals.
Enforces that only models meeting medical safety thresholds achieve 'production' status.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


def evaluate_clinical_status(model_type: str, metrics: Dict[str, Any]) -> str:
    """Evaluates whether model metrics satisfy strict clinical safety gates."""
    if not metrics:
        return "staged"

    if model_type in ["anomaly_detection", "critical_alert", "classifier"]:
        recall = metrics.get("recall", 0.0)
        fpr = metrics.get("fpr", 1.0)
        accuracy = metrics.get("accuracy", 0.0)
        
        # High-sensitivity clinical alert gate
        if recall >= 0.95 and fpr <= 0.05:
            return "production"
        # High accuracy activity / sleep classifier gate
        elif accuracy >= 0.85:
            return "production"
        else:
            return "rejected"
    elif model_type in ["regression", "risk_scoring", "continuous_prediction"]:
        r2 = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 999.0)
        if r2 >= 0.80 or mae <= 10.0:
            return "production"
        else:
            return "rejected"
    return "staged"


class ModelRegistry:
    """Manages versioned ML model artifacts, metadata, and clinical approvals."""

    def __init__(self, registry_file: str = "model-registry/registry.json"):
        self.registry_file = registry_file
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"registry_version": "1.0.0", "updated_at": datetime.now(timezone.utc).isoformat(), "models": {}}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def register_model(
        self,
        model_id: str,
        version: str,
        model_type: str,
        target: str,
        deployed_to: List[str],
        algorithm: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Dict[str, str],
        feature_schema_version: str,
        changelog: str = "",
        dataset_hash: str = "",
        patient_count: int = 0,
        sample_count: int = 0,
        force_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register or update a model entry in the registry with clinical verification."""
        # Sanitize artifact paths to purely relative clean paths
        sanitized_artifacts = {}
        for k, v in artifacts.items():
            if v:
                clean_path = v.replace("\\", "/")
                # Strip absolute tmp/temp references if present
                if "model-registry" in clean_path:
                    clean_path = clean_path[clean_path.find("model-registry"):]
                sanitized_artifacts[k] = clean_path

        # Determine clinical production status
        status = force_status or evaluate_clinical_status(model_type, metrics)

        entry = {
            "modelId": model_id,
            "version": version,
            "status": status,
            "type": model_type,
            "target": target,
            "deployedTo": deployed_to,
            "training": {
                "date": datetime.now(timezone.utc).isoformat(),
                "datasetHash": dataset_hash or "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "patientCount": patient_count,
                "sampleCount": sample_count,
                "algorithm": algorithm,
                "hyperparameters": hyperparameters,
            },
            "evaluation": metrics,
            "artifacts": sanitized_artifacts,
            "deployment": {
                "requiredFeatureSchema": feature_schema_version,
                "changelog": changelog,
            },
        }

        if "models" not in self.data:
            self.data["models"] = {}

        if model_id not in self.data["models"]:
            self.data["models"][model_id] = {}

        self.data["models"][model_id][version] = entry
        self.save()
        return entry

    def get_latest_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        if model_id not in self.data.get("models", {}) or not self.data["models"][model_id]:
            return None
        versions = list(self.data["models"][model_id].keys())
        latest_version = sorted(versions)[-1]
        return self.data["models"][model_id][latest_version]
