"""Model Registry Manager for Health OS ML."""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


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
        return {"registry_version": "1.0.0", "updated_at": datetime.utcnow().isoformat() + "Z", "models": {}}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.data["updated_at"] = datetime.utcnow().isoformat() + "Z"
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
    ) -> Dict[str, Any]:
        """Register or update a model entry in the registry."""
        entry = {
            "modelId": model_id,
            "version": version,
            "status": "production",
            "type": model_type,
            "target": target,
            "deployedTo": deployed_to,
            "training": {
                "date": datetime.utcnow().isoformat() + "Z",
                "datasetHash": dataset_hash,
                "patientCount": patient_count,
                "sampleCount": sample_count,
                "algorithm": algorithm,
                "hyperparameters": hyperparameters,
            },
            "evaluation": metrics,
            "artifacts": artifacts,
            "deployment": {
                "requiredFeatureSchema": feature_schema_version,
                "changelog": changelog,
            },
        }

        if model_id not in self.data["models"]:
            self.data["models"][model_id] = {}

        self.data["models"][model_id][version] = entry
        self.save()
        return entry

    def get_latest_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        if model_id not in self.data["models"] or not self.data["models"][model_id]:
            return None
        versions = list(self.data["models"][model_id].keys())
        latest_version = sorted(versions)[-1]
        return self.data["models"][model_id][latest_version]
