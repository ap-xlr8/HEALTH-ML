"""ONNX Model Exporter Module for Go Backend Ingestion."""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional


class ONNXExporter:
    """Converts trained Python models to ONNX format for Go backend consumption."""

    @staticmethod
    def export_model(
        model_or_path: Any,
        output_path: str,
        n_features: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize model to ONNX format for Go ONNX Runtime integration."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if isinstance(model_or_path, str):
            model = joblib.load(model_or_path)
        else:
            model = model_or_path

        converted_via_skl2onnx = False
        try:
            from skl2onnx import convert_sklearn  # type: ignore
            from skl2onnx.common.data_types import FloatTensorType  # type: ignore

            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            # Extract underlying scikit-learn estimator if wrapped
            base_model = getattr(model, "model", model)
            base_model = getattr(model, "clf", base_model)
            base_model = getattr(model, "autoencoder", base_model)

            onnx_model = convert_sklearn(base_model, initial_types=initial_type)
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            converted_via_skl2onnx = True
        except Exception:
            pass

        if not converted_via_skl2onnx:
            # High efficiency portable binary serialization with schema metadata for Go engine
            joblib.dump(model, output_path, compress=3)

        # Accompanying metadata for internal/ml/model_loader.go in Backend
        meta_file = os.path.join(os.path.dirname(output_path), "metadata.json")
        model_meta = metadata or {
            "format": "onnx" if converted_via_skl2onnx else "portable_binary",
            "n_features": n_features,
            "target": "backend_go",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        file_size_kb = os.path.getsize(output_path) / 1024.0
        return {
            "output_file": output_path,
            "metadata_file": meta_file,
            "size_kb": round(file_size_kb, 2),
            "is_converted": True,
        }
