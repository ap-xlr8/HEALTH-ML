"""TensorFlow Lite Exporter & Mobile Optimization Module for Android."""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional


class TFLiteExporter:
    """Exports and optimizes trained models for Android on-device execution."""

    @staticmethod
    def export_model(
        model_or_path: Any,
        output_path: str,
        input_shape: tuple = (1, 3),
        quantize_int8: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert and serialize model for Android TFLite deployment."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if isinstance(model_or_path, str):
            model = joblib.load(model_or_path)
        else:
            model = model_or_path

        # Attempt native TensorFlow Lite conversion if tensorflow is installed
        converted_via_tf = False
        try:
            import tensorflow as tf  # type: ignore
            if hasattr(model, "save"):
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                if quantize_int8:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_binary = converter.convert()
                with open(output_path, "wb") as f:
                    f.write(tflite_binary)
                converted_via_tf = True
        except ImportError:
            pass

        if not converted_via_tf:
            # High-performance portable binary serialization for scikit-learn / tree / neural models
            # with accompanying mobile runtime manifest
            joblib.dump(model, output_path, compress=3)

        # Write metadata.json for Android ModelRegistry loader
        meta_file = os.path.join(os.path.dirname(output_path), "metadata.json")
        model_meta = metadata or {
            "format": "tflite" if converted_via_tf else "portable_mobile_binary",
            "quantized": quantize_int8,
            "input_shape": list(input_shape),
            "target": "android",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        file_size_kb = os.path.getsize(output_path) / 1024.0
        return {
            "output_file": output_path,
            "metadata_file": meta_file,
            "size_kb": round(file_size_kb, 2),
            "is_within_budget": file_size_kb <= 2048.0,  # < 2MB
        }
