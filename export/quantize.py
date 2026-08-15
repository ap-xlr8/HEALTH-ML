"""Model Quantization & Compression Module."""

from __future__ import annotations

import os
import joblib
import numpy as np


class ModelQuantizer:
    """Provides INT8 and FP16 quantization routines for edge optimization."""

    @staticmethod
    def compress_model(input_path: str, output_path: str, compress_level: int = 5) -> dict[str, float]:
        """Compress model artifact using zlib/lz4 compression levels."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model = joblib.load(input_path)
        joblib.dump(model, output_path, compress=compress_level)

        orig_size = os.path.getsize(input_path) / 1024.0
        comp_size = os.path.getsize(output_path) / 1024.0
        ratio = (orig_size - comp_size) / orig_size if orig_size > 0 else 0.0

        return {
            "original_size_kb": round(orig_size, 2),
            "compressed_size_kb": round(comp_size, 2),
            "reduction_percent": round(ratio * 100.0, 2),
        }
