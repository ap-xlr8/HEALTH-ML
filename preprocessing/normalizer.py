"""Feature Normalization and Scaling Module for Health OS ML."""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class PhysiologicalNormalizer:
    """Manages scaling and normalization of multi-source biometric metrics."""

    def __init__(self, method: str = "robust"):
        self.method = method
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> PhysiologicalNormalizer:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            self.scaler.fit(X.values)
        else:
            self.scaler.fit(X)
            if feature_names:
                self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted yet.")
        vals = X.values if isinstance(X, pd.DataFrame) else X
        return self.scaler.transform(vals)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> PhysiologicalNormalizer:
        return joblib.load(filepath)
