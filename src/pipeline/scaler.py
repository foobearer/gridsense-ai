# ============================================================
# src/pipeline/scaler.py
# ============================================================
# Feature scaling for time-series data.
#
# Why scale?
# Neural networks like LSTM converge much faster and more
# reliably when input features are on the same numeric scale.
# Without scaling, a feature like "consumption_kwh" (range 0–10)
# would dominate a feature like "is_weekend" (range 0–1).
#
# We use Min-Max scaling (normalise to [0, 1]) rather than
# StandardScaler because energy consumption has natural
# bounds (0 = no usage) and we want to preserve that.
#
# IMPORTANT: The scaler is fit ONLY on the training split.
# Fitting on the full dataset would leak future information
# into the training process (data leakage).
# ============================================================

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesScaler:
    """
    Wraps scikit-learn's MinMaxScaler for 3D LSTM input arrays.

    The LSTM expects input shape (samples, timesteps, features).
    MinMaxScaler only handles 2D arrays, so we reshape before
    fitting/transforming and reshape back after.
    """

    def __init__(self):
        # One scaler for the input features (X)
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        # Separate scaler for the target variable (y)
        # We need to invert-transform predictions back to kWh
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesScaler":
        """
        Fit both scalers on training data.

        X shape: (n_samples, lookback, n_features)
        y shape: (n_samples, horizon)

        We reshape X to 2D (n_samples * lookback, n_features)
        to fit the scaler, then it can transform any 3D array.
        """
        n_samples, lookback, n_features = X.shape

        # Reshape 3D → 2D for the scaler, then fit
        X_2d = X.reshape(-1, n_features)
        self.feature_scaler.fit(X_2d)

        # Reshape y to 2D for the target scaler
        y_2d = y.reshape(-1, 1)
        self.target_scaler.fit(y_2d)

        self._fitted = True
        logger.info("scaler_fitted", n_features=n_features)
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Scale feature array from raw units to [0, 1]."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fit before transforming.")
        n_samples, lookback, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_2d)
        # Reshape back to 3D for the LSTM
        return X_scaled.reshape(n_samples, lookback, n_features)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Scale target array from kWh to [0, 1]."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fit before transforming.")
        return self.target_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert model predictions back from [0, 1] to kWh.
        This is called after inference to get human-readable values.
        """
        original_shape = y_scaled.shape
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).reshape(original_shape)

    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler to disk using pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("scaler_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesScaler":
        """Load a previously saved scaler from disk."""
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("scaler_loaded", path=str(path))
        return scaler
