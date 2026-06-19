# ============================================================
# src/models/anomaly_detector.py
# ============================================================
# Unsupervised anomaly detection for energy consumption data.
#
# Algorithm: Isolation Forest (scikit-learn)
#
# How it works:
#   Isolation Forest isolates observations by randomly selecting
#   a feature and a split value. Anomalies — which are rare and
#   numerically extreme — require fewer splits to isolate than
#   normal points. The anomaly score is inversely proportional
#   to the number of splits needed.
#
# Why not a threshold rule?
#   A fixed threshold (e.g. > 10 kWh = anomaly) breaks when
#   consumption patterns change seasonally. Isolation Forest
#   learns the normal distribution from training data and flags
#   deviations automatically — no manual threshold tuning.
#
# Real-world use cases it catches:
#   - Equipment malfunction (sudden spike)
#   - Meter tampering / energy theft (sudden sustained drop)
#   - Sensor failure (flatline or missing data)
#   - Unexpected industrial load (after-hours spike)
# ============================================================

from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Isolation Forest returns: 1 = normal, -1 = anomaly
_IF_ANOMALY_LABEL = -1


class EnergyAnomalyDetector:
    """
    Wraps scikit-learn's IsolationForest for energy anomaly detection.

    The detector expects a feature matrix that includes both the
    raw consumption value AND engineered context features (hour,
    rolling stats) so it can distinguish a genuine spike from a
    normal peak-hour reading.
    """

    def __init__(self):
        # ── Isolation Forest configuration ────────────────────
        # contamination: the proportion of anomalies in training
        # data. We read this from settings so it's tunable via
        # environment variables without changing code.
        #
        # n_estimators: number of isolation trees in the ensemble.
        # More trees = more stable scores but slower training.
        #
        # random_state: fixed seed for reproducibility
        self.model = IsolationForest(
            n_estimators=200,
            contamination=settings.anomaly_contamination,
            random_state=42,
            n_jobs=-1,  # use all CPU cores
        )

        # StandardScaler normalises features before Isolation Forest.
        # Unlike LSTM, Isolation Forest doesn't converge better with
        # MinMax scaling — StandardScaler (zero mean, unit variance)
        # is the standard choice for tree-based models.
        self.scaler = StandardScaler()
        self._fitted = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract a numeric feature matrix from the DataFrame.

        We use a richer feature set here than just raw consumption
        to help the model understand context:
          - consumption_kwh     → the raw value (main signal)
          - hour                → helps distinguish peak vs off-peak
          - dayofweek           → weekday vs weekend patterns
          - rolling_mean_24h    → recent average (context for current value)
          - rolling_std_24h     → recent volatility
          - lag_1h              → comparison to one hour ago
          - lag_24h             → comparison to same hour yesterday
        """
        feature_cols = [
            "consumption_kwh",
            "hour",
            "dayofweek",
            "rolling_mean_24h",
            "rolling_std_24h",
            "lag_1h",
            "lag_24h",
        ]
        # Use only columns that exist in the DataFrame
        available = [c for c in feature_cols if c in df.columns]
        return df[available].values

    def fit(self, df: pd.DataFrame) -> "EnergyAnomalyDetector":
        """
        Fit the Isolation Forest on historical (normal) data.

        Args:
            df: Feature DataFrame (output of engineer_features)
        """
        features = self._extract_features(df)

        # Fit StandardScaler on training features
        features_scaled = self.scaler.fit_transform(features)

        # Train the Isolation Forest
        # fit() builds n_estimators isolation trees on the data
        self.model.fit(features_scaled)
        self._fitted = True

        logger.info(
            "anomaly_detector_fitted",
            n_samples=len(df),
            contamination=settings.anomaly_contamination,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each row in df and flag anomalies.

        Returns a copy of df with two additional columns:
          - anomaly_score: float, lower = more anomalous
          - is_anomaly: bool, True if flagged as anomalous
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before predicting.")

        features = self._extract_features(df)
        features_scaled = self.scaler.transform(features)

        # decision_function returns raw anomaly scores:
        # negative → anomaly, positive → normal
        scores = self.model.decision_function(features_scaled)

        # predict returns 1 (normal) or -1 (anomaly)
        labels = self.model.predict(features_scaled)

        result_df = df.copy()
        result_df["anomaly_score"] = np.round(scores, 4)
        result_df["is_anomaly"] = labels == _IF_ANOMALY_LABEL

        n_anomalies = int(result_df["is_anomaly"].sum())
        pct = round(n_anomalies / len(result_df) * 100, 2)
        logger.info("anomaly_detection_complete", n_anomalies=n_anomalies, pct=pct)

        return result_df

    def get_anomaly_summary(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return only the anomalous rows as a list of dicts.
        Useful for the API response — clients don't need all rows.
        """
        scored = self.predict(df)
        anomalies = scored[scored["is_anomaly"]].copy()

        # Sort by anomaly score ascending (most extreme first)
        anomalies = anomalies.sort_values("anomaly_score", ascending=True)

        return [
            {
                "timestamp": str(row.Index),
                "consumption_kwh": round(float(row.consumption_kwh), 4),
                "anomaly_score": float(row.anomaly_score),
                "hour": int(row.hour) if hasattr(row, "hour") else None,
                "dayofweek": int(row.dayofweek) if hasattr(row, "dayofweek") else None,
            }
            for row in anomalies.itertuples()
        ]

    def save(self, path: str | Path) -> None:
        """Persist the fitted detector and its scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info("anomaly_detector_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "EnergyAnomalyDetector":
        """Load a previously saved detector from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        detector = cls()
        detector.model = state["model"]
        detector.scaler = state["scaler"]
        detector._fitted = True
        logger.info("anomaly_detector_loaded", path=str(path))
        return detector
