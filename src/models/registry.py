# ============================================================
# src/models/registry.py
# ============================================================
# Model registry — single source of truth for loaded models.
#
# This module manages the lifecycle of all ML models in the app:
#   - Loads (or trains from scratch) the LSTM forecaster
#   - Loads (or trains from scratch) the anomaly detector
#   - Caches both as singletons so they load once at startup
#
# Why a registry pattern?
#   FastAPI's dependency injection (Depends) calls get_* functions
#   on every request. Without caching, we'd reload a 100MB model
#   on each API call — catastrophically slow.
#   The registry ensures models are initialised once and reused.
# ============================================================

from __future__ import annotations
from pathlib import Path
import numpy as np
from src.models.lstm_forecaster import LSTMForecaster
from src.models.anomaly_detector import EnergyAnomalyDetector
from src.pipeline.ingestor import run_ingestion_pipeline
from src.pipeline.scaler import TimeSeriesScaler
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── File paths for persisted model artefacts ─────────────────
MODEL_DIR = Path(settings.model_save_dir)
LSTM_PATH = MODEL_DIR / "lstm_forecaster.pt"
DETECTOR_PATH = MODEL_DIR / "anomaly_detector.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ── Singleton instances (None until first load) ───────────────
_forecaster: LSTMForecaster | None = None
_detector: EnergyAnomalyDetector | None = None
_scaler: TimeSeriesScaler | None = None


def _train_and_save_all() -> tuple[LSTMForecaster, EnergyAnomalyDetector, TimeSeriesScaler]:
    """
    Full training pipeline — called when no saved models exist.

    Steps:
      1. Run data ingestion (synthetic data in demo mode)
      2. Train-test split (80/20)
      3. Fit and save the TimeSeriesScaler
      4. Train and save the LSTM forecaster
      5. Fit and save the anomaly detector
    """
    logger.info("no_saved_models_found_training_from_scratch")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and preprocess data
    pipeline_output = run_ingestion_pipeline()
    X = pipeline_output["X"]
    y = pipeline_output["y"]
    df = pipeline_output["dataframe"]
    n_features = pipeline_output["n_features"]

    # Step 2: Chronological train/val split (no shuffling —
    # shuffling would leak future data into the training set)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Step 3: Fit scaler on training data only, then transform both sets
    scaler = TimeSeriesScaler()
    scaler.fit(X_train, y_train)
    X_train_s = scaler.transform_X(X_train)
    X_val_s = scaler.transform_X(X_val)
    y_train_s = scaler.transform_y(y_train)
    y_val_s = scaler.transform_y(y_val)
    scaler.save(SCALER_PATH)

    # Step 4: Train the LSTM forecaster
    # Use fewer epochs in demo mode to keep startup time reasonable
    forecaster = LSTMForecaster(n_features=n_features)
    forecaster.train(
        X_train_s, y_train_s,
        X_val_s, y_val_s,
        epochs=10,       # increase to 50+ for production quality
        batch_size=64,
    )
    forecaster.save(LSTM_PATH)

    # Step 5: Fit anomaly detector on the full featured DataFrame
    detector = EnergyAnomalyDetector()
    detector.fit(df)
    detector.save(DETECTOR_PATH)

    return forecaster, detector, scaler


def initialise_models() -> None:
    """
    Load all models at application startup.
    Called from main.py's lifespan context manager.

    If saved model files exist → load them (fast, <1s).
    If not → train from scratch (slower, ~30s on CPU with demo data).
    """
    global _forecaster, _detector, _scaler

    if LSTM_PATH.exists() and DETECTOR_PATH.exists() and SCALER_PATH.exists():
        logger.info("loading_saved_models")
        _forecaster = LSTMForecaster.load(LSTM_PATH)
        _detector = EnergyAnomalyDetector.load(DETECTOR_PATH)
        _scaler = TimeSeriesScaler.load(SCALER_PATH)
    else:
        _forecaster, _detector, _scaler = _train_and_save_all()

    logger.info("all_models_ready")


def get_forecaster() -> LSTMForecaster:
    """Dependency injection target for FastAPI routes."""
    if _forecaster is None:
        raise RuntimeError("Models not initialised. Call initialise_models() at startup.")
    return _forecaster


def get_detector() -> EnergyAnomalyDetector:
    """Dependency injection target for FastAPI routes."""
    if _detector is None:
        raise RuntimeError("Models not initialised. Call initialise_models() at startup.")
    return _detector


def get_scaler() -> TimeSeriesScaler:
    """Dependency injection target for FastAPI routes."""
    if _scaler is None:
        raise RuntimeError("Models not initialised. Call initialise_models() at startup.")
    return _scaler
