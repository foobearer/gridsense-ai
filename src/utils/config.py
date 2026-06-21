# ============================================================
# src/utils/config.py
# ============================================================
# Centralised application configuration.
#
# Pydantic-Settings reads values from environment variables
# (or a .env file) and validates their types automatically.
# The @lru_cache decorator ensures the Settings object is
# created only once — a singleton pattern for config.
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── Application ─────────────────────────────────────
    app_env: str = Field("development", env="APP_ENV")
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    app_version: str = Field("1.0.0", env="APP_VERSION")

    # ── Forecasting ─────────────────────────────────────
    # forecast_horizon_hours: how many future time steps
    # the model will predict (e.g. 24 = next 24 hours)
    forecast_horizon_hours: int = Field(24, env="FORECAST_HORIZON_HOURS")

    # lookback_window_hours: how many historical hours the
    # LSTM uses as its input sequence (its "memory")
    lookback_window_hours: int = Field(168, env="LOOKBACK_WINDOW_HOURS")

    # ── Anomaly Detection ────────────────────────────────
    # contamination: the proportion of the dataset expected
    # to be anomalous; tunes the Isolation Forest threshold
    anomaly_contamination: float = Field(0.05, env="ANOMALY_CONTAMINATION")

    # ── MLflow ───────────────────────────────────────────
    mlflow_tracking_uri: str = Field("./data/mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        "gridsense-forecasting", env="MLFLOW_EXPERIMENT_NAME"
    )

    # ── Paths ────────────────────────────────────────────
    data_dir: str = Field("./data", env="DATA_DIR")
    model_save_dir: str = Field("./data/models", env="MODEL_SAVE_DIR")

    # ── Logging ──────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        # pydantic-settings will load a .env file if present
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    Using lru_cache means this function is only executed once
    per process — every subsequent call returns the same object.
    """
    return Settings()
