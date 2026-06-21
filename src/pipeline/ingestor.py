# ============================================================
# src/pipeline/ingestor.py
# ============================================================
# Data ingestion and feature engineering for energy time-series.
#
# Pipeline flow:
#   raw CSV / DataFrame
#       → validate & clean
#       → resample to hourly
#       → engineer time features
#       → engineer lag features
#       → return feature-rich DataFrame ready for modelling
#
# Dataset used: UCI Household Power Consumption
# (or any CSV with a datetime index and a kW/kWh column)
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


# ── Column name constants ────────────────────────────────────
# Centralising column names avoids magic strings scattered
# across the codebase — change here, reflected everywhere.
TARGET_COL = "consumption_kwh"   # the value we want to predict
DATETIME_COL = "datetime"


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw energy consumption data from a CSV file.

    Expected columns (flexible naming — we normalise below):
      - A datetime column (named 'datetime', 'Date', 'timestamp', etc.)
      - A numeric consumption column (kW, kWh, power, etc.)

    Returns a DataFrame with a DatetimeIndex and TARGET_COL column.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("loading_csv", path=str(path))
    df = pd.read_csv(path)

    # ── Normalise column names to lowercase with underscores ──
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Find and parse the datetime column ────────────────────
    # Try common naming patterns used in public energy datasets
    datetime_candidates = ["datetime", "date", "timestamp", "time", "date_time"]
    dt_col = next((c for c in datetime_candidates if c in df.columns), None)

    if dt_col is None:
        raise ValueError(
            f"No datetime column found. Columns present: {list(df.columns)}"
        )

    # Parse with infer_datetime_format for speed; errors='coerce'
    # turns unparseable rows into NaT so we can drop them cleanly
    df[DATETIME_COL] = pd.to_datetime(df[dt_col], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=[DATETIME_COL])
    df = df.set_index(DATETIME_COL).sort_index()

    # ── Find and normalise the consumption column ─────────────
    consumption_candidates = [
        "consumption_kwh", "global_active_power", "energy_kwh",
        "kwh", "power_kw", "consumption", "load",
    ]
    cons_col = next((c for c in consumption_candidates if c in df.columns), None)

    if cons_col is None:
        raise ValueError(
            f"No consumption column found. Columns present: {list(df.columns)}"
        )

    df[TARGET_COL] = pd.to_numeric(df[cons_col], errors="coerce")
    df = df[[TARGET_COL]].dropna()

    logger.info("csv_loaded", rows=len(df), start=str(df.index.min()), end=str(df.index.max()))
    return df


def generate_synthetic_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic energy consumption data.

    Used when no real dataset is provided — enables the app
    to run a live demo without requiring external data files.

    The synthetic signal combines:
      - Daily seasonality (peak morning + evening demand)
      - Weekly seasonality (lower weekend consumption)
      - Long-term trend (slight growth over the year)
      - Gaussian noise (natural measurement variance)
    """
    logger.info("generating_synthetic_data", n_days=n_days)

    # Hourly timestamps for the requested period
    dates = pd.date_range(start="2023-01-01", periods=n_days * 24, freq="H")

    # ── Daily seasonality ────────────────────────────────────
    # Two peaks: morning commute (8–9am) and evening (6–8pm)
    hour_of_day = dates.hour
    daily_pattern = (
        3.0 * np.exp(-0.5 * ((hour_of_day - 8) / 2) ** 2)   # morning peak
        + 4.0 * np.exp(-0.5 * ((hour_of_day - 19) / 2) ** 2) # evening peak
        + 1.5                                                  # baseline
    )

    # ── Weekly seasonality ───────────────────────────────────
    # Weekends (dayofweek >= 5) consume ~20% less than weekdays
    weekly_factor = np.where(dates.dayofweek >= 5, 0.8, 1.0)

    # ── Long-term upward trend ───────────────────────────────
    # Represents gradual increase in connected devices / EVs
    trend = np.linspace(0, 0.5, len(dates))

    # ── Gaussian noise ───────────────────────────────────────
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    noise = rng.normal(loc=0, scale=0.3, size=len(dates))

    # ── Inject anomalies ─────────────────────────────────────
    # ~2% of readings are anomalous (spikes or drops) to give
    # the Isolation Forest model real anomalies to detect
    consumption = daily_pattern * weekly_factor + trend + noise
    anomaly_indices = rng.choice(len(dates), size=int(0.02 * len(dates)), replace=False)
    consumption[anomaly_indices] += rng.choice([-3.0, 5.0], size=len(anomaly_indices))
    consumption = np.clip(consumption, 0, None)  # consumption can't be negative

    df = pd.DataFrame({TARGET_COL: consumption}, index=dates)
    logger.info("synthetic_data_generated", rows=len(df))
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the DataFrame to a uniform hourly frequency.

    Energy datasets often have sub-hourly readings (e.g. every
    15 minutes). Resampling to hourly:
      - Reduces noise from very-short-term fluctuations
      - Standardises input shape for the LSTM
      - Forward-fills any gaps of up to 3 consecutive hours
    """
    df_hourly = df[TARGET_COL].resample("H").mean()

    # Forward-fill short gaps (e.g. meter outages under 3 hours)
    # then drop any remaining NaNs (longer outages)
    df_hourly = df_hourly.fillna(method="ffill", limit=3).dropna()

    logger.info("resampled_hourly", rows=len(df_hourly))
    return df_hourly.to_frame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features to the consumption DataFrame.

    Features added:
      Time features  — hour, dayofweek, month, is_weekend
                       These capture recurring human behaviour
                       patterns (daily / weekly seasonality).

      Lag features   — consumption 1h, 24h, and 168h (1 week) ago
                       Lag features give the model access to recent
                       history in a form it can learn from directly.

      Rolling stats  — 24h rolling mean and standard deviation
                       The rolling mean captures trend; std captures
                       how volatile consumption has been recently.
    """
    df = df.copy()

    # ── Time features ────────────────────────────────────────
    # The model can't infer time from the index alone —
    # we must provide it as explicit numeric columns.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek   # 0 = Monday, 6 = Sunday
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # ── Lag features ─────────────────────────────────────────
    # lag_1h  → "what was consumption one hour ago?"
    # lag_24h → "what was consumption at this same hour yesterday?"
    # lag_168h→ "what was consumption at this hour last week?"
    df["lag_1h"] = df[TARGET_COL].shift(1)
    df["lag_24h"] = df[TARGET_COL].shift(24)
    df["lag_168h"] = df[TARGET_COL].shift(168)

    # ── Rolling statistics ────────────────────────────────────
    # min_periods=1 avoids NaN for the first rolling window
    df["rolling_mean_24h"] = (
        df[TARGET_COL].rolling(window=24, min_periods=1).mean()
    )
    df["rolling_std_24h"] = (
        df[TARGET_COL].rolling(window=24, min_periods=1).std().fillna(0)
    )

    # Drop rows where lag features are NaN (first 168 rows)
    df = df.dropna()

    logger.info("features_engineered", columns=list(df.columns), rows=len(df))
    return df


def build_sequences(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature DataFrame into (X, y) sequences for LSTM training.

    The LSTM expects 3D input: (samples, timesteps, features).
    This function slides a window of length `lookback` across the
    data, creating one sample per step.

    Args:
        df:       Feature DataFrame (output of engineer_features)
        lookback: Number of time steps the model looks back
        horizon:  Number of future steps to predict

    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples, horizon)  — future TARGET_COL values
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    target = df[TARGET_COL].values
    features = df[feature_cols].values

    X, y = [], []

    # Slide a window across the time series:
    # each sample covers [i : i+lookback] as input
    # and [i+lookback : i+lookback+horizon] as target
    for i in range(len(df) - lookback - horizon + 1):
        X.append(features[i : i + lookback])
        y.append(target[i + lookback : i + lookback + horizon])

    logger.info(
        "sequences_built",
        n_samples=len(X),
        lookback=lookback,
        horizon=horizon,
    )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_ingestion_pipeline(
    filepath: Optional[str] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
) -> dict:
    """
    Orchestrate the full ingestion pipeline end-to-end.

    Steps:
      1. Load CSV (or generate synthetic data)
      2. Resample to hourly
      3. Engineer time + lag + rolling features
      4. Build LSTM sequences

    Returns a dict with the processed DataFrame and model-ready arrays.
    """
    lookback = lookback or settings.lookback_window_hours
    horizon = horizon or settings.forecast_horizon_hours

    # Step 1: Load or generate data
    if filepath:
        raw_df = load_csv(filepath)
    else:
        logger.info("no_filepath_provided_using_synthetic_data")
        raw_df = generate_synthetic_data(n_days=730)  # 2 years of hourly data

    # Step 2: Resample to uniform hourly grid
    hourly_df = resample_hourly(raw_df)

    # Step 3: Add all model features
    featured_df = engineer_features(hourly_df)

    # Step 4: Build supervised learning sequences for the LSTM
    X, y = build_sequences(featured_df, lookback=lookback, horizon=horizon)

    return {
        "dataframe": featured_df,
        "X": X,
        "y": y,
        "feature_columns": [c for c in featured_df.columns if c != TARGET_COL],
        "n_features": X.shape[2] if len(X.shape) == 3 else 0,
        "n_samples": len(X),
    }
