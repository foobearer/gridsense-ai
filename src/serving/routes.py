# ============================================================
# src/serving/routes.py
# ============================================================
# FastAPI route handlers for GridSense AI.
#
# Endpoints:
#   POST /forecast   → 24–168hr energy consumption forecast
#   POST /anomaly    → anomaly detection on a reading window
#
# Each route uses FastAPI's Depends() for dependency injection:
#   - get_forecaster() → returns the singleton LSTM model
#   - get_detector()   → returns the singleton anomaly detector
#   - get_scaler()     → returns the fitted scaler
#
# This pattern means routes never import models directly —
# they declare what they need and FastAPI resolves it.
# ============================================================

from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from src.serving.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    AnomalyRequest, AnomalyResponse, AnomalyPoint,
)
from src.models.registry import get_forecaster, get_detector, get_scaler
from src.models.lstm_forecaster import LSTMForecaster
from src.models.anomaly_detector import EnergyAnomalyDetector
from src.pipeline.scaler import TimeSeriesScaler
from src.pipeline.ingestor import TARGET_COL
from src.monitoring.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ANOMALY_COUNT, track_time
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Router setup ─────────────────────────────────────────────
# prefix="/v1" namespaces all routes under /v1/forecast etc.
# This allows future API versioning without breaking clients.
router = APIRouter(prefix="/v1", tags=["GridSense"])


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(
    body: ForecastRequest,
    # FastAPI resolves these dependencies on each request call
    forecaster: LSTMForecaster = Depends(get_forecaster),
    scaler: TimeSeriesScaler = Depends(get_scaler),
):
    """
    Forecast energy consumption for the next N hours.

    The client provides a list of recent hourly readings.
    We use the last `lookback_window_hours` of them as
    the LSTM's input sequence and return a forecast.

    Returns hourly predicted kWh values with confidence bounds.
    """
    # Increment Prometheus counter for observability
    REQUEST_COUNT.labels(endpoint="forecast", method="POST").inc()

    try:
        with track_time(REQUEST_LATENCY.labels(endpoint="forecast")):
            horizon = body.horizon_hours or settings.forecast_horizon_hours
            lookback = settings.lookback_window_hours

            readings = np.array(body.readings, dtype=np.float32)

            # We need at least `lookback` readings for the LSTM input window
            if len(readings) < lookback:
                # Pad with the series mean if we have fewer readings than lookback
                pad_length = lookback - len(readings)
                pad_value = float(np.mean(readings))
                readings = np.concatenate([
                    np.full(pad_length, pad_value), readings
                ])

            # Take the last `lookback` readings as the input sequence
            input_sequence = readings[-lookback:]

            # ── Build a minimal feature array ─────────────────
            # The LSTM was trained on multi-feature input, so we
            # reconstruct the same feature set from the raw readings.
            # For the API path we use a simplified feature set.
            n_steps = lookback
            hour_arr = np.arange(n_steps) % 24
            dow_arr = (np.arange(n_steps) // 24) % 7
            is_weekend = (dow_arr >= 5).astype(float)
            lag_1 = np.roll(input_sequence, 1); lag_1[0] = input_sequence[0]
            lag_24 = np.roll(input_sequence, 24); lag_24[:24] = input_sequence[:24].mean()
            rolling_mean = pd.Series(input_sequence).rolling(24, min_periods=1).mean().values
            rolling_std = pd.Series(input_sequence).rolling(24, min_periods=1).std().fillna(0).values

            # Stack features: shape (lookback, n_features)
            # The order here must match the training feature order
            features = np.column_stack([
                hour_arr, dow_arr, is_weekend, lag_1, lag_24, rolling_mean, rolling_std
            ]).astype(np.float32)

            # Add batch dimension: (1, lookback, n_features)
            X = features[np.newaxis, :, :]

            # Scale the input using the fitted scaler
            X_scaled = scaler.transform_X(X)

            # Run the LSTM forecast
            predictions_scaled = forecaster.predict(X_scaled)  # shape: (1, horizon)

            # Inverse transform predictions back to kWh
            preds_kwh = scaler.inverse_transform_y(
                predictions_scaled[:, :horizon]
            )[0]  # shape: (horizon,)

            # Clip to non-negative (consumption cannot be negative)
            preds_kwh = np.clip(preds_kwh, 0, None)

            # ── Build confidence bounds ────────────────────────
            # Simple ±1 std dev of the input series as a proxy
            # for prediction uncertainty. In production, replace
            # with proper quantile regression or MC dropout.
            std = float(np.std(input_sequence))
            lower = np.clip(preds_kwh - std, 0, None)
            upper = preds_kwh + std

            # ── Build timestamp list for forecast window ───────
            now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            forecast_points = [
                ForecastPoint(
                    timestamp=(now + timedelta(hours=i + 1)).isoformat(),
                    predicted_kwh=round(float(preds_kwh[i]), 4),
                    lower_bound=round(float(lower[i]), 4),
                    upper_bound=round(float(upper[i]), 4),
                )
                for i in range(len(preds_kwh))
            ]

        return ForecastResponse(
            horizon_hours=horizon,
            forecast=forecast_points,
            summary={
                "mean_kwh": round(float(np.mean(preds_kwh)), 4),
                "peak_kwh": round(float(np.max(preds_kwh)), 4),
                "min_kwh": round(float(np.min(preds_kwh)), 4),
                "total_kwh": round(float(np.sum(preds_kwh)), 4),
            },
            metadata={
                "model": "LSTM",
                "lookback_hours": lookback,
                "input_readings": len(body.readings),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("forecast_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anomaly", response_model=AnomalyResponse)
async def detect_anomalies(
    body: AnomalyRequest,
    detector: EnergyAnomalyDetector = Depends(get_detector),
):
    """
    Detect anomalies in a window of energy consumption readings.

    Returns a list of flagged anomalies with their timestamps,
    consumption values, and Isolation Forest anomaly scores.
    """
    REQUEST_COUNT.labels(endpoint="anomaly", method="POST").inc()

    if len(body.timestamps) != len(body.readings):
        raise HTTPException(
            status_code=422,
            detail="timestamps and readings must have the same length.",
        )

    try:
        with track_time(REQUEST_LATENCY.labels(endpoint="anomaly")):
            # Build a DataFrame that matches the structure the
            # anomaly detector was trained on (same column names)
            df = pd.DataFrame(
                {TARGET_COL: body.readings},
                index=pd.to_datetime(body.timestamps),
            )

            # Add the same engineered features used during training
            df["hour"] = df.index.hour
            df["dayofweek"] = df.index.dayofweek
            df["rolling_mean_24h"] = (
                df[TARGET_COL].rolling(24, min_periods=1).mean()
            )
            df["rolling_std_24h"] = (
                df[TARGET_COL].rolling(24, min_periods=1).std().fillna(0)
            )
            df["lag_1h"] = df[TARGET_COL].shift(1).fillna(method="bfill")
            df["lag_24h"] = df[TARGET_COL].shift(24).fillna(method="bfill")

            # Run the Isolation Forest detector
            anomaly_list = detector.get_anomaly_summary(df)

        # Update Prometheus counter for detected anomalies
        ANOMALY_COUNT.inc(len(anomaly_list))

        return AnomalyResponse(
            total_readings=len(body.readings),
            anomalies_detected=len(anomaly_list),
            anomaly_rate_pct=round(len(anomaly_list) / len(body.readings) * 100, 2),
            anomalies=[
                AnomalyPoint(
                    timestamp=a["timestamp"],
                    consumption_kwh=a["consumption_kwh"],
                    anomaly_score=a["anomaly_score"],
                    hour=a.get("hour"),
                    dayofweek=a.get("dayofweek"),
                )
                for a in anomaly_list
            ],
            metadata={
                "model": "IsolationForest",
                "contamination": settings.anomaly_contamination,
                "input_readings": len(body.readings),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("anomaly_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
