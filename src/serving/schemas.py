# ============================================================
# src/serving/schemas.py
# ============================================================
# Pydantic models for API request validation and response
# serialisation.
#
# Why Pydantic?
#   FastAPI uses Pydantic schemas to:
#     1. Automatically validate incoming JSON bodies
#     2. Generate the OpenAPI (Swagger) documentation
#     3. Serialise Python objects to JSON responses
#
# Every API endpoint has a Request schema (what we accept)
# and a Response schema (what we return). This makes the
# API self-documenting and catches bad inputs early.
# ============================================================

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ── Request Schemas ──────────────────────────────────────────

class ForecastRequest(BaseModel):
    """
    Request body for the /forecast endpoint.

    The client sends a list of recent hourly consumption
    readings. We use these as the model's input sequence.
    """
    # Recent hourly consumption readings in kWh
    # min_length=24: we need at least 24 hours of history
    readings: List[float] = Field(
        ...,
        min_length=24,
        max_length=720,   # max 30 days of hourly data
        description="Recent hourly energy consumption readings in kWh",
    )
    # How many hours ahead to forecast (overrides settings default)
    horizon_hours: Optional[int] = Field(
        None,
        ge=1,
        le=168,
        description="Forecast horizon in hours (1–168). Defaults to app setting.",
    )

    @field_validator("readings")
    @classmethod
    def readings_must_be_non_negative(cls, v: List[float]) -> List[float]:
        # Energy consumption cannot be negative
        if any(r < 0 for r in v):
            raise ValueError("All consumption readings must be non-negative.")
        return v


class AnomalyRequest(BaseModel):
    """
    Request body for the /anomaly endpoint.

    Accepts a list of timestamped consumption readings.
    Both fields must have the same length.
    """
    timestamps: List[str] = Field(
        ...,
        min_length=48,
        description="ISO-8601 datetime strings for each reading",
    )
    readings: List[float] = Field(
        ...,
        min_length=48,
        description="Corresponding energy consumption readings in kWh",
    )

    @field_validator("readings")
    @classmethod
    def readings_non_negative(cls, v: List[float]) -> List[float]:
        if any(r < 0 for r in v):
            raise ValueError("Readings must be non-negative.")
        return v


# ── Response Schemas ─────────────────────────────────────────

class ForecastPoint(BaseModel):
    """A single point in the forecast output."""
    # The forecast timestamp (e.g. "2024-03-15T14:00:00")
    timestamp: str
    # Predicted consumption in kWh
    predicted_kwh: float
    # Prediction interval lower and upper bounds (±1 std dev)
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    """Full response from the /forecast endpoint."""
    status: str = "success"
    # How many hours ahead was forecast
    horizon_hours: int
    # The list of hourly forecast points
    forecast: List[ForecastPoint]
    # Aggregate stats for the forecast window
    summary: Dict[str, float]
    # Latency and model metadata
    metadata: Dict[str, Any] = {}


class AnomalyPoint(BaseModel):
    """A single detected anomaly."""
    timestamp: str
    consumption_kwh: float
    # Isolation Forest anomaly score (lower = more anomalous)
    anomaly_score: float
    hour: Optional[int] = None
    dayofweek: Optional[int] = None


class AnomalyResponse(BaseModel):
    """Full response from the /anomaly endpoint."""
    status: str = "success"
    total_readings: int
    anomalies_detected: int
    anomaly_rate_pct: float
    anomalies: List[AnomalyPoint]
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    version: str
    models_loaded: bool
    environment: str


class ErrorResponse(BaseModel):
    """Standard error response shape."""
    status: str = "error"
    message: str
    detail: Optional[str] = None
