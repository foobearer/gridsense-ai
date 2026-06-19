# ============================================================
# src/monitoring/metrics.py
# ============================================================
# Prometheus metrics for GridSense AI observability.
#
# Prometheus works by scraping a /metrics endpoint on a schedule
# (e.g. every 15s). The metrics defined here are updated in real
# time as requests are handled, then read by Prometheus.
# Grafana visualises the Prometheus data as dashboards.
#
# Metric types used here:
#   Counter   — monotonically increasing count (requests, errors)
#   Histogram — distribution of values (latency, payload size)
#   Gauge     — current value that can go up/down (model loaded?)
# ============================================================

from contextlib import contextmanager
import time
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Use a custom registry so tests can instantiate it cleanly
# without conflicting with the default global registry
REGISTRY = CollectorRegistry()

# ── Request counter ───────────────────────────────────────────
# Labels allow us to filter by endpoint and HTTP method in Grafana:
# e.g. "show me POST /forecast requests per second"
REQUEST_COUNT = Counter(
    "gridsense_requests_total",
    "Total number of API requests received",
    ["endpoint", "method"],
    registry=REGISTRY,
)

# ── Request latency histogram ─────────────────────────────────
# Buckets define histogram bin edges in seconds.
# This range covers everything from fast cache hits (50ms)
# to slow first-inference model calls (5s).
REQUEST_LATENCY = Histogram(
    "gridsense_request_latency_seconds",
    "End-to-end API request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ── Model load status gauge ───────────────────────────────────
# Set to 1 when models are loaded, 0 if load failed.
# Grafana alert: notify on-call if this drops to 0.
MODEL_LOAD_STATUS = Gauge(
    "gridsense_models_loaded",
    "Whether all ML models are loaded and ready (1=yes, 0=no)",
    registry=REGISTRY,
)

# ── Anomaly count counter ─────────────────────────────────────
# Tracks cumulative anomalies detected across all API calls.
# A sudden spike indicates a data quality issue or real incident.
ANOMALY_COUNT = Counter(
    "gridsense_anomalies_detected_total",
    "Total number of energy anomalies detected by the API",
    registry=REGISTRY,
)

# ── Forecast request counter ──────────────────────────────────
FORECAST_COUNT = Counter(
    "gridsense_forecasts_total",
    "Total number of forecast requests served",
    registry=REGISTRY,
)


@contextmanager
def track_time(histogram_metric):
    """
    Context manager to measure and record execution time.

    Usage:
        with track_time(REQUEST_LATENCY.labels(endpoint="forecast")):
            result = run_expensive_operation()

    This records the wall-clock time of the with-block into
    the given Histogram metric.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        # observe() records the elapsed time in seconds
        histogram_metric.observe(time.perf_counter() - start)
