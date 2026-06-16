# ============================================================
# main.py
# ============================================================
# GridSense AI — FastAPI application entrypoint.
#
# This file wires together:
#   - Application lifecycle (startup / shutdown)
#   - Middleware (CORS)
#   - System routes (health, metrics)
#   - Business routes (forecast, anomaly detection)
#   - Global error handling
#
# Entry point for both local dev and Docker/Render deployment.
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.utils.config import get_settings
from src.utils.logger import setup_logging, get_logger
from src.models.registry import initialise_models
from src.serving.routes import router as gridsense_router
from src.monitoring.metrics import REGISTRY, MODEL_LOAD_STATUS

# ── Initialise logging before anything else ───────────────────
# Must be called before any logger.info() calls so the
# structlog processors are configured first.
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Application lifespan ──────────────────────────────────────
# FastAPI's lifespan context manager replaces the deprecated
# @app.on_event("startup") pattern. Code before `yield` runs
# at startup; code after runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    logger.info(
        "gridsense_starting",
        version=settings.app_version,
        env=settings.app_env,
    )
    try:
        # Load (or train) all ML models.
        # This blocks until models are ready — intentional,
        # as we don't want to serve requests before models load.
        initialise_models()
        # Signal to Prometheus that models are healthy
        MODEL_LOAD_STATUS.set(1)
        logger.info("gridsense_ready")
    except Exception as e:
        logger.error("model_initialisation_failed", error=str(e))
        # Set to 0 so Grafana can alert on model load failure
        MODEL_LOAD_STATUS.set(0)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("gridsense_shutting_down")


# ── FastAPI application ───────────────────────────────────────
app = FastAPI(
    title="GridSense AI",
    description=(
        "End-to-end energy consumption forecasting and anomaly detection API. "
        "Powered by an LSTM forecaster (PyTorch) and Isolation Forest detector "
        "(scikit-learn) with MLflow experiment tracking and Prometheus monitoring."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",     # Swagger UI at /docs
    redoc_url="/redoc",   # ReDoc at /redoc
)

# ── CORS Middleware ───────────────────────────────────────────
# Allows the API to be called from any frontend origin.
# In production, replace allow_origins=["*"] with a list
# of specific trusted domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System endpoints ──────────────────────────────────────────

@app.get("/health", tags=["system"], response_model=dict)
async def health():
    """
    Liveness probe — used by Render, Docker HEALTHCHECK,
    and load balancers to verify the service is running.
    """
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
        "models_loaded": True,
    }


@app.get("/metrics", tags=["system"], include_in_schema=False)
async def metrics():
    """
    Prometheus scrape endpoint.
    Returns all registered metrics in the Prometheus text format.
    Excluded from OpenAPI docs (include_in_schema=False) since
    it's consumed by Prometheus, not API clients.
    """
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ── Global exception handler ──────────────────────────────────
# Catches any unhandled exceptions that propagate past route handlers.
# Returns a consistent JSON error shape instead of a raw 500.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal error occurred.",
            "detail": str(exc),
        },
    )


# ── Register business routes ──────────────────────────────────
# All GridSense business endpoints are prefixed /v1 (defined in routes.py)
app.include_router(gridsense_router)


# ── Local development entrypoint ──────────────────────────────
# When running `python main.py` directly (not via uvicorn CLI),
# reload=True enables hot-reloading of Python files in dev mode.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
