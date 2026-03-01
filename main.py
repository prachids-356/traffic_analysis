"""
main.py – Application entry point.

1. Start the background detection pipeline (daemon thread).
2. Mount static files and the dashboard HTML.
3. Register API routes.
4. Launch uvicorn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import traffic_service
from routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Analysis API",
    description="Real-time vehicle detection & counting dashboard",
    version="1.0.0",
)

# Register API routes
app.include_router(router)

@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Unhandle exception during request: %s", e)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ── Static / dashboard ────────────────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    index = TEMPLATES_DIR / "dashboard.html"
    return FileResponse(str(index), media_type="text/html")


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Starting background detection pipeline …")
    traffic_service.start_pipeline()
    logger.info("FastAPI application ready.")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
