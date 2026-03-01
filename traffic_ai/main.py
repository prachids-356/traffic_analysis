"""
main.py – Application entry point for Traffic AI.

Responsibilities
----------------
1. Bootstrap logging.
2. Create the FastAPI application.
3. Register the API router (REST endpoints).
4. Mount the Jinja2 template for the dashboard at ``GET /``.
5. Mount the ``/static`` directory for CSS/JS assets.
6. Start the background detection pipeline on server startup.
7. Launch uvicorn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.services import traffic_service

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
TEMPLATES_DIR  = BASE_DIR / "templates"
STATIC_DIR     = BASE_DIR / "static"

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic AI",
    description="Real-time YOLOv8 vehicle detection & analytics dashboard",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── Templates (Jinja2) ─────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── API routes ─────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Dashboard route ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    """Serve the live traffic analytics dashboard."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request},
    )


# ── Lifecycle ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting background detection pipeline …")
    traffic_service.start_pipeline()
    logger.info("Traffic AI is ready — http://localhost:8000")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    from app.database import db
    db.flush()
    logger.info("Traffic AI shutdown complete.")


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
