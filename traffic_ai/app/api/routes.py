"""
app/api/routes.py – FastAPI router for the Traffic AI REST API.

Endpoints
---------
GET /stats          → current metrics snapshot (JSON)
GET /vehicles       → last 50 log entries (JSON)
GET /download-csv   → full traffic_logs export (CSV download)
GET /health         → service health (JSON)
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.database import db
from app.services import traffic_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ── /stats ─────────────────────────────────────────────────────────────────────
@router.get("/stats", summary="Current traffic metrics")
async def get_stats() -> Dict[str, Any]:
    """Return total count, density, vehicles/min, and per-type breakdown."""
    return traffic_service.get_state()["metrics"]


# ── /vehicles ─────────────────────────────────────────────────────────────────
@router.get("/vehicles", summary="Recent vehicle log entries")
async def get_vehicles():
    """Return the 50 most-recent crossing events as JSON."""
    return JSONResponse(content=db.fetch_recent(50))


# ── /download-csv ─────────────────────────────────────────────────────────────
@router.get("/download-csv", summary="Export full log as CSV")
async def download_csv():
    """Stream the entire ``traffic_logs`` table as a CSV attachment."""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={"detail": "pandas not installed — run: pip install pandas"},
        )

    rows = db.fetch_all()
    if not rows:
        rows = [{"id": None, "timestamp": None,
                 "vehicle_type": None, "direction": None, "density": None}]

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_logs.csv"},
    )


# ── /health ───────────────────────────────────────────────────────────────────
@router.get("/health", summary="Service health check")
async def health_check() -> Dict[str, Any]:
    """Return basic service health information."""
    state = traffic_service.get_state()
    return {
        "status":           "ok",
        "pipeline_running": state["running"],
        "frames_processed": state["frame_id"],
        "total_counted":    state["metrics"]["total_count"],
    }
