"""
app/services/traffic_service.py – Background detection pipeline.

Runs in a daemon thread.  All shared state is protected by a
``threading.Lock``; API routes read it via ``get_state()``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List

from app.config import VIDEO_PATH
from app.ingestion.video_stream import frame_generator
from app.detection.detector import detect
from app.tracking.tracker import CentroidTracker
from app.analytics.metrics import compute_metrics, density_label
from app.database import db

logger = logging.getLogger(__name__)

# ── Shared state ───────────────────────────────────────────────────────────────
_lock = threading.Lock()

_state: Dict[str, Any] = {
    "running":       False,
    "frame_id":      0,
    "metrics": {
        "total_count":      0,
        "density":          "LOW",
        "vehicles_per_min": 0.0,
        "type_breakdown":   {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    },
    "count_history": [],   # list of {timestamp, vehicle_type, direction}
    "recent_logs":   [],   # cache of db.fetch_recent(50)
}


# ── Public accessors (called from routes) ──────────────────────────────────────

def get_state() -> Dict[str, Any]:
    """Return a thread-safe snapshot of the shared state."""
    with _lock:
        return {
            "running":     _state["running"],
            "frame_id":    _state["frame_id"],
            "metrics":     dict(_state["metrics"]),
            "recent_logs": list(_state["recent_logs"]),
        }


def is_running() -> bool:
    with _lock:
        return _state["running"]


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _run_pipeline() -> None:
    logger.info("Detection pipeline starting …")
    tracker = CentroidTracker()

    with _lock:
        _state["running"] = True

    try:
        for frame_id, frame in frame_generator(VIDEO_PATH):
            # 1. Detect vehicles
            detections = detect(frame)

            # 2. Track & get crossing events
            _, new_counts = tracker.update(detections)

            # 3. Persist crossings + update history
            now = datetime.now()
            for event in new_counts:
                with _lock:
                    history: List[Dict[str, Any]] = _state["count_history"]
                    current_density = density_label(len(history))
                    history.append({
                        "timestamp":    now,
                        "vehicle_type": event["vehicle_type"],
                        "direction":    event["direction"],
                    })

                db.log_crossing(
                    timestamp=now,
                    vehicle_type=event["vehicle_type"],
                    direction=event["direction"],
                    density=current_density,
                )

            # 4. Recompute metrics
            with _lock:
                history_snap: List[Dict[str, Any]] = list(_state["count_history"])

            metrics     = compute_metrics(history_snap)
            recent_logs = db.fetch_recent(50)

            # 5. Publish to shared state
            with _lock:
                _state["frame_id"]    = frame_id
                _state["metrics"]     = metrics
                _state["recent_logs"] = recent_logs

    except Exception:
        logger.exception("Pipeline crashed.")
    finally:
        db.flush()
        with _lock:
            _state["running"] = False
        logger.info("Detection pipeline stopped.")


# ── Public start ───────────────────────────────────────────────────────────────

def start_pipeline() -> threading.Thread:
    """Spawn and return the background daemon thread."""
    t = threading.Thread(
        target=_run_pipeline,
        name="DetectionPipeline",
        daemon=True,
    )
    t.start()
    logger.info("Pipeline thread started (id=%s).", t.ident)
    return t
