"""
traffic_service.py – Background detection pipeline.

Runs in a daemon thread.  Shared state is protected by threading.Lock.
Other modules (routes.py) read state via get_state().
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from config import VIDEO_PATH, SPEED_LIMIT_KMH, PLATE_OCR_ENABLED, RESIZE_WIDTH, RESIZE_HEIGHT
from plate_ocr import recognize_plate
from video_stream import frame_generator
from detector import detect
from tracker import CentroidTracker
from metrics import compute_metrics, density_label, average_confidence, vehicles_per_minute
from heatmap import HeatmapAccumulator
import annotator
import db

logger = logging.getLogger(__name__)

# ── Heatmap accumulator (singleton, always running) ────────────────────────────
_heatmap = HeatmapAccumulator(height=RESIZE_HEIGHT, width=RESIZE_WIDTH)

# ── Shared state ───────────────────────────────────────────────────────────────
_lock = threading.Lock()
_pipeline_thread: threading.Thread | None = None
_current_video_path: str = VIDEO_PATH

_state: Dict[str, Any] = {
    "running": False,
    "frame_id": 0,
    "metrics": {
        "total_count": 0,
        "density": "LOW",
        "vehicles_per_min": 0.0,
        "type_breakdown": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
        "health_score": 100,
        "risk_score": 0,
        "congestion_likely": False,
        "avg_confidence": 0.0,
    },
    "count_history": [],
    "recent_logs": [],
    "latest_frame": None,
    "currently_tracked": 0,
    "speed_violations": 0,
    "speed_distribution": {"0-30": 0, "30-60": 0, "60+": 0},
    "lane_counts": {"L1": 0, "L2": 0, "L3": 0},
    "wrong_way_alerts": 0,
    "heatmap_enabled": False,
    "last_summary_time": time.time(),
    "hourly_counts": {},   # "YYYY-MM-DD HH": count
}

# ── Pipeline state (internal) ──────────────────────────────────────────────────
_last_fps_time: float = 0.0
_last_spike_time: float = 0.0


def _reset_state() -> None:
    """Zero-out all per-session counters so a new source starts clean."""
    with _lock:
        _state["running"] = False
        _state["frame_id"] = 0
        _state["metrics"] = {
            "total_count": 0,
            "density": "LOW",
            "vehicles_per_min": 0.0,
            "type_breakdown": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
            "health_score": 100,
            "risk_score": 0,
            "congestion_likely": False,
            "avg_confidence": 0.0,
        }
        _state["count_history"] = []
        _state["recent_logs"] = []
        _state["latest_frame"] = None
        _state["currently_tracked"] = 0
        _state["speed_violations"] = 0
        _state["speed_distribution"] = {"0-30": 0, "30-60": 0, "60+": 0}
        _state["lane_counts"] = {"L1": 0, "L2": 0, "L3": 0}
        _state["wrong_way_alerts"] = 0
        _state["heatmap_enabled"] = False
        _state["last_summary_time"] = time.time()
        _state["hourly_counts"] = {}

# ── Public accessors ───────────────────────────────────────────────────────────


def get_state() -> Dict[str, Any]:
    with _lock:
        m = dict(_state["metrics"])
        # Always ensure type_breakdown is a proper dict copy
        m["type_breakdown"] = dict(m.get("type_breakdown", {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}))
        return {
            "running": _state["running"],
            "frame_id": _state["frame_id"],
            "metrics": m,
            "recent_logs": list(_state["recent_logs"]),
            "currently_tracked": _state["currently_tracked"],
            "speed_violations": _state["speed_violations"],
            "speed_distribution": dict(_state["speed_distribution"]),
            "lane_counts": dict(_state["lane_counts"]),
            "wrong_way_alerts": _state["wrong_way_alerts"],
            "heatmap_enabled": _state["heatmap_enabled"],
            "hourly_counts": dict(_state["hourly_counts"]),
        }


def toggle_heatmap() -> bool:
    """Flip the heatmap flag; return the new state."""
    with _lock:
        _state["heatmap_enabled"] = not _state["heatmap_enabled"]
        return _state["heatmap_enabled"]


def get_plates() -> List[Dict[str, Any]]:
    return db.fetch_plates(20)


def get_latest_frame() -> bytes | None:
    """Return the most recent annotated JPEG bytes (thread-safe)."""
    with _lock:
        return _state["latest_frame"]


def get_alerts() -> List[Dict[str, Any]]:
    return db.fetch_alerts(20)


def is_running() -> bool:
    with _lock:
        return _state["running"]


# ── Pipeline ───────────────────────────────────────────────────────────────────


def _pipeline() -> None:
    logger.info("Detection pipeline starting …")
    tracker = CentroidTracker()

    with _lock:
        _state["running"] = True
        video_to_process = _current_video_path
        def stop_check():
            return not _state["running"]

    try:
        for frame_id, frame in frame_generator(video_to_process, stop_check=stop_check):
            # 1. Detect
            detections = detect(frame)

            # 2. Track
            objects, new_counts = tracker.update(detections)

            # 2b. Update tracking telemetry: speed + lanes + wrong-way
            dist  = {"0-30": 0, "30-60": 0, "60+": 0}
            lcounts = {"L1": 0, "L2": 0, "L3": 0}
            new_violations = 0
            new_wrong_way  = 0
            for obj in objects.values():
                # Speed distribution
                s = obj.speed_kmh
                if s < 30:
                    dist["0-30"] += 1
                elif s < 60:
                    dist["30-60"] += 1
                else:
                    dist["60+"] += 1
                    if not obj.violated:
                        obj.violated = True
                        new_violations += 1
                # Lane counts
                lkey = obj.lane_name          # 'L1', 'L2', 'L3'
                lcounts[lkey] = lcounts.get(lkey, 0) + 1
                # Wrong-way (count each vehicle once)
                if obj.wrong_way and not obj._ww_counted:
                    obj._ww_counted = True
                    new_wrong_way += 1
                    db.log_alert("WRONG_WAY", "CRITICAL", f"Vehicle {obj.obj_id} moving wrong way in {obj.lane_name}")

                # Stationary Alert
                if obj.stationary_seconds > 10 and not obj.stationary_alert_fired:
                    obj.stationary_alert_fired = True
                    db.log_alert("STATIONARY_VEHICLE", "MEDIUM", f"Vehicle {obj.obj_id} stationary for >10s")

                # Heavy Vehicle Alert
                if obj.class_name in ["bus", "truck"] and not obj._heavy_alerted:
                    obj._heavy_alerted = True
                    db.log_alert("HEAVY_VEHICLE", "LOW", f"{obj.class_name.upper()} detected in {obj.lane_name}")

                # 2b-ii. Plate OCR (run once per vehicle if enabled)
                if PLATE_OCR_ENABLED and obj.class_name in ["car", "bus", "truck", "motorcycle"] and obj.plate_text is None:
                    # Crop and recognize
                    p_text, p_conf = recognize_plate(frame, list(obj.bbox))
                    if p_text and p_conf > 0.4:
                        obj.plate_text = p_text
                        obj.plate_conf = p_conf
                        db.log_plate(obj.obj_id, obj.class_name, p_text, p_conf)
                        logger.info("Plate recognized for vehicle %d: %s (conf=%.2f)", obj.obj_id, p_text, p_conf)

            with _lock:
                if not _state["running"]:
                    break
                _state["currently_tracked"] = tracker.active_count
                _state["speed_violations"]  += new_violations
                _state["speed_distribution"] = dist
                _state["lane_counts"]        = lcounts
                _state["wrong_way_alerts"]  += new_wrong_way
                _state["metrics"]["avg_confidence"] = average_confidence(detections)

            # 2c. FPS Calculation
            global _last_fps_time
            now_time = time.time()
            if _last_fps_time > 0:
                dt = now_time - _last_fps_time
                fps = 1.0 / dt if dt > 0 else 0
            else:
                fps = 0
            _last_fps_time = now_time

            # 3. Log crossings
            now = datetime.now()
            for event in new_counts:
                with _lock:
                    history = _state["count_history"]
                    current_density = density_label(len(history))
                    history.append(
                        {
                            "timestamp": now,
                            "vehicle_type": event["vehicle_type"],
                            "direction": event["direction"],
                        }
                    )
                db.log_crossing(
                    timestamp=now,
                    vehicle_type=event["vehicle_type"],
                    direction=event["direction"],
                    density=current_density,
                )
                # Hourly count for peak detection
                hour_key = now.strftime("%Y-%m-%d %H:00")
                _state["hourly_counts"][hour_key] = _state["hourly_counts"].get(hour_key, 0) + 1

            # 4. Update heatmap + annotate frame
            centroids = [obj.centroid for obj in objects.values()]
            _heatmap.update(centroids)

            with _lock:
                hm_on = _state["heatmap_enabled"]

            objects_snapshot = dict(tracker._objects)  # copy for thread safety
            display_frame = _heatmap.render(frame) if hm_on else frame
            jpeg_bytes = annotator.draw(display_frame, objects_snapshot, fps=fps)
            with _lock:
                _state["latest_frame"] = jpeg_bytes

            # 5. Recompute metrics
            with _lock:
                history_snapshot: List[Dict[str, Any]] = list(_state["count_history"])

            metrics = compute_metrics(history_snapshot)
            recent = db.fetch_recent(50)

            # 5a. Build live type_breakdown from currently tracked objects.
            # This is always populated while vehicles are visible, unlike
            # count_history which only grows when vehicles cross LINE_Y.
            live_breakdown: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for obj in objects_snapshot.values():
                if obj.class_name in live_breakdown:
                    live_breakdown[obj.class_name] += 1
            # Use live breakdown if historical counts are all zero (no crossings yet)
            historical_tb = metrics.get("type_breakdown", {})
            if sum(historical_tb.values()) == 0 and sum(live_breakdown.values()) > 0:
                metrics["type_breakdown"] = live_breakdown

            # 5b. Spike Detection (requires metrics to be computed first)
            global _last_spike_time
            with _lock:
                history_for_spike = list(_state["count_history"])
            vpm = metrics.get("vehicles_per_min", 0.0)
            
            # Use explicit list for slicing to satisfy linter and prevent errors
            avg_vpm = 0.0
            if len(history_for_spike) > 1:
                # Compare current VPM to recent average (last 10 events excluding latest)
                recent_segment = history_for_spike[-11:-1]
                avg_vpm = vehicles_per_minute(recent_segment) if recent_segment else 0.0

            if vpm > avg_vpm * 2 and vpm > 5 and time.time() - _last_spike_time > 60:
                _last_spike_time = time.time()
                db.log_alert("TRAFFIC_SPIKE", "HIGH",
                             f"Sudden surge: {vpm:.1f} vehicles/min (Avg: {avg_vpm:.1f})")

            # 6. Update shared state
            with _lock:
                _state["frame_id"] = frame_id
                _state["metrics"] = metrics
                _state["recent_logs"] = recent

                # 7. Periodic summary logging (every 1 min)
                if now_time - _state["last_summary_time"] > 60:
                    _state["last_summary_time"] = now_time
                    peak_hour = max(_state["hourly_counts"], key=_state["hourly_counts"].get) if _state["hourly_counts"] else "N/A"
                    db.log_summary(
                        total_count=metrics["total_count"],
                        health_score=metrics["health_score"],
                        risk_score=metrics["risk_score"],
                        peak_hour=peak_hour
                    )

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
    finally:
        db.flush()
        with _lock:
            _state["running"] = False
        logger.info("Detection pipeline stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────


def start_pipeline(video_path: str | None = None) -> threading.Thread:
    """Spawn and return the background daemon thread."""
    global _pipeline_thread, _current_video_path
    
    with _lock:
        if _state["running"]:
            logger.warning("Pipeline already running. Stop it first.")
            return _pipeline_thread

        if video_path:
            _current_video_path = video_path

    _pipeline_thread = threading.Thread(target=_pipeline, name="DetectionPipeline", daemon=True)
    _pipeline_thread.start()
    logger.info("Background pipeline thread started (id=%s) for %s.", _pipeline_thread.ident, _current_video_path)
    return _pipeline_thread


def stop_pipeline() -> None:
    """Signal the pipeline to stop, wait for it, then reset all state."""
    global _pipeline_thread
    with _lock:
        _state["running"] = False

    if _pipeline_thread and _pipeline_thread.is_alive():
        logger.info("Waiting for pipeline thread to exit...")
        _pipeline_thread.join(timeout=5.0)
        _pipeline_thread = None

    # Reset all counters so the next source starts from a clean slate
    _reset_state()
    logger.info("Pipeline stopped and state reset.")


def restart_pipeline(video_path: str) -> None:
    """Stop current pipeline and start with new video path."""
    logger.info("Restarting pipeline for: %s", video_path)
    stop_pipeline()
    # Give a tiny bit of time for the OS to release resources if needed
    time.sleep(0.1)
    start_pipeline(video_path)
