"""
detector.py – YOLOv8 vehicle detector with ByteTrack integration.

Uses model.track(persist=True) so ByteTrack assigns stable IDs across
frames without any external dependency (ByteTrack ships with ultralytics).
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, YOLO_IMGSZ, TRACK_ALL_CLASSES

logger = logging.getLogger(__name__)

# ── COCO class IDs for vehicles ────────────────────────────────────────────────
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Singleton model load ───────────────────────────────────────────────────────
_model = None


def _get_model():
    """Return the already-loaded model, loading it on first call."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required.  Install with: pip install ultralytics"
            ) from exc

        logger.info("Loading YOLO model from %s (this happens once) …", MODEL_PATH)
        _model = YOLO(MODEL_PATH)
        logger.info("YOLO model loaded.")
    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def detect(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Run ByteTrack-enhanced inference on *frame* and return vehicle detections.

    Returns
    -------
    list of {
        "bbox":       [x1, y1, x2, y2],   # ints
        "class_id":   int,
        "class_name": str,
        "confidence": float,
        "track_id":   int,   # ByteTrack stable ID (-1 if not yet assigned)
    }
    """
    model = _get_model()

    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        device="cpu",
        verbose=False,
        persist=True,           # ByteTrack maintains state across calls
        tracker="bytetrack.yaml",
    )

    detections: List[Dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if not TRACK_ALL_CLASSES and cls_id not in VEHICLE_CLASSES:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Use model's internal name map if class_id is known
            class_name = VEHICLE_CLASSES.get(cls_id)
            if class_name is None:
                class_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
            # ByteTrack assigns box.id; may be None for unmatched detections
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(float(box.conf[0]), 3),
                    "track_id": track_id,
                }
            )

    return detections
