"""
app/detection/detector.py – YOLOv8n vehicle detector (singleton).

The model is loaded at module import time and never reloaded.
Only COCO vehicle classes are returned; no drawing logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ── COCO vehicle class ids ─────────────────────────────────────────────────────
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Singleton ──────────────────────────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install ultralytics: pip install ultralytics"
            ) from exc
        logger.info("Loading YOLO model from %r … (one-time)", MODEL_PATH)
        _model = YOLO(MODEL_PATH)
        logger.info("YOLO model ready.")
    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def detect(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Run inference on *frame*.

    Returns
    -------
    list of::

        {
            "bbox":       [x1, y1, x2, y2],   # int pixels
            "class_id":   int,
            "class_name": str,
            "confidence": float,
        }

    Only vehicle classes (car / motorcycle / bus / truck) are included.
    """
    model = _get_model()
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "class_id":   cls_id,
                "class_name": VEHICLE_CLASSES[cls_id],
                "confidence": round(float(box.conf[0]), 3),
            })
    return detections
