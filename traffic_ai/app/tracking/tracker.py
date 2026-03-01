"""
app/tracking/tracker.py – Custom centroid-based vehicle tracker.

No external library used (no DeepSORT, no SORT).

Behaviour
---------
* Each bounding-box detection is collapsed to its centroid.
* New centroids are matched to existing objects greedily by minimum
  Euclidean distance (threshold = DISTANCE_THRESHOLD px).
* Objects unseen for MAX_DISAPPEARED frames are pruned.
* When a centroid crosses LINE_Y the object is counted once
  (``counted = True``) and its direction is recorded ("up" / "down").
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import LINE_Y

logger = logging.getLogger(__name__)

DISTANCE_THRESHOLD: int = 50   # px
MAX_DISAPPEARED:    int = 30   # frames


# ── TrackedObject ──────────────────────────────────────────────────────────────

class TrackedObject:
    __slots__ = (
        "obj_id", "centroid", "class_name", "confidence",
        "disappeared", "counted", "direction", "prev_y",
    )

    def __init__(
        self,
        obj_id:     int,
        centroid:   Tuple[int, int],
        class_name: str,
        confidence: float,
    ) -> None:
        self.obj_id:     int              = obj_id
        self.centroid:   Tuple[int, int]  = centroid
        self.class_name: str              = class_name
        self.confidence: float            = confidence
        self.disappeared: int             = 0
        self.counted:    bool             = False
        self.direction:  Optional[str]    = None
        self.prev_y:     int              = centroid[1]


# ── CentroidTracker ────────────────────────────────────────────────────────────

class CentroidTracker:
    """Match detections frame-to-frame and detect LINE_Y crossings."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._objects: Dict[int, TrackedObject] = {}

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _centroid(bbox: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ── update ─────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, TrackedObject], List[Dict[str, Any]]]:
        """Process one frame's detections.

        Returns
        -------
        objects : dict[id → TrackedObject]  – full current object registry
        new_counts : list of {"vehicle_type": str, "direction": str}
            One entry per vehicle that crossed LINE_Y in this frame.
        """
        new_counts: List[Dict[str, Any]] = []

        # ── No detections: age everything ─────────────────────────────────────
        if not detections:
            for obj in self._objects.values():
                obj.disappeared += 1
            self._prune()
            return self._objects, new_counts

        input_centroids = [self._centroid(d["bbox"]) for d in detections]

        # ── Bootstrap ─────────────────────────────────────────────────────────
        if not self._objects:
            for i, det in enumerate(detections):
                self._register(input_centroids[i], det)
            return self._objects, new_counts

        # ── Distance-matrix matching ───────────────────────────────────────────
        existing_ids       = list(self._objects.keys())
        existing_centroids = [self._objects[eid].centroid for eid in existing_ids]

        matched_ex  = set()
        matched_in  = set()
        pairs: List[Tuple[float, int, int]] = [
            (self._dist(ec, ic), ei, ii)
            for ei, ec in enumerate(existing_centroids)
            for ii, ic in enumerate(input_centroids)
        ]
        pairs.sort()

        for dist, ei, ii in pairs:
            if ei in matched_ex or ii in matched_in:
                continue
            if dist > DISTANCE_THRESHOLD:
                break
            obj = self._objects[existing_ids[ei]]
            det = detections[ii]
            obj.prev_y      = obj.centroid[1]
            obj.centroid    = input_centroids[ii]
            obj.class_name  = det["class_name"]
            obj.confidence  = det["confidence"]
            obj.disappeared = 0
            matched_ex.add(ei)
            matched_in.add(ii)

        # Age unmatched existing objects
        for ei, eid in enumerate(existing_ids):
            if ei not in matched_ex:
                self._objects[eid].disappeared += 1

        # Register unmatched new detections
        for ii, det in enumerate(detections):
            if ii not in matched_in:
                self._register(input_centroids[ii], det)

        # ── Line-crossing check ────────────────────────────────────────────────
        for obj in self._objects.values():
            if obj.counted:
                continue
            cy, py = obj.centroid[1], obj.prev_y
            if py < LINE_Y <= cy:
                obj.counted   = True
                obj.direction = "down"
                new_counts.append({"vehicle_type": obj.class_name, "direction": "down"})
            elif py > LINE_Y >= cy:
                obj.counted   = True
                obj.direction = "up"
                new_counts.append({"vehicle_type": obj.class_name, "direction": "up"})

        self._prune()
        return self._objects, new_counts

    # ── internals ──────────────────────────────────────────────────────────────

    def _register(self, centroid: Tuple[int, int], det: Dict[str, Any]) -> None:
        self._objects[self._next_id] = TrackedObject(
            obj_id=self._next_id,
            centroid=centroid,
            class_name=det["class_name"],
            confidence=det["confidence"],
        )
        self._next_id += 1

    def _prune(self) -> None:
        stale = [oid for oid, obj in self._objects.items()
                 if obj.disappeared > MAX_DISAPPEARED]
        for oid in stale:
            del self._objects[oid]

    # ── Drawing (optional visualisation) ──────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay counting line + tracked centroids on *frame* in-place."""
        h, w = frame.shape[:2]
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNT LINE", (5, LINE_Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for obj in self._objects.values():
            cx, cy = obj.centroid
            colour = (0, 200, 0) if obj.counted else (255, 100, 0)
            cv2.circle(frame, (cx, cy), 5, colour, -1)
            cv2.putText(frame, f"ID{obj.obj_id} {obj.class_name}",
                        (cx + 7, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        return frame
