"""
tracker.py – Centroid tracker with ByteTrack ID support, dwell time,
             speed estimation, lane detection, and direction tracking.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from config import LINE_Y, PIXELS_PER_METER, VIDEO_FPS, FRAME_SKIP
import lanes as lane_helpers

logger = logging.getLogger(__name__)

_EFFECTIVE_FPS: float = VIDEO_FPS / max(1, FRAME_SKIP)

DISTANCE_THRESHOLD: int = 40
MAX_DISAPPEARED: int = 15
SPEED_EMA_ALPHA: float = 0.3
MAX_SPEED_KMH: float = 200.0


# ── TrackedObject ──────────────────────────────────────────────────────────────
class TrackedObject:
    __slots__ = (
        "obj_id", "centroid", "prev_centroid", "class_name", "confidence",
        "disappeared", "counted", "direction", "prev_y",
        "first_seen", "speed_kmh", "violated",
        # Lane / direction
        "lane", "horiz_dir", "wrong_way",
        "last_moved_time", "stationary_seconds", "stationary_alert_fired",
        # One-shot alert flags
        "_ww_counted", "_heavy_alerted",
        # License plate info
        "plate_text", "plate_conf",
        "bbox",
    )

    def __init__(self, obj_id: int, centroid: Tuple[int, int],
                 class_name: str, confidence: float, bbox: List[int]) -> None:
        self.obj_id: int        = obj_id
        self.centroid           = centroid
        self.prev_centroid      = centroid
        self.class_name: str    = class_name
        self.confidence: float  = confidence
        self.bbox: List[int]    = bbox
        self.disappeared: int   = 0
        self.counted: bool      = False
        self.direction: Optional[str] = None
        self.prev_y: int        = centroid[1]
        self.first_seen         = datetime.now()
        self.speed_kmh: float   = 0.0
        self.violated: bool     = False
        # Lane
        self.lane: int          = lane_helpers.get_lane(centroid[0])
        self.horiz_dir: str     = "none"
        self.wrong_way: bool    = False
        self.last_moved_time    = datetime.now()
        self.stationary_seconds: float = 0.0
        self.stationary_alert_fired: bool = False
        self._ww_counted: bool      = False
        self._heavy_alerted: bool   = False
        self.plate_text: Optional[str] = None
        self.plate_conf: float      = 0.0

    @property
    def dwell_seconds(self) -> float:
        return (datetime.now() - self.first_seen).total_seconds()

    @property
    def lane_name(self) -> str:
        return lane_helpers.lane_name(self.lane)


# ── Speed helper ───────────────────────────────────────────────────────────────
def _estimate_speed(prev: Tuple[int, int], curr: Tuple[int, int]) -> float:
    px_dist = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
    speed_ms = (px_dist / PIXELS_PER_METER) * _EFFECTIVE_FPS
    return min(speed_ms * 3.6, MAX_SPEED_KMH)


# ── CentroidTracker ────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self) -> None:
        self._next_id: int = 0
        self._objects: Dict[int, TrackedObject] = {}

    @staticmethod
    def _centroid(bbox: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    @property
    def active_count(self) -> int:
        return len(self._objects)

    def update(self, detections: List[Dict[str, Any]]) -> Tuple[Dict, List]:
        new_counts: List[Dict[str, Any]] = []

        if not detections:
            for obj in list(self._objects.values()):
                obj.disappeared += 1
            self._prune()
            return self._objects, new_counts

        input_centroids = [self._centroid(d["bbox"]) for d in detections]

        # ── ByteTrack fast path ───────────────────────────────────────────────
        has_bytetrack = any(d.get("track_id", -1) >= 0 for d in detections)
        matched_input: set = set()

        if has_bytetrack:
            for ii, det in enumerate(detections):
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue
                cx, cy = input_centroids[ii]
                if tid in self._objects:
                    self._update_object(self._objects[tid], (cx, cy), det)
                else:
                    self._register((cx, cy), det, external_id=tid)
                matched_input.add(ii)

        # ── Centroid-distance fallback ────────────────────────────────────────
        unmatched = [i for i in range(len(detections)) if i not in matched_input]
        if unmatched and self._objects:
            existing_ids = list(self._objects.keys())
            existing_centroids = [self._objects[eid].centroid for eid in existing_ids]
            matched_existing: set = set()
            pairs: List[Tuple[float, int, int]] = [
                (self._distance(ec, input_centroids[ii]), ei, ii)
                for ei, ec in enumerate(existing_centroids)
                for ii in unmatched
            ]
            pairs.sort(key=lambda x: x[0])
            for dist, ei, ii in pairs:
                if ei in matched_existing or ii in matched_input:
                    continue
                if dist > DISTANCE_THRESHOLD:
                    break
                self._update_object(self._objects[existing_ids[ei]],
                                    input_centroids[ii], detections[ii])
                matched_existing.add(ei)
                matched_input.add(ii)

        # ── Age unmatched ────────────────────────────────────────────────────
        updated_centroids = {input_centroids[ii] for ii in matched_input}
        for obj in self._objects.values():
            if obj.centroid not in updated_centroids:
                obj.disappeared += 1

        # ── Register new ─────────────────────────────────────────────────────
        for ii in range(len(detections)):
            if ii not in matched_input:
                self._register(input_centroids[ii], detections[ii])

        # ── Line-crossing check ───────────────────────────────────────────────
        for obj in self._objects.values():
            if obj.counted:
                continue
            cy, py = obj.centroid[1], obj.prev_y
            if py < LINE_Y <= cy:
                obj.counted = True; obj.direction = "down"
                new_counts.append({"vehicle_type": obj.class_name, "direction": "down"})
            elif py > LINE_Y >= cy:
                obj.counted = True; obj.direction = "up"
                new_counts.append({"vehicle_type": obj.class_name, "direction": "up"})

        self._prune()
        return self._objects, new_counts

    # ── helpers ────────────────────────────────────────────────────────────────

    def _update_object(self, obj: TrackedObject,
                       new_centroid: Tuple[int, int],
                       detection: Dict[str, Any]) -> None:
        # Speed
        raw_speed = _estimate_speed(obj.centroid, new_centroid)
        obj.speed_kmh = SPEED_EMA_ALPHA * raw_speed + (1 - SPEED_EMA_ALPHA) * obj.speed_kmh

        # Horizontal direction + lane + wrong-way
        dx = new_centroid[0] - obj.centroid[0]
        horiz_dir = lane_helpers.get_horiz_dir(dx)
        lane_idx  = lane_helpers.get_lane(new_centroid[0])
        obj.horiz_dir = horiz_dir
        obj.lane      = lane_idx
        obj.wrong_way = lane_helpers.is_wrong_way(lane_idx, horiz_dir)

        # Stationary check
        if self._distance(obj.centroid, new_centroid) < 2:
            obj.stationary_seconds = (datetime.now() - obj.last_moved_time).total_seconds()
        else:
            obj.last_moved_time = datetime.now()
            obj.stationary_seconds = 0.0

        # Violations
        if obj.speed_kmh > 60 and not obj.violated:
            obj.violated = True

        obj.prev_centroid = obj.centroid
        obj.prev_y        = obj.centroid[1]
        obj.centroid      = new_centroid
        obj.bbox          = detection["bbox"]
        obj.class_name    = detection["class_name"]
        obj.confidence    = detection["confidence"]
        obj.disappeared   = 0

    def _register(self, centroid: Tuple[int, int], detection: Dict[str, Any],
                  external_id: Optional[int] = None) -> None:
        obj_id = external_id if external_id is not None else self._next_id
        self._objects[obj_id] = TrackedObject(
            obj_id=obj_id, centroid=centroid,
            class_name=detection["class_name"], confidence=detection["confidence"],
            bbox=detection["bbox"]
        )
        if external_id is None:
            self._next_id += 1

    def _prune(self) -> None:
        stale = [oid for oid, obj in self._objects.items()
                 if obj.disappeared > MAX_DISAPPEARED]
        for oid in stale:
            del self._objects[oid]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 255), 2)
        for obj in self._objects.values():
            cx, cy = obj.centroid
            color = (0, 80, 220) if obj.speed_kmh > 60 else (0, 200, 0)
            cv2.circle(frame, (cx, cy), 5, color, -1)
        return frame
