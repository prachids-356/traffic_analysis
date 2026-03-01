"""
annotator.py – Draw detection overlays onto BGR frames.

Per-vehicle label format:
    ID:{n}  {class}  {conf}%  [{dwell}s]  L{n}  |  {speed} km/h

Special overlays:
  • Colored vertical lane divider lines (labeled L1/L2/L3)
  • Magenta box + ⚠ WRONG WAY text for wrong-way vehicles
  • Red box + ! icon for speeders
  • Dashed yellow counting line
Returns JPEG bytes.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple

from config import LINE_Y, SPEED_LIMIT_KMH, LANE_BOUNDARIES, RESIZE_WIDTH, SHOW_FPS

# ── Per-class colours (BGR) ────────────────────────────────────────────────────
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car":        (0x88, 0xFF, 0x00),
    "truck":      (0xFF, 0x88, 0x44),
    "bus":        (0x00, 0x88, 0xFF),
    "motorcycle": (0xAA, 0x44, 0xFF),
}
DEFAULT_COLOR          = (0xCC, 0xCC, 0xCC)
SPEED_VIOLATION_COLOR  = (0x00, 0x00, 0xFF)      # red
WRONG_WAY_COLOR        = (0xFF, 0x00, 0xFF)       # magenta

# Lane divider colours (by lane index)
LANE_DIVIDER_COLORS = [
    (0, 220, 255),   # cyan
    (255, 165,  0),  # orange
]

JPEG_QUALITY = 75

# ── Placeholder ────────────────────────────────────────────────────────────────
def _make_placeholder() -> bytes:
    h, w = 720, 1280
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for stream...", (w//2-220, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes()

PLACEHOLDER_FRAME: bytes = _make_placeholder()


# ── Lane divider overlay helper ────────────────────────────────────────────────
def _draw_lanes(out: np.ndarray) -> None:
    """Draw semi-transparent colored vertical lane dividers."""
    h = out.shape[0]
    lane_labels = [f"L{i+1}" for i in range(len(LANE_BOUNDARIES) + 1)]

    for i, bx in enumerate(LANE_BOUNDARIES):
        color = LANE_DIVIDER_COLORS[i % len(LANE_DIVIDER_COLORS)]
        # Dashed vertical line
        y = 0
        dash, gap = 20, 10
        while y < h:
            y2 = min(y + dash, h)
            cv2.line(out, (bx, y), (bx, y2), color, 2)
            y += dash + gap

    # Lane labels near the top
    num_lanes = len(LANE_BOUNDARIES) + 1
    prev_x = 0
    for i in range(num_lanes):
        next_x = LANE_BOUNDARIES[i] if i < len(LANE_BOUNDARIES) else RESIZE_WIDTH
        mid_x  = (prev_x + next_x) // 2
        label  = f"L{i+1}"
        color  = LANE_DIVIDER_COLORS[min(i, len(LANE_DIVIDER_COLORS)-1)]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # semi-transparent pill
        overlay = out.copy()
        cv2.rectangle(overlay, (mid_x - tw//2 - 6, 8), (mid_x + tw//2 + 6, 8 + th + 10), color, -1)
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        cv2.putText(out, label, (mid_x - tw//2, 8 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
        prev_x = next_x


# ── Main draw function ─────────────────────────────────────────────────────────
def draw(frame: np.ndarray, objects: Dict, fps: float = 0.0) -> bytes:
    out = frame.copy()
    h, w = out.shape[:2]

    # FPS overlay
    if SHOW_FPS and fps > 0:
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Lane dividers
    _draw_lanes(out)

    # Dashed counting line
    x = 0
    while x < w:
        cv2.line(out, (x, LINE_Y), (min(x+30, w), LINE_Y), (0, 230, 255), 2)
        x += 45
    cv2.putText(out, "COUNT LINE", (6, LINE_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

    FONT     = cv2.FONT_HERSHEY_SIMPLEX
    BOX_HALF = 40

    for obj_id, obj in objects.items():
        cx, cy   = obj.centroid
        cls      = obj.class_name
        conf     = obj.confidence
        dwell    = obj.dwell_seconds
        speed    = obj.speed_kmh
        speeding = speed > SPEED_LIMIT_KMH
        wrong    = obj.wrong_way
        lname    = obj.lane_name

        # Box color priority: wrong-way > speeding > class
        if wrong:
            color = WRONG_WAY_COLOR
        elif speeding:
            color = SPEED_VIOLATION_COLOR
        else:
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        x1 = max(0, cx - BOX_HALF)
        y1 = max(0, cy - BOX_HALF)
        x2 = min(w-1, cx + BOX_HALF)
        y2 = min(h-1, cy + BOX_HALF)

        thickness = 3 if (speeding or wrong) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Wrong-way alert text
        if wrong:
            cv2.putText(out, "WRONG WAY", (x1, y2 + 16),
                        FONT, 0.45, WRONG_WAY_COLOR, 2, cv2.LINE_AA)
        elif speeding:
            cv2.putText(out, "!", (x2 - 10, y1 + 14),
                        FONT, 0.6, SPEED_VIOLATION_COLOR, 2, cv2.LINE_AA)

        # Label bar: ID:{n}  {cls}  {conf}%  [{dwell}s]  L{n}  |  {speed} km/h
        dir_arrow = {"right": "→", "left": "←", "none": ""}.get(obj.horiz_dir, "")
        speed_str = f"{int(speed)} km/h" if speed > 0 else "-- km/h"
        label = f"ID:{obj_id}  {cls}  {int(conf*100)}%  [{dwell:.1f}s]  {lname}{dir_arrow}  |  {speed_str}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
        bar_y1 = max(0, y1 - th - 8)
        bar_y2 = y1
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, bar_y1), (x1 + tw + 8, bar_y2), color, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.putText(out, label, (x1 + 4, bar_y2 - 3),
                    FONT, 0.38, (10, 10, 10), 1, cv2.LINE_AA)

        # Centroid dot
        cv2.circle(out, (cx, cy), 4, (0, 200, 80) if obj.counted else color, -1)

        # Flow Arrow
        if obj.horiz_dir in ["left", "right"]:
            arrow_len = 30
            dx = arrow_len if obj.horiz_dir == "right" else -arrow_len
            cv2.arrowedLine(out, (cx, cy), (cx + dx, cy), color, 2, tipLength=0.3)

    _, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes()
