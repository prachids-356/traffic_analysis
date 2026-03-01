"""
lanes.py – Pure lane-assignment and direction helpers.

No OpenCV dependency.  All thresholds come from config.py so they can be
tuned without changing code.
"""

from __future__ import annotations

from config import LANE_BOUNDARIES, LANE_EXPECTED_DIRS, DIRECTION_MIN_DX


def get_lane(cx: int) -> int:
    """Return 0-indexed lane number for centroid x-position *cx*."""
    for i, boundary in enumerate(LANE_BOUNDARIES):
        if cx < boundary:
            return i
    return len(LANE_BOUNDARIES)   # last lane


def lane_name(idx: int) -> str:
    """Human-readable lane label, e.g. 0 → 'L1'."""
    return f"L{idx + 1}"


def get_horiz_dir(dx: float) -> str:
    """
    Classify horizontal movement direction.

    Returns
    -------
    'right'  if dx >=  DIRECTION_MIN_DX
    'left'   if dx <= -DIRECTION_MIN_DX
    'none'   otherwise (too little movement to classify)
    """
    if dx >= DIRECTION_MIN_DX:
        return "right"
    if dx <= -DIRECTION_MIN_DX:
        return "left"
    return "none"


def is_wrong_way(lane_idx: int, horiz_dir: str) -> bool:
    """
    Return True if *horiz_dir* conflicts with the expected direction for *lane_idx*.

    'any' lanes never trigger wrong-way.
    'none' direction never triggers wrong-way (vehicle not moving enough).
    """
    if horiz_dir == "none":
        return False
    try:
        expected = LANE_EXPECTED_DIRS[lane_idx]
    except IndexError:
        return False
    if expected == "any":
        return False
    return horiz_dir != expected
