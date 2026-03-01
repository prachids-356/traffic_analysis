"""
app/analytics/metrics.py – Pure statistics functions.

No I/O, no side-effects: these are ordinary Python functions that
transform data and return plain dicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

# ── Density thresholds ─────────────────────────────────────────────────────────
_LOW    = 10
_MEDIUM = 25


def density_label(total: int) -> str:
    """Return ``"LOW"`` / ``"MEDIUM"`` / ``"HIGH"`` based on *total*."""
    if total < _LOW:
        return "LOW"
    if total < _MEDIUM:
        return "MEDIUM"
    return "HIGH"


def vehicles_per_minute(history: List[Dict[str, Any]]) -> float:
    """
    Rolling vehicles-per-minute rate from a list of crossing events.

    Each entry must contain a ``"timestamp"`` key that is either a
    ``datetime`` object or an ISO-8601 string.  Returns ``0.0`` if
    the history is empty or less than 2 seconds old.
    """
    if len(history) < 2:
        return 0.0

    def _ts(entry: Dict[str, Any]) -> datetime:
        ts = entry["timestamp"]
        return datetime.fromisoformat(ts) if isinstance(ts, str) else ts

    try:
        elapsed = (_ts(history[-1]) - _ts(history[0])).total_seconds()
        if elapsed < 1:
            return 0.0
        return round(len(history) / (elapsed / 60), 2)
    except Exception:
        return 0.0


def type_breakdown(history: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count each vehicle type in the crossing history."""
    totals: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    for event in history:
        vtype = event.get("vehicle_type", "")
        if vtype in totals:
            totals[vtype] += 1
    return totals


def compute_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the complete metrics payload consumed by ``GET /stats``.

    Parameters
    ----------
    history:
        List of crossing-event dicts, each with keys:
        ``timestamp``, ``vehicle_type``, ``direction``.

    Returns
    -------
    ::

        {
            "total_count":       int,
            "density":           "LOW" | "MEDIUM" | "HIGH",
            "vehicles_per_min":  float,
            "type_breakdown":    {"car": int, "motorcycle": int,
                                  "bus": int, "truck": int},
        }
    """
    total = len(history)
    return {
        "total_count":      total,
        "density":          density_label(total),
        "vehicles_per_min": vehicles_per_minute(history),
        "type_breakdown":   type_breakdown(history),
    }
