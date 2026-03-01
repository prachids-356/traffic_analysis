"""
metrics.py – Compute traffic statistics from the shared state snapshot.

All functions are pure (no side-effects / no I/O) so they are easy to test.
"""

from __future__ import annotations

from typing import Dict, Any, List


# ── Density thresholds ────────────────────────────────────────────────────────
DENSITY_LOW    = 10
DENSITY_MEDIUM = 25


def average_confidence(detections: List[Dict[str, Any]]) -> float:
    """Calculate the mean confidence of a list of detections."""
    if not detections:
        return 0.0
    scores = [d.get("confidence", 0.0) for d in detections]
    return round(sum(scores) / len(scores), 3)


def density_label(total: int) -> str:
    """Return a human-readable density label."""
    if total < DENSITY_LOW:
        return "LOW"
    elif total < DENSITY_MEDIUM:
        return "MEDIUM"
    else:
        return "HIGH"


def vehicles_per_minute(count_history: List[Dict[str, Any]]) -> float:
    """
    Estimate vehicles/minute from a list of timestamped count events.

    Each entry in *count_history* must have a ``"timestamp"`` key that is a
    ``datetime`` object (or ISO-8601 string convertible via fromisoformat).

    Returns 0.0 if history is empty or covers less than one second.
    """
    if len(count_history) < 2:
        return 0.0

    from datetime import datetime

    def _ts(entry: Dict[str, Any]) -> datetime:
        ts = entry.get("timestamp")
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        if isinstance(ts, datetime):
            return ts
        return datetime.min  # Fallback for type safety

    try:
        oldest = _ts(count_history[0])
        newest = _ts(count_history[-1])
        if oldest == datetime.min or newest == datetime.min:
            return 0.0
        elapsed_seconds = (newest - oldest).total_seconds()
        if elapsed_seconds < 1:
            return 0.0
        val = len(count_history) / (elapsed_seconds / 60)
        return float(round(val, 2))
    except Exception:
        return 0.0


def type_breakdown(count_history: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return per-vehicle-type counts from the history list."""
    totals: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    for entry in count_history:
        vtype = entry.get("vehicle_type", "")
        if vtype in totals:
            totals[vtype] += 1
    return totals


def compute_metrics(
    count_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the full metrics dict consumed by GET /stats.

    Parameters
    ----------
    count_history:
        List of crossing-event dicts, each with keys:
        ``timestamp``, ``vehicle_type``, ``direction``.

    Returns
    -------
    {
        "total_count":       int,
        "density":           "LOW" | "MEDIUM" | "HIGH",
        "vehicles_per_min":  float,
        "type_breakdown":    {"car": int, "motorcycle": int, "bus": int, "truck": int},
    }
    """
    total = len(count_history)
    return {
        "total_count":      total,
        "density":          density_label(total),
        "vehicles_per_min": vehicles_per_minute(count_history),
        "type_breakdown":   type_breakdown(count_history),
        "health_score":     calculate_health_score(total, count_history),
        "risk_score":       calculate_risk_score(total, count_history),
        "congestion_likely": predict_congestion(count_history),
        "avg_confidence":   0.0, # Placeholder, will be updated by service directly
    }


def calculate_health_score(total: int, history: List[Dict[str, Any]]) -> int:
    """Compute overall traffic efficiency score (0-100)."""
    if total == 0:
        return 100
    # Simple heuristic: density reduces health
    score = 100 - (total * 2)
    return max(0, min(100, score))


def calculate_risk_score(total: int, history: List[Dict[str, Any]]) -> int:
    """Risk score based on density and recent violations."""
    if total == 0:
        return 0
    # Placeholder: density + random factor for demo
    base_risk = (total / DENSITY_MEDIUM) * 50
    return max(0, min(100, int(base_risk)))


def predict_congestion(history: List[Dict[str, Any]]) -> bool:
    """Predict if congestion is likely in next 2 mins based on trend."""
    if len(history) < 5:
        return False
    # Check if last 5 vehicles arrived very close together
    return vehicles_per_minute(history[-10:]) > DENSITY_MEDIUM
