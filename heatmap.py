"""
heatmap.py – Gaussian heatmap accumulator for vehicle density visualization.

Usage in the pipeline
---------------------
    accum = HeatmapAccumulator(height=720, width=1280)

    # Each frame:
    accum.update([(cx1, cy1), (cx2, cy2), ...])
    display_frame = accum.render(raw_frame)      # blend, return ndarray
"""

from __future__ import annotations

import threading
import cv2
import numpy as np
from typing import List, Tuple


class HeatmapAccumulator:
    """
    Accumulates vehicle centroid positions into a float32 intensity map.

    Parameters
    ----------
    height, width  : dimensions of the video frame
    decay          : per-frame multiplier applied to the buffer (0.95–0.99)
    blur_ksize     : Gaussian kernel size for spatial smoothing
    alpha          : blend weight of the colorised heatmap over the frame
    threshold      : normalised intensity threshold below which heatmap is invisible
    """

    COLORMAP = cv2.COLORMAP_JET   # looks great; COLORMAP_INFERNO is another good choice

    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        decay: float = 0.97,
        blur_ksize: int = 71,
        alpha: float = 0.55,
        threshold: int = 8,
    ) -> None:
        self._h = height
        self._w = width
        self._decay = decay
        self._blur_ksize = blur_ksize | 1          # must be odd
        self._alpha = alpha
        self._threshold = threshold
        self._buf = np.zeros((height, width), dtype=np.float32)
        self._lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, centroids: List[Tuple[int, int]]) -> None:
        """Decay the buffer and add new vehicle positions."""
        with self._lock:
            self._buf *= self._decay
            for cx, cy in centroids:
                if 0 <= cx < self._w and 0 <= cy < self._h:
                    self._buf[cy, cx] += 1.0

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a copy of *frame* with the heatmap blended on top.
        If the buffer is empty, the original frame is returned unchanged.
        """
        with self._lock:
            buf_copy = self._buf.copy()

        # Spatially smooth with Gaussian
        blurred = cv2.GaussianBlur(buf_copy, (self._blur_ksize, self._blur_ksize), 0)

        if blurred.max() < 1e-6:
            return frame.copy()

        # Normalise to [0, 255]
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colour map
        colored = cv2.applyColorMap(norm, self.COLORMAP)

        # Only paint where intensity is above threshold (keeps background clean)
        mask = norm > self._threshold

        out = frame.copy()
        blended = cv2.addWeighted(frame, 1.0 - self._alpha, colored, self._alpha, 0)
        out[mask] = blended[mask]

        # Add "HEATMAP" watermark
        cv2.putText(
            out, "HEATMAP",
            (out.shape[1] - 110, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return out

    def reset(self) -> None:
        """Clear the accumulation buffer."""
        with self._lock:
            self._buf[:] = 0.0
