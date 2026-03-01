"""
app/ingestion/video_stream.py – Efficient frame generator.

Yields (frame_id, bgr_frame) for every FRAME_SKIP-th decoded frame.
Loops the source video so the pipeline never stalls.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

import cv2
import numpy as np

from app.config import VIDEO_PATH, FRAME_SKIP, RESIZE_WIDTH, RESIZE_HEIGHT

logger = logging.getLogger(__name__)


def frame_generator(
    path: str = VIDEO_PATH,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Open *path* and yield ``(frame_id, resized_frame)`` tuples.

    * Sets ``CAP_PROP_BUFFERSIZE = 1`` so we always work with the most
      recent frame rather than buffered ones.
    * Skips frames that are not multiples of ``FRAME_SKIP``.
    * Loops the file when the end is reached (simulates a continuous CCTV feed).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path!r}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("VideoCapture opened: %s  (buffer=1, skip=%d)", path, FRAME_SKIP)

    frame_id: int = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("End of video — looping back to frame 0.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id += 1
            if frame_id % FRAME_SKIP != 0:
                continue

            resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            yield frame_id, resized
    finally:
        cap.release()
        logger.info("VideoCapture released after %d frames.", frame_id)
