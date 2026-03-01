"""
video_stream.py – Efficient frame ingestion from a video file or camera.

Yields (frame_id, frame) tuples.  Every FRAME_SKIP-th frame is yielded;
the rest are grabbed-but-discarded so the decoder stays in sync.
"""

from __future__ import annotations

import cv2
import logging
from typing import Generator, Tuple

import numpy as np

from config import VIDEO_PATH, FRAME_SKIP, RESIZE_WIDTH, RESIZE_HEIGHT

logger = logging.getLogger(__name__)


def frame_generator(
    path: str = VIDEO_PATH,
    stop_check=None,
    loop: bool = False,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Open *path* and yield (frame_id, resized_bgr_frame) for every
    FRAME_SKIP-th frame.

    Parameters
    ----------
    path:
        File path or camera index (as string, e.g. "0").
    stop_check:
        Callable returning True when the pipeline should stop.
    loop:
        If True, video files restart from frame 0 at EOF (infinite loop).
        If False (default), the generator exits after one full pass.
        Live camera sources (numeric path) always exit on disconnect.
    """
    source = int(path) if path.isdigit() else path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path!r}")

    # Keep the internal decode buffer small so we always get the newest frame.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_id: int = 0
    is_live = isinstance(source, int)

    try:
        while True:
            if stop_check and stop_check():
                logger.info("Stop check triggered. Exiting frame generator.")
                break
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    logger.warning("Live stream disconnected.")
                    break
                if loop:
                    # End of file – loop back to the beginning.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.debug("Video looped back to start.")
                    continue
                else:
                    logger.info("Video finished. Stopping pipeline.")
                    break

            frame_id += 1

            # Skip frames that are not multiples of FRAME_SKIP.
            if frame_id % FRAME_SKIP != 0:
                continue

            resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            yield frame_id, resized
    finally:
        cap.release()
        logger.info("VideoCapture released.")
