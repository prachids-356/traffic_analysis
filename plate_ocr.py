"""
plate_ocr.py – Helper for License Plate Recognition using EasyOCR.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)

# ── Singleton for EasyOCR ──────────────────────────────────────────────────
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        if easyocr is None:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            return None
        logger.info("Initializing EasyOCR reader (English) ...")
        # Optimization: gpu=False for CPU-only systems or if you want to save VRAM
        _reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR initialized.")
    return _reader

def recognize_plate(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
    """
    Crop the image at bbox, perform OCR, and return (text, confidence).
    """
    reader = _get_reader()
    if not reader:
        return None, 0.0

    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within frame
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    crop = frame[y1:y2, x1:x2]
    
    # Preprocessing (optional but usually helps LPR)
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Resize to have a minimum height for better OCR
    if gray.shape[0] < 50:
        scale = 50.0 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(gray)
    
    if not results:
        return None, 0.0

    # results is a list of [ [bbox], text, confidence ]
    # Pick the one with highest confidence or longest text? Usually highest confidence.
    best_text = None
    best_conf = 0.0

    for (_, text, conf) in results:
        # Simple heuristic: plates usually contain letters and numbers, avoid very short strings
        text = text.replace(" ", "").upper()
        if len(text) < 3:
            continue
        if conf > best_conf:
            best_conf = conf
            best_text = text

    return best_text, best_conf
