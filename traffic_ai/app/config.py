"""
app/config.py – Central configuration for all tunable parameters.
Override any value via environment variables before starting the server.
"""

import os

# ── Video source ───────────────────────────────────────────────────────────────
VIDEO_PATH: str = os.getenv("VIDEO_PATH", "traffic.mp4")

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", "yolov8n.pt")

# ── Frame processing ───────────────────────────────────────────────────────────
FRAME_SKIP: int = 3          # yield every FRAME_SKIP-th frame
RESIZE_WIDTH: int = 640
RESIZE_HEIGHT: int = 480

# ── Detection ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.4

# ── Counting line ─────────────────────────────────────────────────────────────
LINE_Y: int = 240            # y-pixel position in the resized frame

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH: str = os.getenv("DB_PATH", "traffic.db")
