import os

# ── Video source ───────────────────────────────────────────────────────────────
# To use a local webcam, set VIDEO_PATH="0" or use the Switch Camera button on the dashboard.
VIDEO_PATH: str = os.getenv("VIDEO_PATH", "traffic.mp4")

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", "yolov8n.pt")

# ── Detection ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.15  # lowered for aerial/drone footage
FRAME_SKIP: int = 3          # process every Nth frame (1 = every frame)
RESIZE_WIDTH: int = 640     # optimized for CPU
RESIZE_HEIGHT: int = 480
YOLO_IMGSZ: int = 640       # inference image size for YOLO
TRACK_ALL_CLASSES: bool = False # set to True to track any object
PLATE_OCR_ENABLED: bool = True  # set to True for License Plate recognition

# ── Counting line ─────────────────────────────────────────────────────────────
LINE_Y: int = 240            # y-position in 640x480 resized frame (middle)
SHOW_FPS: bool = True

# ── Speed estimation ──────────────────────────────────────────────────────────
# Pixels per real-world metre at the camera's altitude/zoom.
# Tune this value: measure a known object (e.g. a car ≈4.5m long) in pixels.
# Default: ~13 px/m estimated for 1280-px-wide aerial intersection (≈100m wide).
PIXELS_PER_METER: float = float(os.getenv("PIXELS_PER_METER", "13.0"))
VIDEO_FPS: float = float(os.getenv("VIDEO_FPS", "30.0"))   # native video fps
SPEED_LIMIT_KMH: float = float(os.getenv("SPEED_LIMIT_KMH", "60.0"))

# ── Lane detection ─────────────────────────────────────────────────────────────
# X-pixel boundaries between lanes (in terms of RESIZE_WIDTH px).
# Override e.g. LANE_BOUNDARIES="320,640" as a comma-separated env var.
_lb = os.getenv("LANE_BOUNDARIES", "427,854")
LANE_BOUNDARIES: list = [int(x) for x in _lb.split(",")]

# Expected horizontal direction per lane ("right", "left", "any")
_ld = os.getenv("LANE_EXPECTED_DIRS", "right,any,left")
LANE_EXPECTED_DIRS: list = [d.strip() for d in _ld.split(",")]

# Minimum horizontal pixel displacement per frame to call a direction
DIRECTION_MIN_DX: int = int(os.getenv("DIRECTION_MIN_DX", "6"))

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH: str = os.getenv("DB_PATH", "traffic.db")
