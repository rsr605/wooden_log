"""
Wooden Log Detection - Shared Utilities
Common helper functions used across the project.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported file types
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Detection defaults — tuned for high-recall wooden log detection on real
# photos. Multi-scale inference + WBF compensates for the synthetic→real
# domain gap. conf=0.25 with WBF gives the best F1 on real test images.
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.50
DEFAULT_MODEL_VARIANT = "models/wooden_log_best.pt"

# Upload limits
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# Class names for this project
CLASS_NAMES = ["wooden_log"]


# ---------------------------------------------------------------------------
# Coordinate / Bounding Box Helpers
# ---------------------------------------------------------------------------

def compute_aspect_ratio(width: float, height: float) -> float:
    """
    Compute aspect ratio (width / height) of a bounding box.

    A ratio near 1.0 indicates a nearly circular cross-section.
    Clamps height to a tiny epsilon to avoid division by zero.

    Args:
        width:  Bounding box width in pixels.
        height: Bounding box height in pixels.

    Returns:
        Aspect ratio rounded to 2 decimal places.
    """
    if height == 0:
        height = 1e-6
    return round(width / height, 2)


def compute_diameter(width: float, height: float) -> int:
    """
    Compute the average diameter (in pixels) of a detected log cross-section,
    approximated as the average of the bounding box width and height.

    Args:
        width:  Bounding box width in pixels.
        height: Bounding box height in pixels.

    Returns:
        Average diameter as a rounded integer.
    """
    return round((width + height) / 2.0)


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) → (x_center, y_center, width, height)."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert (x_center, y_center, width, height) → (x1, y1, x2, y2)."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def normalize_bbox(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    source_format: str = "xyxy",
) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates to [0, 1] range.

    Args:
        bbox: Either (x1,y1,x2,y2) or (cx,cy,w,h) in pixel coordinates.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        source_format: 'xyxy' or 'xywh'.

    Returns:
        Normalized (cx, cy, w, h) in [0, 1] range (YOLO format).
    """
    if source_format == "xyxy":
        cx, cy, w, h = xyxy_to_xywh(*bbox)
    else:
        cx, cy, w, h = bbox
    return cx / img_width, cy / img_height, w / img_width, h / img_height


def denormalize_bbox(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO (cx, cy, w, h) to pixel (x1, y1, x2, y2).

    Args:
        bbox: Normalized (cx, cy, w, h) in [0, 1].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Pixel (x1, y1, x2, y2).
    """
    cx, cy, w, h = bbox
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height
    return xywh_to_xyxy(cx, cy, w, h)


# ---------------------------------------------------------------------------
# File / Path Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension including the dot (e.g. '.jpg')."""
    return Path(filename).suffix.lower()


def is_allowed_image(filename: str) -> bool:
    """Check if the filename has an allowed image extension."""
    return get_file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS


def is_allowed_video(filename: str) -> bool:
    """Check if the filename has an allowed video extension."""
    return get_file_extension(filename) in ALLOWED_VIDEO_EXTENSIONS


def generate_hashed_filename(filename: str, prefix: str = "") -> str:
    """
    Generate a unique filename using a timestamp + short hash.

    Preserves the original extension.
    """
    ext = get_file_extension(filename)
    raw = f"{filename}{time.time()}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}{short_hash}{ext}"


# ---------------------------------------------------------------------------
# Color Helpers
# ---------------------------------------------------------------------------

# Distinct colors for drawing bounding boxes
BBOX_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue (BGR)
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Get a deterministic BGR color for a class ID."""
    return BBOX_COLORS[class_id % len(BBOX_COLORS)]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for measuring elapsed time in milliseconds."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


# ---------------------------------------------------------------------------
# Detection Result Serialization
# ---------------------------------------------------------------------------

def format_detection_result(
    image_path: str,
    image_width: int,
    image_height: int,
    detections: List[Dict[str, Any]],
    processing_time_ms: float,
) -> Dict[str, Any]:
    """Build the standardized detection result dictionary."""
    return {
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
        "count": len(detections),
        "processing_time_ms": round(processing_time_ms, 2),
    }
