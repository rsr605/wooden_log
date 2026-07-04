"""
Unit tests for src/utils.py — aspect ratio, diameter, and bbox helpers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    compute_aspect_ratio,
    compute_diameter,
    xyxy_to_xywh,
    xywh_to_xyxy,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Aspect Ratio
# ---------------------------------------------------------------------------

class TestAspectRatio:
    def test_square_bbox(self):
        """Square bbox should have ratio 1.0."""
        assert compute_aspect_ratio(100, 100) == 1.0

    def test_wide_bbox(self):
        """Wide bbox should have ratio > 1.0."""
        assert compute_aspect_ratio(200, 100) == 2.0

    def test_tall_bbox(self):
        """Tall bbox should have ratio < 1.0."""
        assert compute_aspect_ratio(100, 200) == 0.5

    def test_zero_height(self):
        """Zero height should not crash (returns large value)."""
        result = compute_aspect_ratio(100, 0)
        assert result > 100  # very large

    def test_rounded_to_2_decimals(self):
        """Result should be rounded to 2 decimal places."""
        result = compute_aspect_ratio(100, 3)
        assert result == 33.33


# ---------------------------------------------------------------------------
# Diameter
# ---------------------------------------------------------------------------

class TestDiameter:
    def test_equal_dimensions(self):
        """Equal w and h should give that value as diameter."""
        assert compute_diameter(100, 100) == 100

    def test_different_dimensions(self):
        """Diameter should be average of w and h."""
        assert compute_diameter(100, 200) == 150

    def test_returns_int(self):
        """Result should be a rounded int."""
        result = compute_diameter(100, 103)
        assert isinstance(result, int)
        assert result == 102


# ---------------------------------------------------------------------------
# IoU Threshold Default
# ---------------------------------------------------------------------------

class TestThresholdDefaults:
    def test_iou_threshold_is_050(self):
        """IoU threshold should default to 0.50 for WBF fusion + NMS."""
        assert DEFAULT_IOU_THRESHOLD == 0.50

    def test_confidence_threshold_default(self):
        """Confidence threshold default should be 0.25 for high recall on real photos."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.25


# ---------------------------------------------------------------------------
# BBox Conversion (ensure no regression)
# ---------------------------------------------------------------------------

class TestBBoxConversions:
    def test_xyxy_to_xywh(self):
        cx, cy, w, h = xyxy_to_xywh(10, 20, 110, 220)
        assert cx == 60
        assert cy == 120
        assert w == 100
        assert h == 200

    def test_xywh_to_xyxy(self):
        x1, y1, x2, y2 = xywh_to_xyxy(60, 120, 100, 200)
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110
        assert y2 == 220
