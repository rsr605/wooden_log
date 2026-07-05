"""
Unit tests for the LogAnalyzer class (app/segmentation.py).

Tests use synthetic images with known shapes so geometric measurements
can be verified precisely. After the move to cv2.fitEllipse, every
result now exposes length_px (major axis) and breadth_px (minor axis).
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers — create synthetic images with known shapes
# ---------------------------------------------------------------------------

def make_circle_image(
    width: int = 200,
    height: int = 200,
    cx: int = 100,
    cy: int = 100,
    radius: int = 40,
    color: tuple = (28, 58, 86),  # dark brown wood BGR (corrected)
    bg_color: tuple = (60, 70, 55),
) -> np.ndarray:
    """Create an image with a filled brown circle on a green background."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, color, -1)
    return img


def make_ellipse_image(
    width: int = 300,
    height: int = 200,
    cx: int = 150,
    cy: int = 100,
    axes: tuple = (80, 25),
    color: tuple = (28, 58, 86),
    bg_color: tuple = (60, 70, 55),
) -> np.ndarray:
    """Create an image with a filled brown ellipse on a green background."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, color, -1)
    return img


def make_detection(bbox: tuple, conf: float = 0.9, cls: str = "wooden_log") -> dict:
    """Create a detection dict matching LogDetector.detect() output."""
    x1, y1, x2, y2 = bbox
    return {
        "class_id": 0,
        "class_name": cls,
        "confidence": conf,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "aspect_ratio": round((x2 - x1) / max(1, y2 - y1), 2),
        "diameter_px": round(((x2 - x1) + (y2 - y1)) / 2),
    }


# ---------------------------------------------------------------------------
# LogAnalyzer Initialization
# ---------------------------------------------------------------------------

class TestLogAnalyzerInit:
    def test_default_pad(self):
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer()
        assert analyzer.pad == 5

    def test_custom_pad(self):
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=10)
        assert analyzer.pad == 10


# ---------------------------------------------------------------------------
# Shared extract_wood_contour helper
# ---------------------------------------------------------------------------

class TestExtractWoodContour:
    def test_returns_contour_for_brown_shape(self):
        """extract_wood_contour should find a contour for a brown shape."""
        from app.segmentation import extract_wood_contour

        img = make_circle_image(200, 200, 100, 100, 40)
        bbox = {"x1": 55, "y1": 55, "x2": 145, "y2": 145}
        contour, rx1, ry1 = extract_wood_contour(img, bbox, pad=5)

        assert contour is not None
        assert len(contour) >= 5

    def test_returns_none_for_no_wood(self):
        """extract_wood_contour should return None when there are no wood pixels."""
        from app.segmentation import extract_wood_contour

        img = np.zeros((200, 200, 3), dtype=np.uint8)  # all black
        bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        contour, _, _ = extract_wood_contour(img, bbox, pad=5)
        assert contour is None

    def test_contour_in_global_coordinates(self):
        """The returned contour should be shifted to full-image coordinates."""
        from app.segmentation import extract_wood_contour

        img = make_circle_image(200, 200, 100, 100, 40)
        bbox = {"x1": 55, "y1": 55, "x2": 145, "y2": 145}
        contour, _, _ = extract_wood_contour(img, bbox, pad=5)

        # The contour centroid should be near (100, 100) — the circle centre
        # in full-image coords, proving the shift was applied.
        m = cv2.moments(contour)
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        assert abs(cx - 100) < 8
        assert abs(cy - 100) < 8

    def test_empty_roi_does_not_crash(self):
        """A zero-size ROI should not crash."""
        from app.segmentation import extract_wood_contour

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        # bbox completely outside the image
        bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        contour, _, _ = extract_wood_contour(img, bbox, pad=5)
        assert contour is None


# ---------------------------------------------------------------------------
# Contour Extraction (legacy instance helpers — still present)
# ---------------------------------------------------------------------------

class TestContourExtraction:
    def test_wood_mask_produces_nonempty_mask(self):
        """_wood_mask should return a non-empty mask for brown pixels."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer()

        roi = make_circle_image(100, 100, 50, 50, 30)
        mask = analyzer._wood_mask(roi)

        assert mask.size > 0
        assert mask.sum() > 0  # some pixels should be white

    def test_wood_mask_empty_for_green_image(self):
        """_wood_mask should return mostly-empty mask for non-wood colours."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer()

        roi = np.full((100, 100, 3), (60, 70, 55), dtype=np.uint8)  # green bg
        mask = analyzer._wood_mask(roi)

        assert mask.sum() == 0

    def test_largest_contour_returns_valid_contour(self):
        """_largest_contour should return the biggest contour."""
        from app.segmentation import LogAnalyzer

        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        cv2.circle(mask, (30, 30), 10, 255, -1)

        contour = LogAnalyzer._largest_contour(mask)
        assert contour is not None
        assert len(contour) >= 3
        area = cv2.contourArea(contour)
        assert area > 1000  # the big circle

    def test_largest_contour_returns_none_for_empty(self):
        """_largest_contour should return None for an empty mask."""
        from app.segmentation import LogAnalyzer

        mask = np.zeros((100, 100), dtype=np.uint8)
        contour = LogAnalyzer._largest_contour(mask)
        assert contour is None

    def test_largest_contour_returns_none_for_tiny(self):
        """_largest_contour should return None for tiny contours (<10 px)."""
        from app.segmentation import LogAnalyzer

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)  # tiny speck
        contour = LogAnalyzer._largest_contour(mask)
        assert contour is None


# ---------------------------------------------------------------------------
# Geometry — ellipse fitting (NEW primary tests)
# ---------------------------------------------------------------------------

class TestEllipseFitting:
    def test_length_and_breadth_for_circle(self):
        """A circle should give length ≈ breadth ≈ 2 × radius."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert abs(res["length_px"] - 80) < 6
        assert abs(res["breadth_px"] - 80) < 6

    def test_length_ge_breadth(self):
        """length_px must always be ≥ breadth_px (major ≥ minor axis)."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        # Elongated ellipse
        img = make_ellipse_image(300, 200, 150, 100, (80, 25))
        det = make_detection((65, 70, 235, 130))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["length_px"] >= res["breadth_px"]

    def test_length_matches_major_axis_for_ellipse(self):
        """length_px should ≈ 2 × major axis for a drawn ellipse."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        # axes=(80, 25) → major=80, so length_px ≈ 160
        img = make_ellipse_image(300, 200, 150, 100, (80, 25))
        det = make_detection((65, 70, 235, 130))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert abs(res["length_px"] - 160) < 10
        assert abs(res["breadth_px"] - 50) < 8

    def test_ellipse_angle_is_none_in_fallback(self):
        """When no wood contour is found, ellipse_angle should be None."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        det = make_detection((50, 50, 150, 150))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["ellipse_angle"] is None


# ---------------------------------------------------------------------------
# Geometry — circle shapes (compatibility)
# ---------------------------------------------------------------------------

class TestCircleGeometry:
    def test_center_for_circular_log(self):
        """A perfect circle should yield center ≈ known."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert abs(res["center"]["x"] - 100) < 5
        assert abs(res["center"]["y"] - 100) < 5

    def test_circularity_near_1_for_circle(self):
        """A perfect circle should have circularity near 1.0."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(250, 250, 125, 125, 60)
        det = make_detection((60, 60, 190, 190))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["circularity"] > 0.85

    def test_area_ratio_near_1_for_circle(self):
        """A perfect circle should have area_ratio near 1.0."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(250, 250, 125, 125, 60)
        det = make_detection((60, 60, 190, 190))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["area_ratio"] > 0.7

    def test_contour_area_positive(self):
        """Contour area should be a positive number for a real shape."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["contour_area"] > 100


# ---------------------------------------------------------------------------
# Geometry — elongated shapes
# ---------------------------------------------------------------------------

class TestEllipseGeometry:
    def test_elongated_shape_has_low_circularity(self):
        """An elongated ellipse should have circularity < 0.85."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        # Axes (80, 25) — very elongated
        img = make_ellipse_image(300, 200, 150, 100, (80, 25))
        det = make_detection((65, 70, 235, 130))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["circularity"] < 0.85

    def test_length_much_greater_than_breadth(self):
        """An elongated ellipse should have length >> breadth."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_ellipse_image(300, 200, 150, 100, (80, 25))
        det = make_detection((65, 70, 235, 130))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["length_px"] > res["breadth_px"] * 2


# ---------------------------------------------------------------------------
# Multiple logs
# ---------------------------------------------------------------------------

class TestMultipleLogs:
    def test_multiple_logs_get_sequential_ids(self):
        """Each log should get an incrementing ID."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = np.full((400, 400, 3), (60, 70, 55), dtype=np.uint8)
        cv2.circle(img, (100, 100), 30, (86, 58, 28), -1)
        cv2.circle(img, (250, 200), 40, (86, 58, 28), -1)
        cv2.circle(img, (300, 350), 25, (86, 58, 28), -1)

        dets = [
            make_detection((65, 65, 135, 135)),
            make_detection((205, 155, 295, 245)),
            make_detection((270, 320, 330, 380)),
        ]

        results, _ = analyzer.analyze(img, dets)

        assert len(results) == 3
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[2]["id"] == 3

    def test_empty_detections(self):
        """An empty detection list should return empty results."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer()

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        results, annotated = analyzer.analyze(img, [])

        assert results == []
        np.testing.assert_array_equal(img, annotated)

    def test_annotated_image_differs_from_original(self):
        """Annotated image should differ when ellipses are drawn."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        _, annotated = analyzer.analyze(img, [det])
        assert not np.array_equal(img, annotated)


# ---------------------------------------------------------------------------
# JSON serialization structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_has_all_required_keys(self):
        """Each result dict must contain all required keys per spec."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        required_keys = {
            "id", "class_name", "confidence", "bbox",
            "center", "radius", "length_px", "breadth_px",
            "contour_area", "ellipse_area", "circularity", "area_ratio",
        }
        assert required_keys.issubset(set(res.keys()))

    def test_result_does_not_contain_ellipse_box(self):
        """The raw ellipse_box (cv2 tuple) must NOT leak into the JSON dict."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        assert "ellipse_box" not in results[0]

    def test_center_has_x_and_y(self):
        """center should be a dict with x and y."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert "x" in res["center"]
        assert "y" in res["center"]
        assert isinstance(res["center"]["x"], (int, float))
        assert isinstance(res["center"]["y"], (int, float))

    def test_bbox_preserved_from_detection(self):
        """The original bbox should be preserved in the result."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["bbox"] == {"x1": 55, "y1": 55, "x2": 145, "y2": 145}

    def test_json_serializable(self):
        """Results should be JSON serializable."""
        import json
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = make_circle_image(200, 200, 100, 100, 40)
        det = make_detection((55, 55, 145, 145))

        results, _ = analyzer.analyze(img, [det])

        json_str = json.dumps(results)
        decoded = json.loads(json_str)
        assert len(decoded) == 1
        assert decoded[0]["id"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_wood_pixels_uses_bbox_fallback(self):
        """When no wood-coloured pixels found, fall back to bbox-based circle."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        # All-black image — no wood tones
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        det = make_detection((50, 50, 150, 150))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        # Should still produce valid numbers
        assert res["length_px"] > 0
        assert res["breadth_px"] > 0
        assert res["ellipse_area"] > 0
        # Fallback circularity is 0 since no contour found
        assert res["circularity"] == 0.0

    def test_fallback_length_equals_breadth(self):
        """Fallback circle should give length == breadth."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        det = make_detection((50, 50, 150, 150))

        results, _ = analyzer.analyze(img, [det])
        res = results[0]

        assert res["length_px"] == res["breadth_px"]

    def test_roi_padding_respected(self):
        """Analyzer should use the configured pad value."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=15)
        assert analyzer.pad == 15

    def test_results_list_length_matches_detections(self):
        """One result per detection."""
        from app.segmentation import LogAnalyzer
        analyzer = LogAnalyzer(pad=2)

        img = np.full((300, 300, 3), (60, 70, 55), dtype=np.uint8)
        cv2.circle(img, (80, 80), 25, (86, 58, 28), -1)
        cv2.circle(img, (200, 200), 30, (86, 58, 28), -1)

        dets = [
            make_detection((50, 50, 110, 110)),
            make_detection((165, 165, 235, 235)),
        ]

        results, _ = analyzer.analyze(img, dets)
        assert len(results) == len(dets)
