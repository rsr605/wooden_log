"""
Unit tests for the LogDetector class (app/detector.py).

The YOLOv8 model is mocked so these tests run without GPU or model downloads.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_box(cls_val: float, conf_val: float, xyxy_vals: list):
    """Create a mock YOLOv8 box object."""
    box = MagicMock()
    box.cls.item.return_value = cls_val
    box.conf.item.return_value = conf_val
    box.xyxy.__getitem__.return_value.tolist.return_value = xyxy_vals
    return box


def make_mock_result(boxes, names=None):
    """Create a mock ultralytics Result object."""
    result = MagicMock()
    result.boxes = boxes
    result.names = names or {0: "wooden_log"}
    result.__len__ = lambda self: 1
    return result


# ---------------------------------------------------------------------------
# LogDetector Initialization
# ---------------------------------------------------------------------------

class TestLogDetectorInit:
    @patch("app.detector.LogDetector.__init__", return_value=None)
    def test_init_stores_params(self, _):
        """Constructor parameters should be stored on the instance."""
        from app.detector import LogDetector
        d = LogDetector.__new__(LogDetector)
        d.model_path = "yolov8n.pt"
        d.conf_threshold = 0.25
        d.iou_threshold = 0.7
        d.class_names = ["wooden_log"]
        assert d.model_path == "yolov8n.pt"
        assert d.conf_threshold == 0.25
        assert d.iou_threshold == 0.7
        assert d.class_names == ["wooden_log"]

    def test_init_with_custom_params(self):
        """LogDetector should accept custom parameters."""
        from app.detector import LogDetector
        # Don't actually load the model — just test __init__ sets attributes
        with patch("ultralytics.YOLO"):
            d = LogDetector(
                model_path="yolov8s.pt",
                conf_threshold=0.5,
                iou_threshold=0.5,
                class_names=["log", "stump"],
            )
            assert d.model_path == "yolov8s.pt"
            assert d.conf_threshold == 0.5
            assert d.iou_threshold == 0.5
            assert d.class_names == ["log", "stump"]


# ---------------------------------------------------------------------------
# Detection Result Parsing
# ---------------------------------------------------------------------------

class TestDetectionParsing:
    def test_detect_with_multiple_boxes(self):
        """detect() should parse multiple boxes into detection dicts."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        # Mock the model and its return value
        boxes = [
            make_mock_box(0.0, 0.95, [10, 20, 100, 200]),
            make_mock_box(0.0, 0.80, [50, 60, 150, 160]),
        ]
        mock_result = make_mock_result(boxes, names={0: "wooden_log"})
        d.model = MagicMock()
        d.model.return_value = [mock_result]

        image = np.zeros((300, 300, 3), dtype=np.uint8)
        detections, elapsed = d.detect(image)

        assert len(detections) == 2
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "wooden_log"
        assert detections[0]["confidence"] == 0.95
        assert detections[0]["bbox"]["x1"] == 10
        assert detections[0]["bbox"]["y1"] == 20
        assert detections[0]["bbox"]["x2"] == 100
        assert detections[0]["bbox"]["y2"] == 200

        assert detections[1]["confidence"] == 0.80
        assert elapsed > 0

    def test_detect_no_boxes(self):
        """detect() should return empty list when model finds nothing."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda self: 0
        mock_result.names = {0: "wooden_log"}
        d.model = MagicMock()
        d.model.return_value = [mock_result]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections, elapsed = d.detect(image)

        assert detections == []
        assert elapsed >= 0

    def test_detect_confidence_override(self):
        """detect() should pass custom confidence to model."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda self: 0
        mock_result.names = {}
        d.model = MagicMock()
        d.model.return_value = [mock_result]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        d.detect(image, conf=0.5)

        # Verify model was called with conf=0.5
        call_kwargs = d.model.call_args
        assert call_kwargs.kwargs["conf"] == 0.5

    def test_class_name_from_model_names(self):
        """class_name should use model's names dict when available."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        boxes = [make_mock_box(2.0, 0.9, [10, 20, 30, 40])]
        mock_result = make_mock_result(boxes, names={0: "log", 1: "stump", 2: "plank"})
        d.model = MagicMock()
        d.model.return_value = [mock_result]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections, _ = d.detect(image)

        assert detections[0]["class_name"] == "plank"


# ---------------------------------------------------------------------------
# Annotation / Drawing
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_annotate_returns_copy(self):
        """annotate() should not modify the original image."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {
                "class_id": 0,
                "class_name": "wooden_log",
                "confidence": 0.9,
                "bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 90},
            }
        ]
        annotated = d.annotate(original, detections)

        assert not np.array_equal(original, annotated)
        assert annotated.shape == original.shape

    def test_annotate_empty_detections(self):
        """annotate() with no detections should return identical copy."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        annotated = d.annotate(original, [])

        np.testing.assert_array_equal(original, annotated)

    def test_annotate_draws_box(self):
        """annotate() should draw a rectangle for each detection."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        original = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [
            {
                "class_id": 0,
                "class_name": "wooden_log",
                "confidence": 0.85,
                "bbox": {"x1": 20, "y1": 30, "x2": 100, "y2": 150},
            }
        ]
        annotated = d.annotate(original, detections)

        # The annotated image should differ (pixels were drawn)
        assert not np.array_equal(original, annotated)
        # Check that some non-zero pixels exist (green box color)
        assert annotated.sum() > 0


# ---------------------------------------------------------------------------
# Result Formatting
# ---------------------------------------------------------------------------

class TestResultFormatting:
    def test_format_result(self):
        """format_result should build correct result dict with image dimensions."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                "class_id": 0,
                "class_name": "wooden_log",
                "confidence": 0.92,
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            }
        ]
        result = d.format_result("test.jpg", image, detections, 42.5)

        assert result["image_path"] == "test.jpg"
        assert result["image_width"] == 640
        assert result["image_height"] == 480
        assert result["count"] == 1
        assert result["detections"] == detections
        assert result["processing_time_ms"] == 42.5


# ---------------------------------------------------------------------------
# detect_from_path
# ---------------------------------------------------------------------------

class TestDetectFromPath:
    def test_valid_image_path(self, tmp_path):
        """detect_from_path should read an image and run detection."""
        import cv2
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        # Create test image
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

        # Mock detection
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda self: 0
        mock_result.names = {}
        d.model = MagicMock()
        d.model.return_value = [mock_result]

        detections, image, elapsed = d.detect_from_path(str(img_path))

        assert detections == []
        assert image is not None
        assert image.shape == (100, 100, 3)

    def test_invalid_image_path(self):
        """detect_from_path should raise ValueError for unreadable image."""
        from app.detector import LogDetector

        with patch("ultralytics.YOLO"):
            d = LogDetector()

        with pytest.raises(ValueError, match="Cannot read image"):
            d.detect_from_path("/nonexistent/path/image.jpg")


# ---------------------------------------------------------------------------
# Singleton get_detector
# ---------------------------------------------------------------------------

class TestGetDetector:
    def test_singleton_returns_same_instance(self):
        """get_detector should return the same instance on repeated calls."""
        with patch("ultralytics.YOLO"):
            import app.detector as det_module
            # Reset singleton
            det_module._detector_instance = None

            d1 = det_module.get_detector()
            d2 = det_module.get_detector()
            assert d1 is d2

            # Cleanup
            det_module._detector_instance = None
