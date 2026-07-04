"""
Tests for the YOLOv8 detector module.

Covers:
- Weighted Box Fusion (WBF) algorithm correctness
- IoU matrix computation
- LogDetector initialization and lazy model loading
- Multi-scale detect() returns well-formed detection dicts
- annotate() draws bounding boxes and returns same-shape image
- Default thresholds are correctly tuned for high-recall detection
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# WBF unit tests (pure functions, no model needed)
# ---------------------------------------------------------------------------

class TestWeightedBoxFusion:
    """Unit tests for the Weighted Box Fusion algorithm."""

    def test_empty_input(self):
        """WBF on empty input returns empty arrays."""
        from app.detector import weighted_box_fusion

        boxes, scores, classes = weighted_box_fusion(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(classes) == 0

    def test_single_box_passthrough(self):
        """A single box passes through unchanged."""
        from app.detector import weighted_box_fusion

        boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        classes = np.array([0], dtype=np.int32)

        f_boxes, f_scores, f_classes = weighted_box_fusion(boxes, scores, classes)

        assert len(f_boxes) == 1
        assert np.allclose(f_boxes[0], boxes[0])
        assert f_scores[0] == pytest.approx(0.9)
        assert f_classes[0] == 0

    def test_overlapping_boxes_merged(self):
        """Two highly-overlapping boxes should be merged into one."""
        from app.detector import weighted_box_fusion

        boxes = np.array([
            [10, 10, 110, 110],
            [15, 15, 105, 105],  # IoU with first ~0.83
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

        f_boxes, f_scores, f_classes = weighted_box_fusion(
            boxes, scores, classes, iou_thr=0.5
        )

        assert len(f_boxes) == 1, "Two overlapping boxes should merge into 1"

    def test_non_overlapping_boxes_kept(self):
        """Non-overlapping boxes should all be preserved."""
        from app.detector import weighted_box_fusion

        boxes = np.array([
            [0, 0, 50, 50],
            [200, 200, 250, 250],
            [400, 400, 450, 450],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.int32)

        f_boxes, f_scores, f_classes = weighted_box_fusion(
            boxes, scores, classes, iou_thr=0.5
        )

        assert len(f_boxes) == 3, "Non-overlapping boxes should all be kept"

    def test_fused_score_is_average(self):
        """The fused score should be the mean of merged member scores."""
        from app.detector import weighted_box_fusion

        boxes = np.array([
            [10, 10, 110, 110],
            [12, 12, 108, 108],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

        _, f_scores, _ = weighted_box_fusion(boxes, scores, classes, iou_thr=0.5)

        assert f_scores[0] == pytest.approx(0.8, abs=0.01)

    def test_fused_box_is_weighted_average(self):
        """The fused box coordinates should be confidence-weighted average."""
        from app.detector import weighted_box_fusion

        boxes = np.array([
            [10, 10, 110, 110],
            [10, 10, 110, 110],  # identical → average is same
        ], dtype=np.float32)
        scores = np.array([0.9, 0.9], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

        f_boxes, _, _ = weighted_box_fusion(boxes, scores, classes, iou_thr=0.5)

        assert np.allclose(f_boxes[0], [10, 10, 110, 110])

    def test_sorting_by_confidence(self):
        """Output should preserve a reasonable order (highest confidence first)."""
        from app.detector import weighted_box_fusion

        boxes = np.array([
            [0, 0, 50, 50],
            [200, 200, 250, 250],
            [400, 400, 450, 450],
        ], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.int32)

        f_boxes, f_scores, _ = weighted_box_fusion(boxes, scores, classes, iou_thr=0.5)

        assert f_scores[0] == pytest.approx(0.9), "Highest score should come first"


class TestIoUMatrix:
    """Unit tests for IoU matrix computation."""

    def test_identical_boxes_iou_one(self):
        """IoU of identical boxes should be 1.0."""
        from app.detector import _iou_matrix

        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        iou = _iou_matrix(boxes, boxes)
        assert iou[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_non_overlapping_iou_zero(self):
        """IoU of non-overlapping boxes should be 0.0."""
        from app.detector import _iou_matrix

        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[100, 100, 110, 110]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        from app.detector import _iou_matrix

        a = np.array([[0, 0, 20, 20]], dtype=np.float32)
        b = np.array([[10, 10, 30, 30]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        assert 0.0 < iou[0, 0] < 1.0

    def test_empty_input(self):
        """Empty input should return empty matrix."""
        from app.detector import _iou_matrix

        empty = np.zeros((0, 4), dtype=np.float32)
        iou = _iou_matrix(empty, empty)
        assert iou.shape == (0, 0)


# ---------------------------------------------------------------------------
# Detector initialization tests
# ---------------------------------------------------------------------------

class TestDetectorInit:
    """Test LogDetector initialization (without loading the actual model)."""

    def test_detector_init_attributes(self):
        """Detector stores config without loading model."""
        from app.detector import LogDetector

        detector = LogDetector(
            model_path="dummy.pt",
            conf_threshold=0.25,
            iou_threshold=0.50,
        )
        assert detector.model_path == "dummy.pt"
        assert detector.conf_threshold == 0.25
        assert detector.iou_threshold == 0.50
        assert detector.model is None  # lazy loading

    def test_detector_default_thresholds(self):
        """Detector uses high-recall thresholds by default."""
        from app.detector import LogDetector

        detector = LogDetector(model_path="dummy.pt")
        assert detector.conf_threshold == 0.25, "Conf threshold should be 0.25 for recall"
        assert detector.iou_threshold == 0.50, "IoU threshold should be 0.50"

    def test_infer_scales_defined(self):
        """Detector has multi-scale inference configured."""
        from app.detector import INFER_SCALES

        assert isinstance(INFER_SCALES, list)
        assert len(INFER_SCALES) >= 2, "Multi-scale requires at least 2 scales"
        for s in INFER_SCALES:
            assert s % 32 == 0, "Inference size must be multiple of 32"


# ---------------------------------------------------------------------------
# Integration tests (require actual model)
# ---------------------------------------------------------------------------

class TestDetectorInference:
    """Integration tests that actually load the model and run inference."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Load the real model for integration tests."""
        from app.detector import LogDetector
        model_path = str(Path("models/wooden_log_best.pt"))
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return LogDetector(model_path=model_path, conf_threshold=0.25, iou_threshold=0.50)

    def test_detect_returns_tuple(self, detector):
        """detect() returns (list, float)."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections, elapsed = detector.detect(img)
        assert isinstance(detections, list)
        assert isinstance(elapsed, float)
        assert elapsed > 0

    def test_detection_structure(self, detector):
        """Each detection has all required keys with correct types."""
        img = np.full((480, 640, 3), 60, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (33, 67, 101), -1)
        detections, _ = detector.detect(img)
        for det in detections:
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert "aspect_ratio" in det
            assert "diameter_px" in det
            assert isinstance(det["bbox"], dict)
            assert all(k in det["bbox"] for k in ["x1", "y1", "x2", "y2"])
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detect_on_sample_image(self, detector):
        """Detector finds multiple logs in a sample image."""
        sample_path = Path("data_v3/sample_images/sample_0.jpg")
        if not sample_path.exists():
            pytest.skip("Sample image not found")
        img = cv2.imread(str(sample_path))
        detections, _ = detector.detect(img)
        assert len(detections) > 0, "Should detect at least one log"

    def test_detect_on_blank_image_no_false_positives(self, detector):
        """A blank/near-blank image should produce zero or very few detections."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections, _ = detector.detect(img)
        assert len(detections) == 0, f"Expected 0 detections on black image, got {len(detections)}"

    def test_detect_on_real_image(self, detector):
        """Multi-scale detection finds logs on real-world images."""
        test_path = Path("test_end_view.png")
        if not test_path.exists():
            pytest.skip("Real test image not found")
        img = cv2.imread(str(test_path))
        detections, _ = detector.detect(img)
        assert len(detections) >= 5, f"Should detect multiple logs, got {len(detections)}"

    def test_annotate_preserves_shape(self, detector):
        """annotate() returns an image with the same dimensions."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{
            "class_id": 0,
            "class_name": "wooden_log",
            "confidence": 0.95,
            "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
            "aspect_ratio": 1.0,
            "diameter_px": 100,
        }]
        annotated = detector.annotate(img, detections)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == img.shape
