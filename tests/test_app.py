"""
Integration tests for the Flask application (app/main.py).

These tests use the Flask test client without starting a real server.
The YOLOv8 model is mocked so tests run without GPU/model dependencies.
"""

import sys
import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def flask_app():
    """Create a Flask app instance for testing."""
    from app.main import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(flask_app):
    """Flask test client."""
    return flask_app.test_client()


@pytest.fixture
def sample_jpeg_bytes():
    """Generate a small JPEG image as bytes."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [100, 60, 30]  # brown-ish rectangle
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Basic Route Tests
# ---------------------------------------------------------------------------

class TestBasicRoutes:
    def test_home_page(self, client):
        """GET / should return 200 and contain the upload form."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Wooden Log Detection" in response.data
        assert b"Detect" in response.data

    def test_health_check(self, client):
        """GET /health should return JSON status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "wooden-log-detection"


# ---------------------------------------------------------------------------
# Upload / Detect Tests (with mocked detector)
# ---------------------------------------------------------------------------

class TestImageUpload:
    def test_detect_no_file(self, client):
        """POST /detect without a file should redirect with flash."""
        response = client.post("/detect", follow_redirects=True)
        assert response.status_code == 200
        # Should show error flash or redirect to form
        assert b"Wooden Log Detection" in response.data

    def test_detect_empty_filename(self, client):
        """POST /detect with empty filename should redirect."""
        data = {"file": (io.BytesIO(b""), "")}
        response = client.post(
            "/detect",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200

    def test_detect_invalid_extension(self, client):
        """POST /detect with .txt should redirect with error."""
        data = {"file": (io.BytesIO(b"hello"), "test.txt")}
        response = client.post(
            "/detect",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        # Should contain error about unsupported type
        assert b"Unsupported" in response.data or b"error" in response.data.lower()

    @patch("app.detector.get_detector")
    def test_detect_valid_image(self, mock_get_detector, client, sample_jpeg_bytes):
        """POST /detect with a valid JPEG should return results page."""
        # Mock the detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (
            [
                {
                    "class_id": 0,
                    "class_name": "wooden_log",
                    "confidence": 0.92,
                    "bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 90},
                    "aspect_ratio": 1.06,
                    "diameter_px": 70,
                }
            ],
            45.5,  # processing time ms
        )
        mock_detector.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_get_detector.return_value = mock_detector

        data = {"file": (io.BytesIO(sample_jpeg_bytes), "test.jpg")}
        response = client.post(
            "/detect",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert b"Detection Results" in response.data
        assert b"wooden_log" in response.data
        assert b"0.92" in response.data

    @patch("app.detector.get_detector")
    def test_detect_no_detections(self, mock_get_detector, client, sample_jpeg_bytes):
        """POST /detect with no detections should show 'No wooden logs detected'."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], 10.0)
        mock_detector.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_get_detector.return_value = mock_detector

        data = {"file": (io.BytesIO(sample_jpeg_bytes), "test.jpg")}
        response = client.post(
            "/detect",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert b"No wooden logs detected" in response.data


# ---------------------------------------------------------------------------
# API Tests (with mocked detector)
# ---------------------------------------------------------------------------

class TestAPI:
    @patch("app.detector.get_detector")
    def test_api_detect_success(self, mock_get_detector, client, sample_jpeg_bytes):
        """POST /api/detect should return JSON with detections."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (
            [
                {
                    "class_id": 0,
                    "class_name": "wooden_log",
                    "confidence": 0.85,
                    "bbox": {"x1": 5, "y1": 10, "x2": 50, "y2": 60},
                }
            ],
            30.0,
        )
        mock_get_detector.return_value = mock_detector

        data = {"file": (io.BytesIO(sample_jpeg_bytes), "test.jpg")}
        response = client.post(
            "/api/detect",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["count"] == 1
        assert result["detections"][0]["class_name"] == "wooden_log"
        assert result["detections"][0]["confidence"] == 0.85
        assert "processing_time_ms" in result

    def test_api_detect_no_file(self, client):
        """POST /api/detect without file should return 400."""
        response = client.post("/api/detect")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    @patch("app.detector.get_detector")
    def test_api_info(self, mock_get_detector, client):
        """GET /api/info should return model configuration."""
        mock_detector = MagicMock()
        mock_detector.model_path = "yolov8n.pt"
        mock_detector.conf_threshold = 0.25
        mock_detector.iou_threshold = 0.7
        mock_detector.class_names = ["wooden_log"]
        mock_get_detector.return_value = mock_detector

        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_path"] == "yolov8n.pt"
        assert data["confidence_threshold"] == 0.25
        assert "wooden_log" in data["class_names"]
