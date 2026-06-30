"""
Flask Web Application for Wooden Log Detection.

Routes:
  GET  /            → Upload form (index)
  POST /detect      → Upload + detect image or video, display annotated result
  POST /api/detect  → API endpoint (POST multipart, JSON response)
  GET  /api/info    → Model configuration info (JSON)
  GET  /health      → Health check (JSON)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    flash,
    abort,
)
from werkzeug.utils import secure_filename

# Ensure both app/ and src/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    ensure_dir,
    is_allowed_image,
    is_allowed_video,
    generate_hashed_filename,
    MAX_UPLOAD_SIZE,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER", str(BASE_DIR / "static" / "uploads")
)
RESULTS_FOLDER = os.environ.get(
    "RESULTS_FOLDER", str(BASE_DIR / "static" / "results")
)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "wooden-log-dev-secret-key")

    # Ensure directories exist
    ensure_dir(UPLOAD_FOLDER)
    ensure_dir(RESULTS_FOLDER)

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(file_storage, folder: str) -> str:
    """Validate and save an uploaded file. Returns the saved filename."""
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided")
    filename = secure_filename(file_storage.filename)
    if not filename:
        raise ValueError("Invalid filename")
    # Validate by extension
    if not (is_allowed_image(filename) or is_allowed_video(filename)):
        raise ValueError(
            f"Unsupported file type. Allowed: jpg, jpeg, png, bmp, webp, "
            f"mp4, avi, mov, mkv"
        )
    # Validate by content type (defense-in-depth against extension spoofing)
    content_type = (file_storage.content_type or "").lower()
    valid_types = {
        "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
        "video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska",
        "application/octet-stream",  # some browsers send this for unknown video
    }
    if content_type and content_type not in valid_types:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Please upload an image or video file."
        )
    hashed_name = generate_hashed_filename(filename, prefix="upload_")
    file_storage.save(os.path.join(folder, hashed_name))
    return hashed_name


def _get_result_stats(detections: list) -> dict:
    """Compute summary statistics from a detection list."""
    if not detections:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "class_distribution": {},
        }
    confidences = [d["confidence"] for d in detections]
    class_dist = {}
    for d in detections:
        name = d["class_name"]
        class_dist[name] = class_dist.get(name, 0) + 1
    return {
        "count": len(detections),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "class_distribution": class_dist,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        """Render the upload form."""
        return render_template(
            "index.html",
            classes=CLASS_NAMES,
        )

    @app.route("/detect", methods=["POST"])
    def detect_image():
        """Handle image upload → detection → result display."""
        if "file" not in request.files:
            flash("No file selected.", "error")
            return redirect(url_for("index"))

        file_storage = request.files["file"]
        try:
            filename = _save_upload(file_storage, app.config["UPLOAD_FOLDER"])
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # If it's a video, redirect to video handler
        if is_allowed_video(filename):
            return _handle_video_detection(app, filename, filepath)

        # Read image
        image = cv2.imread(filepath)
        if image is None:
            flash("Could not read image file. It may be corrupted.", "error")
            return redirect(url_for("index"))

        # Optional confidence override
        conf_override = None
        if "confidence" in request.form:
            try:
                conf_override = float(request.form["confidence"])
            except (ValueError, TypeError):
                pass

        # Detect
        from app.detector import get_detector

        detector = get_detector()
        detections, elapsed = detector.detect(image, conf=conf_override)
        annotated = detector.annotate(image, detections)

        # Save annotated result
        result_filename = generate_hashed_filename(filename, prefix="result_")
        result_path = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
        cv2.imwrite(result_path, annotated)

        stats = _get_result_stats(detections)
        h, w = image.shape[:2]

        return render_template(
            "result.html",
            original_image=f"uploads/{filename}",
            result_image=f"results/{result_filename}",
            detections=detections,
            stats=stats,
            processing_time_ms=round(elapsed, 2),
            image_width=w,
            image_height=h,
            is_video=False,
        )

    def _handle_video_detection(app: Flask, filename: str, filepath: str) -> str:
        """Process uploaded video and render result."""
        from app.detector import get_detector

        detector = get_detector()
        result_filename = generate_hashed_filename(filename, prefix="result_")
        output_path = os.path.join(app.config["RESULTS_FOLDER"], result_filename)

        # Limit to 200 frames for web to avoid long processing
        summary = detector.detect_video(filepath, output_path, max_frames=200)
        stats = {
            "count": summary["total_detections"],
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "class_distribution": {},
        }

        return render_template(
            "result.html",
            original_image=f"uploads/{filename}",
            result_image=f"results/{result_filename}",
            detections=[],
            stats=stats,
            processing_time_ms=summary["avg_processing_time_ms"],
            image_width=summary["width"],
            image_height=summary["height"],
            is_video=True,
            video_summary=summary,
        )

    @app.route("/api/detect", methods=["POST"])
    def api_detect():
        """Programmatic detection endpoint (JSON response)."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file_storage = request.files["file"]
        try:
            filename = _save_upload(file_storage, app.config["UPLOAD_FOLDER"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({"error": "Cannot read image"}), 400

        from app.detector import get_detector

        detector = get_detector()
        detections, elapsed = detector.detect(image)

        h, w = image.shape[:2]
        result = {
            "filename": filename,
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "count": len(detections),
            "processing_time_ms": round(elapsed, 2),
            "stats": _get_result_stats(detections),
        }
        return jsonify(result), 200

    @app.route("/api/info", methods=["GET"])
    def api_info():
        """Return model and configuration info."""
        from app.detector import get_detector

        detector = get_detector()
        return jsonify({
            "model_path": detector.model_path,
            "confidence_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "class_names": detector.class_names,
        }), 200

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "service": "wooden-log-detection"}), 200

    @app.route("/results/<path:filename>")
    def serve_result(filename: str):
        return send_from_directory(app.config["RESULTS_FOLDER"], filename)

    @app.errorhandler(413)
    def too_large(error):
        flash("File too large. Maximum size is 50 MB.", "error")
        return redirect(url_for("index")), 413

    @app.errorhandler(404)
    def not_found(error):
        return render_template("index.html", classes=CLASS_NAMES, error="Page not found"), 404


# ---------------------------------------------------------------------------
# Create the Flask app instance for gunicorn
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
