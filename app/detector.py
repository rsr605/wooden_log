"""
YOLOv8 Detection Wrapper for Wooden Log Detection.

Provides the LogDetector class that wraps the ultralytics YOLOv8 model,
runs inference on images and video, and draws annotated results with OpenCV.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np

# Ensure src is importable when running from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    Timer,
    get_color_for_class,
    format_detection_result,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_VARIANT,
    CLASS_NAMES,
)


class LogDetector:
    """
    YOLOv8-based wooden log detector.

    Wraps the ultralytics YOLO model. Handles model loading, inference,
    and OpenCV annotation of bounding boxes.

    Attributes:
        model: The loaded YOLOv8 model.
        model_path: Path to the model weights file.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        class_names: List of class names the model can detect.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_VARIANT,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path or name of YOLOv8 weights (e.g. 'yolov8n.pt').
            conf_threshold: Minimum confidence for a detection to be kept.
            iou_threshold: IoU threshold for NMS.
            class_names: Override class names; defaults to CLASS_NAMES from utils.
        """
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or CLASS_NAMES
        self._yolo_cls = YOLO  # store class reference for lazy loading
        self.model = None

    def load_model(self) -> None:
        """Load the YOLOv8 model (lazy loading)."""
        if self.model is None:
            self.model = self._yolo_cls(self.model_path)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run detection on a single image (BGR numpy array).

        Args:
            image: Input image as a BGR numpy array (OpenCV format).
            conf: Override confidence threshold.
            iou: Override IoU threshold.

        Returns:
            Tuple of (detections_list, processing_time_ms).
            Each detection dict has keys: class_id, class_name, confidence, bbox.
        """
        self.load_model()

        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold

        with Timer() as t:
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                verbose=False,
            )

        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Map class ID to name
                    if hasattr(result, "names") and result.names:
                        class_name = result.names.get(class_id, f"class_{class_id}")
                    elif class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        },
                    })

        return detections, round(t.elapsed_ms, 2)

    def detect_from_path(self, image_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
        """
        Load an image from disk and run detection.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (detections_list, original_image, processing_time_ms).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        detections, elapsed = self.detect(image)
        return detections, image, elapsed

    def format_result(
        self,
        image_path: str,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        processing_time_ms: float,
    ) -> Dict[str, Any]:
        """Build a standardized result dict from detection output."""
        h, w = image.shape[:2]
        return format_detection_result(image_path, w, h, detections, processing_time_ms)

    # ------------------------------------------------------------------
    # Annotation / Drawing
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        box_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.

        Args:
            image: Original BGR image.
            detections: Detection list from detect().
            box_thickness: Line thickness for bounding boxes.
            font_scale: Font scale for labels.

        Returns:
            Annotated image (new copy; original is not modified).
        """
        annotated = image.copy()

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = get_color_for_class(class_id)

            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Label text
            label = f"{class_name} {confidence:.2f}"

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_y = max(y1, th + 4)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color,
                -1,  # filled
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # white text
                1,
                cv2.LINE_AA,
            )

        return annotated

    def detect_and_annotate(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convenience: run detection + annotation in one call.

        Args:
            image: Input BGR image.
            save_path: If provided, save the annotated image to this path.

        Returns:
            Tuple of (annotated_image, result_dict).
        """
        detections, elapsed = self.detect(image)
        annotated = self.annotate(image, detections)

        h, w = image.shape[:2]
        result = format_detection_result("", w, h, detections, elapsed)

        if save_path:
            cv2.imwrite(save_path, annotated)

        return annotated, result

    # ------------------------------------------------------------------
    # Video Processing
    # ------------------------------------------------------------------

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run detection on a video file frame by frame.

        Args:
            video_path: Path to the input video.
            output_path: If provided, write annotated video to this path.
            max_frames: Maximum number of frames to process (None = all).

        Returns:
            Summary dict with frame count, detections, avg processing time, etc.
        """
        self.load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections = 0
        total_time_ms = 0.0
        frames_processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frames_processed >= max_frames:
                    break

                detections, elapsed = self.detect(frame)
                annotated = self.annotate(frame, detections)
                total_detections += len(detections)
                total_time_ms += elapsed
                frames_processed += 1

                if writer:
                    writer.write(annotated)
        finally:
            cap.release()
            if writer:
                writer.release()

        return {
            "video_path": video_path,
            "output_path": output_path,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "avg_processing_time_ms": (
                round(total_time_ms / frames_processed, 2) if frames_processed else 0.0
            ),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialized for Flask)
# ---------------------------------------------------------------------------

_detector_instance: Optional[LogDetector] = None


def get_detector() -> LogDetector:
    """Return a module-level singleton detector (created on first use)."""
    global _detector_instance
    if _detector_instance is None:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_VARIANT)
        conf = float(os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))
        iou = float(os.environ.get("IOU_THRESHOLD", DEFAULT_IOU_THRESHOLD))
        _detector_instance = LogDetector(
            model_path=model_path,
            conf_threshold=conf,
            iou_threshold=iou,
        )
    return _detector_instance
