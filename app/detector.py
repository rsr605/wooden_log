"""
YOLOv8 Detection Wrapper for Wooden Log Detection.

Multi-scale inference with Weighted Box Fusion (WBF) for high-recall log
detection. The model is trained on synthetic wooden log images, so we
compensate for the domain gap to real photos by:

  1. Running inference at two scales (640 + 960) with TTA augmentation.
  2. Fusing all scale predictions via Weighted Box Fusion, which averages
     overlapping boxes by confidence rather than greedily suppressing them.
  3. A final NMS pass to clean up any residual near-duplicates.

This achieves significantly higher recall on real-world images while keeping
false positives low.
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
    compute_aspect_ratio,
    compute_diameter,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_VARIANT,
    CLASS_NAMES,
)


# ---------------------------------------------------------------------------
# Inference scales — multiple resolutions for multi-scale fusion
# ---------------------------------------------------------------------------

INFER_SCALES = [640, 960]


# ---------------------------------------------------------------------------
# Weighted Box Fusion (WBF)
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of [x1, y1, x2, y2] boxes.

    Args:
        boxes_a: (N, 4) array.
        boxes_b: (M, 4) array.

    Returns:
        (N, M) IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand for broadcasting
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def weighted_box_fusion(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thr: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse overlapping detection boxes via Weighted Box Fusion.

    Boxes that overlap above ``iou_thr`` are grouped, and each group is
    replaced by a single box whose coordinates are the confidence-weighted
    average of its members. The fused confidence is the arithmetic mean
    of the member confidences.

    Args:
        boxes:    (N, 4) array of [x1, y1, x2, y2].
        scores:   (N,) confidence scores.
        class_ids:(N,) integer class IDs.
        iou_thr:  IoU threshold for grouping.

    Returns:
        Tuple of (fused_boxes, fused_scores, fused_class_ids).
    """
    n = len(boxes)
    if n == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
    if n == 1:
        return boxes.copy(), scores.copy(), class_ids.copy()

    # Sort by score descending so highest-confidence boxes are clustered first
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    class_ids = class_ids[order]

    fused_boxes: List[np.ndarray] = []
    fused_scores: List[float] = []
    fused_classes: List[int] = []

    used = np.zeros(n, dtype=bool)

    for i in range(n):
        if used[i]:
            continue

        # Start a new cluster with box i
        cluster_boxes = [boxes[i]]
        cluster_scores = [scores[i]]
        cluster_classes = [class_ids[i]]
        used[i] = True

        # Find all remaining boxes that overlap with this cluster
        for j in range(i + 1, n):
            if used[j]:
                continue

            # Check IoU against all current cluster members
            ious = _iou_matrix(
                boxes[j:j + 1],
                np.array(cluster_boxes),
            )
            if ious.max() >= iou_thr:
                cluster_boxes.append(boxes[j])
                cluster_scores.append(scores[j])
                cluster_classes.append(class_ids[j])
                used[j] = True

        # Weighted average of the cluster
        weights = np.array(cluster_scores, dtype=np.float32)
        weight_sum = weights.sum()

        if weight_sum > 0:
            w = weights / weight_sum
            fused_box = np.sum(
                np.array(cluster_boxes, dtype=np.float32) * w[:, None], axis=0
            )
        else:
            fused_box = np.mean(cluster_boxes, axis=0)

        # Fused confidence: arithmetic mean of member scores.
        fused_score = float(np.mean(cluster_scores))
        fused_class = int(cluster_classes[np.argmax(cluster_scores)])

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_classes.append(fused_class)

    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_classes, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# LogDetector
# ---------------------------------------------------------------------------

class LogDetector:
    """
    YOLOv8-based wooden log detector with multi-scale inference + WBF.

    Wraps the ultralytics YOLO model. Runs inference at two image scales
    (640 and 960) to catch both large and small logs, then fuses the
    results with Weighted Box Fusion to eliminate duplicates while
    preserving true positives.
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
            model_path: Path or name of YOLOv8 weights.
            conf_threshold: Minimum confidence for a detection to be kept.
            iou_threshold: IoU threshold for NMS and WBF grouping.
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
    # Core inference (multi-scale + WBF)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run multi-scale YOLOv8 detection with Weighted Box Fusion.

        Runs inference at both 640px and 960px (with TTA), then fuses
        overlapping detections via WBF to maximize recall while avoiding
        duplicate boxes.

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
            all_boxes: List[np.ndarray] = []
            all_scores: List[float] = []
            all_classes: List[int] = []

            for scale in INFER_SCALES:
                results = self.model(
                    image,
                    conf=conf,
                    iou=iou,
                    imgsz=scale,
                    max_det=300,
                    verbose=False,
                    augment=True,  # TTA for better recall
                )
                r = results[0] if results else None
                if r is not None and r.boxes is not None and len(r.boxes) > 0:
                    all_boxes.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
                    all_scores.extend(r.boxes.conf.cpu().numpy().astype(np.float32).tolist())
                    all_classes.extend(r.boxes.cls.cpu().numpy().astype(np.int32).tolist())

            # Fuse multi-scale results via WBF
            if all_boxes:
                merged_boxes = np.vstack(all_boxes)
                merged_scores = np.array(all_scores, dtype=np.float32)
                merged_classes = np.array(all_classes, dtype=np.int32)

                fused_boxes, fused_scores, fused_classes = weighted_box_fusion(
                    merged_boxes,
                    merged_scores,
                    merged_classes,
                    iou_thr=iou,
                )

                # Second-pass NMS to remove any residual near-duplicate boxes
                # that WBF didn't merge (can happen with boxes at IoU ~0.3-0.5).
                # Use a slightly lower threshold than WBF to catch marginal
                # overlaps without over-suppressing genuine adjacent logs.
                if len(fused_boxes) > 1:
                    import torch
                    from torchvision.ops import nms as tv_nms
                    nms_thr = max(0.35, iou - 0.1)
                    t_boxes = torch.from_numpy(fused_boxes)
                    t_scores = torch.from_numpy(fused_scores)
                    keep = tv_nms(t_boxes, t_scores, iou_threshold=nms_thr)
                    keep_idx = keep.cpu().numpy()
                    fused_boxes = fused_boxes[keep_idx]
                    fused_scores = fused_scores[keep_idx]
                    fused_classes = fused_classes[keep_idx]
            else:
                fused_boxes = np.zeros((0, 4), dtype=np.float32)
                fused_scores = np.zeros(0, dtype=np.float32)
                fused_classes = np.zeros(0, dtype=np.int32)

        # Build detection dicts
        detections = self._build_detection_dicts(
            fused_boxes, fused_scores, fused_classes, conf
        )

        return detections, round(t.elapsed_ms, 2)

    def _build_detection_dicts(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        conf: float,
    ) -> List[Dict[str, Any]]:
        """Convert raw arrays into detection dicts with computed metrics."""
        names_map = {}
        if self.model is not None:
            try:
                names_map = self.model.names or {}
            except Exception:
                pass

        detections: List[Dict[str, Any]] = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            class_id = int(class_ids[i])
            confidence = float(scores[i])

            if confidence < conf:
                continue

            if class_id in names_map:
                class_name = names_map[class_id]
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
                "aspect_ratio": compute_aspect_ratio(
                    int(x2) - int(x1), int(y2) - int(y1)
                ),
                "diameter_px": compute_diameter(
                    int(x2) - int(x1), int(y2) - int(y1)
                ),
            })

        return detections

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
        draw_circle: bool = True,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, tight fitted ellipses, and labels on the image.

        For each detection the ellipse/circle is fitted to the **actual wood
        contour** (HSV-segmented inside the ROI), not to the loose YOLO
        bounding box — so the drawn shape traces the real wood edge and shows
        the exact length × breadth of each log.

        For each detection:
          - Bounding box rectangle (optional).
          - Fitted ellipse (``cv2.fitEllipse`` on the wood contour) hugging
            the actual wood edge, with major (length, orange) and minor
            (breadth, blue) axes drawn through the centre.
          - Center point dot + class/confidence/length×breadth label.

        Args:
            image: Original BGR image.
            detections: Detection list from detect().
            box_thickness: Line thickness for bounding boxes and ellipses.
            font_scale: Font scale for labels.
            draw_circle: If True, draw the fitted ellipse + axes.
            draw_bbox: If True, draw the bounding box rectangle.

        Returns:
            Annotated image (new copy; original is not modified).
        """
        import math

        # Lazy import to avoid a circular import at module load time
        from app.segmentation import extract_wood_contour

        annotated = image.copy()

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = get_color_for_class(class_id)

            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]

            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            # Draw bounding box
            if draw_bbox:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Draw fitted ellipse / circle traced on the ACTUAL wood contour
            length_px = 0.0
            breadth_px = 0.0
            if draw_circle and w > 0 and h > 0:
                contour_global, _, _ = extract_wood_contour(image, bbox, pad=5)
                drawn = False

                if contour_global is not None and len(contour_global) >= 5:
                    try:
                        ellipse_box = cv2.fitEllipse(contour_global)
                        cv2.ellipse(annotated, ellipse_box, color, box_thickness)
                        (ec_x, ec_y), (emaj, emin), ea = ellipse_box
                        length_px = float(max(emaj, emin))
                        breadth_px = float(min(emaj, emin))
                        cx_e = int(ec_x)
                        cy_e = int(ec_y)

                        # Draw length (major, orange) + breadth (minor, blue)
                        # axes through the ellipse centre.
                        angle_rad = math.radians(ea)
                        dx_l = (length_px / 2.0) * math.cos(angle_rad)
                        dy_l = (length_px / 2.0) * math.sin(angle_rad)
                        dx_b = (breadth_px / 2.0) * math.cos(
                            angle_rad + math.pi / 2
                        )
                        dy_b = (breadth_px / 2.0) * math.sin(
                            angle_rad + math.pi / 2
                        )
                        cv2.line(
                            annotated,
                            (int(cx_e - dx_l), int(cy_e - dy_l)),
                            (int(cx_e + dx_l), int(cy_e + dy_l)),
                            (0, 200, 255),  # orange — length axis
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            annotated,
                            (int(cx_e - dx_b), int(cy_e - dy_b)),
                            (int(cx_e + dx_b), int(cy_e + dy_b)),
                            (255, 100, 0),  # blue — breadth axis
                            1,
                            cv2.LINE_AA,
                        )
                        # center dot at the true ellipse centre
                        cv2.circle(annotated, (cx_e, cy_e), 2, color, -1)
                        drawn = True
                    except cv2.error:
                        drawn = False

                if not drawn:
                    # Fallback: tight circle using the smaller bbox dimension
                    # (tighter than the old minEnclosingCircle approach).
                    radius = int(min(w, h) / 2)
                    cv2.circle(annotated, (cx, cy), radius, color, box_thickness)
                    cv2.circle(annotated, (cx, cy), 2, color, -1)
                    length_px = float(radius * 2)
                    breadth_px = float(radius * 2)

            # Label text: class + confidence + length×breadth
            if length_px > 0:
                label = (
                    f"{class_name} {confidence:.2f} "
                    f"L={int(length_px)}xB={int(breadth_px)}"
                )
            else:
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
