"""
Log Segmentation & Geometric Analysis.

Provides the LogAnalyzer class that takes YOLOv8 bounding-box detections,
extracts individual log contours via colour-based segmentation within each
ROI, fits a tight ellipse (length × breadth) to the actual wood pixels,
and computes geometric measurements:
center, length (major axis), breadth (minor axis), contour area,
ellipse area, circularity, area ratio.
"""

import math
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np

from src.utils import get_color_for_class


# ---------------------------------------------------------------------------
# HSV colour ranges for wood / brown tones (OpenCV HSV: H 0–179, S 0–255, V 0–255)
# ---------------------------------------------------------------------------

# Brown wood tones: Hue ~10-25 in OpenCV HSV (real hue 20-50° = brown/amber).
# Since the synthetic generator now produces correctly-coloured brown logs.

# Range 1: primary brown / tan spectrum (Hue 8–25)
_WOOD_HSV_LOWER_1 = np.array([8, 40, 40])
_WOOD_HSV_UPPER_1 = np.array([25, 255, 255])

# Range 2: dark reddish-brown (Hue wraps near 0/180)
_WOOD_HSV_LOWER_2 = np.array([170, 40, 20])
_WOOD_HSV_UPPER_2 = np.array([179, 255, 255])

# Range 3: light tan / bleached wood (low saturation)
_WOOD_HSV_LOWER_3 = np.array([10, 20, 80])
_WOOD_HSV_UPPER_3 = np.array([30, 100, 255])


class LogAnalyzer:
    """
    Instance segmentation + geometric analysis for individual wooden logs.

    Given a list of YOLOv8 detection dicts (with bbox), this class:
      1. Crops each detection ROI from the source image.
      2. Creates a binary mask via HSV colour filtering for wood tones.
      3. Extracts the largest contour in each ROI mask.
      4. Fits ``cv2.fitEllipse()`` on the contour points to get a tight
         ellipse hugging the actual wood edge (length × breadth).
      5. Computes contour area, ellipse area, circularity (4πA / P²),
         and area_ratio (contour_area / ellipse_area).
    """

    # Padding (pixels) added around each bbox before colour segmentation
    # so edge pixels are included.
    ROI_PADDING: int = 5

    def __init__(self, pad: int = 5):
        self.pad = pad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run geometric analysis on each detection and annotate the image.

        Args:
            image: Source BGR image.
            detections: Detection list from LogDetector.detect().

        Returns:
            Tuple of (analysis_results, annotated_image).
            ``analysis_results`` is a list of dicts, one per detected log,
            with keys: id, center, length_px, breadth_px, ellipse_angle,
            radius, contour_area, ellipse_area, circularity, area_ratio,
            confidence, bbox.
        """
        annotated = image.copy()
        results: List[Dict[str, Any]] = []
        ellipse_boxes: List[Optional[Any]] = []

        for idx, det in enumerate(detections):
            analysis, ellipse_box = self._analyze_single(image, det, log_id=idx + 1)
            results.append(analysis)
            ellipse_boxes.append(ellipse_box)

        self._draw(annotated, results, ellipse_boxes)

        return results, annotated

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _draw(
        self,
        annotated: np.ndarray,
        results: List[Dict[str, Any]],
        ellipse_boxes: Optional[List[Any]] = None,
    ) -> None:
        """Draw fitted ellipses, length/breadth axes, center dots, and labels *in-place*."""
        for idx, res in enumerate(results):
            cx = res["center"]["x"]
            cy = res["center"]["y"]
            log_id = res["id"]

            color = get_color_for_class(0)  # green for wooden_log

            # Draw the fitted ellipse (or fallback circle)
            ellipse_box = ellipse_boxes[idx] if ellipse_boxes else None
            if ellipse_box is not None:
                cv2.ellipse(annotated, ellipse_box, color, 2)
            else:
                cv2.circle(annotated, (int(cx), int(cy)), int(res["radius"]), color, 2)

            # Draw length (major) and breadth (minor) axes through the centre
            # so the exact wood length × breadth is visible at a glance.
            angle_deg = res.get("ellipse_angle")
            length = res.get("length_px", 0.0)
            breadth = res.get("breadth_px", 0.0)
            if angle_deg is not None and length > 0 and breadth > 0:
                angle_rad = math.radians(angle_deg)
                # Major axis endpoints (length)
                dx_l = (length / 2.0) * math.cos(angle_rad)
                dy_l = (length / 2.0) * math.sin(angle_rad)
                # Minor axis endpoints (breadth)
                dx_b = (breadth / 2.0) * math.cos(angle_rad + math.pi / 2)
                dy_b = (breadth / 2.0) * math.sin(angle_rad + math.pi / 2)
                cv2.line(
                    annotated,
                    (int(cx - dx_l), int(cy - dy_l)),
                    (int(cx + dx_l), int(cy + dy_l)),
                    (0, 200, 255),  # orange — length axis
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    annotated,
                    (int(cx - dx_b), int(cy - dy_b)),
                    (int(cx + dx_b), int(cy + dy_b)),
                    (255, 100, 0),  # blue — breadth axis
                    1,
                    cv2.LINE_AA,
                )

            # Center point dot
            cv2.circle(annotated, (int(cx), int(cy)), 3, color, -1)

            # ID label placed just outside the ellipse
            r_out = max(length, breadth, res["radius"]) / 2.0 + 6
            label = f"Log #{log_id}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_x = int(cx) + int(r_out)
            label_y = int(cy) - int(r_out) - 2
            if label_y < th + 4:
                label_y = int(cy) + int(r_out) + th + 6

            cv2.rectangle(
                annotated,
                (label_x - 2, label_y - th - 4),
                (label_x + tw + 4, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (label_x, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Contour extraction + geometry
    # ------------------------------------------------------------------

    def _analyze_single(
        self,
        image: np.ndarray,
        det: Dict[str, Any],
        log_id: int,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """Analyze a single detection.

        Returns:
            Tuple of (result_dict, ellipse_box). ``ellipse_box`` is the raw
            ``cv2.fitEllipse`` output (kept out of the JSON-serializable
            result_dict) or None if no ellipse was fit.
        """
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        # Extract the actual wood contour in global image coordinates
        contour_global, rx1, ry1 = extract_wood_contour(image, bbox, self.pad)

        ellipse_box: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None
        contour_area = 0.0
        perimeter = 0.0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        length = 0.0
        breadth = 0.0
        angle_deg: Optional[float] = None
        ellipse_area = 0.0

        if contour_global is not None and len(contour_global) >= 5:
            # Fit a tight ellipse to the actual wood pixels
            try:
                ellipse_box = cv2.fitEllipse(contour_global)
                (ec_x, ec_y), (emaj, emin), ea = ellipse_box
                # cv2.fitEllipse returns (MAJOR, MINOR) where these are the
                # full axis lengths (2 × semi-axis) of the ellipse's
                # rotated bounding rectangle.
                length = float(max(emaj, emin))
                breadth = float(min(emaj, emin))
                cx = float(ec_x)
                cy = float(ec_y)
                angle_deg = float(ea)
                ellipse_area = math.pi * (length / 2.0) * (breadth / 2.0)
            except cv2.error:
                ellipse_box = None

            contour_area = float(cv2.contourArea(contour_global))
            perimeter = float(cv2.arcLength(contour_global, True))

        if ellipse_box is None:
            # Fallback: tight circle using the smaller bbox dimension.
            # Tighter than minEnclosingCircle because we use min(w,h), not
            # the diagonal-encompassing enclosing circle.
            radius = min(x2 - x1, y2 - y1) / 2.0
            length = float(radius * 2.0)
            breadth = float(radius * 2.0)
            ellipse_area = math.pi * (radius ** 2)

        # Circularity: 1.0 = perfect circle, < 1 for non-circular shapes
        if perimeter > 0:
            circularity = (4 * math.pi * contour_area) / (perimeter ** 2)
        else:
            circularity = 0.0

        # Area ratio: how much of the fitted ellipse the log fills
        if ellipse_area > 0:
            area_ratio = contour_area / ellipse_area
        else:
            area_ratio = 0.0

        # Backwards-compatible radius (average of length/2 and breadth/2)
        radius_compat = round((length + breadth) / 4.0, 2)

        result = {
            "id": log_id,
            "class_name": det.get("class_name", "wooden_log"),
            "confidence": det.get("confidence", 0.0),
            "bbox": bbox,
            "center": {
                "x": round(cx, 2),
                "y": round(cy, 2),
            },
            "radius": radius_compat,
            "length_px": round(length, 2),
            "breadth_px": round(breadth, 2),
            "ellipse_angle": round(angle_deg, 2) if angle_deg is not None else None,
            "contour_area": round(contour_area, 2),
            "circle_area": round(ellipse_area, 2),  # keep old key for template compat
            "ellipse_area": round(ellipse_area, 2),
            "circularity": round(circularity, 4),
            "area_ratio": round(area_ratio, 4),
        }
        return result, ellipse_box

    # ------------------------------------------------------------------
    # Colour-based wood segmentation
    # ------------------------------------------------------------------

    def _wood_mask(self, roi: np.ndarray) -> np.ndarray:
        """Create a binary mask of wood-coloured pixels in the ROI."""
        return _build_wood_mask(roi)

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        """Return the largest external contour from a binary mask, or None."""
        return _largest_contour_from_mask(mask)


# ---------------------------------------------------------------------------
# Module-level shared helpers (used by both segmentation.py and detector.py)
# ---------------------------------------------------------------------------

def _build_wood_mask(roi: np.ndarray) -> np.ndarray:
    """Create a binary mask of wood-coloured pixels in the ROI."""
    if roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, _WOOD_HSV_LOWER_1, _WOOD_HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, _WOOD_HSV_LOWER_2, _WOOD_HSV_UPPER_2)
    mask3 = cv2.inRange(hsv, _WOOD_HSV_LOWER_3, _WOOD_HSV_UPPER_3)
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    # Clean up the mask: remove small noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour from a binary mask, or None."""
    if mask.size == 0:
        return None
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    return largest


def extract_wood_contour(
    image: np.ndarray,
    bbox: Dict[str, int],
    pad: int = 5,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Extract the largest wood-coloured contour within a detection ROI.

    Crops the ROI defined by ``bbox`` (expanded by ``pad`` pixels and clamped
    to image bounds), builds an HSV wood-colour mask, and returns the largest
    external contour **shifted back to full-image coordinates**.

    This is the shared contour source used by both ``LogAnalyzer`` (segmentation
    analysis) and ``LogDetector.annotate()`` (precise ellipse drawing on the
    /detect page), so both code paths trace the same actual wood edge.

    Args:
        image: Full source BGR image.
        bbox:  Dict with keys x1, y1, x2, y2 (pixel coords).
        pad:   Pixels of padding to add around the bbox before masking.

    Returns:
        Tuple of (contour_global, roi_x1, roi_y1).
        ``contour_global`` is None if no wood contour was found.
        ``roi_x1, roi_y1`` are the top-left of the cropped ROI in image coords.
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    h, w = image.shape[:2]

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)

    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None, rx1, ry1

    mask = _build_wood_mask(roi)
    contour = _largest_contour_from_mask(mask)

    if contour is None:
        return None, rx1, ry1

    # Shift contour points back to full-image coordinates
    contour_global = contour.copy()
    contour_global[:, 0, 0] += rx1
    contour_global[:, 0, 1] += ry1

    return contour_global, rx1, ry1
