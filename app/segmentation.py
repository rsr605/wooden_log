"""
Log Segmentation & Geometric Analysis.

Provides the LogAnalyzer class that takes YOLOv8 bounding-box detections,
extracts individual log contours via colour-based segmentation within each
ROI, fits a minimum enclosing circle, and computes geometric measurements:
center, radius, contour area, circle area, circularity, area ratio.
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
      4. Fits ``cv2.minEnclosingCircle()`` on the contour points.
      5. Computes contour area, circle area, circularity (4πA / P²),
         and area_ratio (contour_area / circle_area).
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
            with keys: id, center, radius, contour_area, circle_area,
            circularity, area_ratio, confidence, bbox.
        """
        annotated = image.copy()
        results: List[Dict[str, Any]] = []

        for idx, det in enumerate(detections):
            analysis = self._analyze_single(image, det, log_id=idx + 1)
            results.append(analysis)

        self._draw(annotated, results)

        return results, annotated

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _draw(
        self,
        annotated: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> None:
        """Draw minimum enclosing circles, center dots, and labels *in-place*."""
        for res in results:
            cx = res["center"]["x"]
            cy = res["center"]["y"]
            r = res["radius"]
            log_id = res["id"]

            color = get_color_for_class(0)  # green for wooden_log

            # Minimum enclosing circle
            cv2.circle(annotated, (int(cx), int(cy)), int(r), color, 2)

            # Center point dot
            cv2.circle(annotated, (int(cx), int(cy)), 3, color, -1)

            # ID label
            label = f"Log #{log_id}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_x = int(cx) + int(r) + 6
            label_y = int(cy) - int(r) - 2
            if label_y < th + 4:
                label_y = int(cy) + int(r) + th + 6

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
    ) -> Dict[str, Any]:
        """Analyze a single detection and return its measurement dict."""
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        h, w = image.shape[:2]

        # Expand ROI with padding, clamped to image bounds
        rx1 = max(0, x1 - self.pad)
        ry1 = max(0, y1 - self.pad)
        rx2 = min(w, x2 + self.pad)
        ry2 = min(h, y2 + self.pad)

        roi = image[ry1:ry2, rx1:rx2]

        # Build binary mask via HSV colour filtering
        mask = self._wood_mask(roi)

        # Find the largest contour within the ROI
        contour = self._largest_contour(mask)

        if contour is not None and len(contour) >= 3:
            # Shift contour points back to full-image coordinates
            contour_global = contour.copy()
            contour_global[:, 0, 0] += rx1
            contour_global[:, 0, 1] += ry1

            center, radius = cv2.minEnclosingCircle(contour_global)
            contour_area = cv2.contourArea(contour_global)
            perimeter = cv2.arcLength(contour_global, True)
            circle_area = math.pi * (radius ** 2)
        else:
            # Fallback: use bbox center and min dimension as radius
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = min(x2 - x1, y2 - y1) / 2.0
            contour_area = 0.0
            perimeter = 0.0
            circle_area = math.pi * (radius ** 2)
            center = (cx, cy)

        # Circularity: 1.0 = perfect circle, < 1 for non-circular shapes
        if perimeter > 0:
            circularity = (4 * math.pi * contour_area) / (perimeter ** 2)
        else:
            circularity = 0.0

        # Area ratio: how much of the enclosing circle the log fills
        if circle_area > 0:
            area_ratio = contour_area / circle_area
        else:
            area_ratio = 0.0

        return {
            "id": log_id,
            "class_name": det.get("class_name", "wooden_log"),
            "confidence": det.get("confidence", 0.0),
            "bbox": bbox,
            "center": {
                "x": round(center[0], 2),
                "y": round(center[1], 2),
            },
            "radius": round(radius, 2),
            "contour_area": round(contour_area, 2),
            "circle_area": round(circle_area, 2),
            "circularity": round(circularity, 4),
            "area_ratio": round(area_ratio, 4),
        }

    # ------------------------------------------------------------------
    # Colour-based wood segmentation
    # ------------------------------------------------------------------

    def _wood_mask(self, roi: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
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
