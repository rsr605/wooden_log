# Segmentation & Geometric Analysis Feature

## Overview
Add instance segmentation + geometric log analysis to the Wooden Log Detection app.
Each detected log gets contour extraction, minimum enclosing circle fitting, and
geometric measurements (center, radius, areas, circularity, area ratio).

## Files to Change
- `app/segmentation.py` (NEW) — LogAnalyzer class: contour extraction, circle fitting, geometry
- `app/main.py` — add `/analyze` route (HTML) + `/api/analyze` endpoint (JSON)
- `app/templates/analysis.html` (NEW) — analysis result page with circles + measurement table
- `app/static/css/style.css` — styles for analysis view
- `tests/test_segmentation.py` (NEW) — unit tests for LogAnalyzer

## Algorithm
1. YOLOv8 detects bounding boxes (existing `detect()`)
2. For each detection bbox:
   a. Crop the ROI (with small padding)
   b. Apply HSV color filtering for brown/wood tones → binary mask
   c. Find largest contour in the mask
   d. Adjust contour coordinates back to full-image space
   e. Fit `cv2.minEnclosingCircle()` on the contour points
   f. Compute contour area (`cv2.contourArea`), perimeter (`cv2.arcLength`)
   g. Compute circle area = π * r²
   h. Compute circularity = 4π * area / perimeter²
   i. Compute area_ratio = contour_area / circle_area
3. Annotate image: draw minimum enclosing circle + center dot + ID label
4. Return structured JSON + annotated image

## Acceptance Criteria
- [x] `POST /api/analyze` accepts an image file, returns JSON with per-log measurements
- [x] `POST /analyze` accepts an image file, renders HTML result page with annotated image + table
- [x] Each log entry in JSON has: id, center {x,y}, radius, contour_area, circle_area, circularity, area_ratio, confidence, bbox
- [x] Annotated image draws minimum enclosing circle (not inscribed) for each log
- [x] Center point dot drawn on each log
- [x] Log ID label drawn next to each circle
- [x] Circular log → circularity near 1.0, area_ratio near 1.0
- [x] Elongated log → circularity < 0.8, area_ratio < 0.7
- [x] Unit tests pass for contour extraction, circle fitting, geometry calculations
- [x] Existing detection routes (`/detect`, `/api/detect`) unchanged
- [x] Existing tests still pass

## Test Strategy
- Unit tests with synthetic images (draw a known circle/shape, verify measurements)
- Mock the YOLO model, test pure geometry functions
- Integration test: POST to `/api/analyze` with a sample image
