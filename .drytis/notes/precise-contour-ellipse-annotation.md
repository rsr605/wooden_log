# Precise Contour-Based Ellipse Annotation

## Date: 2026-07-04

## Problem
User reported the drawn circles were "big big" and didn't trace the actual
wood precisely — they wanted each circle to match the exact wood length and
breadth.

## Root Cause
- `/detect` mode drew circles/ellipses *inscribed in the YOLO bounding box*.
  The YOLO bbox is loose (includes padding/background), so the circle was
  much bigger than the real wood.
- `/analyze` mode used `cv2.minEnclosingCircle()` which returns the smallest
  circle containing ALL pixels — still too big for elongated logs and showed
  nothing about length vs breadth.

## Solution
Replaced both annotation paths with **contour-based ellipse fitting**:

1. **Shared `extract_wood_contour(image, bbox, pad)`** in `app/segmentation.py`
   — crops the ROI, builds an HSV wood-colour mask, finds the largest contour,
   and shifts it back to global image coordinates. Used by BOTH code paths.

2. **`cv2.fitEllipse()`** on the actual wood contour — produces a tight ellipse
   that hugs the real wood edge. Returns `(center, (MAJOR, MINOR), angle)`.

3. **Length (major, orange) + Breadth (minor, blue) axes** drawn through the
   ellipse centre so exact L×B is visible at a glance.

4. **Fallback**: when no wood contour is found (< 5 pts), use a tight circle
   with radius = min(bbox_w, bbox_h) / 2 (tighter than minEnclosingCircle).

## Files Changed
- `app/segmentation.py` — `extract_wood_contour()` + `_build_wood_mask()` +
  `_largest_contour_from_mask()` module helpers; `_analyze_single()` returns
  `(dict, ellipse_box)` tuple; `_draw()` renders fitted ellipse + axes;
  result dict has `length_px`, `breadth_px`, `ellipse_angle`, `ellipse_area`.
  Raw `ellipse_box` deliberately kept OUT of the JSON dict.
- `app/detector.py` — `annotate()` rewritten to call `extract_wood_contour` +
  `fitEllipse`. Label now shows `L=...xB=...`.
- `app/templates/analysis.html` — Length (px) + Breadth (px) columns; updated
  info card describing the fitted ellipse with orange/blue axes.
- `tests/test_segmentation.py` — new `TestExtractWoodContour`,
  `TestEllipseFitting` classes; `test_result_does_not_contain_ellipse_box`.
- `tests/test_detector.py` — new `TestAnnotateEllipseDrawing` class.

## Verification
- 131 tests pass.
- Live API: `/api/analyze` on `test_end_view.png` → 20 logs, all with
  `length_px ≥ breadth_px`, `ellipse_angle` non-null, `ellipse_box` not in JSON.
- Side view logs now show realistic L×B (e.g. 440×193, 676×407) instead of
  giant circles.
- reviewer: PASS (all acceptance criteria).
- infra_verifier: PASS (0 failures).
- tester: PASS (source-level; browser tools unavailable).

## Known Issues (pre-existing, unchanged)
- `/workspace/yolov8n.pt` has XFS corruption (cannot delete/rename/stat).
- Git index is corrupt (`bad signature 0x2e302030`) — would block production
  deploy until rebuilt.
