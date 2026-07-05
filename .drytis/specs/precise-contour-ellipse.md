# Task: Precise Contour-Based Ellipse Annotation

## Problem
The circles drawn on detected logs are **too big** and don't trace the actual
wood edge. Two issues:

1. **`/detect` mode** (`app/detector.py::annotate`) draws a circle/ellipse
   *inscribed in the YOLO bounding box*. The YOLO bbox is loose — it includes
   padding/background around the log — so the circle is much bigger than the
   actual wood.

2. **`/analyze` mode** (`app/segmentation.py::_analyze_single`) uses
   `cv2.minEnclosingCircle()` on the wood-colour contour. That returns the
   smallest circle that *contains all pixels*, which is still too big for
   elongated logs and shows nothing about length vs breadth.

## Goal
Every circle/ellipse should trace the **actual wood edge precisely**, so the
user sees the exact **length (major axis) × breadth (minor axis)** of each log.

## Approach
Replace both annotation paths with a **shared contour extraction + ellipse
fit**:

1. For each detection ROI (bbox + small padding), build an HSV wood-colour
   mask and extract the largest contour (reuse existing segmentation logic).
2. If the contour has ≥ 5 points, call `cv2.fitEllipse()` to fit a tight
   ellipse to the actual wood pixels. Store `(center, (MAJOR, MINOR), angle)`.
3. If fitEllipse fails or contour is too small, fall back to a circle whose
   radius = min(bbox_w, bbox_h) / 2 (tighter than minEnclosingCircle).
4. Draw the fitted ellipse directly on the image at `box_thickness=2`.
5. Overlay the **major (length) and minor (breadth) axes** as coloured line
   segments through the centre so length × breadth is visible.

### Shared module
Move the wood-mask + largest-contour logic into a shared helper so both
`detector.py` and `segmentation.py` use the *same* contour source. Add it to
`app/segmentation.py` as a public function `extract_wood_contour(image, bbox, pad)`
returning `(contour_global, roi_x1, roi_y1)`, and have both call sites use it.

## Files to Change
- `app/segmentation.py` — expose `extract_wood_contour()`; rewrite
  `_analyze_single()` to use `fitEllipse` instead of `minEnclosingCircle`;
  add `length_px`, `breadth_px`, `ellipse_angle` to the result dict.
- `app/detector.py` — rewrite `annotate()` so the ellipse/circle is computed
  from the wood contour + `fitEllipse` (import from segmentation), not the
  loose bbox. Draw L×B axes.
- `app/templates/analysis.html` — show Length (px), Breadth (px) columns.
- `tests/test_segmentation.py` — assert `fitEllipse` path, length ≥ breadth.
- `tests/test_detector.py` — assert annotate draws an ellipse from contour.

## Acceptance Criteria
- [ ] `extract_wood_contour(image, bbox, pad)` exists in segmentation.py
      and returns a contour in global image coordinates.
- [ ] `detector.annotate()` draws an ellipse fit to the wood contour (not the
      loose bbox) — the ellipse visibly hugs the actual wood edge.
- [ ] When fitEllipse is unavailable (< 5 contour pts), code falls back
      gracefully without crashing.
- [ ] `/analyze` results include `length_px` and `breadth_px` (length ≥ breadth)
      for every log where a contour was found.
- [ ] Major (length) and minor (breadth) axes are drawn on the annotated
      image as visible line segments.
- [ ] `analysis.html` shows Length (px) and Breadth (px) columns.
- [ ] All existing unit + integration tests pass; new tests added and green.
- [ ] No new env keys, services, or proxy changes (annotation-only change).
- [ ] Reviewer PASS, tester browser-test PASS (home page loads, detection
      produces tighter ellipses than before).

## Tests
- Unit: synthetic image with a brown ellipse on a dark background →
  `fitEllipse` returns major ≥ minor; `extract_wood_contour` finds a contour.
- Unit: tiny/empty ROI → fallback path returns a sensible circle, no crash.
- Integration: `GET /` loads; `POST /analyze` with a sample image returns 200
  and JSON/HTML containing length_px + breadth_px.
