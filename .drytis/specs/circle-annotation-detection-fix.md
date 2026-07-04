# Task: Circle Annotation + Detection Improvement

## Problem
1. **Not detecting all logs**: IoU threshold of 0.7 causes NMS to suppress overlapping wooden logs. Many visible logs are not detected.
2. **No circle annotation**: Only rectangular bounding boxes are drawn. User wants an inscribed circle/ellipse fitted to each detected log cross-section.
3. **No aspect ratio**: No width/height ratio or diameter information is shown for each detection.

## Changes

### 1. Fix `data/data.yaml` (corrupted binary file → valid YAML text)
- File is currently binary garbage (249 bytes of non-UTF-8 data).
- Must be valid YAML pointing to dataset dirs with `nc: 1`, `names: ['wooden_log']`.

### 2. Lower IoU threshold (better NMS, detect overlapping logs)
- `IOU_THRESHOLD` env default: `0.7` → `0.45`
- `DEFAULT_IOU_THRESHOLD` in `src/utils.py`: `0.7` → `0.45`
- Lower IoU = NMS keeps boxes that overlap less aggressively, so closely-packed logs survive.

### 3. Add ellipse/circle annotation to `LogDetector.annotate()`
- Draw an inscribed ellipse (fitted to the bounding box) on every detection.
- If the bbox is nearly square (ratio 0.85–1.18), draw a circle; otherwise draw an ellipse matching the bbox.
- Show the circle/ellipse in the same class color, with a center point dot.

### 4. Add aspect ratio + diameter to detection output
- Each detection dict gains: `aspect_ratio` (w/h, 2 decimals), `diameter_px` (average of w,h, rounded).
- The annotate method and the API both surface these.

### 5. Update `result.html` detection table
- Add columns: **Aspect Ratio (w/h)** and **Diameter (px)**.
- Update info text to mention circle annotation.

### 6. Update `index.html` 
- Mention that detections include circle overlay + aspect ratio.

## Acceptance Criteria
- [ ] `data/data.yaml` is valid UTF-8 YAML text (parseable by Python yaml)
- [ ] IoU threshold default = 0.45 in both env keys and utils.py
- [ ] `annotate()` draws an inscribed ellipse on each detection bbox
- [ ] Each detection dict includes `aspect_ratio` and `diameter_px`
- [ ] result.html shows aspect ratio + diameter columns
- [ ] More logs detected on multi-log images (count increases vs before)
- [ ] Unit tests pass for: aspect_ratio computation, diameter computation, ellipse drawing
- [ ] Web app serves correctly, preview loads
