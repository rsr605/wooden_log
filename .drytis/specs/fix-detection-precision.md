# Fix Detection Precision — Remove False Positives

## Problem
The detector "detects randomly anything" — produces many false positive detections
on images. Root causes identified:

1. **CONFIDENCE_THRESHOLD = 0.08** (absurdly low) — at this threshold the model
   emits huge numbers of low-confidence false-positive boxes.
2. **Multi-scale + TTA + WBF pipeline** (4 inference passes per image) — each pass
   generates its own set of false positives at 0.08 conf; WBF does not suppress
   them well because non-overlapping noise boxes survive the fusion.
3. **IOU_THRESHOLD = 0.30** — too aggressive for NMS, keeps duplicate boxes.

## Model Validation Data (from `m.val()` on data_v2 val split)
- conf=0.25, iou=0.45 → P=0.9827, R=0.9077, F1=0.9437  ← optimal balance
- conf=0.10, iou=0.45 → P=0.9770, R=0.9165, F1=0.9458
- The model (models/wooden_log_best.pt, YOLOv8s, 11M params) is well-trained.

## Solution
1. Simplify `detector.py` to **single-pass inference** at 640px (no TTA, no WBF,
   no multi-scale). This eliminates the noise amplification and is 4× faster.
2. Raise **DEFAULT_CONFIDENCE_THRESHOLD** from 0.08 → 0.40.
   - Initially tried 0.25, but testing showed a false positive on a solid black
     image (0.37 conf edge artifact). 0.40 eliminates this while maintaining
     89.2% recall. P=99.2% at conf=0.40 vs P=98.3% at conf=0.25.
3. Raise **DEFAULT_IOU_THRESHOLD** from 0.30 → 0.45.
4. Update env keys to match.
5. Update tests to reflect the simplified pipeline.

## Files to Change
- `app/detector.py` — remove WBF/TTA/multi-scale; single-pass `detect()`
- `src/utils.py` — update DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD
- `.env` via env-key tool — update CONFIDENCE_THRESHOLD=0.25, IOU_THRESHOLD=0.45
- `tests/test_detector.py` — update tests for simplified detector

## Acceptance Criteria
- [ ] `detector.py` has no WBF, TTA, or multi-scale code
- [ ] Single-pass inference at imgsz=640
- [ ] DEFAULT_CONFIDENCE_THRESHOLD = 0.40 in utils.py
- [ ] DEFAULT_IOU_THRESHOLD = 0.45 in utils.py
- [ ] CONFIDENCE_THRESHOLD=0.40 in env keys
- [ ] IOU_THRESHOLD=0.45 in env keys
- [ ] Detection on sample images produces clean, precise detections (no random false positives)
- [ ] All unit tests pass
- [ ] App runs on preview URL and detects logs correctly
