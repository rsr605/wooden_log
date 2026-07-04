# Detection Improvement v2

## Problem

The model fails to detect wooden logs properly on real-world images:
- Synthetic training data (digital-art style) creates a domain gap with real photos
- Confidence threshold (0.40) too high → missed logs
- Inference at fixed 640px → misses small logs and catches fewer logs overall
- NMS at IoU 0.45 → duplicate overlapping detections on clustered log piles

## Evidence (measured on real test images)

| Image | conf=0.40 | conf=0.25 | conf=0.25, imgsz=960 | conf=0.25, imgsz=1280 |
|-------|-----------|-----------|----------------------|-----------------------|
| test_side_view.png | 4 | 7 | 11 | 10 |
| test_end_view.png | 20 | 20 | 20 | 23 |
| test_mixed.png | 25 | 25 | 25 | 33 |

TTA + imgsz=1280 catches the most logs.

## Solution: Multi-Scale Inference + Smart Post-Processing

### 1. Multi-scale TTA inference
Run the model at two scales (640, 960) with augmentation, fuse results via WBF.
This catches small logs (large imgsz) and maintains high recall on large logs (640).

### 2. Lowered thresholds
- conf 0.40 → 0.25
- iou 0.45 → 0.50

### 3. Post-processing NMS
After WBF fusion, apply a second NMS pass to remove residual overlapping boxes.

### 4. Improved synthetic data generator
- More realistic textures: add Gaussian blur, stronger HSV jitter, CLAHE
- Photo-realistic backgrounds: add natural-looking forest/sawmill textures
- Photo-like noise: camera noise, JPEG compression artifacts
- These force the model to learn log features, not synthetic artifacts

## Files to change
- `app/detector.py` — multi-scale inference + WBF + post-NMS
- `src/utils.py` — update default thresholds
- `src/generate_samples.py` — more realistic synthetic generation
- `src/train.py` — improved training preset
- `.env` — updated thresholds (via `update_environment_key`)

## Acceptance criteria
- [ ] Multi-scale inference (640 + 960) implemented in `detect()`
- [ ] WBF fusion of multi-scale results
- [ ] Post-processing NMS removes overlapping duplicates
- [ ] Confidence threshold lowered to 0.25
- [ ] Real-world test images: detections at conf≥0.25 match or exceed visual log count
- [ ] No false positive explosions on synthetic sample images
- [ ] Unit tests pass for new logic
- [ ] App runs and preview URL responds
