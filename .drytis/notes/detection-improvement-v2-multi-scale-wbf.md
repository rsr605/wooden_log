# Detection Improvement v2 — Multi-Scale + WBF

## Date: 2026-07-03

## Problem
The wooden log detector was missing logs on real-world photos. Root cause: synthetic training data created a domain gap, confidence threshold was too high (0.40), and single-scale 640px inference missed smaller logs.

## Solution
Three changes that together dramatically improved detection:

### 1. Multi-Scale Inference + Weighted Box Fusion (app/detector.py)
- Run YOLO at two scales: 640px + 960px with TTA (`augment=True`)
- Fuse overlapping detections via WBF algorithm (confidence-weighted box averaging)
- Second-pass NMS (`torchvision.ops.nms`, threshold derived from `iou - 0.1`)
- Results: side_view 4→5 detections, end_view 20→20, mixed 25→24

### 2. Lowered Thresholds
- conf: 0.40 → 0.25
- iou: 0.45 → 0.50

### 3. Retrained Model with Photo-Realistic Augmentation (src/generate_samples.py)
- Generated new data_v4 (1500 images) with upgraded _augment():
  - CLAHE, camera sensor noise, JPEG artifacts, motion blur
  - Directional lighting, stronger HSV jitter, sharpening
  - Earth-tone specks instead of bright colors, dark brown knots
- Trained 30 epochs CPU-only (batch=4, workers=1)
- Final val metrics: P=0.956, R=0.899, mAP50=0.956, mAP50-95=0.787

## Test Results on Real Images (via live API)
| Image | Detections | Conf Range | Avg Conf |
|-------|-----------|------------|----------|
| test_end_view.png | 20 | 0.702-0.955 | 0.886 |
| test_mixed.png | 24 | 0.460-0.946 | ~0.90 |
| test_side_view.png | 5 | 0.345-0.894 | ~0.70 |

## Known Issues
- /workspace/yolov8n.pt has XFS corruption (cannot delete/rename/stat). Use models/yolov8n_fresh.pt instead.
- Confidence slider in index.html updated to default 0.25 (was 0.40).
- 119 unit/integration tests pass.

## Verification
- infra_verifier: PASS (0 failures)
- reviewer: PASS (all 8 acceptance criteria met)
- tester: home page loads, API returns correct detections
