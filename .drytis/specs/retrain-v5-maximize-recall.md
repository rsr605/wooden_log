# Task: Retrain Model v5 — Maximize Log Detection Recall

## Problem
The user reports the detector "does not detect all" logs. The current model
(`models/wooden_log_best.pt`, trained on `data_v4`, 30 epochs CPU) achieves:
- Precision: 0.956
- **Recall: 0.899** ← too low; ~10% of logs are missed.
- mAP50: 0.956
- mAP50-95: 0.787

The recall gap comes from:
1. Not enough small/tiny log examples in training data (1500 images).
2. Moderate epoch count (30) — undertrained for hard examples.
3. No explicit recall-focused loss weighting.

## Goal
Retrain to maximize recall (detect every visible log) while keeping precision
acceptable. **Target: Recall ≥ 0.95, mAP50 ≥ 0.96.**

## Strategy (CPU-only, 4 cores, 6 GB RAM — must be efficient)

### 1. Enhanced generator (`src/generate_samples.py`)
- **More small/tiny logs**: increase the weight of tiny+small size classes
  from (12%, 25%) → (18%, 30%). Small logs are the hardest to detect.
- **Denser end-view piles**: push max logs from 40 → 55, with tighter packing
  and more overlap/occlusion (simulates real log piles with hidden edges).
- **More partial/edge logs**: increase edge-partial probability from 30% → 40%.
- **Add "partially occluded" annotations**: when logs overlap, keep BOTH
  annotations (already the case) but increase the overlap frequency.
- **More side+end mixed scenes**: increase mixed weight from 20% → 30%.
- **More variety in log count per image**: add sparse scenes (1-3 logs) so
  the model also learns isolated logs, not just dense piles.

### 2. Larger dataset (`data_v5`)
- **2500 images** (up from 1500): 2000 train + 500 val.
- Multiple resolutions: 640×480, 800×600, 512×512, 1024×768.

### 3. Recall-focused training (`src/train.py` — new "recall" preset)
- **Epochs: 60** (up from 30) — CPU-friendly but enough to converge.
- **imgsz: 640** (matches inference).
- **batch: 4, workers: 1** (CPU constraints).
- **optimizer: AdamW, lr0: 0.001**.
- **label_smoothing: 0.05** — prevents overconfident false negatives.
- **box loss gain boost**: `box=8.0` (default 7.5) — better localization.
- **cls loss gain reduced**: `cls=0.4` (default 0.5) — single class, less
  classification pressure, more focus on finding boxes.
- **close_mosaic: 10** — turn off mosaic in last 10 epochs for clean learning.
- **patience: 20** — early stop if no improvement.
- **Augmentation**: mosaic=1.0, mixup=0.1, copy_paste=0.1, fliplr=0.5,
  scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.5, degrees=10, translate=0.1.

### 4. Inference threshold
Keep confidence threshold at 0.25 (already low for max recall). The
multi-scale + WBF inference pipeline (640+960 with TTA) stays as-is.

## Files to Change
- `src/generate_samples.py` — tuning weights for size classes, log counts,
  edge partials, mixed scenes, denser piles.
- `src/train.py` — add "recall" preset with the hyperparameters above.
- `data_v5/` — new dataset (generated).
- `models/wooden_log_best.pt` — replaced with the newly trained weights.

## Acceptance Criteria
- [ ] `data_v5` has ≥ 2500 images (2000 train, 500 val).
- [ ] New "recall" preset exists in `src/train.py`.
- [ ] Trained model achieves **Recall ≥ 0.95** on `data_v5` val set.
- [ ] Trained model achieves **mAP50 ≥ 0.96** on `data_v5` val set.
- [ ] `models/wooden_log_best.pt` is replaced with the new weights.
- [ ] App detects ≥ as many logs as before on the 3 real test images
      (test_end_view, test_side_view, test_mixed).
- [ ] All existing unit + integration tests pass.
- [ ] Reviewer PASS, tester browser-test PASS.

## Tests
- Existing test suite continues to pass (no API changes).
- Validation metrics printed and recorded.
- Manual comparison on 3 real test images (before vs after detection counts).
