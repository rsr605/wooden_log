# Task: Improve Detection Coverage & Accuracy

## Goal
Maximize the number of wooden logs detected in every image — including small, distant, overlapping, and densely packed logs — while keeping false positives low.

## Problem
- Current model (yolov8n, 200-epoch run) misses logs, especially small/dense ones.
- Default thresholds (conf=0.15, iou=0.40) are tuned conservatively.
- Single-scale inference at 640px misses small objects.
- Single-pass inference — no test-time augmentation.

## Changes Required

### 1. `app/detector.py` — Multi-Scale + TTA Detection
- [x] Add multi-scale inference: run at imgsz 640, 960, and 1280, merge results.
- [x] Add Test-Time Augmentation (TTA): run on original + horizontally-flipped image, merge.
- [x] Lower default confidence threshold to 0.08 for maximum recall.
- [x] Lower IoU/NMS threshold to 0.30 to keep densely overlapping logs.
- [x] Increase max_det to 500.
- [x] Implement Weighted Box Fusion (WBF) to merge overlapping detections across scales/augmentations without killing overlapping logs (unlike NMS).
- [x] Keep the public API of LogDetector and get_detector() unchanged so the Flask app needs no changes.

### 2. `.env` — Tuned Thresholds
- [x] CONFIDENCE_THRESHOLD = 0.08
- [x] IOU_THRESHOLD = 0.30

### 3. Training — Upgraded Run
- [x] Launch a new training run with yolov8s (11M params, better accuracy than yolov8n's 3M).
- [x] More aggressive augmentation for small-object detection.
- [x] When complete, copy best.pt → models/wooden_log_best.pt and restart gunicorn.

## Acceptance Criteria
- [x] Multi-scale + TTA inference runs without error on sample images.
- [x] WBF correctly merges duplicate boxes from multiple scales.
- [x] Detection count on sample images increases vs old single-scale run.
- [x] Small/distant logs that were previously missed are now detected.
- [x] Web app loads and responds (HTTP 200).
- [x] All unit tests pass.
- [x] No false positives on background regions (empty images).
