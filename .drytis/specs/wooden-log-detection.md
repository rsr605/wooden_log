# Task Spec: Wooden Log Detection with YOLOv8

## Overview
Build a complete wooden log detection system using YOLOv8 + OpenCV + Flask. Includes web UI for image/video inference, training pipeline, data augmentation toolkit, dataset preprocessing utilities, and a sample dataset generator.

## Files to Create

### Core Detection
- `app/detector.py` — YOLOv8 inference wrapper with OpenCV annotation
- `app/__init__.py` — package init
- `src/__init__.py` — package init

### Web App
- `app/main.py` — Flask application (upload, detect, display results)
- `app/templates/base.html` — base template with nav
- `app/templates/index.html` — upload form
- `app/templates/result.html` — detection results display
- `app/static/css/style.css` — styling
- `app/static/js/app.js` — frontend interactions

### Training Pipeline
- `src/train.py` — YOLOv8 training script with hyperparameter presets
- `src/augment.py` — Data augmentation toolkit (flip, rotate, brightness, blur, mosaic)
- `src/preprocess.py` — Dataset preprocessing + YOLO label conversion
- `src/generate_samples.py` — Synthetic sample data generator
- `src/predict.py` — Batch/video prediction CLI
- `src/export.py` — Model export (ONNX)
- `src/utils.py` — Shared utilities

### Config & Data
- `data/data.yaml` — YOLO dataset configuration
- `requirements.txt` — Python dependencies
- `README.md` — Documentation

### Tests
- `tests/test_augment.py` — Unit tests for augmentation functions
- `tests/test_preprocess.py` — Unit tests for preprocessing/label conversion
- `tests/test_detector.py` — Unit tests for detector logic
- `tests/test_app.py` — Integration tests for Flask endpoints

## Acceptance Criteria

- [ ] YOLOv8 detector wrapper loads model and runs inference on images
- [ ] OpenCV draws bounding boxes with labels and confidence scores
- [ ] Flask web app accepts image upload and displays annotated results
- [ ] Video detection processes uploaded video and returns annotated frames
- [ ] Training pipeline script runs with configurable hyperparameters
- [ ] Data augmentation applies transformations with correct YOLO bbox transformation
- [ ] Dataset preprocessing converts annotations to YOLO format
- [ ] Train/val/test split creates correct directory structure
- [ ] Batch prediction processes folders of images
- [ ] Sample data generator produces images with YOLO annotations
- [ ] Model export utility exports to ONNX format
- [ ] All unit tests pass
- [ ] Flask integration tests pass
- [ ] Web app is accessible at the preview URL

## Edge Cases
- Empty image upload → graceful error
- No detections found → informative message
- Corrupted image file → error handling
- Very large video → size limit + feedback
- Missing model weights → fallback to base YOLOv8 model

## Testing Strategy
- **Unit tests**: augmentation transforms, label format conversion, coordinate calculation
- **Integration tests**: Flask endpoints (upload, detect, results)
- **Smoke test**: detection runs on generated sample images
