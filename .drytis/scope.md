# Scope — Wooden Log Detection with YOLOv8

## Features By Module

### 1. Detection Web App (app/)
- Image upload with drag-and-drop
- Real-time YOLOv8 inference
- Annotated result with bounding boxes, labels, confidence scores
- Video upload and frame-by-frame detection
- Detection statistics (object count, avg confidence, processing time)
- Model selection (nano/small/medium/large)

### 2. Training Pipeline (src/train.py)
- YOLOv8 training with configurable hyperparameters
- Hyperparameter presets (fast, balanced, accurate)
- Augmentation parameter control
- Resume from checkpoint
- Training metrics output

### 3. Data Augmentation (src/augment.py)
- Horizontal/vertical flip (with bbox transformation)
- Random rotation (with bbox transformation)
- Brightness/contrast adjustment
- Gaussian blur / noise
- Mosaic / mixup helpers
- Batch processing of dataset folders

### 4. Dataset Preprocessing (src/preprocess.py)
- Convert raw annotations (XML/COCO JSON) to YOLO format
- Train/val/test split (configurable ratios)
- Image resizing and normalization
- Generate data.yaml configuration
- Dataset statistics reporting

### 5. Batch & Video Prediction (src/predict.py)
- Process entire folders of images
- Video stream detection with output video save
- Export detection results to JSON/CSV
- Confidence threshold control

### 6. Sample Data Generator (src/generate_samples.py)
- Generate synthetic wooden log images using OpenCV drawing
- Produce YOLO-format annotations for generated images
- Create train/val split automatically

### 7. Model Export (src/export.py)
- Export trained model to ONNX format
- Export to TorchScript format

## In Scope
- All modules above
- Flask web UI accessible at preview URL
- Unit + integration tests
- Documentation

## Out of Scope
- Real-time webcam/camera stream in web UI
- Distributed training across GPUs
- Production deployment with ONNX runtime serving
- User authentication
