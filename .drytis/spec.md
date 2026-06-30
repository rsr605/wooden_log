# Spec — Wooden Log Detection with YOLOv8

## Overview
Computer vision learning project that detects wooden logs in images and video using YOLOv8 object detection and OpenCV. Includes a Flask web interface, full training pipeline with data augmentation, and CLI tools.

## Tech Stack
- **Language**: Python 3
- **Object Detection**: YOLOv8 (ultralytics package)
- **Image Processing**: OpenCV (cv2)
- **Augmentation**: Albumentations + YOLOv8 built-in augmentations
- **Web Framework**: Flask + Jinja2
- **ML Utilities**: NumPy, Pillow, Matplotlib

## Key Decisions
1. Use pre-trained YOLOv8n (nano) as default model for fast inference; allow swapping to larger models
2. Flask instead of Django for simplicity — this is a learning project
3. Synthetic sample data generator so the app works out-of-the-box without external datasets
4. Support both image and video inference in the web UI
5. Data augmentation as standalone toolkit (usable independently of training)
