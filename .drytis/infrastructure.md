# Infrastructure — Wooden Log Detection with YOLOv8

## Proxy Routes
- **Caddy reverse proxy** at `/` → port 5000 (Flask app)

## Background Services
- **wooden-log-app**: Flask app served via gunicorn on port 5000
  - Command: `cd /workspace && gunicorn --bind 0.0.0.0:5000 --timeout 120 --workers 2 app.main:app`
  - Working directory: `/workspace`

## Environment Variables
- `FLASK_ENV` — Flask environment (production/development)
- `MODEL_PATH` — Path to YOLOv8 model weights (default: yolov8n.pt)
- `CONFIDENCE_THRESHOLD` — Detection confidence threshold (default: 0.25)
- `IOU_THRESHOLD` — NMS IoU threshold (default: 0.7)
- `UPLOAD_FOLDER` — Path for uploaded files
- `RESULTS_FOLDER` — Path for annotated results
- `MAX_CONTENT_LENGTH` — Max upload size in bytes (default: 52428800 = 50MB)

## Ports
- 5000 — Flask/gunicorn (internal, proxied by Caddy)

## Dependencies
- ultralytics (YOLOv8)
- opencv-python
- flask
- gunicorn
- numpy
- pillow
- albumentations
- matplotlib
- pyyaml
