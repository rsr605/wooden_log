# Architecture — Wooden Log Detection with YOLOv8

## Directory Structure
```
/workspace/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── main.py                  # Flask routes & main entry
│   ├── detector.py              # YOLOv8 detection wrapper
│   ├── templates/
│   │   ├── base.html            # Base layout with nav
│   │   ├── index.html           # Upload form
│   │   └── result.html          # Detection results
│   └── static/
│       ├── css/style.css        # Styling
│       ├── js/app.js            # Frontend JS
│       ├── uploads/             # Uploaded images
│       └── results/             # Annotated results
├── src/
│   ├── __init__.py
│   ├── train.py                 # Training script
│   ├── augment.py               # Augmentation toolkit
│   ├── preprocess.py            # Dataset preprocessing
│   ├── generate_samples.py      # Sample data generator
│   ├── predict.py               # CLI prediction
│   ├── export.py                # Model export
│   └── utils.py                 # Shared utilities
├── data/
│   ├── data.yaml                # YOLO dataset config
│   └── sample_images/           # Generated demo images
├── models/                      # Weight files
├── tests/                       # Unit & integration tests
├── requirements.txt
└── README.md
```

## Data Flow

### Web App Detection Flow
```
User uploads image
  → Flask route receives file
  → app/detector.py: LogDetector.detect(image)
    → YOLOv8 model inference
    → Extract bounding boxes, confidences, class names
    → OpenCV draws annotations
  → Save annotated image to static/results/
  → Render result.html with image + detection stats
```

### Training Flow
```
User runs: python src/train.py --data data/data.yaml --epochs 100
  → Load YOLOv8 model (pre-trained or from checkpoint)
  → Apply augmentation from src/augment.py
  → Train via ultralytics API
  → Save weights to models/
  → Output metrics
```

### Preprocessing Flow
```
Raw dataset (images + XML/JSON annotations)
  → src/preprocess.py: convert to YOLO format
  → Split into train/val/test
  → Generate data.yaml
  → Ready for training
```

## Component Responsibilities
- **LogDetector** (`app/detector.py`): Wraps YOLOv8, runs inference, draws boxes
- **Flask App** (`app/main.py`): HTTP routes, file handling, template rendering
- **DataAugmentor** (`src/augment.py`): Applies image + label transforms
- **DatasetPreprocessor** (`src/preprocess.py`): Format conversion, splitting
- **SampleGenerator** (`src/generate_samples.py`): Synthetic data creation
- **PredictionCLI** (`src/predict.py`): Batch & video processing
- **ModelExporter** (`src/export.py`): Format conversion

## Ports
- Flask app listens on **port 5000** (internal)
- Caddy reverse proxy at `/` → port 5000
