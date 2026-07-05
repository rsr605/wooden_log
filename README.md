# 🪵 Wooden Log Detection with YOLOv8

A computer vision project that detects wooden logs in images and video using **YOLOv8** object detection and **OpenCV** for image processing. Built as a learning exercise with a full training pipeline, data augmentation toolkit, and a Flask web interface.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Web Detection UI** | Upload images/video → get annotated bounding boxes with confidence scores |
| **Training Pipeline** | Full YOLOv8 training with configurable hyperparameter presets |
| **Data Augmentation** | 8 transforms (flip, rotate, brightness, contrast, blur, noise, grayscale) with YOLO bbox adjustment |
| **Dataset Preprocessing** | Convert Pascal VOC XML / COCO JSON → YOLO format, train/val/test split |
| **Batch Prediction** | Process entire directories, export JSON/CSV results |
| **Sample Data Generator** | Synthetic wooden log images with annotations — works out of the box |
| **Model Export** | Export trained model to ONNX, TorchScript, OpenVINO |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **YOLOv8** (ultralytics) — object detection
- **OpenCV** (cv2) — image processing & annotation
- **Flask** — web application
- **NumPy** — numerical operations
- **Albumentations** — augmentation library support

---

## 📁 Project Structure

```
├── app/                    # Flask web application
│   ├── main.py             # Flask routes & entry point
│   ├── detector.py         # YOLOv8 detection wrapper
│   ├── templates/          # HTML templates (Jinja2)
│   └── static/             # CSS, JS, uploads, results
├── src/                    # Core ML utilities
│   ├── train.py            # Training pipeline
│   ├── augment.py          # Data augmentation toolkit
│   ├── preprocess.py       # Dataset preprocessing & format conversion
│   ├── generate_samples.py # Synthetic data generator
│   ├── predict.py          # Batch & video prediction CLI
│   ├── export.py           # Model export (ONNX, etc.)
│   └── utils.py            # Shared utilities
├── data/                   # Dataset & configuration
│   ├── data.yaml           # YOLO dataset config
│   └── sample_images/      # Generated demo images
├── models/                 # Trained model weights
├── tests/                  # Unit & integration tests
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Web App
```bash
python app/main.py
# OR with gunicorn (production):
gunicorn --bind 0.0.0.0:5000 --timeout 120 --workers 2 app.main:app
```
Then open `http://localhost:5000` in your browser.

### Generate Sample Data
```bash
python src/generate_samples.py --output data --n 50
```

### Train the Model
```bash
# Using a preset (fast / balanced / accurate)
python src/train.py --data data/data.yaml --preset balanced

# With custom epochs
python src/train.py --data data/data.yaml --preset fast --epochs 50 --batch 32
```

---

## 📖 Usage Guide

### Data Augmentation

```python
from src.augment import DataAugmentor

augmentor = DataAugmentor(seed=42)
img_aug, bbox_aug = augmentor.apply(
    image, bboxes,
    transforms=["hflip", "brightness", "blur"]
)
```

Available transforms: `hflip`, `vflip`, `rotate`, `brightness`, `contrast`, `blur`, `noise`, `grayscale`.

### Dataset Preprocessing

```python
from src.preprocess import DatasetPreprocessor

pp = DatasetPreprocessor()

# Split into train/val/test
pp.split_dataset("images/", "labels/", "dataset/", train_ratio=0.7)

# Convert COCO to YOLO
pp.coco_json_to_yolo("annotations.json", "yolo_labels/")

# Generate data.yaml
pp.generate_data_yaml("/path/to/dataset")
```

### Batch Prediction

```bash
# Process a directory
python src/predict.py batch --input images/ --output results/

# Process a video
python src/predict.py video --input input.mp4 --output output.mp4
```

### Model Export

```bash
python src/export.py --model models/best.pt --format onnx
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Upload form |
| `POST` | `/detect` | Upload & detect (web UI) |
| `POST` | `/api/detect` | Programmatic detection (JSON) |
| `GET` | `/api/info` | Model configuration info (JSON) |
| `GET` | `/health` | Health check |

### Example API Call
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "file=@test_image.jpg"
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_augment.py -v
```

---

## 📝 Training Presets

| Preset | Epochs | Image Size | Batch | Best For |
|--------|--------|-----------|-------|----------|
| `fast` | 50 | 416 | 32 | Quick experimentation |
| `balanced` | 100 | 640 | 16 | Good accuracy/time tradeoff |
| `accurate` | 200 | 640 | 8 | Maximum accuracy |

---

## 📄 License

This Project Created By Rohit
