# Schema — Wooden Log Detection with YOLOv8

## YOLO Annotation Format
Each `.txt` file corresponds to one image. One line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1] relative to image dimensions.

For this project:
- **Class 0**: `wooden_log`

## data.yaml Structure
```yaml
path: /workspace/data
train: images/train
val: images/val
test: images/test

nc: 1
names: ['wooden_log']
```

## Dataset Directory Structure
```
data/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  sample_images/
```

## Detection Result Schema (JSON output)
```json
{
  "image_path": "uploads/test.jpg",
  "image_width": 640,
  "image_height": 480,
  "detections": [
    {
      "class_id": 0,
      "class_name": "wooden_log",
      "confidence": 0.92,
      "bbox": {
        "x1": 100, "y1": 200,
        "x2": 300, "y2": 450
      }
    }
  ],
  "count": 1,
  "processing_time_ms": 45.3
}
```

## Database
This project does not use a relational database. All data is file-based (images, annotations, model weights).
