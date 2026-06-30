"""
Unit tests for Dataset Preprocessing (src/preprocess.py) and Utils (src/utils.py).
"""

import sys
import os
import json
import shutil
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    normalize_bbox,
    denormalize_bbox,
    get_file_extension,
    is_allowed_image,
    is_allowed_video,
    generate_hashed_filename,
    ensure_dir,
    format_detection_result,
)
from src.preprocess import DatasetPreprocessor


# ---------------------------------------------------------------------------
# Coordinate Conversion Tests
# ---------------------------------------------------------------------------

class TestCoordinateConversion:
    def test_xyxy_to_xywh(self):
        """Convert (0,0)-(100,100) → center (50,50), size (100,100)."""
        cx, cy, w, h = xyxy_to_xywh(0, 0, 100, 100)
        assert cx == 50.0
        assert cy == 50.0
        assert w == 100.0
        assert h == 100.0

    def test_xywh_to_xyxy(self):
        """Reverse conversion."""
        x1, y1, x2, y2 = xywh_to_xyxy(50, 50, 100, 100)
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 100.0
        assert y2 == 100.0

    def test_round_trip(self):
        """xyxy → xywh → xyxy should be identity."""
        import random
        rng = random.Random(42)
        for _ in range(50):
            x1, y1 = rng.uniform(0, 500), rng.uniform(0, 500)
            x2, y2 = rng.uniform(x1, 1000), rng.uniform(y1, 1000)
            cx, cy, w, h = xyxy_to_xywh(x1, y1, x2, y2)
            rx1, ry1, rx2, ry2 = xywh_to_xyxy(cx, cy, w, h)
            assert abs(rx1 - x1) < 1e-6
            assert abs(ry1 - y1) < 1e-6
            assert abs(rx2 - x2) < 1e-6
            assert abs(ry2 - y2) < 1e-6


class TestNormalization:
    def test_normalize_xyxy(self):
        """Normalize a 100x100 image bbox to [0,1]."""
        cx, cy, w, h = normalize_bbox(
            (0, 0, 100, 100), img_width=200, img_height=200, source_format="xyxy"
        )
        assert abs(cx - 0.25) < 1e-6  # center 50 / 200
        assert abs(cy - 0.25) < 1e-6
        assert abs(w - 0.5) < 1e-6  # width 100 / 200
        assert abs(h - 0.5) < 1e-6

    def test_denormalize(self):
        """Denormalize YOLO [0.5, 0.5, 0.4, 0.4] on 200x200."""
        x1, y1, x2, y2 = denormalize_bbox(
            (0.5, 0.5, 0.4, 0.4), img_width=200, img_height=200
        )
        # cx=100, cy=100, w=80, h=80 → x1=60, y1=60, x2=140, y2=140
        assert abs(x1 - 60) < 1e-6
        assert abs(y1 - 60) < 1e-6
        assert abs(x2 - 140) < 1e-6
        assert abs(y2 - 140) < 1e-6


# ---------------------------------------------------------------------------
# File Helper Tests
# ---------------------------------------------------------------------------

class TestFileHelpers:
    def test_get_extension(self):
        assert get_file_extension("image.JPG") == ".jpg"
        assert get_file_extension("file.tar.gz") == ".gz"
        assert get_file_extension("noext") == ""

    def test_is_allowed_image(self):
        assert is_allowed_image("photo.jpg") is True
        assert is_allowed_image("photo.PNG") is True
        assert is_allowed_image("photo.gif") is False
        assert is_allowed_image("doc.pdf") is False

    def test_is_allowed_video(self):
        assert is_allowed_video("clip.mp4") is True
        assert is_allowed_video("clip.AVI") is True
        assert is_allowed_video("photo.jpg") is False

    def test_generate_hashed_filename_preserves_extension(self):
        name = generate_hashed_filename("test.jpg", prefix="up_")
        assert name.startswith("up_")
        assert name.endswith(".jpg")
        assert len(name) > 8

    def test_ensure_dir(self, tmp_path):
        test_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()


# ---------------------------------------------------------------------------
# Detection Result Format Test
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_format_result(self):
        detections = [
            {
                "class_id": 0,
                "class_name": "wooden_log",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            }
        ]
        result = format_detection_result("img.jpg", 640, 480, detections, 42.5)
        assert result["image_path"] == "img.jpg"
        assert result["image_width"] == 640
        assert result["image_height"] == 480
        assert result["count"] == 1
        assert result["detections"] == detections
        assert result["processing_time_ms"] == 42.5


# ---------------------------------------------------------------------------
# Dataset Preprocessor Tests
# ---------------------------------------------------------------------------

class TestDatasetPreprocessor:
    def test_generate_data_yaml(self, tmp_path):
        """data.yaml should contain correct paths and class config."""
        pp = DatasetPreprocessor()
        yaml_path = pp.generate_data_yaml(
            str(tmp_path), output_path=str(tmp_path / "data.yaml")
        )
        content = Path(yaml_path).read_text()
        assert "train: images/train" in content
        assert "val: images/val" in content
        assert "test: images/test" in content
        assert "nc: 1" in content
        assert "wooden_log" in content

    def test_coco_to_yolo(self, tmp_path):
        """Convert a small COCO JSON to YOLO labels."""
        coco = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img2.jpg", "width": 200, "height": 200},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 60]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 30, 100, 80]},
            ],
            "categories": [
                {"id": 1, "name": "wooden_log"},
            ],
        }
        coco_path = tmp_path / "coco.json"
        coco_path.write_text(json.dumps(coco))

        out_dir = tmp_path / "labels"
        pp = DatasetPreprocessor()
        result = pp.coco_json_to_yolo(str(coco_path), str(out_dir))

        assert result["labels_created"] == 2
        # Check img1 label
        lbl1 = (out_dir / "img1.txt").read_text().strip()
        parts = lbl1.split()
        assert len(parts) == 5
        assert int(parts[0]) == 0  # class_id = 0
        # bbox center should be (10+25)/100 = 0.35
        assert abs(float(parts[1]) - 0.35) < 1e-6

    def test_split_dataset(self, tmp_path):
        """Split a small dataset into train/val/test."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create 10 image-label pairs
        import cv2
        import numpy as np
        for i in range(10):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"img{i}.jpg"), img)
            (labels_dir / f"img{i}.txt").write_text(
                f"0 0.5 0.5 0.3 0.3\n"
            )

        output_dir = tmp_path / "output"
        pp = DatasetPreprocessor(seed=42)
        result = pp.split_dataset(
            str(images_dir), str(labels_dir), str(output_dir),
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        )

        assert result["train"] == 7
        assert result["val"] == 2
        assert result["test"] == 1
        # Verify files exist
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "labels" / "train").exists()

    def test_split_invalid_ratios(self, tmp_path):
        """Ratios not summing to 1.0 should raise ValueError."""
        pp = DatasetPreprocessor()
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            pp.split_dataset(
                str(tmp_path), str(tmp_path), str(tmp_path),
                train_ratio=0.5, val_ratio=0.2, test_ratio=0.5,
            )

    def test_get_dataset_stats(self, tmp_path):
        """Stats should correctly count images, labels, and objects."""
        # Create a mini dataset
        root = tmp_path / "dataset"
        for split in ["train", "val"]:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            import cv2
            import numpy as np
            for i in range(3):
                img = np.zeros((50, 50, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)
                (lbl_dir / f"img{i}.txt").write_text(
                    f"0 0.5 0.5 0.3 0.3\n0 0.2 0.2 0.1 0.1\n"
                )

        pp = DatasetPreprocessor()
        stats = pp.get_dataset_stats(str(root))
        assert "train" in stats["splits"]
        assert stats["splits"]["train"]["images"] == 3
        assert stats["splits"]["train"]["total_objects"] == 6  # 2 per file
