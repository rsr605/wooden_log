"""
Unit tests for the Sample Data Generator (src/generate_samples.py).
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate_samples import SampleGenerator


class TestSampleGenerator:
    def test_generate_single_image_shape(self):
        """Generated image should have the requested dimensions."""
        gen = SampleGenerator(seed=42)
        image, bboxes = gen.generate_single(width=640, height=480)
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8

    def test_generate_single_has_bboxes(self):
        """At least one bounding box should be generated for a non-empty image."""
        gen = SampleGenerator(seed=42)
        _, bboxes = gen.generate_single(n_logs=3)
        assert len(bboxes) > 0

    def test_bboxes_in_valid_range(self):
        """All normalized bbox values should be in [0, 1]."""
        gen = SampleGenerator(seed=42)
        _, bboxes = gen.generate_single(width=320, height=240, n_logs=3)
        for b in bboxes:
            cx, cy, w, h = b[1], b[2], b[3], b[4]
            assert 0.0 <= cx <= 1.0, f"cx out of range: {cx}"
            assert 0.0 <= cy <= 1.0, f"cy out of range: {cy}"
            assert 0.0 < w <= 1.0, f"w out of range: {w}"
            assert 0.0 < h <= 1.0, f"h out of range: {h}"

    def test_class_id_is_zero(self):
        """All logs should have class_id = 0 (wooden_log)."""
        gen = SampleGenerator(seed=42)
        _, bboxes = gen.generate_single(n_logs=5)
        for b in bboxes:
            assert b[0] == 0.0

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        gen1 = SampleGenerator(seed=99)
        gen2 = SampleGenerator(seed=99)
        img1, bbox1 = gen1.generate_single(n_logs=3)
        img2, bbox2 = gen2.generate_single(n_logs=3)
        np.testing.assert_array_equal(img1, img2)
        assert bbox1 == bbox2

    def test_generate_dataset(self, tmp_path):
        """Dataset generation should create proper directory structure."""
        gen = SampleGenerator(seed=42)
        result = gen.generate_dataset(str(tmp_path), n_images=10, width=320, height=240)

        assert result["train"] == 8  # 80% of 10
        assert result["val"] == 2    # 20% of 10
        assert result["sample"] > 0

        # Check directories exist
        assert (tmp_path / "images" / "train").exists()
        assert (tmp_path / "images" / "val").exists()
        assert (tmp_path / "labels" / "train").exists()
        assert (tmp_path / "labels" / "val").exists()

        # Check images were created
        train_images = list((tmp_path / "images" / "train").glob("*.jpg"))
        assert len(train_images) == 8

    def test_generate_sample_images(self, tmp_path):
        """Standalone sample image generation."""
        gen = SampleGenerator(seed=42)
        paths = gen.generate_sample_images(str(tmp_path), n_images=3, width=200, height=150)
        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()
            img = cv2.imread(p)
            assert img is not None
            assert img.shape == (150, 200, 3)
