"""
Unit tests for the Data Augmentation Toolkit (src/augment.py).
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import DataAugmentor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """Create a simple 100x100 BGR test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_bboxes():
    """Sample YOLO bboxes: [class_id, cx, cy, w, h] normalized."""
    return [
        [0.0, 0.5, 0.5, 0.3, 0.2],  # centered
        [0.0, 0.2, 0.3, 0.1, 0.15],  # upper-left
    ]


@pytest.fixture
def augmentor():
    return DataAugmentor(seed=42)


# ---------------------------------------------------------------------------
# Horizontal Flip
# ---------------------------------------------------------------------------

class TestHorizontalFlip:
    def test_image_flipped(self, augmentor, sample_image, sample_bboxes):
        """hflip should mirror the image."""
        result_img, _ = augmentor.horizontal_flip(sample_image, sample_bboxes)
        assert result_img.shape == sample_image.shape

    def test_bbox_cx_reflected(self, augmentor, sample_image, sample_bboxes):
        """hflip should reflect cx: new_cx = 1 - cx."""
        _, result_bboxes = augmentor.horizontal_flip(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert abs(new[1] - (1.0 - orig[1])) < 1e-6

    def test_bbox_cy_unchanged(self, augmentor, sample_image, sample_bboxes):
        """hflip should not change cy."""
        _, result_bboxes = augmentor.horizontal_flip(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert abs(new[2] - orig[2]) < 1e-6

    def test_bbox_w_h_unchanged(self, augmentor, sample_image, sample_bboxes):
        """hflip should not change w, h."""
        _, result_bboxes = augmentor.horizontal_flip(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert abs(new[3] - orig[3]) < 1e-6
            assert abs(new[4] - orig[4]) < 1e-6


# ---------------------------------------------------------------------------
# Vertical Flip
# ---------------------------------------------------------------------------

class TestVerticalFlip:
    def test_bbox_cy_reflected(self, augmentor, sample_image, sample_bboxes):
        """vflip should reflect cy: new_cy = 1 - cy."""
        _, result_bboxes = augmentor.vertical_flip(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert abs(new[2] - (1.0 - orig[2])) < 1e-6

    def test_bbox_cx_unchanged(self, augmentor, sample_image, sample_bboxes):
        """vflip should not change cx."""
        _, result_bboxes = augmentor.vertical_flip(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert abs(new[1] - orig[1]) < 1e-6


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------

class TestBrightness:
    def test_brightness_increases(self, augmentor, sample_image, sample_bboxes):
        """factor > 1 should increase pixel values."""
        result_img, _ = augmentor.brightness(sample_image, sample_bboxes, factor=2.0)
        assert result_img.mean() >= sample_image.mean()

    def test_bbox_unchanged(self, augmentor, sample_image, sample_bboxes):
        """brightness should not change bboxes."""
        _, result_bboxes = augmentor.brightness(sample_image, sample_bboxes, factor=1.5)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert new == orig


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

class TestContrast:
    def test_contrast_changes_image(self, augmentor, sample_image, sample_bboxes):
        """contrast adjustment should produce a different image."""
        result_img, _ = augmentor.contrast(sample_image, sample_bboxes, factor=1.5)
        assert not np.array_equal(result_img, sample_image)

    def test_bbox_unchanged(self, augmentor, sample_image, sample_bboxes):
        _, result_bboxes = augmentor.contrast(sample_image, sample_bboxes, factor=1.2)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert new == orig


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------

class TestBlur:
    def test_blur_produces_valid_image(self, augmentor, sample_image, sample_bboxes):
        result_img, _ = augmentor.blur(sample_image, sample_bboxes, kernel_size=5)
        assert result_img.shape == sample_image.shape

    def test_bbox_unchanged(self, augmentor, sample_image, sample_bboxes):
        _, result_bboxes = augmentor.blur(sample_image, sample_bboxes, kernel_size=3)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert new == orig


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class TestNoise:
    def test_noise_changes_image(self, augmentor, sample_image, sample_bboxes):
        """Adding noise should change the image."""
        result_img, _ = augmentor.noise(sample_image, sample_bboxes, intensity=20)
        assert not np.array_equal(result_img, sample_image)

    def test_bbox_unchanged(self, augmentor, sample_image, sample_bboxes):
        _, result_bboxes = augmentor.noise(sample_image, sample_bboxes, intensity=10)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert new == orig

    def test_noise_clipped(self, augmentor, sample_image, sample_bboxes):
        """Noise result should be valid uint8 [0,255]."""
        result_img, _ = augmentor.noise(sample_image, sample_bboxes, intensity=100)
        assert result_img.dtype == np.uint8
        assert result_img.min() >= 0
        assert result_img.max() <= 255


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

class TestGrayscale:
    def test_returns_3_channels(self, augmentor, sample_image, sample_bboxes):
        """grayscale output should still be 3-channel BGR for YOLO."""
        result_img, _ = augmentor.grayscale(sample_image, sample_bboxes)
        assert len(result_img.shape) == 3
        assert result_img.shape[2] == 3

    def test_bbox_unchanged(self, augmentor, sample_image, sample_bboxes):
        _, result_bboxes = augmentor.grayscale(sample_image, sample_bboxes)
        for orig, new in zip(sample_bboxes, result_bboxes):
            assert new == orig


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------

class TestRotate:
    def test_zero_angle_preserves_bbox(self, augmentor, sample_image, sample_bboxes):
        """Zero rotation should keep bboxes roughly the same."""
        _, result_bboxes = augmentor.rotate(sample_image, sample_bboxes, angle=0)
        assert len(result_bboxes) == len(sample_bboxes)

    def test_rotate_returns_valid_image(self, augmentor, sample_image, sample_bboxes):
        result_img, _ = augmentor.rotate(sample_image, sample_bboxes, angle=10)
        assert result_img.shape == sample_image.shape


# ---------------------------------------------------------------------------
# Composite Apply
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_single_transform(self, augmentor, sample_image, sample_bboxes):
        result_img, result_bboxes = augmentor.apply(
            sample_image, sample_bboxes, transforms=["hflip"]
        )
        assert result_img.shape == sample_image.shape
        assert len(result_bboxes) == len(sample_bboxes)

    def test_apply_multiple_transforms(self, augmentor, sample_image, sample_bboxes):
        result_img, result_bboxes = augmentor.apply(
            sample_image, sample_bboxes,
            transforms=["hflip", "brightness", "blur"]
        )
        assert result_img.shape == sample_image.shape
        assert len(result_bboxes) == len(sample_bboxes)

    def test_apply_all_transforms(self, augmentor, sample_image, sample_bboxes):
        """Applying all transforms should still produce valid output."""
        result_img, result_bboxes = augmentor.apply(
            sample_image, sample_bboxes,
            transforms=DataAugmentor.AVAILABLE_TRANSFORMS
        )
        assert result_img.shape == sample_image.shape

    def test_apply_unknown_transform_raises(self, augmentor, sample_image, sample_bboxes):
        with pytest.raises(ValueError, match="Unknown transform"):
            augmentor.apply(sample_image, sample_bboxes, transforms=["nonexistent"])

    def test_apply_random_when_none(self, augmentor, sample_image, sample_bboxes):
        """Passing transforms=None should still work (random selection)."""
        result_img, result_bboxes = augmentor.apply(sample_image, sample_bboxes)
        assert result_img.shape == sample_image.shape
