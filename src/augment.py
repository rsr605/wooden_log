"""
Data Augmentation Toolkit for Wooden Log Detection.

Provides the DataAugmentor class with image + label transformations
that work together to preserve YOLO bounding box accuracy.
Designed to be used standalone or as a pre-training pipeline.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.utils import xywh_to_xyxy, xyxy_to_xywh, ensure_dir


class DataAugmentor:
    """
    Augmentation toolkit for YOLO-format datasets.

    Each transform takes an image (np.ndarray, BGR) and a list of YOLO
    bboxes (each: [class_id, cx, cy, w, h] normalized) and returns the
    transformed image and adjusted bboxes.

    Usage:
        augmentor = DataAugmentor()
        img_aug, bbox_aug = augmentor.apply(
            image, bboxes, transforms=["hflip", "brightness", "blur"]
        )
    """

    # Available transform names
    AVAILABLE_TRANSFORMS = [
        "hflip", "vflip", "rotate", "brightness",
        "contrast", "blur", "noise", "grayscale",
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    def horizontal_flip(
        self, image: np.ndarray, bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Flip image horizontally and adjust bboxes."""
        flipped = cv2.flip(image, 1)  # 1 = horizontal
        out = []
        for b in bboxes:
            cls, cx, cy, w, h = b
            new_cx = 1.0 - cx
            out.append([cls, new_cx, cy, w, h])
        return flipped, out

    def vertical_flip(
        self, image: np.ndarray, bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Flip image vertically and adjust bboxes."""
        flipped = cv2.flip(image, 0)  # 0 = vertical
        out = []
        for b in bboxes:
            cls, cx, cy, w, h = b
            new_cy = 1.0 - cy
            out.append([cls, cx, new_cy, w, h])
        return flipped, out

    def rotate(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        angle: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Rotate the image by a given angle (degrees).

        For angle=None, a random angle in [-15, 15] is used.
        Bounding boxes are rotated to match; boxes that move mostly
        out-of-frame are dropped.
        """
        if angle is None:
            angle = self.rng.uniform(-15, 15)

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, matrix, (w, h))

        # Compute new bounding box corners for each YOLO bbox
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])

        out = []
        for b in bboxes:
            cls, cx, cy, bw, bh = b
            # Convert to absolute pixel coords
            abs_w = bw * w
            abs_h = bh * h
            abs_cx = cx * w
            abs_cy = cy * h

            # Rotate center point
            new_cx = matrix[0, 0] * abs_cx + matrix[0, 1] * abs_cy + matrix[0, 2]
            new_cy = matrix[1, 0] * abs_cx + matrix[1, 1] * abs_cy + matrix[1, 2]

            # New bbox width/height (bounding rect of rotated box)
            new_w = abs_w * cos_a + abs_h * sin_a
            new_h = abs_w * sin_a + abs_h * cos_a

            # Normalize
            ncx = new_cx / w
            ncy = new_cy / h
            nw = new_w / w
            nh = new_h / h

            # Clamp center; skip if mostly outside
            if ncx < -0.1 or ncx > 1.1 or ncy < -0.1 or ncy > 1.1:
                continue
            ncx = max(0.0, min(1.0, ncx))
            ncy = max(0.0, min(1.0, ncy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            out.append([cls, ncx, ncy, nw, nh])

        return rotated_img, out

    def brightness(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust brightness. factor > 1 brightens, < 1 darkens."""
        if factor is None:
            factor = self.rng.uniform(0.6, 1.4)
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted, [list(b) for b in bboxes]

    def contrast(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust contrast."""
        if factor is None:
            factor = self.rng.uniform(0.7, 1.5)
        # alpha controls contrast, beta keeps mean roughly the same
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=128 * (1 - factor))
        return adjusted, [list(b) for b in bboxes]

    def blur(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        kernel_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Apply Gaussian blur."""
        if kernel_size is None:
            kernel_size = self.rng.choice([3, 5, 7])
        # Kernel must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred, [list(b) for b in bboxes]

    def noise(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        intensity: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Add Gaussian noise."""
        if intensity is None:
            intensity = self.rng.uniform(5, 25)
        noise_map = np.random.normal(0, intensity, image.shape)
        noisy = np.clip(image.astype(np.float64) + noise_map, 0, 255).astype(np.uint8)
        return noisy, [list(b) for b in bboxes]

    def grayscale(
        self, image: np.ndarray, bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Convert to grayscale (output kept as 3-channel BGR for YOLO)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray_3ch, [list(b) for b in bboxes]

    # ------------------------------------------------------------------
    # Composite transform application
    # ------------------------------------------------------------------

    # Map of transform name → method
    _TRANSFORM_MAP = {
        "hflip": "horizontal_flip",
        "vflip": "vertical_flip",
        "rotate": "rotate",
        "brightness": "brightness",
        "contrast": "contrast",
        "blur": "blur",
        "noise": "noise",
        "grayscale": "grayscale",
    }

    def apply(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        transforms: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply a sequence of transforms to an image + bboxes.

        Args:
            image: BGR numpy array.
            bboxes: List of [class_id, cx, cy, w, h] (normalized YOLO).
            transforms: List of transform names. If None, applies all random.

        Returns:
            (transformed_image, transformed_bboxes).
        """
        if transforms is None:
            transforms = self.AVAILABLE_TRANSFORMS.copy()
            self.rng.shuffle(transforms)
            # Apply 2-3 random transforms
            transforms = transforms[: self.rng.randint(2, 3)]

        # Validate
        for t in transforms:
            if t not in self._TRANSFORM_MAP:
                raise ValueError(f"Unknown transform: {t}. Available: {list(self._TRANSFORM_MAP.keys())}")

        current_img = image.copy()
        current_bboxes = [list(b) for b in bboxes]

        for t_name in transforms:
            method = getattr(self, self._TRANSFORM_MAP[t_name])
            current_img, current_bboxes = method(current_img, current_bboxes)

        return current_img, current_bboxes

    # ------------------------------------------------------------------
    # Dataset-level augmentation
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        n_augments: int = 3,
        transforms: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Augment an entire dataset folder.

        Args:
            images_dir: Path to source images.
            labels_dir: Path to source YOLO labels.
            output_images_dir: Where to write augmented images.
            output_labels_dir: Where to write augmented labels.
            n_augments: Number of augmented versions per source image.
            transforms: Specific transforms; None = random subset.

        Returns:
            Dict with 'images_created', 'labels_created', 'skipped'.
        """
        ensure_dir(output_images_dir)
        ensure_dir(output_labels_dir)

        images_created = 0
        labels_created = 0
        skipped = 0

        img_paths = sorted(Path(images_dir).glob("*"))
        for img_path in img_paths:
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                skipped += 1
                continue

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                skipped += 1
                continue

            # Read labels
            label_path = Path(labels_dir) / (img_path.stem + ".txt")
            bboxes = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            bboxes.append([
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                            ])

            # Generate augmentations
            for i in range(n_augments):
                aug_img, aug_bboxes = self.apply(image, bboxes, transforms=transforms)
                out_name = f"{img_path.stem}_aug{i}"
                out_img_path = Path(output_images_dir) / f"{out_name}{img_path.suffix}"
                out_lbl_path = Path(output_labels_dir) / f"{out_name}.txt"

                cv2.imwrite(str(out_img_path), aug_img)
                with open(out_lbl_path, "w") as f:
                    for b in aug_bboxes:
                        f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                images_created += 1
                labels_created += 1

        return {
            "images_created": images_created,
            "labels_created": labels_created,
            "skipped": skipped,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry: augment a dataset folder."""
    import argparse

    parser = argparse.ArgumentParser(description="Augment a YOLO dataset")
    parser.add_argument("--images", required=True, help="Source images directory")
    parser.add_argument("--labels", required=True, help="Source labels directory")
    parser.add_argument("--out-images", required=True, help="Output images directory")
    parser.add_argument("--out-labels", required=True, help="Output labels directory")
    parser.add_argument("--n", type=int, default=3, help="Augmentations per image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=None,
        help=f"Transforms: {DataAugmentor.AVAILABLE_TRANSFORMS}",
    )
    args = parser.parse_args()

    augmentor = DataAugmentor(seed=args.seed)
    result = augmentor.augment_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_images_dir=args.out_images,
        output_labels_dir=args.out_labels,
        n_augments=args.n,
        transforms=args.transforms,
    )
    print(f"Augmentation complete: {result}")


if __name__ == "__main__":
    main()
