"""
Synthetic Wooden Log Sample Data Generator.

Generates synthetic images containing wooden log shapes using OpenCV drawing
functions, along with corresponding YOLO-format annotations. This allows the
detection app to demonstrate functionality out-of-the-box without requiring
an external dataset.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from src.utils import ensure_dir, xyxy_to_xywh


class SampleGenerator:
    """
    Generate synthetic wooden log images for demonstration.

    Creates images with cylindrical log shapes (drawn as ellipses + rectangles)
    in various colors, sizes, positions, and backgrounds. Produces matching
    YOLO-format annotations.
    """

    # Log color palette (BGR)
    LOG_COLORS = [
        (101, 67, 33),    # brown
        (86, 58, 28),     # dark brown
        (120, 80, 40),    # light brown
        (65, 45, 20),     # dark walnut
        (140, 95, 50),    # tan
        (90, 60, 30),     # medium brown
    ]

    # Background colors (BGR, muted natural tones)
    BG_COLORS = [
        (60, 70, 55),     # forest
        (80, 90, 70),     # olive green
        (90, 100, 80),    # moss
        (70, 80, 65),     # dark green
        (100, 110, 90),   # sage
        (50, 60, 45),     # dark forest
    ]

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def _generate_background(self, width: int, height: int) -> np.ndarray:
        """Create a textured background (grass/forest-like)."""
        base_color = self.rng.choice(self.BG_COLORS)
        # Create base image
        bg = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add texture: random noise
        noise = self._np_rng.randint(-20, 20, (height, width, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some random darker patches (shadows)
        for _ in range(self.rng.randint(3, 8)):
            px = self.rng.randint(0, width)
            py = self.rng.randint(0, height)
            radius = self.rng.randint(20, 60)
            overlay = bg.copy()
            cv2.circle(overlay, (px, py), radius, (30, 40, 25), -1)
            cv2.addWeighted(overlay, 0.3, bg, 0.7, 0, bg)

        return bg

    def _draw_log(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        angle: Optional[float] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Draw a wooden log (horizontal cylinder shape) on the image.

        Args:
            image: Image to draw on (modified in-place).
            x, y: Top-left position.
            w, h: Width and height of the log.
            angle: Rotation angle in degrees (None = random).

        Returns:
            Bounding box (x1, y1, x2, y2) in pixel coordinates.
        """
        if angle is None:
            angle = self.rng.choice([0, 0, 0, 15, -15, 90, -90])

        color = self.rng.choice(self.LOG_COLORS)

        # Draw log body: rounded rectangle (ellipse + rectangle)
        # Create a rotated log on a separate canvas, then overlay
        log_canvas = np.zeros((h + 40, w + 40, 3), dtype=np.uint8)
        cx_local = (w + 40) // 2
        cy_local = (h + 40) // 2

        # Main body (rectangle with rounded ends)
        cv2.rectangle(log_canvas, (20, 15), (20 + w, 15 + h), color, -1)
        # Left cap (ellipse)
        cv2.ellipse(log_canvas, (20, cy_local), (h // 2, h // 2), 0, 90, 270, color, -1)
        # Right cap (ellipse)
        cv2.ellipse(log_canvas, (20 + w, cy_local), (h // 2, h // 2), 0, -90, 90, color, -1)

        # Add wood grain lines
        grain_color = tuple(max(0, c - 30) for c in color)
        for i in range(3):
            gy = cy_local - h // 4 + i * (h // 4)
            cv2.line(log_canvas, (25, gy), (15 + w, gy), grain_color, 1, cv2.LINE_AA)

        # Add end ring pattern (concentric ellipses on the caps)
        ring_color = tuple(min(255, c + 20) for c in color)
        for r_factor in [0.3, 0.5, 0.7]:
            r = int(h // 2 * r_factor)
            cv2.ellipse(log_canvas, (20, cy_local), (r, r), 0, 90, 270, ring_color, 1)
            cv2.ellipse(log_canvas, (20 + w, cy_local), (r, r), 0, -90, 90, ring_color, 1)

        # Rotate if needed
        if angle != 0:
            rot_matrix = cv2.getRotationMatrix2D((cx_local, cy_local), angle, 1.0)
            log_canvas = cv2.warpAffine(log_canvas, rot_matrix, (w + 40, h + 40))

        # Overlay onto main image
        # Determine paste region
        paste_x = max(0, x - 20)
        paste_y = max(0, y - 20)
        paste_w = min(w + 40, image.shape[1] - paste_x)
        paste_h = min(h + 40, image.shape[0] - paste_y)

        if paste_w <= 0 or paste_h <= 0:
            return (x, y, x + w, y + h)

        src_region = log_canvas[:paste_h, :paste_w]
        dst_region = image[paste_y: paste_y + paste_h, paste_x: paste_x + paste_w]

        # Blend only non-black pixels (mask-based overlay)
        mask = cv2.cvtColor(src_region, cv2.COLOR_BGR2GRAY)
        mask = mask > 0
        dst_region[mask] = src_region[mask]

        return (paste_x, paste_y, paste_x + paste_w, paste_y + paste_h)

    def generate_single(
        self,
        width: int = 640,
        height: int = 480,
        n_logs: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Generate a single synthetic image with wooden logs.

        Args:
            width: Image width.
            height: Image height.
            n_logs: Number of logs to place (None = random 1-5).

        Returns:
            Tuple of (image, yolo_bboxes) where yolo_bboxes is a list of
            [class_id, cx, cy, w, h] normalized.
        """
        image = self._generate_background(width, height)

        if n_logs is None:
            n_logs = self.rng.randint(1, 5)

        bboxes: List[List[float]] = []

        for _ in range(n_logs):
            min_w, max_w = 40, max(60, width // 3)
            min_h, max_h = 20, max(40, height // 4)
            log_w = self.rng.randint(min_w, max(min_w, min(250, max_w)))
            log_h = self.rng.randint(min_h, max(min_h, min(100, max_h)))
            max_x = max(11, width - log_w - 10)
            max_y = max(11, height - log_h - 10)
            x = self.rng.randint(10, max_x)
            y = self.rng.randint(10, max_y)

            x1, y1, x2, y2 = self._draw_log(image, x, y, log_w, log_h)

            # Convert to YOLO normalized format
            cx, cy, bw, bh = xyxy_to_xywh(x1, y1, x2, y2)
            bboxes.append([
                0.0,  # class_id = 0 (wooden_log)
                cx / width,
                cy / height,
                bw / width,
                bh / height,
            ])

        return image, bboxes

    def generate_dataset(
        self,
        output_dir: str,
        n_images: int = 50,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, int]:
        """
        Generate a full synthetic dataset with train/val split.

        Creates images and labels directories with an 80/20 split.

        Args:
            output_dir: Root output directory.
            n_images: Total number of images to generate.
            width: Image width.
            height: Image height.

        Returns:
            Dict with generation stats.
        """
        root = Path(output_dir)

        # Create split directories
        for split in ["train", "val"]:
            ensure_dir(root / "images" / split)
            ensure_dir(root / "labels" / split)

        # Also keep a flat sample_images directory for easy web demo
        sample_dir = root / "sample_images"
        ensure_dir(sample_dir)

        n_train = int(n_images * 0.8)
        splits = {
            "train": n_train,
            "val": n_images - n_train,
        }

        counts = {"train": 0, "val": 0, "sample": 0}
        idx = 0

        for split, count in splits.items():
            for _ in range(count):
                image, bboxes = self.generate_single(width, height)

                name = f"log_{idx:04d}"
                img_path = root / "images" / split / f"{name}.jpg"
                lbl_path = root / "labels" / split / f"{name}.txt"

                cv2.imwrite(str(img_path), image)
                with open(lbl_path, "w") as f:
                    for b in bboxes:
                        f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                counts[split] += 1
                idx += 1

        # Generate a few sample images for the web demo (no labels needed)
        for i in range(min(5, n_images)):
            image, _ = self.generate_single(width, height)
            cv2.imwrite(str(sample_dir / f"sample_{i}.jpg"), image)
            counts["sample"] += 1

        return counts

    def generate_sample_images(
        self,
        output_dir: str,
        n_images: int = 5,
        width: int = 640,
        height: int = 480,
    ) -> List[str]:
        """
        Generate sample images for web demo (no labels).

        Args:
            output_dir: Output directory for images.
            n_images: Number of images.
            width: Image width.
            height: Image height.

        Returns:
            List of generated file paths.
        """
        ensure_dir(output_dir)
        paths = []
        for i in range(n_images):
            image, _ = self.generate_single(width, height)
            path = str(Path(output_dir) / f"sample_{i}.jpg")
            cv2.imwrite(path, image)
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry: generate synthetic sample data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic wooden log samples")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--n", type=int, default=50, help="Number of images")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gen = SampleGenerator(seed=args.seed)
    result = gen.generate_dataset(
        output_dir=args.output,
        n_images=args.n,
        width=args.width,
        height=args.height,
    )
    print(f"Sample data generated: {result}")

    # Also generate data.yaml
    root = Path(args.output).resolve()
    yaml_content = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names: ['wooden_log']\n"
    )
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    main()
