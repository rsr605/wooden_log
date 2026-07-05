"""
Synthetic Wooden Log Sample Data Generator — v5 (Recall-Focused).

Evolution from v4: tuned specifically to maximize detection RECALL so the
model learns to find every visible log, including small, occluded, and
edge-partial ones.

Key changes over v4:
- More small/tiny logs (weight shifted toward smaller size classes)
- Denser end-view piles (up to 55 logs with tighter packing)
- More partial/edge logs (40% vs 30%)
- More mixed side+end scenes (30% vs 20%)
- Sparse scenes (1-3 logs) so the model also learns isolated logs
- Higher base radius range for end-view variety
- More overlap/occlusion between logs
"""

import os
import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from src.utils import ensure_dir, xyxy_to_xywh

# Reuse all the rendering methods from v3/v4 generator — only the
# scene-layout parameters change. Import the full class.
from src.generate_samples import SampleGenerator as _BaseGenerator


class SampleGeneratorV5(_BaseGenerator):
    """
    v5 generator — recall-focused scene layout.

    Inherits all rendering methods (bark texture, rings, knots, augmentation)
    from the base v3/v4 generator. Only overrides the scene-generation logic
    to produce:
      - More small/tiny logs (the hardest to detect)
      - Denser piles with more overlap
      - More edge-partials and occluded logs
      - Sparse isolated-log scenes
      - More mixed side+end compositions
    """

    def generate_single(
        self,
        width: int = 640,
        height: int = 480,
        n_logs: Optional[int] = None,
        view_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Generate a single synthetic image with wooden logs (v5 layout).

        Changes from v4:
        - view_mode weights: side 25%, end 45%, mixed 30% (was 30/50/20)
        - End-view: up to 55 logs (was 40), tighter packing
        - Side-view: more small logs, more edge partials
        - New sparse-scene mode (10% of side-view scenes have 1-3 logs)
        """
        image = self._generate_background(width, height)

        if view_mode is None:
            view_mode = self.rng.choices(
                population=["side", "end", "mixed"],
                weights=[25, 45, 30],  # v5: more mixed
            )[0]

        # End-view scenes get dense packing (18-55 logs) — v5: higher max
        if view_mode in ("end", "mixed") and n_logs is None:
            n_logs = self.rng.choices(
                population=[18, 20, 22, 25, 28, 30, 35, 40, 45, 50, 55],
                weights=[6, 8, 10, 12, 12, 12, 10, 8, 6, 4, 3],
            )[0]
        elif n_logs is None:
            # Side-view: 10% chance of sparse scene (1-3 logs)
            if self.rng.random() < 0.10:
                n_logs = self.rng.choices(
                    population=[1, 2, 3],
                    weights=[3, 4, 3],
                )[0]
            else:
                n_logs = self.rng.choices(
                    population=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                    weights=[4, 8, 10, 12, 12, 12, 10, 8, 6, 4],
                )[0]

        if view_mode == "end":
            bboxes = self._generate_end_view_pile_v5(image, width, height, n_logs)
        elif view_mode == "mixed":
            n_side = max(1, n_logs // 4)
            bboxes = self._generate_side_view_scene_v5(image, width, height, n_side)
            n_end = n_logs - n_side
            bboxes += self._generate_end_view_pile_v5(image, width, height, n_end)
        else:
            bboxes = self._generate_side_view_scene_v5(image, width, height, n_logs)

        # Ensure at least one log
        if not bboxes:
            log_w = self.rng.randint(60, 120)
            log_h = self.rng.randint(40, 60)
            x = self.rng.randint(50, max(51, width - log_w - 50))
            y = self.rng.randint(50, max(51, height - log_h - 50))
            x1, y1, x2, y2 = self._draw_log(image, x, y, log_w, log_h)
            cx, cy, bw, bh = xyxy_to_xywh(x1, y1, x2, y2)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])

        # Apply augmentation
        image = self._augment(image)

        return image, bboxes

    def _generate_side_view_scene_v5(
        self, image: np.ndarray, width: int, height: int, n_logs: int
    ) -> List[List[float]]:
        """
        Side-view scene — v5: more small/tiny logs, more edge partials.

        Size class weights shifted toward smaller sizes:
          v4: tiny 12%, small 25%, medium 30%, large 20%, xlarge 13%
          v5: tiny 18%, small 30%, medium 27%, large 15%, xlarge 10%
        """
        bboxes: List[List[float]] = []

        for _ in range(n_logs):
            size_class = self.rng.choices(
                population=["tiny", "small", "medium", "large", "xlarge"],
                weights=[18, 30, 27, 15, 10],  # v5: more small/tiny
            )[0]

            if size_class == "tiny":
                log_w = self.rng.randint(22, 42)    # v5: slightly smaller floor
                log_h = self.rng.randint(16, 28)
            elif size_class == "small":
                log_w = self.rng.randint(42, 72)
                log_h = self.rng.randint(26, 48)
            elif size_class == "medium":
                log_w = self.rng.randint(72, 125)
                log_h = self.rng.randint(42, 72)
            elif size_class == "large":
                log_w = self.rng.randint(125, 195)
                log_h = self.rng.randint(62, 100)
            else:
                log_w = self.rng.randint(195, 310)
                log_h = self.rng.randint(88, 135)

            # v5: 40% edge-partial (was 30%)
            allow_edge = self.rng.random() < 0.40
            if allow_edge:
                x = self.rng.randint(-35, max(1, width - log_w + 35))
                y = self.rng.randint(-35, max(1, height - log_h + 35))
            else:
                max_x = max(11, width - log_w - 10)
                max_y = max(11, height - log_h - 10)
                x = self.rng.randint(5, max_x)
                y = self.rng.randint(5, max_y)

            x1, y1, x2, y2 = self._draw_log(image, x, y, log_w, log_h)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            if (x2_c - x1_c) < 10 or (y2_c - y1_c) < 7:
                continue  # too small after clipping

            cx, cy, bw, bh = xyxy_to_xywh(x1_c, y1_c, x2_c, y2_c)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])

        return bboxes

    def _generate_end_view_pile_v5(
        self, image: np.ndarray, width: int, height: int, n_logs: int
    ) -> List[List[float]]:
        """
        End-view pile — v5: denser packing, more size variety, more overlap.

        Changes from v4:
        - Wider base radius range (20-60 vs 25-55)
        - More attempts (n_logs * 4 vs * 3) to pack tighter
        - Smaller minimum visible area threshold (10 vs 12) so occluded
          logs are still annotated
        - More elliptical variation (35% vs 30%)
        """
        bboxes: List[List[float]] = []

        # v5: wider base radius range
        base_radius = self.rng.randint(20, 60)

        placed = 0
        max_attempts = n_logs * 4  # v5: more attempts for denser packing
        attempts = 0

        while placed < n_logs and attempts < max_attempts:
            attempts += 1

            # Random position within the image
            gx = self.rng.randint(15, max(16, width - 15))
            gy = self.rng.randint(15, max(16, height - 15))

            # Size variation within the pile — v5: wider variation
            size_var = self.rng.uniform(0.45, 1.5)
            log_w = int(base_radius * 2 * size_var)
            log_h = int(base_radius * 2 * size_var)

            # v5: more logs slightly elliptical (35% vs 30%)
            if self.rng.random() < 0.35:
                log_h = int(log_h * self.rng.uniform(0.70, 0.95))

            x = gx - log_w // 2
            y = gy - log_h // 2

            x1, y1, x2, y2 = self._draw_end_view_log(image, x, y, log_w, log_h)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            # v5: smaller minimum (10 vs 12) so occluded logs still annotated
            if (x2_c - x1_c) < 10 or (y2_c - y1_c) < 10:
                continue

            cx, cy, bw, bh = xyxy_to_xywh(x1_c, y1_c, x2_c, y2_c)
            bboxes.append([0.0, cx / width, cy / height, bw / width, bh / height])
            placed += 1

        return bboxes

    def generate_dataset(
        self,
        output_dir: str,
        n_images: int = 2500,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, int]:
        """Generate a full synthetic dataset with train/val split (v5)."""
        root = Path(output_dir)

        for split in ["train", "val"]:
            ensure_dir(root / "images" / split)
            ensure_dir(root / "labels" / split)

        sample_dir = root / "sample_images"
        ensure_dir(sample_dir)

        n_train = int(n_images * 0.8)
        splits = {"train": n_train, "val": n_images - n_train}

        counts = {"train": 0, "val": 0, "sample": 0}
        idx = 0

        # v5: more resolution variety
        resolutions = [(640, 480), (800, 600), (512, 512), (1024, 768)]

        for split, count in splits.items():
            for _ in range(count):
                # 60% standard resolution, 40% varied (v5: more variety)
                if self.rng.random() < 0.60:
                    iw, ih = width, height
                else:
                    iw, ih = self.rng.choice(resolutions)

                image, bboxes = self.generate_single(iw, ih)

                name = f"log_{idx:04d}"
                img_path = root / "images" / split / f"{name}.jpg"
                lbl_path = root / "labels" / split / f"{name}.txt"

                cv2.imwrite(str(img_path), image)
                with open(lbl_path, "w") as f:
                    for b in bboxes:
                        f.write(
                            f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} "
                            f"{b[3]:.6f} {b[4]:.6f}\n"
                        )

                counts[split] += 1
                idx += 1

        # Generate sample images for the web demo
        for i in range(min(5, n_images)):
            image, _ = self.generate_single(width, height)
            cv2.imwrite(str(sample_dir / f"sample_{i}.jpg"), image)
            counts["sample"] += 1

        return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry: generate v5 synthetic sample data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate v5 synthetic wooden log samples (recall-focused)"
    )
    parser.add_argument("--output", default="data_v5", help="Output directory")
    parser.add_argument("--n", type=int, default=2500, help="Number of images")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gen = SampleGeneratorV5(seed=args.seed)
    result = gen.generate_dataset(
        output_dir=args.output,
        n_images=args.n,
        width=args.width,
        height=args.height,
    )
    print(f"v5 sample data generated: {result}")

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
