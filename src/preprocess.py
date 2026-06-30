"""
Dataset Preprocessing for YOLO-format Datasets.

Converts various annotation formats (Pascal VOC XML, COCO JSON) to YOLO format,
creates train/val/test splits, generates data.yaml configuration, and reports
dataset statistics.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    ensure_dir,
    xyxy_to_xywh,
    CLASS_NAMES,
)


class DatasetPreprocessor:
    """
    Convert raw datasets to YOLO format and create train/val/test splits.

    Supports input formats:
      - Pascal VOC XML annotations
      - COCO JSON annotations
      - Already-YOLO format (just splitting)

    Output structure (YOLO standard):
      output_dir/
        images/ train/ val/ test/
        labels/ train/ val/ test/
        data.yaml
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Args:
            class_names: Ordered list of class names (index = class_id).
            seed: Random seed for reproducible splits.
        """
        self.class_names = class_names or CLASS_NAMES
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Format Conversion: Pascal VOC XML → YOLO
    # ------------------------------------------------------------------

    def voc_xml_to_yolo(self, xml_path: str, img_width: int, img_height: int) -> List[List[float]]:
        """
        Parse a Pascal VOC XML annotation file and convert to YOLO format.

        Args:
            xml_path: Path to the XML file.
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            List of [class_id, cx, cy, w, h] (normalized).
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        yolo_labels: List[List[float]] = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            class_name = name_elem.text.strip()

            # Map class name to ID
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
            else:
                # Unknown class — skip
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            cx, cy, w, h = xyxy_to_xywh(xmin, ymin, xmax, ymax)
            yolo_labels.append([
                float(class_id),
                cx / img_width,
                cy / img_height,
                w / img_width,
                h / img_height,
            ])

        return yolo_labels

    # ------------------------------------------------------------------
    # Format Conversion: COCO JSON → YOLO
    # ------------------------------------------------------------------

    def coco_json_to_yolo(
        self,
        coco_json_path: str,
        output_labels_dir: str,
    ) -> Dict[str, int]:
        """
        Convert a COCO-format annotation file to YOLO label files.

        Args:
            coco_json_path: Path to the COCO JSON file.
            output_labels_dir: Directory to write .txt label files.

        Returns:
            Dict with conversion stats.
        """
        ensure_dir(output_labels_dir)

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        # Build category map: category_id → class index
        cat_map: Dict[int, int] = {}
        for i, cat in enumerate(coco.get("categories", [])):
            if cat["name"] in self.class_names:
                cat_map[cat["id"]] = self.class_names.index(cat["name"])

        # Build image map: image_id → (filename, width, height)
        img_map: Dict[int, Tuple[str, int, int]] = {}
        for img in coco.get("images", []):
            img_map[img["id"]] = (img["file_name"], img["width"], img["height"])

        # Group annotations by image
        annotations_by_image: Dict[int, List[List[float]]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_map:
                continue
            cat_id = ann["category_id"]
            if cat_id not in cat_map:
                continue
            class_id = cat_map[cat_id]

            # COCO bbox = [x, y, width, height] (top-left corner)
            x, y, w, h = ann["bbox"]
            filename, img_w, img_h = img_map[img_id]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append([float(class_id), cx, cy, nw, nh])

        # Write label files
        files_created = 0
        for img_id, labels in annotations_by_image.items():
            filename, _, _ = img_map[img_id]
            stem = Path(filename).stem
            label_path = Path(output_labels_dir) / f"{stem}.txt"
            with open(label_path, "w") as f:
                for l in labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
            files_created += 1

        return {
            "labels_created": files_created,
            "total_images": len(img_map),
            "total_annotations": len(coco.get("annotations", [])),
        }

    # ------------------------------------------------------------------
    # Train / Val / Test Split
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        copy: bool = True,
    ) -> Dict[str, int]:
        """
        Split a dataset into train/val/test sets.

        Args:
            images_dir: Source images directory.
            labels_dir: Source labels directory.
            output_dir: Root output directory.
            train_ratio: Fraction for training (default 0.7).
            val_ratio: Fraction for validation (default 0.2).
            test_ratio: Fraction for testing (default 0.1).
            copy: If True, copy files; if False, move them.

        Returns:
            Dict with split counts.
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01:
            raise ValueError("Ratios must sum to 1.0")

        # Create output directories
        for split in ["train", "val", "test"]:
            ensure_dir(Path(output_dir) / "images" / split)
            ensure_dir(Path(output_dir) / "labels" / split)

        # Collect valid image-label pairs
        img_dir = Path(images_dir)
        lbl_dir = Path(labels_dir)

        valid_pairs: List[Tuple[Path, Optional[Path]]] = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            valid_pairs.append((img_path, label_path if label_path.exists() else None))

        self.rng.shuffle(valid_pairs)
        n = len(valid_pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": valid_pairs[:n_train],
            "val": valid_pairs[n_train : n_train + n_val],
            "test": valid_pairs[n_train + n_val :],
        }

        counts = {"train": 0, "val": 0, "test": 0}
        for split_name, pairs in splits.items():
            for img_path, label_path in pairs:
                if copy:
                    shutil.copy2(img_path, Path(output_dir) / "images" / split_name / img_path.name)
                    if label_path:
                        shutil.copy2(label_path, Path(output_dir) / "labels" / split_name / label_path.name)
                else:
                    shutil.move(str(img_path), str(Path(output_dir) / "images" / split_name / img_path.name))
                    if label_path:
                        shutil.move(str(label_path), str(Path(output_dir) / "labels" / split_name / label_path.name))
                counts[split_name] += 1

        return counts

    # ------------------------------------------------------------------
    # Generate data.yaml
    # ------------------------------------------------------------------

    def generate_data_yaml(
        self,
        dataset_root: str,
        output_path: Optional[str] = None,
        nc: Optional[int] = None,
        names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a YOLO data.yaml configuration file.

        Args:
            dataset_root: Absolute path to the dataset root.
            output_path: Path for data.yaml; defaults to dataset_root/data.yaml.
            nc: Number of classes; defaults to len(self.class_names).
            names: Class names; defaults to self.class_names.

        Returns:
            Path to the generated data.yaml.
        """
        if nc is None:
            nc = len(self.class_names)
        if names is None:
            names = self.class_names

        if output_path is None:
            output_path = str(Path(dataset_root) / "data.yaml")

        yaml_content = (
            f"# YOLOv8 Dataset Configuration\n"
            f"# Generated by Wooden Log Detection preprocessor\n\n"
            f"path: {dataset_root}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"nc: {nc}\n"
            f"names: {names}\n"
        )

        with open(output_path, "w") as f:
            f.write(yaml_content)

        return output_path

    # ------------------------------------------------------------------
    # Dataset Statistics
    # ------------------------------------------------------------------

    def get_dataset_stats(self, dataset_root: str) -> Dict[str, Any]:
        """
        Compute statistics for a YOLO-format dataset.

        Args:
            dataset_root: Path to the dataset root (with images/ and labels/).

        Returns:
            Dict with per-split image/label counts and class distribution.
        """
        root = Path(dataset_root)
        stats: Dict[str, Any] = {"splits": {}}

        for split in ["train", "val", "test"]:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split

            if not img_dir.exists():
                continue

            images = list(img_dir.glob("*"))
            labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

            class_counts: Dict[int, int] = {}
            total_objects = 0
            for lbl in labels:
                with open(lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid = int(float(parts[0]))
                            class_counts[cid] = class_counts.get(cid, 0) + 1
                            total_objects += 1

            stats["splits"][split] = {
                "images": len(images),
                "labels": len(labels),
                "total_objects": total_objects,
                "class_counts": class_counts,
            }

        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry for dataset preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset for YOLO training")
    sub = parser.add_subparsers(dest="command")

    # Convert VOC XML
    p_voc = sub.add_parser("voc", help="Convert Pascal VOC XML to YOLO")
    p_voc.add_argument("--xml", required=True, help="XML annotations directory")
    p_voc.add_argument("--images", required=True, help="Images directory")
    p_voc.add_argument("--output", required=True, help="Output labels directory")

    # Split dataset
    p_split = sub.add_parser("split", help="Create train/val/test split")
    p_split.add_argument("--images", required=True)
    p_split.add_argument("--labels", required=True)
    p_split.add_argument("--output", required=True)
    p_split.add_argument("--train", type=float, default=0.7)
    p_split.add_argument("--val", type=float, default=0.2)
    p_split.add_argument("--test", type=float, default=0.1)

    # Generate data.yaml
    p_yaml = sub.add_parser("yaml", help="Generate data.yaml")
    p_yaml.add_argument("--root", required=True, help="Dataset root")

    # Stats
    p_stats = sub.add_parser("stats", help="Print dataset statistics")
    p_stats.add_argument("--root", required=True, help="Dataset root")

    args = parser.parse_args()
    pp = DatasetPreprocessor()

    if args.command == "split":
        result = pp.split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
        )
        print(f"Split complete: {result}")
    elif args.command == "yaml":
        path = pp.generate_data_yaml(args.root)
        print(f"data.yaml created at: {path}")
    elif args.command == "stats":
        stats = pp.get_dataset_stats(args.root)
        print(json.dumps(stats, indent=2))
    elif args.command == "voc":
        # Batch convert all XML files
        xml_dir = Path(args.xml)
        ensure_dir(args.output)
        for xml_file in xml_dir.glob("*.xml"):
            # Would need image dimensions — parse from XML size element
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            labels = pp.voc_xml_to_yolo(str(xml_file), w, h)
            out_path = Path(args.output) / f"{xml_file.stem}.txt"
            with open(out_path, "w") as f:
                for l in labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
        print(f"Converted VOC XML files to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
