"""
Batch & Video Prediction CLI for Wooden Log Detection.

Processes folders of images or video streams, saves annotated outputs,
and exports detection results to JSON/CSV.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.utils import (
    ensure_dir,
    is_allowed_image,
    is_allowed_video,
    generate_hashed_filename,
    format_detection_result,
    DEFAULT_MODEL_VARIANT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)


class BatchPredictor:
    """
    Batch prediction for images and video.

    Wraps LogDetector for processing entire directories or video files
    with output saving and result export.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_VARIANT,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._detector = None

    @property
    def detector(self):
        """Lazy-load the LogDetector."""
        if self._detector is None:
            # Import here to avoid loading ultralytics at module import time
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from app.detector import LogDetector

            self._detector = LogDetector(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
        return self._detector

    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        conf: Optional[float] = None,
        save_json: bool = True,
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images.
            output_dir: Directory to save annotated images + results.
            conf: Override confidence threshold.
            save_json: Save results.json with all detections.
            save_csv: Save results.csv with tabular detection data.

        Returns:
            Summary dict.
        """
        ensure_dir(output_dir)

        all_results: List[Dict[str, Any]] = []
        csv_rows: List[Dict[str, Any]] = []
        total_images = 0
        total_detections = 0
        total_time_ms = 0.0

        input_path = Path(input_dir)
        image_files = [
            f for f in sorted(input_path.glob("*"))
            if is_allowed_image(f.name)
        ]

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            detections, elapsed = self.detector.detect(image, conf=conf)
            annotated = self.detector.annotate(image, detections)

            out_name = img_file.name
            cv2.imwrite(str(Path(output_dir) / out_name), annotated)

            h, w = image.shape[:2]
            result = format_detection_result(str(img_file), w, h, detections, elapsed)
            all_results.append(result)

            for det in detections:
                csv_rows.append({
                    "image": img_file.name,
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "x1": det["bbox"]["x1"],
                    "y1": det["bbox"]["y1"],
                    "x2": det["bbox"]["x2"],
                    "y2": det["bbox"]["y2"],
                })

            total_images += 1
            total_detections += len(detections)
            total_time_ms += elapsed

        # Save JSON
        if save_json:
            json_path = Path(output_dir) / "results.json"
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Save CSV
        if save_csv and csv_rows:
            csv_path = Path(output_dir) / "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image", "class_id", "class_name",
                        "confidence", "x1", "y1", "x2", "y2",
                    ],
                )
                writer.writeheader()
                writer.writerows(csv_rows)

        return {
            "images_processed": total_images,
            "total_detections": total_detections,
            "avg_processing_time_ms": (
                round(total_time_ms / total_images, 2) if total_images else 0.0
            ),
            "output_dir": output_dir,
            "json_saved": save_json,
            "csv_saved": save_csv and bool(csv_rows),
        }

    def predict_video(
        self,
        video_path: str,
        output_path: str,
        conf: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file and write annotated output.

        Args:
            video_path: Path to input video.
            output_path: Path for output video.
            conf: Override confidence threshold.

        Returns:
            Summary dict.
        """
        return self.detector.detect_video(video_path, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry for batch prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch prediction for wooden log detection")
    parser.add_argument("--model", default=DEFAULT_MODEL_VARIANT, help="Model weights path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help="IoU threshold")

    sub = parser.add_subparsers(dest="command")

    # Batch images
    p_batch = sub.add_parser("batch", help="Process a directory of images")
    p_batch.add_argument("--input", required=True, help="Input images directory")
    p_batch.add_argument("--output", required=True, help="Output directory")

    # Video
    p_vid = sub.add_parser("video", help="Process a video file")
    p_vid.add_argument("--input", required=True, help="Input video path")
    p_vid.add_argument("--output", required=True, help="Output video path")

    args = parser.parse_args()

    predictor = BatchPredictor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    if args.command == "batch":
        result = predictor.predict_directory(args.input, args.output)
        print(f"Batch prediction complete: {json.dumps(result, indent=2)}")
    elif args.command == "video":
        result = predictor.predict_video(args.input, args.output)
        print(f"Video prediction complete: {json.dumps(result, indent=2)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
