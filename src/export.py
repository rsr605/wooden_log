"""
Model Export Utility for YOLOv8.

Exports a trained YOLOv8 model to deployment-ready formats:
  - ONNX
  - TorchScript
  - OpenVINO
  - TensorFlow Lite (optional)

Useful when you want to deploy the model outside the Python training
environment.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir


# Supported export formats
SUPPORTED_FORMATS = ["onnx", "torchscript", "openvino", "tflite", "coreml", "engine"]


class ModelExporter:
    """Export YOLOv8 models to various deployment formats."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Args:
            model_path: Path to the model weights.
        """
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def export(
        self,
        fmt: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Export the model to a target format.

        Args:
            fmt: Target format ('onnx', 'torchscript', 'openvino', etc.).
            imgsz: Input image size for the exported model.
            half: Use FP16 half-precision (for onnx/torchscript).
            dynamic: Dynamic axes (for onnx).
            simplify: Simplify the ONNX model graph.
            output_dir: Directory for the exported model; defaults to models/.

        Returns:
            Path to the exported model.
        """
        fmt = fmt.lower()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}")

        model = self._load_model()
        export_kwargs = {
            "format": fmt,
            "imgsz": imgsz,
            "half": half,
            "dynamic": dynamic,
            "simplify": simplify,
        }

        if output_dir:
            ensure_dir(output_dir)

        result = model.export(**export_kwargs)
        return str(result)

    def export_all(
        self,
        formats: Optional[List[str]] = None,
        imgsz: int = 640,
    ) -> Dict[str, str]:
        """
        Export the model to multiple formats at once.

        Args:
            formats: List of formats; defaults to ['onnx', 'torchscript'].
            imgsz: Input image size.

        Returns:
            Dict mapping format → exported file path.
        """
        if formats is None:
            formats = ["onnx", "torchscript"]

        results: Dict[str, str] = {}
        for fmt in formats:
            try:
                path = self.export(fmt=fmt, imgsz=imgsz)
                results[fmt] = path
            except Exception as exc:
                results[fmt] = f"ERROR: {exc}"

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry for model export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 model")
    parser.add_argument("--model", default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--format", default="onnx", choices=SUPPORTED_FORMATS,
                        help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes (ONNX)")
    parser.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification")
    parser.add_argument("--all", action="store_true", help="Export all common formats")
    args = parser.parse_args()

    exporter = ModelExporter(model_path=args.model)

    if args.all:
        results = exporter.export_all(imgsz=args.imgsz)
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
    else:
        path = exporter.export(
            fmt=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            simplify=not args.no_simplify,
        )
        print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
