"""
YOLOv8 Training Pipeline for Wooden Log Detection.

Provides the LogTrainer class for configuring and running YOLOv8 training
on a custom wooden log dataset, plus hyperparameter presets.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, CLASS_NAMES


# ---------------------------------------------------------------------------
# Hyperparameter Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "description": "Quick training for experimentation (lower epochs, small images).",
        "epochs": 50,
        "imgsz": 416,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "mosaic": 1.0,
        "mixup": 0.0,
        "fliplr": 0.5,
        "scale": 0.3,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
    },
    "balanced": {
        "description": "Balanced training — good accuracy/time tradeoff.",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "mosaic": 1.0,
        "mixup": 0.1,
        "fliplr": 0.5,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
    },
    "accurate": {
        "description": "High-accuracy training (more epochs, larger images).",
        "epochs": 200,
        "imgsz": 640,
        "batch": 8,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "mosaic": 1.0,
        "mixup": 0.15,
        "fliplr": 0.5,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
        "degrees": 10.0,
        "translate": 0.1,
        "perspective": 0.0,
    },
}


class LogTrainer:
    """
    YOLOv8 training manager for wooden log detection.

    Wraps the ultralytics training API with sensible presets and
    project-specific configuration.
    """

    def __init__(
        self,
        data_yaml: str = "data/data.yaml",
        model: str = "yolov8n.pt",
        preset: str = "balanced",
        project_dir: str = "runs/detect",
        name: str = "wooden_log_train",
    ):
        """
        Args:
            data_yaml: Path to the dataset data.yaml file.
            model: Base model to train from (e.g. 'yolov8n.pt').
            preset: Hyperparameter preset ('fast', 'balanced', 'accurate').
            project_dir: Directory to save training runs.
            name: Run name (subdirectory under project_dir).
        """
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(PRESETS.keys())}")
        self.data_yaml = data_yaml
        self.model = model
        self.preset = preset
        self.project_dir = project_dir
        self.name = name
        self.params = PRESETS[preset].copy()
        self._yolo_model = None

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return the current hyperparameters (without 'description')."""
        return {k: v for k, v in self.params.items() if k != "description"}

    def override(self, **kwargs) -> "LogTrainer":
        """
        Override specific hyperparameters.

        Returns self for chaining.
        """
        for k, v in kwargs.items():
            self.params[k] = v
        return self

    def train(self, resume: bool = False) -> Dict[str, Any]:
        """
        Start training.

        Args:
            resume: If True, resume from the last checkpoint.

        Returns:
            Dict with training results (path to weights, metrics if available).
        """
        from ultralytics import YOLO

        ensure_dir(self.project_dir)

        yolo_model = YOLO(self.model)
        self._yolo_model = yolo_model

        hyperparams = self.get_hyperparameters()
        results = yolo_model.train(
            data=self.data_yaml,
            project=self.project_dir,
            name=self.name,
            resume=resume,
            **hyperparams,
        )
        return {"results": str(results), "params": hyperparams}

    def validate(self, weights_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run validation on the trained model.

        Args:
            weights_path: Path to weights; defaults to best.pt from training.

        Returns:
            Validation metrics dict.
        """
        from ultralytics import YOLO

        if weights_path is None:
            weights_path = str(
                Path(self.project_dir) / self.name / "weights" / "best.pt"
            )
        yolo_model = YOLO(weights_path)
        metrics = yolo_model.val(data=self.data_yaml)
        return {"metrics": str(metrics)}


def list_presets() -> str:
    """Return a human-readable summary of available presets."""
    lines = ["Available Training Presets:"]
    for name, preset in PRESETS.items():
        lines.append(f"\n  {name}: {preset['description']}")
        for k, v in preset.items():
            if k == "description":
                continue
            lines.append(f"    {k:12s}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry: train YOLOv8 on the wooden log dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 for wooden log detection")
    parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--preset", default="balanced", choices=list(PRESETS.keys()),
                        help="Hyperparameter preset")
    parser.add_argument("--project", default="runs/detect", help="Output project dir")
    parser.add_argument("--name", default="wooden_log_train", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    args = parser.parse_args()

    trainer = LogTrainer(
        data_yaml=args.data,
        model=args.model,
        preset=args.preset,
        project_dir=args.project,
        name=args.name,
    )

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz

    if overrides:
        trainer.override(**overrides)

    print(f"Starting training with preset: {args.preset}")
    print(f"Hyperparameters: {trainer.get_hyperparameters()}")
    result = trainer.train(resume=args.resume)
    print(f"\nTraining complete: {result}")


if __name__ == "__main__":
    main()
