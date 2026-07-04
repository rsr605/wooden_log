"""Conftest to handle corrupted yolov8n.pt file in workspace."""
import os
import errno
from pathlib import Path

def pytest_ignore_collect(collection_path, config):
    """Ignore the corrupted yolov8n.pt/yolov8s.pt files."""
    try:
        collection_path.is_dir()
    except OSError:
        return True  # ignore files that cause OS errors
    return None
