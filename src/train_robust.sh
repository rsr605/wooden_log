#!/bin/bash
# Robust YOLOv8 training script with auto-resume.
# Designed to survive container restarts by detecting partial training runs.
# Runs from /workspace/training_workdir to avoid filesystem corruption in /workspace root.

set -e

WORKDIR="/workspace/training_workdir"
DATA_YAML="$WORKDIR/data/data.yaml"
PROJECT_DIR="$WORKDIR/runs/detect"
RUN_NAME="wooden_log_v4"
EPOCHS=200
BATCH=4
IMGSZ=640
PRESET="accurate"

echo "=============================================="
echo "  Wooden Log Detection — Robust Training v4"
echo "=============================================="
echo "  Workdir:  $WORKDIR"
echo "  Epochs:   $EPOCHS"
echo "  Batch:    $BATCH"
echo "  Imgsz:    $IMGSZ"
echo "  Preset:   $PRESET"
echo "=============================================="
echo ""

# Ensure workdir exists
mkdir -p "$WORKDIR"

# Copy dataset if not present
if [ ! -f "$DATA_YAML" ]; then
    echo "Copying dataset to workdir..."
    mkdir -p "$WORKDIR/data"
    cp -r /workspace/data/images "$WORKDIR/data/"
    cp -r /workspace/data/labels "$WORKDIR/data/"
    cp /workspace/data/data.yaml "$WORKDIR/data/"
    # Fix path in data.yaml to point to workdir
    sed -i "s|path:.*|path: $WORKDIR/data|" "$DATA_YAML"
    echo "Dataset copied."
fi

# Check if we have a checkpoint to resume from
LAST_CKPT="$PROJECT_DIR/$RUN_NAME/weights/last.pt"
RESUME_FLAG=""

if [ -f "$LAST_CKPT" ]; then
    echo "Found checkpoint: $LAST_CKPT"
    echo "Resuming training from last checkpoint..."
    RESUME_FLAG="--resume"
    cd "$WORKDIR"
    python3 /workspace/src/train.py \
        --data "$DATA_YAML" \
        --model "$LAST_CKPT" \
        --preset "$PRESET" \
        --project "$PROJECT_DIR" \
        --name "$RUN_NAME" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --imgsz "$IMGSZ" \
        $RESUME_FLAG 2>&1
else
    echo "No checkpoint found. Starting fresh training..."
    # Copy base model to workdir to avoid corrupted inode in /workspace root
    if [ ! -f "$WORKDIR/yolov8n.pt" ]; then
        echo "Copying base model..."
        cp /workspace/models/yolov8n.pt "$WORKDIR/yolov8n.pt" 2>/dev/null || true
    fi

    cd "$WORKDIR"
    python3 /workspace/src/train.py \
        --data "$DATA_YAML" \
        --model "yolov8n.pt" \
        --preset "$PRESET" \
        --project "$PROJECT_DIR" \
        --name "$RUN_NAME" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --imgsz "$IMGSZ" 2>&1
fi

echo ""
echo "Training complete!"

# Copy best weights to production model path
BEST="$PROJECT_DIR/$RUN_NAME/weights/best.pt"
if [ -f "$BEST" ]; then
    echo "Copying best.pt to models/wooden_log_best.pt"
    cp "$BEST" /workspace/models/wooden_log_best.pt
    echo "Model deployed!"
    ls -la /workspace/models/wooden_log_best.pt
else
    echo "WARNING: best.pt not found at $BEST"
    echo "Checking for any weights..."
    find "$PROJECT_DIR/$RUN_NAME" -name "*.pt" 2>/dev/null
fi
