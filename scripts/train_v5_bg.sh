#!/bin/bash
# Train v5 model in background — survives container pause via procmgr.
# Outputs to a log file we can poll.
cd /workspace

LOG=/workspace/training_v5.log

# Clean any previous incomplete runs
rm -rf /workspace/runs/detect/runs/detect/v5_recall 2>/dev/null

echo "[$(date)] Starting v5 recall training..." > "$LOG"

python3 -m src.train \
    --data data_v5/data.yaml \
    --model models/yolov8n_fresh.pt \
    --preset recall \
    --project runs/detect \
    --name v5_recall \
    --epochs 50 \
    --batch 8 \
    --imgsz 640 >> "$LOG" 2>&1

EXIT_CODE=$?
echo "[$(date)] Training finished with exit code $EXIT_CODE" >> "$LOG"

# Copy the best weights to the production model location
if [ -f runs/detect/runs/detect/v5_recall/weights/best.pt ]; then
    cp runs/detect/runs/detect/v5_recall/weights/best.pt models/wooden_log_best_v5.pt
    echo "[$(date)] Best weights copied to models/wooden_log_best_v5.pt" >> "$LOG"
else
    echo "[$(date)] WARNING: best.pt not found!" >> "$LOG"
fi
