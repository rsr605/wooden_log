# Improve Detection on Real Log Piles — End-View Cross-Sections

## Problem
The user's example image shows a real log pile photographed from the end —
circular cross-sections with tree rings, ~20-25 logs densely packed. The model
was trained only on synthetic side-view cylinder logs (horizontal rectangles
with bark texture, max 10 per image). Result: only 6 of ~25 logs detected.

## Root Cause
Domain mismatch between training data and real-world images:
- Training: side-view horizontal cylinders, avg 6 logs/image, max 10
- Real: end-view circular cross-sections, 20-40 logs/image, tightly packed

## Solution
1. Update `src/generate_samples.py` to add **end-view log generation**:
   - Circular/elliptical log cross-sections with realistic tree ring patterns
   - Dense hexagonal packing (like real log piles)
   - Mix of sizes (small/distant to large/close)
   - 15-40 logs per image for pile scenes
2. Generate new training dataset (1500 images) with the updated generator
3. Retrain YOLOv8s model on the new data
4. Test on the example image — target: detect 20+ of ~25 logs

## Files to Change
- `src/generate_samples.py` — add end-view log rendering + dense pile layouts
- `models/wooden_log_best.pt` — retrained model

## Acceptance Criteria
- [ ] Generator produces end-view circular logs with tree rings
- [ ] Generator produces dense pile scenes (15-40 logs)
- [ ] New training dataset generated (1500+ images)
- [ ] Model retrained on new data
- [ ] Detection on example image finds 18+ of ~25 logs
- [ ] No false positives on blank/noise images
- [ ] All tests pass
