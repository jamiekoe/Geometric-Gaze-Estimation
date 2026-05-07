# Example Experiment

This folder contains the inputs needed to run the example from scratch:

- `training/`: 25 calibration frames and `pog.csv` point-of-gaze labels
- `test/`: 200 held-out evaluation frames and `pog.csv` point-of-gaze labels
- `eye_scan/`: 5 RGB-D eye-scan frames used by ClearDepth
- `camera/intrinsics.json`: depth camera intrinsics
- `camera/display.json`: tablet display geometry

Depth maps are stored in millimeters and aligned to the RGB frames. Missing depth values are encoded as `0`.

`pog.csv` stores point-of-gaze labels: the known on-screen target for each labeled frame. The `frame` column is a zero-based frame index. The `x` and `y` columns are normalized screen fractions, not pixels and not millimeters: `x=0` is the left edge, `x=1` is the right edge, `y=0` is the top edge, and `y=1` is the bottom edge. The pipeline converts those fractions to millimeter screen targets using `camera/display.json`.

Run the example from the repository root:

```
python src/main.py \
  --exp-path data/exp \
  --output results/comparison.csv \
  --maxiter 30000
```
