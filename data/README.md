# Data

The example experiment is stored in `data/exp`.

- `training/`: calibration RGB frames, depth maps, and `pog.csv` point-of-gaze labels
- `test/`: held-out RGB frames, depth maps, and `pog.csv` point-of-gaze labels used for evaluation
- `eye_scan/`: RGB-D frames used as the ClearDepth source scan
- `camera/intrinsics.json`: depth camera intrinsics
- `camera/display.json`: tablet display geometry used to convert point-of-gaze labels to millimeters
- `visualizations/`: example ClearDepth pixel-selection outputs

Depth maps are stored in millimeters and aligned to the RGB frames. Missing depth values are encoded as `0`.

`pog.csv` stores point-of-gaze labels: the known on-screen target for each labeled frame. The `frame` column is a zero-based frame index. The `x` and `y` columns are normalized screen fractions, not pixels and not millimeters: `x=0` is the left edge, `x=1` is the right edge, `y=0` is the top edge, and `y=1` is the bottom edge. The pipeline converts those fractions to millimeter screen targets using `camera/display.json`.
