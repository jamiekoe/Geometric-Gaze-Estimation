# Geometric Gaze Estimation with ClearDepth

This repository implements a geometric gaze-estimation pipeline for RGB-D eye images. It includes ClearDepth, a pupil-depth replacement method that uses an eye scan to reduce depth artifacts in the pupil region.

## Paper

ClearDepth is described in:

Jamie Koerner and Vivienne Sze, "ClearDepth: Addressing Depth Distortions Caused By Eyelashes For Accurate Geometric Gaze Estimation On Mobile Devices," ICIP 2024.

[Paper](https://doi.org/10.1109/ICIP51287.2024.10647998)

## Required Packages

Install the Python packages needed to run the code:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Eye Landmark Model

Eye landmark detection uses the model from [`david-wb/gaze-estimation`](https://github.com/david-wb/gaze-estimation), included as a git submodule. After cloning this repository, run these commands from the repository root:

```
git submodule update --init --recursive
cd third_party/gaze-estimation
bash scripts/fetch_models.sh
cd ../..
```

This initializes `third_party/gaze-estimation` and downloads the pretrained EyeNet checkpoint used by the pipeline.

## Example Data

The repository includes one example experiment in `data/exp`. It contains:

- RGB eye/face frames
- compressed depth maps
- `pog.csv` point-of-gaze label files with normalized screen-fraction targets
- camera intrinsics
- display geometry
- a short eye scan used by ClearDepth

## Data Format

Depth maps must be stored in millimeters. The code unprojects image pixels with:

```
X = (x - cx) / fx * depth
Y = (y - cy) / fy * depth
Z = depth
```

Because `depth` is expected in millimeters, all 3D coordinates and reported gaze errors are also in millimeters. Missing depth values should be encoded as `0`; the pipeline interpolates missing values around sampled landmarks.

Each `depth/` folder may contain `.npy` files or compressed `.npz` files. Each file should contain one depth map aligned to the corresponding RGB frame.

### Point-of-Gaze Labels

`pog` means point of gaze: the known screen target the participant was looking at for each labeled frame. These labels are ground truth for calibration/training and error evaluation.

`pog.csv` does not store pixels or millimeters. It stores normalized screen fractions:

- `x = 0.0` is the left edge of the display, and `x = 1.0` is the right edge
- `y = 0.0` is the top edge of the display, and `y = 1.0` is the bottom edge
- `x = 0.5, y = 0.5` is the center of the display

The code converts these normalized fractions to display pixels and then to millimeter screen targets using `camera/display.json`, which should match the tablet used during data collection:

```
{
  "width_px": 2048,
  "height_px": 2732,
  "ppi": 264,
  "camera_to_screen_distance_mm": 5.0
}
```

The example values are for the sample iPad recording. If you use a different iPad or tablet, update the display resolution, ppi, and camera-to-screen offset before training or evaluating. An incorrect display config changes the physical target locations and therefore the fitted model and reported gaze errors. If your recording software uses a different origin or axis direction, convert its labels to this convention before creating `pog.csv`.

Use this format for `training/pog.csv` and `test/pog.csv`:

```
frame,x,y
0,0.1,0.1
1,0.1,0.1
```

Each row labels one RGB/depth frame. `frame` is a zero-based index after natural filename sorting.

## Run

```
python src/main.py \
  --exp-path data/exp \
  --output results/comparison.csv \
  --maxiter 30000
```

This runs the full pipeline from the example inputs: landmark detection, pupil-depth extraction, ClearDepth replacement, geometric model fitting, gaze prediction, and raw-depth versus ClearDepth error reporting.

`results/comparison.csv` contains one row for each dataset (`training`, `test`) and eye (`left`, `right`, `both`). It reports raw-depth and ClearDepth gaze error in millimeters, plus the ClearDepth-minus-raw-depth difference. Negative delta values mean ClearDepth improved the error.

## Predict Without Ground Truth

After fitting a model, use prediction-only mode for unlabeled RGB-D image data:

```
python src/main.py \
  --exp-path data/exp \
  --predict-only \
  --predict-input path/to/rgbd_folder \
  --predict-output-dir results/predictions \
  --depth-mode cleardepth
```

Prediction-only mode loads saved model parameters from `model_cleardepth/` or `model_rawdepth/` and uses the head-pose reference created during training. It writes `preds_left_*`, `preds_right_*`, and `preds_both_*` files into the output folder. If the input has no point-of-gaze labels, the script saves predictions and skips error reporting.

The prediction input should mirror the example experiment structure: a folder with `rgb/` and `depth/` subfolders. The script checks that every RGB image has a corresponding `.npy` or `.npz` depth map before running prediction. RGB-only images are not sufficient for this geometric pipeline because pupil depth and camera intrinsics are required.

## Command-Line Options

`src/main.py`:

- `--exp-path`: experiment folder used for training/evaluation; defaults to `data/exp`
- `--evaluation-split`: dataset used for error evaluation; defaults to `test`
- `--output`: optional CSV path for raw-depth versus ClearDepth error metrics
- `--maxiter`: maximum optimizer iterations for each geometric model fit
- `--reuse-models`: load existing model parameters when available
- `--predict-only`: skip fitting and run prediction with saved model parameters
- `--predict-split`: prediction-only split name; defaults to the input folder name when `--predict-input` is set, otherwise `inference`
- `--predict-input`: RGB-D folder for prediction; it must contain `rgb/` and `depth/` subfolders
- `--predict-output-dir`: directory where prediction `.npy` files are written
- `--depth-mode`: prediction depth mode; one of `rawdepth`, `cleardepth`, or `both`
- `--model-root`: folder containing `model_rawdepth/`, `model_cleardepth/`, and training landmarks
- `--model-path`: folder containing `params_left.json` and `params_right.json` for one depth mode
- `--reference-landmarks`: path to the trained model's `landmarks_3d_ref.npy`
- `--intrinsics`: camera intrinsics JSON path
- `--display-config`: display geometry JSON path
- `--eye-scan-path`: RGB-D eye-scan folder used by ClearDepth

## Visualize ClearDepth Pixels

```
python src/visualize_cleardepth.py \
  --exp-path data/exp \
  --output-dir results/visualizations
```

The visualization marks the eye-scan pixels selected by ClearDepth for pupil-depth replacement.

`src/visualize_cleardepth.py`:

- `--exp-path`: experiment folder to visualize; defaults to `data/exp`
- `--splits`: experiment splits to visualize; defaults to `training test`
- `--output-dir`: directory where visualization PNG/CSV/NPZ files are written
- `--display-config`: display geometry JSON path

## Code Layout

- `src/main.py`: pipeline entrypoint
- `src/geometric_model.py`: geometric gaze model
- `src/data_extraction/cleardepth.py`: ClearDepth implementation
- `src/visualize_cleardepth.py`: ClearDepth pixel visualization

## Citation

```
@inproceedings{koerner2024cleardepth,
  title={ClearDepth: Addressing Depth Distortions Caused By Eyelashes For Accurate Geometric Gaze Estimation On Mobile Devices},
  author={Koerner, Jamie and Sze, Vivienne},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={2135--2141},
  year={2024},
  organization={IEEE}
}
```
