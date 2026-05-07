"""Train or run geometric gaze estimation with and without ClearDepth.

The expected experiment folder contains `training/`, `test/`, `eye_scan/`,
and `camera/` subfolders. Each data split stores RGB frames, depth maps, and
optional point-of-gaze labels. The script extracts
landmarks and pupil positions, fits one geometric model per eye, averages the
left/right predictions, and reports raw-depth versus ClearDepth errors. It can
also load saved model parameters and predict on unlabeled data.
"""

import argparse
import csv
import glob
import os
import json
import natsort
import numpy as np
import cv2
from pathlib import Path
from data_extraction.utils import Display
from data_extraction import DataExtractor
from geometric_model import GeometricModel

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEPTH_EXTENSIONS = (".npy", ".npz")
POG_CSV_NAME = "pog.csv"
TRAINING_SPLIT = "training"
DEFAULT_EVALUATION_SPLIT = "test"
DEFAULT_PREDICTION_SPLIT = "inference"


def create_directory(path):
    """Create `path` and return it as a string/path-like value."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_npy_arrays(paths):
    """Load a sequence of `.npy`/`.npz` arrays and stack them on axis 0."""
    return np.stack([load_array(path) for path in paths])


def load_array(path):
    """Load one NumPy array from `.npy` or single-array `.npz` storage."""
    if str(path).endswith(".npz"):
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(f"Expected one array in {path}, found keys {data.files}.")
            return data[data.files[0]]
    return np.load(path)


def load_imgs(paths):
    """Load BGR images with OpenCV in sorted frame order supplied by caller."""
    imgs = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image file {path}.")
        imgs.append(img)
    return np.stack(imgs)


def load_json(path):
    """Read a JSON file and return the decoded object."""
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_display(exp_path, display_config=None):
    """Load display geometry from JSON or fall back to the default example device."""
    if display_config is None:
        candidate = Path(exp_path) / "camera" / "display.json"
        display_config = candidate if candidate.exists() else None
    if display_config is None:
        return Display()
    return Display.from_json(display_config)


def list_files_by_extension(path, extensions):
    """Return naturally sorted files in `path` with one of `extensions`."""
    files = []
    for extension in extensions:
        files.extend(glob.glob(os.path.join(path, f"*{extension}")))
        files.extend(glob.glob(os.path.join(path, f"*{extension.upper()}")))
    return natsort.natsorted(files)


def resolve_rgb_depth_dirs(path):
    """Return the required `rgb/` and `depth/` folders for one RGB-D split."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {path}")
    rgb_dir = path / "rgb"
    depth_dir = path / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Expected {path} to contain rgb/ and depth/ subfolders.")
    return rgb_dir, depth_dir


def load_rgb_frames(rgb_dir):
    """Load RGB frames from a required `rgb/` folder."""
    rgb_files = list_files_by_extension(rgb_dir, IMAGE_EXTENSIONS)
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}.")
    return load_imgs(rgb_files), rgb_files


def validate_depth_alignment(path, rgb_files, depth_files, num_rgb_frames):
    """Ensure every RGB image has one corresponding depth map."""
    if len(depth_files) != num_rgb_frames:
        raise ValueError(
            f"Found {num_rgb_frames} RGB images but {len(depth_files)} depth maps for {path}."
        )
    rgb_stems = [Path(item).stem for item in rgb_files]
    depth_stems = [Path(item).stem for item in depth_files]
    if rgb_stems != depth_stems:
        raise ValueError(
            "RGB image files and depth maps must have matching stems after natural sorting. "
            f"First RGB stems: {rgb_stems[:5]}; first depth stems: {depth_stems[:5]}."
        )


def load_pog_csv(path, rgb_files, num_frames):
    """Load point-of-gaze labels as normalized screen fractions from `pog.csv`."""
    csv_path = Path(path) / POG_CSV_NAME
    rows = []
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if not {"x", "y"}.issubset(fieldnames):
            raise ValueError(f"{csv_path} must contain at least x and y columns.")
        for row in reader:
            rows.append(row)

    if len(rows) != num_frames:
        raise ValueError(f"Found {num_frames} RGB images but {len(rows)} POG rows in {csv_path}.")

    validate_pog_frame_indices(rows, csv_path)

    return np.array([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)


def validate_pog_frame_indices(rows, csv_path):
    """Validate optional zero-based frame indices in `pog.csv`."""
    if not rows or "frame" not in rows[0]:
        return
    frames = [str(row.get("frame", "")).strip() for row in rows]
    if not any(frames):
        return
    for expected, frame in enumerate(frames):
        try:
            frame_index = int(frame)
        except ValueError as error:
            raise ValueError(f"Frame column in {csv_path} must contain zero-based integer indices.") from error
        if frame_index != expected:
            raise ValueError(
                f"Frame column in {csv_path} must be 0, 1, 2, ... in frame order; "
                f"found {frame_index} at row {expected}."
            )


def load_pog_labels(path, rgb_files, num_frames):
    """Load point-of-gaze labels as normalized screen fractions from `pog.csv`."""
    csv_path = Path(path) / POG_CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Point-of-gaze labels must be stored in {csv_path}.")
    return load_pog_csv(path, rgb_files, num_frames)


def get_raw_data(path, get_pog=False):
    """Load RGB frames, depth maps, and optionally point-of-gaze labels."""
    rgb_dir, depth_dir = resolve_rgb_depth_dirs(path)
    imgs, rgb_files = load_rgb_frames(rgb_dir)
    depth_files = list_files_by_extension(depth_dir, DEPTH_EXTENSIONS)
    if not depth_files:
        raise FileNotFoundError(f"No depth maps found in {depth_dir}.")
    validate_depth_alignment(path, rgb_files, depth_files, len(imgs))
    depth_maps = load_npy_arrays(depth_files)
    data = {"img": imgs, "depth": depth_maps}
    if get_pog:
        pog_normalized = load_pog_labels(path, rgb_files, len(imgs))
    else:
        pog_normalized = None
    data["pog_normalized"] = pog_normalized
    return data


def has_pog(path):
    """Return whether a split folder contains point-of-gaze labels."""
    return (Path(path) / POG_CSV_NAME).is_file()


def get_initial_eye_center(pupil):
    """Initialize eyeball center from the first pupil point and eye radius."""
    eye_center = pupil.copy()
    eye_center[2] += 12.4  # mean human eyeball radius (mm)
    return eye_center


def plot_gaze_predictions(pog_pred, pog_gt, save_path, filename="vis.png", display=None):
    """Save a quick screen-space plot of predicted and target gaze points."""
    if pog_gt is None:
        return
    display = Display() if display is None else display
    screen_width, screen_height = display.dim_pix
    pog_pred = np.array([display.mm_to_pix(float(pog_pred[i, 0]), float(pog_pred[i, 1])) for i in range(pog_pred.shape[0])])
    pog_gt = np.array([display.mm_to_pix(float(pog_gt[i, 0]), float(pog_gt[i, 1])) for i in range(pog_gt.shape[0])])

    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    pog_gt_unique = np.unique(pog_gt, axis=0)

    for (x, y) in pog_gt_unique:
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, radius=10, color=(255, 255, 255), thickness=2)

    for (x, y) in pog_pred:
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(os.path.join(save_path, filename), image)


def mean_abs_error_mm(pog_pred, pog_gt):
    """Mean absolute point-of-gaze error across screen x/y/z coordinates."""
    return float(np.mean(np.abs(pog_pred - pog_gt)))


def mean_2d_error_mm(pog_pred, pog_gt):
    """Mean Euclidean point-of-gaze error in the screen plane."""
    return float(np.mean(np.linalg.norm(pog_pred[:, :2] - pog_gt[:, :2], axis=1)))


def train_model_with_starts(side, initial_eye_center, model_path, dataset_training, starts, maxiter):
    """Train from multiple initializations and keep the lowest calibration loss."""
    best = None
    for start_label, initial_params in starts:
        print(f"Training {side} eye model from {start_label} start.")
        candidate = GeometricModel(side, initial_eye_center, model_path)
        candidate.train(dataset_training, initial_params=initial_params, maxiter=maxiter)
        loss = candidate.compute_training_loss(dataset_training)
        print(f"{side} {start_label} calibration loss: {loss:.3f} mm")
        if best is None or loss < best[0]:
            best = (loss, start_label, candidate.params)

    model = GeometricModel(side, initial_eye_center, model_path)
    model.set_params(best[2])
    print(f"Selected {side} eye {best[1]} start with calibration loss {best[0]:.3f} mm.")
    return model


def load_eye_scan(eye_scan_path):
    """Load the eye scan and return it with its landmark cache path."""
    landmark_path_scan = create_directory(os.path.join(eye_scan_path, "landmarks"))
    data_scan = get_raw_data(eye_scan_path)
    return {"rgb": data_scan["img"], "depth": data_scan["depth"]}, landmark_path_scan


def build_dataset_from_path(data_path, split_name, f, c, reference_landmarks,
                            apply_cleardepth=False, eye_scan=None, landmark_path_scan=None,
                            landmark_cache_dir=None, display=None,
                            save_reference_landmarks=False):
    """Build a geometric dataset from one RGB-D split folder.

    Evaluation and prediction-only use require the saved training reference
    landmarks, which define the trained model's head-coordinate frame.
    """
    dataset_raw = get_raw_data(data_path, get_pog=has_pog(data_path))
    if landmark_cache_dir is None:
        landmark_cache_dir = os.path.join(data_path, "landmarks")
    landmark_path = create_directory(landmark_cache_dir)
    reference_file = str(reference_landmarks)
    if not save_reference_landmarks and not os.path.exists(reference_file):
        raise FileNotFoundError(
            f"Missing {reference_file}. Run training once before prediction-only mode "
            "so the saved model and head-pose reference use the same coordinate frame."
        )
    data_extractor = DataExtractor(landmark_path, f, c, landmark_path_scan=landmark_path_scan,
                                   display=display)
    return data_extractor.get_geometric_dataset(
        dataset_raw["img"],
        dataset_raw["depth"],
        dataset_raw["pog_normalized"],
        reference_file,
        apply_cleardepth=apply_cleardepth,
        eye_scan=eye_scan,
        save_reference_landmarks=save_reference_landmarks,
    )


def build_split_dataset(exp_path, split, f, c, apply_cleardepth=False, eye_scan=None,
                        landmark_path_scan=None, display=None):
    """Build a geometric dataset for a split inside an experiment folder."""
    data_path = os.path.join(exp_path, split)
    reference_landmarks = os.path.join(exp_path, TRAINING_SPLIT, "landmarks", "landmarks_3d_ref.npy")
    return build_dataset_from_path(
        data_path,
        split,
        f,
        c,
        reference_landmarks,
        apply_cleardepth=apply_cleardepth,
        eye_scan=eye_scan,
        landmark_path_scan=landmark_path_scan,
        display=display,
        save_reference_landmarks=(split == TRAINING_SPLIT),
    )


def save_predictions(output_dir, label, preds_by_side):
    """Save left, right, and averaged-both-eye predictions."""
    create_directory(output_dir)
    for side, preds in preds_by_side.items():
        np.save(os.path.join(output_dir, f"preds_{side}_{label}.npy"), preds)
    preds_both = (preds_by_side["left"] + preds_by_side["right"]) / 2
    np.save(os.path.join(output_dir, f"preds_both_{label}.npy"), preds_both)
    return preds_both


def print_prediction_errors(label, split, dataset, preds_by_side, preds_both):
    """Print error metrics for predictions when ground truth is available."""
    if dataset["pog"] is None:
        print(f"No point-of-gaze labels found for {split}; saved {label} predictions without error metrics.")
        return
    for side, pog_pred in [*preds_by_side.items(), ("both", preds_both)]:
        print(f"{split.capitalize()} error {label} {side}: "
              f"{mean_abs_error_mm(pog_pred, dataset['pog']):.3f} mm mean-abs, "
              f"{mean_2d_error_mm(pog_pred, dataset['pog']):.3f} mm 2D")


def predict_depth_mode(input_path, output_dir, f, c, split_name, label, apply_cleardepth,
                       model_path, reference_landmarks, eye_scan_path, display=None):
    """Load saved model parameters and predict one split without training."""
    print(f"\n=== Predicting {split_name} with {label} ===")
    eye_scan, landmark_path_scan = (None, None)
    if apply_cleardepth:
        eye_scan, landmark_path_scan = load_eye_scan(eye_scan_path)
    dataset = build_dataset_from_path(
        input_path,
        split_name,
        f,
        c,
        reference_landmarks,
        apply_cleardepth=apply_cleardepth,
        eye_scan=eye_scan,
        landmark_path_scan=landmark_path_scan,
        landmark_cache_dir=os.path.join(output_dir, "landmarks"),
        display=display,
    )

    preds_by_side = {}
    for side in ["left", "right"]:
        initial_eye_center = get_initial_eye_center(dataset["pupil"][side][0])
        model = GeometricModel(side, initial_eye_center, model_path)
        model.load_params()
        preds_by_side[side] = model.predict(dataset)
    preds_both = save_predictions(output_dir, label, preds_by_side)
    print_prediction_errors(label, split_name, dataset, preds_by_side, preds_both)
    print(f"Saved {label} predictions to {output_dir}.")


def resolve_model_path(label, exp_path, model_root=None, model_path=None):
    """Resolve the directory containing params_left/right JSON files."""
    if model_path is not None:
        return str(model_path)
    root = str(model_root) if model_root is not None else exp_path
    return os.path.join(root, f"model_{label}")


def resolve_reference_landmarks(exp_path, model_root=None, reference_landmarks=None):
    """Resolve the head-pose reference landmark file used by a trained model."""
    if reference_landmarks is not None:
        return str(reference_landmarks)
    root = str(model_root) if model_root is not None else exp_path
    return os.path.join(root, TRAINING_SPLIT, "landmarks", "landmarks_3d_ref.npy")


def run_predict_only(exp_path, f, c, split, depth_mode, predict_input=None,
                     predict_output_dir=None, model_root=None, model_path=None,
                     reference_landmarks=None, eye_scan_path=None,
                     display=None):
    """Run saved model parameters on a split that may not have labels."""
    if model_path is not None and depth_mode == "both":
        raise ValueError("--model-path can only be used with --depth-mode rawdepth or cleardepth.")
    if split is None:
        split = Path(predict_input).stem if predict_input is not None else DEFAULT_PREDICTION_SPLIT
    input_path = str(predict_input) if predict_input is not None else os.path.join(exp_path, split)
    if predict_output_dir is not None:
        output_dir = str(predict_output_dir)
    else:
        output_dir = input_path
    reference_landmarks = resolve_reference_landmarks(exp_path, model_root, reference_landmarks)
    eye_scan_path = str(eye_scan_path) if eye_scan_path is not None else os.path.join(exp_path, "eye_scan")
    labels = ["rawdepth", "cleardepth"] if depth_mode == "both" else [depth_mode]
    for label in labels:
        predict_depth_mode(
            input_path,
            output_dir,
            f,
            c,
            split,
            label,
            apply_cleardepth=(label == "cleardepth"),
            model_path=resolve_model_path(label, exp_path, model_root, model_path),
            reference_landmarks=reference_landmarks,
            eye_scan_path=eye_scan_path,
            display=display,
        )


def run_depth_mode(exp_path, f, c, apply_cleardepth, label, force_retrain=True,
                   initial_params_by_side=None, maxiter=100000,
                   evaluation_split=DEFAULT_EVALUATION_SPLIT, display=None):
    """Run one full pipeline pass, either using raw depth or ClearDepth.

    Returns per-split error metrics and fitted parameters. When ClearDepth is
    enabled, raw-depth parameters can be supplied as an additional optimizer
    starting point because that often avoids poor local minima.
    """
    print(f"\n=== Running {label} ===")
    modes = [TRAINING_SPLIT]
    if evaluation_split != TRAINING_SPLIT:
        modes.append(evaluation_split)
    datasets = {}
    eye_scan, landmark_path_scan = None, None
    if apply_cleardepth:
        eye_scan, landmark_path_scan = load_eye_scan(os.path.join(exp_path, "eye_scan"))

    for mode in modes:
        datasets[mode] = build_split_dataset(
            exp_path,
            mode,
            f,
            c,
            apply_cleardepth=apply_cleardepth,
            eye_scan=eye_scan,
            landmark_path_scan=landmark_path_scan,
            display=display,
        )

    dataset_training = datasets[TRAINING_SPLIT]
    dataset_evaluation = datasets[evaluation_split]
    if force_retrain and dataset_training["pog"] is None:
        raise ValueError("Training POG labels are required unless you use --reuse-models or --predict-only.")
    sides = ["left", "right"]
    preds_training = {}
    preds_evaluation = {}
    errors = {TRAINING_SPLIT: {}, evaluation_split: {}}
    params_by_side = {}
    model_path = create_directory(os.path.join(exp_path, f"model_{label}"))
    for side in sides:
        initial_eye_center = get_initial_eye_center(dataset_training["pupil"][side][0])
        model = GeometricModel(side, initial_eye_center, model_path)
        if force_retrain:
            starts = [("initial", None)]
            if apply_cleardepth and initial_params_by_side and side in initial_params_by_side:
                starts.append(("rawdepth", initial_params_by_side[side]))
            model = train_model_with_starts(side, initial_eye_center, model_path,
                                            dataset_training, starts, maxiter)
            model.save_params()
        else:
            try:
                model.load_params()
            except FileNotFoundError:
                print(f"No {label} parameters found: training {side} eye model from scratch.")
                starts = [("initial", None)]
                if apply_cleardepth and initial_params_by_side and side in initial_params_by_side:
                    starts.append(("rawdepth", initial_params_by_side[side]))
                model = train_model_with_starts(side, initial_eye_center, model_path,
                                                dataset_training, starts, maxiter)
                model.save_params()
        params_by_side[side] = model.params

        pog_pred_train = model.predict(dataset_training)
        pog_pred_evaluation = model.predict(dataset_evaluation)

        for split, dataset, pog_pred in [
            (TRAINING_SPLIT, dataset_training, pog_pred_train),
            (evaluation_split, dataset_evaluation, pog_pred_evaluation),
        ]:
            if dataset["pog"] is None:
                continue
            errors[split][side] = {
                "mean_abs_mm": mean_abs_error_mm(pog_pred, dataset["pog"]),
                "mean_2d_mm": mean_2d_error_mm(pog_pred, dataset["pog"]),
            }
            print(f"{split.capitalize()} error {label} {side}: "
                  f"{errors[split][side]['mean_abs_mm']:.3f} mm mean-abs, "
                  f"{errors[split][side]['mean_2d_mm']:.3f} mm 2D")

        preds_training[side] = pog_pred_train
        preds_evaluation[side] = pog_pred_evaluation

    pog_pred_train_both = save_predictions(os.path.join(exp_path, TRAINING_SPLIT), label, preds_training)
    pog_pred_evaluation_both = save_predictions(os.path.join(exp_path, evaluation_split), label, preds_evaluation)

    for split, dataset, pog_pred in [
        (TRAINING_SPLIT, dataset_training, pog_pred_train_both),
        (evaluation_split, dataset_evaluation, pog_pred_evaluation_both),
    ]:
        if dataset["pog"] is None:
            continue
        errors[split]["both"] = {
            "mean_abs_mm": mean_abs_error_mm(pog_pred, dataset["pog"]),
            "mean_2d_mm": mean_2d_error_mm(pog_pred, dataset["pog"]),
        }
        print(f"{split.capitalize()} error {label} both: "
              f"{errors[split]['both']['mean_abs_mm']:.3f} mm mean-abs, "
              f"{errors[split]['both']['mean_2d_mm']:.3f} mm 2D")
    plot_gaze_predictions(pog_pred_train_both, dataset_training["pog"], exp_path,
                          f"vis_{label}.png", display=display)
    return errors, params_by_side


def comparison_splits(errors_by_label, evaluation_split):
    """Return split names to include in printed and CSV comparisons."""
    splits = [TRAINING_SPLIT]
    if evaluation_split != TRAINING_SPLIT:
        splits.append(evaluation_split)
    for split in errors_by_label["rawdepth"].keys():
        if split not in splits:
            splits.append(split)
    return splits


def flatten_errors(errors_by_label, evaluation_split):
    """Flatten nested raw-depth/ClearDepth error dictionaries into CSV rows."""
    rows = []
    for split in comparison_splits(errors_by_label, evaluation_split):
        for side in ["left", "right", "both"]:
            raw = errors_by_label["rawdepth"].get(split, {}).get(side)
            clear = errors_by_label["cleardepth"].get(split, {}).get(side)
            if raw is None or clear is None:
                continue
            rows.append({
                "split": split,
                "eye": side,
                "rawdepth_mean_abs_mm": raw["mean_abs_mm"],
                "cleardepth_mean_abs_mm": clear["mean_abs_mm"],
                "delta_mean_abs_mm": clear["mean_abs_mm"] - raw["mean_abs_mm"],
                "rawdepth_2d_mm": raw["mean_2d_mm"],
                "cleardepth_2d_mm": clear["mean_2d_mm"],
                "delta_2d_mm": clear["mean_2d_mm"] - raw["mean_2d_mm"],
            })
    return rows


def write_comparison_csv(errors_by_label, output_path, evaluation_split):
    """Write raw-depth versus ClearDepth error metrics to a CSV file."""
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "eye",
        "rawdepth_mean_abs_mm",
        "cleardepth_mean_abs_mm",
        "delta_mean_abs_mm",
        "rawdepth_2d_mm",
        "cleardepth_2d_mm",
        "delta_2d_mm",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flatten_errors(errors_by_label, evaluation_split))
    print(f"Wrote comparison CSV to {output_path}")


def print_comparison(errors_by_label, evaluation_split):
    """Print raw-depth versus ClearDepth error metrics to stdout."""
    for split in comparison_splits(errors_by_label, evaluation_split):
        if not errors_by_label["rawdepth"].get(split):
            continue
        print(f"\n=== {split.capitalize()} error comparison ===")
        print("eye,rawdepth_mean_abs_mm,cleardepth_mean_abs_mm,delta_mean_abs_mm,"
              "rawdepth_2d_mm,cleardepth_2d_mm,delta_2d_mm")
        for side in ["left", "right", "both"]:
            raw = errors_by_label["rawdepth"][split][side]
            clear = errors_by_label["cleardepth"][split][side]
            print(f"{side},"
                  f"{raw['mean_abs_mm']:.3f},"
                  f"{clear['mean_abs_mm']:.3f},"
                  f"{clear['mean_abs_mm'] - raw['mean_abs_mm']:.3f},"
                  f"{raw['mean_2d_mm']:.3f},"
                  f"{clear['mean_2d_mm']:.3f},"
                  f"{clear['mean_2d_mm'] - raw['mean_2d_mm']:.3f}")


def main():
    """Parse CLI arguments and run raw-depth and ClearDepth comparisons."""
    parser = argparse.ArgumentParser(description="Train or run geometric gaze estimation with and without ClearDepth.")
    parser.add_argument("--exp-path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "exp",
                        help="Experiment folder containing training, test/evaluation, eye_scan, and camera data.")
    parser.add_argument("--evaluation-split", default=DEFAULT_EVALUATION_SPLIT,
                        help="Dataset used for error evaluation; defaults to test.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional CSV path for the raw-depth versus ClearDepth comparison.")
    parser.add_argument("--maxiter", type=int, default=100000,
                        help="Maximum optimizer iterations for each geometric model fit.")
    parser.add_argument("--reuse-models", action="store_true",
                        help="Reuse existing model parameters if present instead of retraining.")
    parser.add_argument("--predict-only", action="store_true",
                        help="Load saved model parameters and predict one split without fitting.")
    parser.add_argument("--predict-split", default=None,
                        help=("Prediction-only split name. Defaults to the input folder name when "
                              "--predict-input is set, otherwise inference."))
    parser.add_argument("--predict-input", type=Path, default=None,
                        help="RGB-D folder to process in --predict-only mode. It must contain rgb/ and depth/.")
    parser.add_argument("--predict-output-dir", type=Path, default=None,
                        help="Directory for prediction .npy outputs. Defaults to the prediction input folder.")
    parser.add_argument("--depth-mode", choices=["rawdepth", "cleardepth", "both"], default="cleardepth",
                        help="Depth mode to use in --predict-only mode.")
    parser.add_argument("--model-root", type=Path, default=None,
                        help="Folder containing model_rawdepth/model_cleardepth and training landmarks.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Folder containing params_left.json and params_right.json for one depth mode.")
    parser.add_argument("--reference-landmarks", type=Path, default=None,
                        help="Path to landmarks_3d_ref.npy saved when the model was trained.")
    parser.add_argument("--intrinsics", type=Path, default=None,
                        help="Camera intrinsics JSON. Defaults to EXP_PATH/camera/intrinsics.json.")
    parser.add_argument("--display-config", type=Path, default=None,
                        help="Display geometry JSON. Defaults to EXP_PATH/camera/display.json when present.")
    parser.add_argument("--eye-scan-path", type=Path, default=None,
                        help="Eye-scan RGB-D folder for ClearDepth. Defaults to EXP_PATH/eye_scan.")
    args = parser.parse_args()

    exp_path = str(args.exp_path)
    intrinsics_path = args.intrinsics or Path(exp_path) / "camera" / "intrinsics.json"
    intrinsics = load_json(intrinsics_path)
    f = np.array([intrinsics["fx"], intrinsics["fy"]])
    c = np.array([intrinsics["cx"], intrinsics["cy"]])
    display = load_display(exp_path, args.display_config)

    if args.predict_only:
        run_predict_only(
            exp_path,
            f,
            c,
            args.predict_split,
            args.depth_mode,
            predict_input=args.predict_input,
            predict_output_dir=args.predict_output_dir,
            model_root=args.model_root,
            model_path=args.model_path,
            reference_landmarks=args.reference_landmarks,
            eye_scan_path=args.eye_scan_path,
            display=display,
        )
        return

    raw_errors, raw_params_by_side = run_depth_mode(
        exp_path,
        f,
        c,
        apply_cleardepth=False,
        label="rawdepth",
        force_retrain=not args.reuse_models,
        maxiter=args.maxiter,
        evaluation_split=args.evaluation_split,
        display=display,
    )
    cleardepth_errors, _ = run_depth_mode(
        exp_path,
        f,
        c,
        apply_cleardepth=True,
        label="cleardepth",
        force_retrain=not args.reuse_models,
        initial_params_by_side=raw_params_by_side,
        maxiter=args.maxiter,
        evaluation_split=args.evaluation_split,
        display=display,
    )
    errors_by_label = {"rawdepth": raw_errors, "cleardepth": cleardepth_errors}
    print_comparison(errors_by_label, args.evaluation_split)
    write_comparison_csv(errors_by_label, args.output, args.evaluation_split)


if __name__ == "__main__":
    main()
