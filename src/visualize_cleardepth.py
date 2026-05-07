"""Visualize which eye-scan pixels ClearDepth uses for replacement depth.

The output images are diagnostic overlays on the averaged eye-scan RGB frames:
candidate pixels inside the eye boundary are shown as small blue points, and
the per-frame pixels selected as nearest projected replacements are shown with
the image legend's early-to-late color ramp.
"""

import csv
import glob
import json
import os
import pickle as pkl
from pathlib import Path

import cv2
import natsort
import numpy as np

from data_extraction import DataExtractor
from data_extraction.utils import Display, methods

CANDIDATE_PIXEL_COLOR = (255, 180, 0)
EYE_BOUNDARY_COLOR = (0, 255, 255)
POG_CSV_NAME = "pog.csv"


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
        return json.load(file)


def load_display(exp_path, display_config=None):
    """Load display geometry from JSON or fall back to the default example device."""
    if display_config is None:
        candidate = Path(exp_path) / "camera" / "display.json"
        display_config = candidate if candidate.exists() else None
    if display_config is None:
        return Display()
    return Display.from_json(display_config)


def has_pog(path):
    """Return whether a split folder contains point-of-gaze labels."""
    return (Path(path) / POG_CSV_NAME).is_file()


def load_pog_csv(path, rgb_files):
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

    if len(rows) != len(rgb_files):
        raise ValueError(f"Found {len(rgb_files)} RGB images but {len(rows)} POG rows in {csv_path}.")

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


def load_pog_labels(path, rgb_files):
    """Load point-of-gaze labels as normalized screen fractions from `pog.csv`."""
    csv_path = Path(path) / POG_CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Point-of-gaze labels must be stored in {csv_path}.")
    return load_pog_csv(path, rgb_files)


def get_raw_data(path, get_pog=False):
    """Load RGB frames, depth maps, and optionally point-of-gaze labels."""
    rgb_files = natsort.natsorted(glob.glob(os.path.join(path, "rgb", "*.png")))
    depth_files = natsort.natsorted(
        glob.glob(os.path.join(path, "depth", "*.npy")) +
        glob.glob(os.path.join(path, "depth", "*.npz"))
    )
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {os.path.join(path, 'rgb')}.")
    if len(depth_files) != len(rgb_files):
        raise ValueError(f"Found {len(rgb_files)} RGB images but {len(depth_files)} depth maps in {path}.")
    data = {
        "img": load_imgs(rgb_files),
        "depth": load_npy_arrays(depth_files),
        "pog_normalized": None,
    }
    if get_pog:
        data["pog_normalized"] = load_pog_labels(path, rgb_files)
    return data


def load_eye_landmarks(path):
    """Load cached EyeNet landmarks for one data split."""
    with open(os.path.join(path, "landmarks", "eye_landmarks_pix.npy"), "rb") as file:
        return pkl.load(file)


def compute_selected_eye_scan_pixels(cleardepth, pupil_pix, eye_scan, side):
    """Recompute the ClearDepth nearest-neighbor pixel selection for plotting.

    This mirrors `ClearDepth.run`, but returns intermediate candidate pixels,
    selected pixels, depths, and distances so they can be visualized.
    """
    if pupil_pix.ndim == 3 and pupil_pix.shape[1] == 1:
        pupil_pix = pupil_pix[:, 0]

    eye_boundary_landmarks = eye_scan["eye_landmarks"][side][:, :16].mean(axis=0)
    grid, x_mesh, y_mesh, x_l, x_h, y_l, y_h = cleardepth.get_grid(eye_boundary_landmarks)
    depth = np.mean([cleardepth.interpolate_depth(dm[y_l: y_h, x_l: x_h].copy(), grid)
                     for dm in eye_scan["depth"]], axis=0)

    grid = grid.reshape(-1)
    depth = depth.reshape(-1)
    eye_pix = np.stack((x_mesh, y_mesh), axis=-1).reshape(-1, 2)
    candidate_pix = eye_pix[grid]
    candidate_depth = depth[grid]
    if len(candidate_pix) == 0:
        raise ValueError(f"ClearDepth found no candidate pixels for {side} eye.")
    eye_3d = methods.unproject(candidate_pix[:, np.newaxis], candidate_depth[:, np.newaxis],
                               cleardepth.f, cleardepth.c)[:, 0]
    eye_3d_transformed = np.stack([eye_3d @ R.T + t for R, t in eye_scan["head_poses"]])
    eye_pix_transformed = methods.project(eye_3d_transformed, cleardepth.f, cleardepth.c)

    pupil_pix_repeated = np.repeat(pupil_pix[:, np.newaxis, :], eye_pix_transformed.shape[1], axis=1)
    dists = np.linalg.norm(pupil_pix_repeated - eye_pix_transformed, axis=-1)
    inds = np.argmin(dists, axis=-1)

    return {
        "candidate_pix": candidate_pix,
        "selected_pix": candidate_pix[inds],
        "selected_depth": eye_3d_transformed[np.arange(len(inds)), inds, 2],
        "nearest_distance_pix": dists[np.arange(len(inds)), inds],
        "boundary": eye_boundary_landmarks,
    }


def crop_bounds(points, image_shape, pad=35):
    """Return a padded image crop around a set of x/y points."""
    h, w = image_shape[:2]
    x_min, y_min = np.floor(points.min(axis=0)).astype(int) - pad
    x_max, y_max = np.ceil(points.max(axis=0)).astype(int) + pad
    return max(0, x_min), min(w, x_max), max(0, y_min), min(h, y_max)


def draw_points(image, candidate_pix, selected_pix, boundary, title=None):
    """Draw candidate pixels, selected pixels, and the eye boundary."""
    output = image.copy()
    overlay = output.copy()
    for x, y in np.round(candidate_pix).astype(int):
        cv2.circle(overlay, (x, y), 1, CANDIDATE_PIXEL_COLOR, -1)
    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

    cv2.polylines(output, [np.round(boundary).astype(np.int32)], isClosed=True,
                  color=EYE_BOUNDARY_COLOR, thickness=1)

    selected_int = np.round(selected_pix).astype(int)
    colors = cv2.applyColorMap(np.linspace(0, 255, len(selected_int)).astype(np.uint8),
                               cv2.COLORMAP_TURBO)[:, 0, :]
    for i, ((x, y), color) in enumerate(zip(selected_int, colors)):
        color = tuple(int(v) for v in color)
        cv2.circle(output, (x, y), 2, color, -1)

    if title is not None:
        cv2.putText(output, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def append_legend(image):
    """Append a compact legend that explains the overlay colors."""
    h, w = image.shape[:2]
    out_w = max(w, 310)
    x_image = (out_w - w) // 2
    legend_h = 72
    output = np.zeros((h + legend_h, out_w, 3), dtype=image.dtype)
    output[:h] = (12, 12, 12)
    output[:h, x_image:x_image + w] = image
    output[h:] = (28, 28, 28)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.42, max(0.32, out_w / 700))
    thickness = 1
    text_color = (245, 245, 245)
    x0 = 10
    y0 = h + 17
    row_gap = 21

    cv2.line(output, (x0, y0 - 4), (x0 + 24, y0 - 4), EYE_BOUNDARY_COLOR, 2)
    cv2.putText(output, "boundary", (x0 + 34, y0), font, scale, text_color, thickness, cv2.LINE_AA)

    cv2.circle(output, (x0 + 12, y0 + row_gap - 5), 3, CANDIDATE_PIXEL_COLOR, -1)
    cv2.putText(output, "candidate pixels", (x0 + 34, y0 + row_gap), font, scale, text_color, thickness, cv2.LINE_AA)

    gradient_w = 70
    gradient_y = y0 + 2 * row_gap - 13
    gradient = cv2.applyColorMap(np.linspace(0, 255, gradient_w).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)[:, 0, :]
    for x, color in enumerate(gradient):
        output[gradient_y:gradient_y + 8, x0 + x] = color
    cv2.rectangle(output, (x0, gradient_y), (x0 + gradient_w - 1, gradient_y + 7),
                  (245, 245, 245), 1)
    cv2.putText(output, "selected: early -> late", (x0 + gradient_w + 10, y0 + 2 * row_gap),
                font, scale, text_color, thickness, cv2.LINE_AA)
    return output


def save_side_visualization(mean_rgb, results, mode, side, out_dir):
    """Save a cropped visualization for one split and one eye."""
    points = np.concatenate([results["candidate_pix"], results["selected_pix"], results["boundary"]], axis=0)
    x_l, x_h, y_l, y_h = crop_bounds(points, mean_rgb.shape)
    annotated = draw_points(mean_rgb, results["candidate_pix"], results["selected_pix"],
                            results["boundary"], f"{mode} {side}")
    crop = annotated[y_l:y_h, x_l:x_h]
    crop = append_legend(crop)
    cv2.imwrite(os.path.join(out_dir, f"{mode}_{side}_used_eyescan_pixels.png"), crop)


def save_combined_visualization(mean_rgb, results_by_side, mode, out_dir):
    """Save one cropped visualization containing both eyes."""
    annotated = mean_rgb.copy()
    for side, results in results_by_side.items():
        annotated = draw_points(annotated, results["candidate_pix"], results["selected_pix"],
                                results["boundary"])
    cv2.putText(annotated, f"{mode} both", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    points = np.concatenate([item["candidate_pix"] for item in results_by_side.values()] +
                            [item["selected_pix"] for item in results_by_side.values()] +
                            [item["boundary"] for item in results_by_side.values()], axis=0)
    x_l, x_h, y_l, y_h = crop_bounds(points, mean_rgb.shape, pad=45)
    crop = append_legend(annotated[y_l:y_h, x_l:x_h])
    cv2.imwrite(os.path.join(out_dir, f"{mode}_both_used_eyescan_pixels.png"), crop)


def save_selected_pixels_csv(results_by_side, mode, out_dir):
    """Save selected ClearDepth source pixels and depths as tabular data."""
    path = os.path.join(out_dir, f"{mode}_selected_eyescan_pixels.csv")
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "side", "frame_index", "x_pix", "y_pix", "replacement_depth", "nearest_distance_pix"])
        for side, results in results_by_side.items():
            for i, (pix, depth, dist) in enumerate(zip(results["selected_pix"],
                                                       results["selected_depth"],
                                                       results["nearest_distance_pix"])):
                writer.writerow([mode, side, i, float(pix[0]), float(pix[1]), float(depth), float(dist)])


def visualize_mode(exp_path, f, c, mode, out_dir, display=None):
    """Generate ClearDepth source-pixel visualizations for one data split."""
    eye_scan_path = os.path.join(exp_path, "eye_scan")
    landmark_path_scan = create_directory(os.path.join(eye_scan_path, "landmarks"))
    data_scan = get_raw_data(eye_scan_path)
    eye_scan = {"rgb": data_scan["img"], "depth": data_scan["depth"]}

    data_path = os.path.join(exp_path, mode)
    dataset_raw = get_raw_data(data_path, get_pog=has_pog(data_path))
    landmark_path = create_directory(os.path.join(data_path, "landmarks"))
    data_extractor = DataExtractor(landmark_path, f, c, landmark_path_scan=landmark_path_scan,
                                   display=display)
    data_extractor.get_geometric_dataset(dataset_raw["img"], dataset_raw["depth"],
                                         dataset_raw["pog_normalized"],
                                         os.path.join(exp_path, "training", "landmarks"),
                                         apply_cleardepth=True, eye_scan=eye_scan,
                                         save_reference_landmarks=(mode == "training"))

    runtime_eye_landmarks = load_eye_landmarks(data_path)
    results_by_side = {}
    for side in ["left", "right"]:
        pupil_pix = runtime_eye_landmarks[side][:, -1:]
        results_by_side[side] = compute_selected_eye_scan_pixels(data_extractor.cleardepth,
                                                                 pupil_pix, eye_scan, side)

    mean_rgb = np.mean(eye_scan["rgb"], axis=0).astype(np.uint8)
    for side, results in results_by_side.items():
        save_side_visualization(mean_rgb, results, mode, side, out_dir)
    save_combined_visualization(mean_rgb, results_by_side, mode, out_dir)
    save_selected_pixels_csv(results_by_side, mode, out_dir)
    np.savez(os.path.join(out_dir, f"{mode}_selected_eyescan_pixels.npz"), **{
        f"{side}_{key}": value
        for side, results in results_by_side.items()
        for key, value in results.items()
    })


def main():
    """Parse CLI arguments and write ClearDepth visualization artifacts."""
    import argparse
    parser = argparse.ArgumentParser(description="Visualize ClearDepth source pixels.")
    parser.add_argument("--exp-path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "exp",
                        help="Experiment folder containing training, test/evaluation, eye_scan, and camera data.")
    parser.add_argument("--splits", nargs="+", default=["training", "test"],
                        help="Experiment splits to visualize; defaults to training test.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory where PNG/CSV/NPZ visualization outputs are written.")
    parser.add_argument("--display-config", type=Path, default=None,
                        help="Display geometry JSON. Defaults to EXP_PATH/camera/display.json when present.")
    args = parser.parse_args()

    exp_path = str(args.exp_path)
    out_dir = create_directory(args.output_dir or os.path.join(exp_path, "visualizations"))
    intrinsics = load_json(os.path.join(exp_path, "camera", "intrinsics.json"))
    f = np.array([intrinsics["fx"], intrinsics["fy"]])
    c = np.array([intrinsics["cx"], intrinsics["cy"]])
    display = load_display(exp_path, args.display_config)
    for mode in args.splits:
        visualize_mode(exp_path, f, c, mode, out_dir, display=display)
    print(f"Saved ClearDepth visualizations to {out_dir}")


if __name__ == "__main__":
    main()
