"""Build geometric-model datasets from RGB-D frames.

This module converts raw RGB/depth frames and point-of-gaze screen-fraction labels
into the quantities consumed by `GeometricModel`: 3D pupil centers, head poses,
screen targets in millimeters, and stable face anchors.
"""

import os
import pickle as pkl
from pathlib import Path
import numpy as np
import warnings
from .landmark_detection import LandmarkDetector
from .utils import Display, methods
from .head_pose_extraction import HeadPoseExtractor
from .cleardepth import ClearDepth


class DataExtractor:
    """Extract face landmarks, eye landmarks, head pose, and pupil coordinates."""

    def __init__(self, landmark_path, f, c, landmark_path_scan=None, wearing_glasses=False,
                 display=None):
        """Create an extractor for one experiment split.

        Args:
            landmark_path: Folder used to cache landmarks for the current split.
            f: Camera focal lengths `[fx, fy]` in pixels.
            c: Camera optical center `[cx, cy]` in pixels.
            landmark_path_scan: Cache folder for eye-scan landmarks when using
                ClearDepth.
            wearing_glasses: If true, avoid eye-corner landmarks for head pose.
            display: Display geometry used to convert point-of-gaze screen
                fractions into millimeters.
        """
        self.landmark_path = landmark_path
        self.landmark_path_scan = landmark_path_scan
        self.landmark_detector = None
        self.display = Display() if display is None else display
        self.cleardepth = ClearDepth(f, c)
        self.f = f  # camera's focal length
        self.c = c  # camera's optical center
        self.anchor_inds = self.get_anchor_inds()
        self.head_pose_inds = self.get_head_pose_inds(wearing_glasses)
        self.wanted_inds = np.concatenate([self.head_pose_inds, self.anchor_inds])

    def get_landmark_detector(self):
        """Create landmark detectors only when cached landmarks are unavailable."""
        if self.landmark_detector is None:
            self.landmark_detector = LandmarkDetector()
        return self.landmark_detector

    @staticmethod
    def get_head_pose_inds(wearing_glasses):
        """Return 68-point face-landmark indices used for rigid head pose."""
        face_parts = ["mouth", "nose_base", "nose_ridge", "eyebrow_left", "eyebrow_right"]
        if not wearing_glasses:
            face_parts += ["inner_eye_corner_left", "inner_eye_corner_right",
            "outer_eye_corner_left", "outer_eye_corner_right"]
        landmarks_inds = []
        for part in face_parts:
            landmarks_inds.append(methods.get_landmark_inds(part))
        landmarks_inds = np.concatenate(landmarks_inds)
        return landmarks_inds

    @staticmethod
    def get_anchor_inds():
        """Return stable face-landmark indices used as positional anchors."""
        return [methods.get_landmark_inds("nose_ridge")[0]]

    def get_screen_coords_pix(self, coords_normalized):
        """Convert normalized screen-fraction targets to display pixels."""
        coords_pix = self.display.normalized_to_pix(coords_normalized)
        return coords_pix

    def get_screen_coords_3d(self, coords_normalized):
        """Convert normalized screen-fraction targets to 3D millimeters."""
        coords_pix = self.get_screen_coords_pix(coords_normalized)
        O, Nx, Ny = self.display.O, self.display.Nx, self.display.Ny
        w_pix, h_pix = self.display.dim_pix
        w_mm, h_mm = self.display.dim_mm
        coords_mm = O + coords_pix[:, 0, np.newaxis] * (w_mm / w_pix) * Nx + coords_pix[:, 1, np.newaxis] * (h_mm / h_pix) * Ny
        return coords_mm

    def detect_face_landmarks(self, imgs, path):
        """Detect or load cached 2D face landmarks for all frames."""
        path = os.path.join(path, "face_landmarks_pix.npy")
        try:
            coords_pix = np.load(path)
            print("Loaded face landmarks of shape", coords_pix.shape)
        except FileNotFoundError:
            coords_pix = self.get_landmark_detector().detect_face_landmarks(imgs)
            np.save(path, coords_pix)
        return coords_pix

    def get_face_landmarks(self, imgs, depth_maps, landmark_path):
        """Return face landmarks in camera-space millimeters and image pixels."""
        print("Preparing face landmarks...")
        coords_pix = self.detect_face_landmarks(imgs, landmark_path)
        depth = self.get_depth_of_pix_coords(coords_pix, depth_maps, self.wanted_inds)
        coords_3d = methods.unproject(coords_pix, depth, self.f, self.c, self.wanted_inds)
        return coords_3d, coords_pix

    def detect_eye_landmarks(self, imgs, face_landmarks, landmark_path=None):
        """Detect or load cached 2D eye landmarks for both eyes."""
        landmark_path = self.landmark_path if landmark_path is None else landmark_path
        os.makedirs(landmark_path, exist_ok=True)
        path = os.path.join(landmark_path, "eye_landmarks_pix.npy")
        try:
            with open(path, "rb") as file:
                coords_pix = pkl.load(file)
            for side, coords_pix_side in coords_pix.items():
                print(f"Loaded {side} eye landmarks of shape", coords_pix_side.shape)
        except FileNotFoundError:
            coords_pix = self.get_landmark_detector().detect_eye_landmarks(imgs, face_landmarks)
            with open(path, "wb") as file:
                pkl.dump(coords_pix, file)
        return coords_pix

    def get_pupils(self, imgs, face_landmarks, depth_maps, apply_cleardepth, eye_scan):
        """Return per-eye 3D pupil centers, optionally using ClearDepth depths."""
        print("Preparing pupil landmarks...")
        coords_pix = self.detect_eye_landmarks(imgs, face_landmarks)
        if apply_cleardepth:
            if self.landmark_path_scan is None:
                raise ValueError("Need eye scan landmark path for ClearDepth.")
            coords_pix_scan = self.detect_eye_landmarks(eye_scan["rgb"], eye_scan["face_landmarks"],
                                                        self.landmark_path_scan)
            eye_scan["eye_landmarks"] = coords_pix_scan
        coords_3d = {}
        for side, coords_pix_side in coords_pix.items():
            coords_pix_side = coords_pix_side[:, -1:] # last landmark is pupil/iris center
            if not apply_cleardepth:
                depth = self.get_depth_of_pix_coords(coords_pix_side, depth_maps)
            else:
                depth = self.cleardepth.run(coords_pix_side, eye_scan, side)  # undistorted pupil depth
            coords_3d[side] = methods.unproject(coords_pix_side, depth, self.f, self.c)[:, 0]
        return coords_3d

    @staticmethod
    def get_depth_of_pix_coords(coords_pix, depth_maps, wanted_inds=None):
        """Sample depth maps at subpixel landmark coordinates.

        Missing depth values are encoded as zero. Before bilinear interpolation,
        zero-valued neighborhood samples are filled from the nearest non-zero
        surrounding window so unprojection does not receive invalid depth.
        """
        print("Sampling depth...")
        wanted_inds = np.arange(coords_pix.shape[1]) if wanted_inds is None else wanted_inds

        def bilinear_interpolation(x_, y_, values):
            return (values[0][0] * (1 - x_) * (1 - y_) +
                    values[0][1] * (1 - x_) * y_ +
                    values[1][0] * x_ * (1 - y_) +
                    values[1][1] * x_ * y_)

        def linear_interpolation(x_, values):
            return values[0] * (1 - x_) + values[1] * x_

        def interpolate_depth(x_, y_, xl, yl, xr, yr, dm_):
            if xl == xr:
                if yl == yr:
                    return dm_[yl, xl]
                else:
                    values = [dm_[yl, xl],
                              dm_[yr, xr]]
                    return linear_interpolation(y_ - yl, values)
            else:
                if yl == yr:
                    values = [dm_[yl, xl],
                              dm_[yr, xr]]
                    return linear_interpolation(x_ - xl, values)
                else:
                    values = ((dm_[yl, xl], dm_[yr, xl]),
                              (dm_[yl, xr], dm_[yr, xr]))
                    return bilinear_interpolation(x_ - xl, y_ - yl, values)

        # Ensure neighborhood's depth values are non-zero (i.e. not missing)
        def replace_zero_depth(neighborhood_, dm_, mask_):
            if np.all(mask_):
                raise ValueError("Cannot interpolate depth because the depth map contains no valid values.")
            for k, (col, row) in enumerate(neighborhood_):
                if 0 < dm_[row, col]:
                    continue
                wl = 1  # window length for interpolating zero depth
                while True:
                    br, tr = np.maximum(0, row - wl), row + wl + 1
                    bc, tc = np.maximum(0, col - wl), col + wl + 1

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dm_[row, col] = np.ma.array(dm_[br: tr, bc: tc], mask=mask_[br: tr, bc: tc]).mean()
                    if np.isnan(dm_[row, col]):
                        if br == 0 and bc == 0 and tr >= dm_.shape[0] and tc >= dm_.shape[1]:
                            raise ValueError("Cannot interpolate a missing depth value from an all-zero neighborhood.")
                        wl += 1  # all neighbors of zero depth are zero too -> try with larger window
                    else:
                        break
        depth = np.zeros(coords_pix.shape[:2])
        for i, (coords, dm_raw) in enumerate(zip(coords_pix, depth_maps)):
            dm = dm_raw.copy()
            mask = dm == 0  # zero means depth is missing
            h, w = dm.shape
            for j, (x, y) in enumerate(coords):
                if j not in wanted_inds:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    raise ValueError(f"Landmark {j} in frame {i} is not finite: ({x}, {y}).")
                if not (0 <= x < w and 0 <= y < h):
                    raise ValueError(
                        f"Landmark {j} in frame {i} is outside the depth map: "
                        f"({x:.2f}, {y:.2f}) for map size {w}x{h}."
                    )
                floor_x, floor_y = int(np.floor(x)), int(np.floor(y))
                ceil_x, ceil_y = min(w - 1, int(np.ceil(x))), min(h - 1, int(np.ceil(y)))
                neighborhood = [(floor_x, floor_y), (ceil_x, ceil_y), (ceil_x, floor_y), (floor_x, ceil_y)]
                # After calling replace_zero_depth, neighborhood's depth values in dm are non-zero
                replace_zero_depth(neighborhood, dm, mask)
                # Interpolate coord's depth using neighborhood's depth
                depth[i, j] = interpolate_depth(x, y, floor_x, floor_y, ceil_x, ceil_y, dm)

        return depth

    def get_head_poses(self, face_landmarks_3d, coords_ref):
        """Estimate a rigid head pose for each frame relative to `coords_ref`."""
        head_pose_extractor = HeadPoseExtractor(coords_ref, self.head_pose_inds)
        head_poses = [head_pose_extractor.get_head_pose(coords) for coords in face_landmarks_3d]
        return head_poses

    def get_anchors(self, face_landmarks_3d):
        """Return stable per-frame face anchor points in camera coordinates."""
        return face_landmarks_3d[:, self.anchor_inds].mean(axis=1)

    def get_geometric_dataset(self, imgs, depth_maps, pog_normalized, landmarks_3d_ref_path,
                              apply_cleardepth=False, eye_scan=None,
                              save_reference_landmarks=False):
        """Assemble all geometric-model inputs for one split.

        When `save_reference_landmarks` is true, the first frame's 3D face
        landmarks become the reference head pose. Evaluation and prediction-only
        data should load that saved reference so the fitted model sees a
        consistent head-coordinate system.
        """
        pog_3d = self.get_screen_coords_3d(pog_normalized) if pog_normalized is not None else None
        face_landmarks_3d, face_landmarks_pix = self.get_face_landmarks(imgs, depth_maps, self.landmark_path)
        landmarks_3d_ref_path = Path(landmarks_3d_ref_path)
        if landmarks_3d_ref_path.suffix != ".npy":
            landmarks_3d_ref_path = landmarks_3d_ref_path / "landmarks_3d_ref.npy"
        if save_reference_landmarks:
            landmarks_3d_ref_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(landmarks_3d_ref_path, face_landmarks_3d[0])
        head_poses = self.get_head_poses(face_landmarks_3d, str(landmarks_3d_ref_path))
        if apply_cleardepth:
            if eye_scan is None:
                raise ValueError("Need eye scan for ClearDepth.")
            imgs_scan, depth_maps_scan = eye_scan["rgb"], eye_scan["depth"]
            landmarks_scan = self.get_face_landmarks(imgs_scan, depth_maps_scan, self.landmark_path_scan)
            face_landmarks_3d_scan, face_landmarks_pix_scan = landmarks_scan
            head_poses_scan = self.get_head_poses(face_landmarks_3d, face_landmarks_3d_scan.mean(axis=0))
            eye_scan["face_landmarks"] = face_landmarks_pix_scan
            eye_scan["head_poses"] = head_poses_scan
        pupils_3d = self.get_pupils(imgs, face_landmarks_pix, depth_maps, apply_cleardepth, eye_scan)
        dataset = {"pog": pog_3d,
                   "head_pose": {"R": np.stack([hp[0] for hp in head_poses]),
                                 "t": np.stack([hp[1] for hp in head_poses])},
                   "pupil": pupils_3d,
                   "anchor": self.get_anchors(face_landmarks_3d)}

        return dataset
