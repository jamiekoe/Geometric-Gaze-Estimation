"""ClearDepth pupil-depth replacement.

ClearDepth uses an eye scan recorded with wide-open eyes as an undistorted depth
source. For each runtime frame, the eye-scan depth points are transformed with
the current head pose, projected into the image, and the point nearest the
detected pupil supplies the replacement pupil depth.
"""

import numpy as np
import matplotlib.path as matplot_path
from .utils import methods


class ClearDepth:
    """Compute replacement pupil depths from an eye-scan depth map."""

    def __init__(self, f, c):
        """Store camera intrinsics.

        Args:
            f: Focal lengths `[fx, fy]` in pixels.
            c: Optical center `[cx, cy]` in pixels.
        """
        self.f = f
        self.c = c

    @staticmethod
    def get_grid(eye_boundary_landmarks):
        """Return all integer pixels inside the averaged eye boundary."""
        x_min, y_min = np.min(np.floor(eye_boundary_landmarks).astype(int), axis=0)
        x_max, y_max = np.max(np.ceil(eye_boundary_landmarks).astype(int), axis=0) + 1
        xv, yv = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        eye_boundary = matplot_path.Path([tuple(coord) for coord in eye_boundary_landmarks])
        flags = eye_boundary.contains_points(np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis])))
        grid = np.zeros((y_max - y_min, x_max - x_min), dtype='bool')
        grid[(yv.flatten() - y_min).astype('int'), (xv.flatten() - x_min).astype('int')] = flags
        return grid, xv, yv, x_min, x_max, y_min, y_max

    @staticmethod
    def interpolate_depth(dm, grid):
        """Fill missing zero-depth pixels inside the ClearDepth eye mask."""
        h, w = dm.shape
        if not np.any(grid):
            raise ValueError("ClearDepth eye mask is empty.")
        if np.all(dm[grid] == 0):
            raise ValueError("ClearDepth cannot interpolate an eye-scan region with no valid depth.")
        mask = dm == 0
        for row in range(h):
            for col in range(w):
                if 0 < dm[row, col] or not grid[row, col]:
                    continue
                wl = 1  # window length for interpolating zero depth
                while True:
                    br, tr = np.maximum(0, row - wl), row + wl + 1
                    bc, tc = np.maximum(0, col - wl), col + wl + 1
                    dm[row, col] = np.ma.array(dm[br: tr, bc: tc], mask=mask[br: tr, bc: tc]).mean()
                    if np.isnan(dm[row, col]):
                        if br == 0 and bc == 0 and tr >= h and tc >= w:
                            raise ValueError("ClearDepth cannot fill a missing depth value from an all-zero region.")
                        wl += 1
                    else:
                        break
        return dm

    def run(self, pupil_pix, eye_scan, side):
        """Return ClearDepth replacement depths for detected pupil pixels.

        Args:
            pupil_pix: Runtime pupil/iris centers in image pixels.
            eye_scan: Dict containing scan RGB/depth frames, scan eye landmarks,
                and head poses from scan coordinates into runtime coordinates.
            side: `"left"` or `"right"`.
        """
        if pupil_pix.ndim == 3 and pupil_pix.shape[1] == 1:
            pupil_pix = pupil_pix[:, 0]
        eye_boundary_landmarks = eye_scan["eye_landmarks"][side][:, :16].mean(axis=0)
        grid, x_mesh, y_mesh, x_l, x_h, y_l, y_h = self.get_grid(eye_boundary_landmarks)
        depth = np.mean([self.interpolate_depth(dm[y_l: y_h, x_l: x_h].copy(), grid) for dm in eye_scan["depth"]], axis=0)
        grid = grid.reshape(-1)
        depth = depth.reshape(-1)
        if not np.any(grid):
            raise ValueError(f"ClearDepth found no candidate pixels for {side} eye.")
        eye_pix = np.stack((x_mesh, y_mesh), axis=-1).reshape(-1, 2)
        eye_3d = methods.unproject(eye_pix[grid][:, np.newaxis], depth[grid][:, np.newaxis], self.f, self.c)[:, 0]
        # Transform all usable eye-scan points into each runtime head pose.
        eye_3d_transformed = np.stack([eye_3d @ R.T + t for R, t in eye_scan["head_poses"]])  # N x M x 3
        eye_pix_transformed = methods.project(eye_3d_transformed, self.f, self.c)  # N x M x 2
        pupil_pix = np.repeat(pupil_pix[:, np.newaxis, :], eye_pix_transformed.shape[1], axis=1)  # N x M x 2
        dists = np.linalg.norm(pupil_pix - eye_pix_transformed, axis=-1)  # N x M
        inds = np.argmin(dists, axis=-1)  # N
        pupil_depth_clear = eye_3d_transformed[np.arange(len(inds)), inds, 2]  # extract depth
        return pupil_depth_clear[:, np.newaxis]
