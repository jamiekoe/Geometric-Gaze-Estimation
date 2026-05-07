"""Rigid head-pose estimation from 3D face landmarks."""

import os
import numpy as np


class HeadPoseExtractor:
    """Estimate rotation/translation from a reference face-landmark set."""

    def __init__(self, coords_or_path_ref, head_pose_inds):
        """Store centered reference landmarks used for Procrustes alignment."""
        if isinstance(coords_or_path_ref, (str, os.PathLike)):
            coords_ref = np.load(coords_or_path_ref)
        else:
            coords_ref = coords_or_path_ref
        self.head_pose_inds = head_pose_inds
        coords_ref = coords_ref[head_pose_inds]
        self.centroid_ref = np.mean(coords_ref, axis=0)
        self.coords_ref_centered = (coords_ref - self.centroid_ref).transpose()

    def get_head_pose(self, coords):
        """Return `(R, t)` mapping reference landmarks into current landmarks."""
        coords = coords[self.head_pose_inds]
        centroid = np.mean(coords, axis=0)
        H = np.matmul(self.coords_ref_centered, coords - centroid)
        U, _, Vt = np.linalg.svd(H)
        R = np.matmul(Vt.transpose(), U.transpose())
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.matmul(Vt.transpose(), U.transpose())
        t = centroid - np.matmul(R, self.centroid_ref)
        return R, t
