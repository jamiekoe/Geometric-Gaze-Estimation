"""Geometry helper functions for camera projection and landmark groups."""

import numpy as np


def get_rays(coords_pix, f, c):
    """Convert image pixels to normalized camera rays."""
    f = np.copy(f)
    f[1] *= -1
    rays = (coords_pix - c) / f
    return rays


def unproject(coords_pix, depth, f, c, wanted_inds=None):
    """Convert image pixels and depths to 3D camera coordinates in millimeters."""
    wanted_inds = np.arange(coords_pix.shape[1]) if wanted_inds is None else wanted_inds
    if np.any(depth[:, wanted_inds] == 0):
        raise ValueError("Depth must be non-zero for all unprojected coordinates.")
    rays = get_rays(coords_pix, f, c)
    coords_3d = np.stack((rays[..., 0] * depth, rays[..., 1] * depth, np.atleast_1d(depth)), axis=-1)  # (-1, 3)
    return coords_3d if coords_3d.shape[0] > 1 else coords_3d[0]


def project(P, f, c):
    """Project 3D camera coordinates into image pixels."""
    f = np.copy(f)
    f[1] *= -1
    p = f * P[..., :2] / P[..., 2:] + c
    return p


def get_landmark_inds(name):
    """Return 68-point face-landmark indices for a named facial region."""
    match name:
        case "nose_ridge":
            return np.array([27, 28, 29, 30])
        case "nose_base":
            return np.array([31, 32, 33, 34, 35])
        case "inner_eye_corner_left":
            return np.array([42])
        case "inner_eye_corner_right":
            return np.array([39])
        case "outer_eye_corner_left":
            return np.array([45])
        case "outer_eye_corner_right":
            return np.array([36])
        case "mouth":
            return np.arange(48, 67 + 1)
        case "mouth_corners":
            return np.array([48, 54])
        case "eyebrow_left":
            return np.array([22, 23, 24, 25])
        case "eyebrow_right":
            return np.array([18, 19, 20, 21])
        case _:
            raise ValueError(f"Unknown landmark name {name}.")
