"""Display geometry for converting between screen pixels and millimeters."""

import json
import numpy as np
from typing import NamedTuple

class Coordinate(NamedTuple):
    """2D coordinate container used by display conversion helpers."""
    x: int | float
    y: int | float

class Display:
    """
    Represents the tablet display used during data collection.

    Defaults match the example data: iPad Pro 12.9-inch in portrait
    orientation, 2048 px wide by 2732 px tall, 264 pixels per inch. Use
    `from_json` or constructor arguments for a different device.
    """

    def __init__(self, width_px=2048, height_px=2732, ppi=264,
                 camera_to_screen_distance_mm=5.0) -> None:
        self.dim_pix = int(width_px), int(height_px)  # w, h
        self.ppi = float(ppi)
        self.mm_per_inch = 25.4
        self.pix_per_mm = self.ppi / self.mm_per_inch
        self.mm_per_pix = self.mm_per_inch / self.ppi
        self.dim_mm = tuple([pix / self.pix_per_mm for pix in self.dim_pix])
        self.camera_to_screen_distance = float(camera_to_screen_distance_mm)
        # From camera's POV: pixel origin is top right screen corner
        self.O = np.array([self.dim_mm[0] / 2, -self.camera_to_screen_distance, 0])
        # From camera's POV: pixel x-axis points left but Camera CS x-axis points right
        self.Nx = np.array([-1, 0, 0])
        # From camera's POV: pixel y-axis points down but Camera CS y-axis points up
        self.Ny = np.array([0, -1, 0])
        # From camera's POV: Camera CS z-axis points towards user
        self.Nz = np.array([0, 0, 1])

    @classmethod
    def from_dict(cls, config):
        """Create a display from a dictionary loaded from JSON."""
        return cls(
            width_px=config["width_px"],
            height_px=config["height_px"],
            ppi=config["ppi"],
            camera_to_screen_distance_mm=config.get("camera_to_screen_distance_mm", 5.0),
        )

    @classmethod
    def from_json(cls, path):
        """Create a display from a JSON config file."""
        with open(path, "r") as file:
            config = json.load(file)
        return cls.from_dict(config)

    def get_screen_coordinates_3d(self):
        """Return the display corners in camera coordinates."""
        upper_right = self.O
        upper_left = self.O + self.Nx * self.dim_mm[0]
        lower_right = self.O + self.Ny * self.dim_mm[1]
        lower_left = self.O + self.Nx * self.dim_mm[0] + self.Ny * self.dim_mm[1]
        return upper_left, upper_right, lower_left, lower_right

    def pix_to_mm(self, x_pix: int, y_pix: int) -> Coordinate:
        """Convert display pixel coordinates to display-plane millimeters."""
        x_mm = (self.dim_pix[0] / 2 - x_pix) / self.pix_per_mm
        y_mm = -self.camera_to_screen_distance - y_pix / self.pix_per_mm
        return Coordinate(x_mm, y_mm)

    def mm_to_pix(self, x_mm: float, y_mm: float) -> Coordinate:
        """Convert display-plane millimeters to display pixel coordinates."""
        x_pix = int(np.round(self.dim_pix[0] / 2 - x_mm * self.pix_per_mm))
        y_pix = int(np.round((-y_mm - self.camera_to_screen_distance) * self.pix_per_mm))
        return Coordinate(x_pix, y_pix)

    def normalized_to_pix(self, coords_normalized):
        """Convert `[0, 1]` display-width/height fractions to pixels."""
        return np.round(coords_normalized * np.array(self.dim_pix)).astype(int)
