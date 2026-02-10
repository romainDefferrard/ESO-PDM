"""
Filename: raster_loader.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the RasterLoader class, which builds a blank grid
    for the Generic footprint mode.

    The main output is an instance providing:
        - self.raster: 2D numpy array (zeros).
        - self.x_mesh, self.y_mesh: 2D arrays of projected coordinates.
        - self.map_bounds: Buffered bounding box derived from flight area.
"""
import numpy as np
from typing import List


class RasterLoader:
    def __init__(self, config: dict, flight_bounds: List[float]) -> None:
        """
        Initializes RasterLoader and builds the grid.

        Input:
            config (dict): configuration dictionary.
                - GRID_RES (float): Grid resolution [m].
                - GRID_BUFFER (float): Buffer distance [m].
            flight_bounds (list[float]): [E_min, E_max, N_min, N_max] bounds of flight area.

        Output:
            None (but sets self.raster, self.x_mesh, self.y_mesh, self.map_bounds)
        """
        self.grid_res = config["GRID_RES"]
        self.buffer = config["GRID_BUFFER"]
        self.flight_bounds = flight_bounds
        
        self.map_bounds = {}

        self.raster: np.ndarray
        self.x_mesh: np.ndarray
        self.y_mesh: np.ndarray

        self.compute_map_bounds()
        self.load()

    def load(self) -> np.ndarray:
        """
        Builds a blank grid (Generic).

        Input:
            None

        Output:
            np.ndarray: A blank raster array aligned with the grid.
        """
        res = float(self.grid_res)
        x_coords = np.arange(self.map_bounds[0], self.map_bounds[1] + res, res)
        y_coords = np.arange(self.map_bounds[3], self.map_bounds[2] - res, -res)
        self.x_mesh, self.y_mesh = np.meshgrid(x_coords, y_coords)
        self.raster = np.zeros_like(self.x_mesh, dtype=float)
        return self.raster

    def compute_map_bounds(self) -> None:
        """
        Computes a buffered bounding box around the flight area.

        Input:
            None

        Output:
            None (updates self.map_bounds as [E_min, E_max, N_min, N_max])
        """
        buffer_coef = np.array([-self.buffer, self.buffer, -self.buffer, self.buffer])
        bounds_array = np.array(self.flight_bounds)
        self.map_bounds = bounds_array + buffer_coef
