"""
Filename: raster_loader.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the RasterLoader class, which either loads a buffered
    DTM raster window (SwissDTM mode) or builds a blank grid (Generic mode).

    The main output is an instance providing:
        - self.raster: 2D numpy array.
        - self.x_mesh, self.y_mesh: 2D arrays of Swiss projected coordinates (e.g., LV95 or LV03).
        - self.map_bounds: Buffered bounding box derived from flight area.
"""
import numpy as np
import rasterio
from typing import List


class RasterLoader:
    def __init__(self, config: dict, flight_bounds: List[float]) -> None:
        """
        Initializes RasterLoader and loads the raster or builds the grid.

        Input:
            config (dict): configuration dictionary.
                - FOOTPRINT_MODE (str): "Generic" or "SwissDTM".
                - GRID_RES (float): Grid resolution [m] (Generic).
                - GRID_BUFFER (float): Buffer distance [m] (Generic).
                - DTM_PATH (str): Path to raster file (SwissDTM).
                - RASTER_BUFFER (float): Buffer distance [m] (SwissDTM).
            flight_bounds (list[float]): [E_min, E_max, N_min, N_max] bounds of flight area.

        Output:
            None (but sets self.raster, self.x_mesh, self.y_mesh, self.map_bounds)
        """
        self.mode = config.get("FOOTPRINT_MODE", "Generic")
        if self.mode == "SwissDTM":
            self.file_path = config["DTM_PATH"]
            self.buffer = config["RASTER_BUFFER"]
        else:
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
        Loads a DTM window (SwissDTM) or builds a blank grid (Generic).

        Input:
            None

        Output:
            np.ndarray: A blank raster array aligned with the grid.
        """
        if self.mode == "SwissDTM":
            with rasterio.open(self.file_path) as src:
                res_x, res_y = src.res
                x_coords = np.arange(self.map_bounds[0], self.map_bounds[1] + res_x, res_x)
                y_coords = np.arange(self.map_bounds[3], self.map_bounds[2] - res_y, -res_y)
                self.x_mesh, self.y_mesh = np.meshgrid(x_coords, y_coords)

                row_start, col_start = src.index(x_coords[0], y_coords[0])
                row_end, col_end = src.index(x_coords[-1], y_coords[-1])

                window = rasterio.windows.Window.from_slices(
                    (row_start, row_end + 1), (col_start, col_end + 1)
                )
                self.raster = src.read(1, window=window)
                return self.raster

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
