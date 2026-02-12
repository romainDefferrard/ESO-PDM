"""
Filename: footprint_generator.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the Footprint class, which computes observed raster zones for a set of flights
    using point cloud occupancy on a raster grid (Generic mode).

    The main outputs are:
        - self.observed_masks: List[np.ndarray] of boolean masks (one per flight).
        - self.superpos_masks: List[np.ndarray] of boolean masks (one per flight pair).
        - self.superpos_flight_pairs: List[Tuple[str, str]] flight ID pairs used in superposition analysis.
"""
import numpy as np
from itertools import combinations
from more_itertools import pairwise
from typing import Dict, Tuple
from tqdm import tqdm
import laspy
import pandas as pd

def get_footprint_wrapper(args):
    return Footprint.get_footprint(*args)


def downsample_points(coords: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return coords
    return coords[::step]


class Footprint:
    def __init__(
        self,
        raster: np.ndarray,
        raster_mesh: Tuple[np.ndarray, np.ndarray],
        flights: Dict[str, dict],
        flight_files: Dict[str, str],
        config: dict,
    ) -> None:
        """
        Initializes the Footprint object and computes visibility masks for all flights.

        Inputs:
            raster (np.ndarray): 2D array (blank grid for Generic).
            raster_mesh (tuple of np.ndarray): x and y coordinate grids [m].
            flights (dict[str, dict]): Dictionary of flight trajectory data.
            flight_files (dict[str, str]): Mapping of flight_id -> file path (Generic).
            config (dict): Configuration dictionary.
                - PAIR_MODE ("successive" or "all")
                - POINTCLOUD_DOWNSAMPLING (Generic)
        """
        self.raster_map = raster
        self.flights = flights
        self.flight_files = flight_files
        self.x_mesh, self.y_mesh = raster_mesh

        # Modes 
        self.pair_mode = config["PAIR_MODE"]
        self.scan_mode = config["SCAN_MODE"]

        # Generic params
        self.pc_downsample = config.get("POINTCLOUD_DOWNSAMPLING", 1)

        # Masks
        self.superpos_masks = []
        self.observed_masks = []
        self.superpos_flight_pairs = []

        # MLS extras
        self.tmin_grids = []
        self.tmax_grids = []
        self.superpos_time_windows = []

        self.get_superpos_zones()

    def get_superpos_zones(self) -> None:
        """
        Computes all footprint masks and overlaps between flights.
        """
        flight_ids = []
        flight_key_order = list(self.flights.keys())

        tasks = [
            (
                flight_key,
                self.flight_files,
                self.raster_map,
                self.x_mesh,
                self.y_mesh,
                self.pc_downsample,
                self.scan_mode, 
            )
            for flight_key in flight_key_order
        ]

        results = []
        for task in tqdm(tasks, total=len(tasks), desc="Génération des footprints"):
            results.append(get_footprint_wrapper(task))

        result_dict = dict(results)
        for flight_key in flight_key_order:
            res = result_dict[flight_key]

            if self.scan_mode == "MLS":
                mask, tmin_gps, tmax_gps = res
                self.observed_masks.append(mask)
                self.tmin_grids.append(tmin_gps)
                self.tmax_grids.append(tmax_gps)
            else:
                self.observed_masks.append(res) # No modification in ALS method
        
            flight_id = flight_key.split("_")[-1]
            flight_ids.append(flight_id)

        flight_id_to_index = {flight_id: idx for idx, flight_id in enumerate(flight_ids)}

        if self.pair_mode == "successive":
            flight_pairs = pairwise(flight_ids)
        elif self.pair_mode == "all":
            flight_pairs = combinations(flight_ids, 2)
        else:
            raise ValueError(f"Invalid mode: {self.pair_mode}. Choose 'successive' or 'all'.")

        for flight_id_1, flight_id_2 in flight_pairs:
            idx_1 = flight_id_to_index[flight_id_1]
            idx_2 = flight_id_to_index[flight_id_2]

            combined_mask = self.observed_masks[idx_1] & self.observed_masks[idx_2]
            self.superpos_masks.append(combined_mask)
            self.superpos_flight_pairs.append((flight_id_1, flight_id_2))

            
            if self.scan_mode == "MLS":
                if not np.any(combined_mask):
                    self.superpos_time_windows.append((np.nan, np.nan))
                else:
                    tmin_i = np.nanmin(self.tmin_grids[idx_1][combined_mask])
                    tmax_i = np.nanmax(self.tmax_grids[idx_1][combined_mask])

                    tmin_j = np.nanmin(self.tmin_grids[idx_2][combined_mask])
                    tmax_j = np.nanmax(self.tmax_grids[idx_2][combined_mask])

                    # one single window (intersection)
                    t0 = max(tmin_i, tmin_j)
                    t1 = min(tmax_i, tmax_j)

                    # if empty intersection -> NaNs
                    if not np.isfinite(t0) or not np.isfinite(t1) or t0 >= t1:
                        self.superpos_time_windows.append((np.nan, np.nan))
                    else:
                        self.superpos_time_windows.append((t0, t1))

        if self.scan_mode == "MLS":
            self.build_temporal_footprint()   

    def build_temporal_footprint(self):
        """
        For MLS only:
        - Re-open point clouds for each pair
        - Build point masks based on (t0,t1)
        - Build GUI raster coverage mask as union of both flights (time-filtered)
        """
        self.temporal_gui_masks = []          # list[bool raster] per pair
        self.temporal_point_windows = []      # list[(t0,t1)] per pair (same as superpos_time_windows)
        self.temporal_point_masks_meta = []   # optional: store how to rebuild / where saved

        for pair_idx, (fi, fj) in enumerate(self.superpos_flight_pairs):
            t0, t1 = self.superpos_time_windows[pair_idx]

            if not np.isfinite(t0) or not np.isfinite(t1):
                self.temporal_gui_masks.append(np.zeros_like(self.raster_map, dtype=bool))
                self.temporal_point_windows.append((np.nan, np.nan))
                self.temporal_point_masks_meta.append(None)
                continue

            file_i = self.flight_files[fi]
            file_j = self.flight_files[fj]

            # Raster masks for GUI based on time-filtered points
            mask_i = Footprint.rasterize_time_window(
                file_i, self.raster_map, self.x_mesh, self.y_mesh, self.pc_downsample, t0, t1
            )
            mask_j = Footprint.rasterize_time_window(
                file_j, self.raster_map, self.x_mesh, self.y_mesh, self.pc_downsample, t0, t1
            )

            self.temporal_gui_masks.append(mask_i | mask_j)
            self.temporal_point_windows.append((t0, t1))

            # Optional: you can store only the window; point mask computed later in LasExtractor
            self.temporal_point_masks_meta.append({"fi": fi, "fj": fj, "t0": float(t0), "t1": float(t1)})


    @staticmethod
    def get_footprint(
        flight_key: str,
        flight_files: Dict[str, str],
        raster_map: np.ndarray,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        pc_downsample: int,
        scan_mode: str = "ALS",
    ) -> Tuple[str, np.ndarray]:
        """
        Computes the observed LiDAR footprint for a single flight.
        """
        
        # Generic: rasterize point cloud occupancy
        flight_id = flight_key.split("_")[-1]
        input_file = flight_files[flight_id]

        xy, times = Footprint._read_xy_and_time(input_file, scan_mode)
        xy, times = Footprint._clean_and_downsample(xy, times, pc_downsample)

        mask = Footprint._rasterize_xy(xy, raster_map.shape, x_mesh, y_mesh)

        if scan_mode == "ALS":
            return flight_key, mask

        # MLS: time grids
        if times is None:
            raise ValueError(f"[MLS] GPS times not available for file: {input_file}")

        # compute indices again (cheap) for min/max per cell
        x0 = x_mesh[0, 0]
        y0 = y_mesh[0, 0]
        res_x = float(abs(x_mesh[0, 1] - x_mesh[0, 0]))
        res_y = float(abs(y_mesh[1, 0] - y_mesh[0, 0]))

        cols = np.floor((xy[:, 0] - x0) / res_x).astype(int)
        rows = np.floor((y0 - xy[:, 1]) / res_y).astype(int)

        valid = (
            (rows >= 0) & (rows < raster_map.shape[0]) &
            (cols >= 0) & (cols < raster_map.shape[1])
        )

        rv = rows[valid]
        cv = cols[valid]
        tv = times[valid]

        tmin_grid = np.full(raster_map.shape, np.inf, dtype=np.float64)
        tmax_grid = np.full(raster_map.shape, -np.inf, dtype=np.float64)

        np.minimum.at(tmin_grid, (rv, cv), tv)
        np.maximum.at(tmax_grid, (rv, cv), tv)

        tmin_grid[np.isinf(tmin_grid)] = np.nan
        tmax_grid[np.isinf(tmax_grid)] = np.nan

        return flight_key, (mask, tmin_grid, tmax_grid)

    @staticmethod
    def rasterize_time_window(input_file, raster_map, x_mesh, y_mesh, pc_downsample, t0, t1):
        xy, times = Footprint._read_xy_and_time(input_file, "MLS")
        xy, times = Footprint._clean_and_downsample(xy, times, pc_downsample)
        m = (times >= t0) & (times <= t1)
        return Footprint._rasterize_xy(xy[m], raster_map.shape, x_mesh, y_mesh)


    @staticmethod
    def _read_xy_and_time(input_file: str, scan_mode: str):
        """
        Returns:
        xy: (N,2) float64
        times: (N,) float64 or None
        """
        xy = None
        times = None

        f = input_file.lower()

        if f.endswith((".laz", ".las")):
            with laspy.open(input_file) as fh:
                las = fh.read()
            xy = np.column_stack((np.asarray(las.x, np.float64), np.asarray(las.y, np.float64)))

            if scan_mode == "MLS":
                if not hasattr(las, "gps_time"):
                    raise ValueError(f"[MLS] gps_time missing in file: {input_file}")
                times = np.asarray(las.gps_time, np.float64)

        elif f.endswith((".txyzs", ".txt")):
            df = pd.read_csv(input_file, sep=None, engine="python", header=None)
            xy = df.iloc[:, 1:3].astype(float).values

            if scan_mode == "MLS":
                if df.shape[1] < 5:
                    raise ValueError(
                        f"[MLS] No time column found in {input_file}. "
                        f"Need >= 5 columns or adapt the parser to your format."
                    )
                times = df.iloc[:, -1].astype(float).values

        else:
            raise ValueError(f"Unsupported file format for footprint: {input_file}")

        return xy, times


    @staticmethod
    def _clean_and_downsample(xy: np.ndarray, times, step: int):
        if step and step > 1:
            xy = xy[::step]
            if times is not None:
                times = times[::step]

        xy = np.asarray(xy, dtype=np.float64)
        xy = np.ascontiguousarray(xy)

        valid = np.isfinite(xy).all(axis=1)
        xy = xy[valid]
        if times is not None:
            times = np.asarray(times, dtype=np.float64)[valid]

        return xy, times


    @staticmethod
    def _rasterize_xy(xy: np.ndarray, raster_shape, x_mesh, y_mesh) -> np.ndarray:
        """Occupancy rasterization (bool mask) for xy points."""
        if xy.shape[0] == 0:
            return np.zeros(raster_shape, dtype=bool)

        x0 = x_mesh[0, 0]
        y0 = y_mesh[0, 0]
        res_x = float(abs(x_mesh[0, 1] - x_mesh[0, 0]))
        res_y = float(abs(y_mesh[1, 0] - y_mesh[0, 0]))

        cols = np.floor((xy[:, 0] - x0) / res_x).astype(int)
        rows = np.floor((y0 - xy[:, 1]) / res_y).astype(int)

        valid = (
            (rows >= 0) & (rows < raster_shape[0]) &
            (cols >= 0) & (cols < raster_shape[1])
        )

        mask = np.zeros(raster_shape, dtype=bool)
        mask[rows[valid], cols[valid]] = True
        return mask
