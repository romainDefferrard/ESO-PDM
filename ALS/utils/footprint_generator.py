"""
Filename: footprint_generator.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the Footprint class, which computes observed raster zones for a set of flights.
    Two modes are supported:
        - "SwissDTM": uses trajectory + DTM + scan geometry (legacy method)
        - "Generic": uses point cloud occupancy on a raster grid

    The main outputs are:
        - self.observed_masks: List[np.ndarray] of boolean masks (one per flight).
        - self.superpos_masks: List[np.ndarray] of boolean masks (one per flight pair).
        - self.superpos_flight_pairs: List[Tuple[str, str]] flight ID pairs used in superposition analysis.
"""
import numpy as np
from itertools import combinations, pairwise
from typing import Dict, Tuple
from tqdm import tqdm
from shapely.vectorized import contains
from shapely.geometry import Polygon


def get_footprint_wrapper(args):
    return Footprint.get_footprint(*args)


def downsample_points(coords: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return coords
    return coords[::step]


def build_polygon_from_masks(left_mask, right_mask, x_mesh, y_mesh):
    def extract_coords(mask):
        y_idx, x_idx = np.where(mask)
        x = x_mesh[y_idx, x_idx]
        y = y_mesh[y_idx, x_idx]
        return np.column_stack((x, y))

    left_coords = extract_coords(left_mask)
    right_coords = extract_coords(right_mask)

    left_sorted = left_coords[np.argsort(left_coords[:, 0])]
    right_sorted = right_coords[np.argsort(right_coords[:, 0])]

    polygon_coords = np.concatenate([left_sorted, right_sorted[::-1]])
    polygon = Polygon(polygon_coords)
    return polygon


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
            raster (np.ndarray): 2D array (DTM for SwissDTM, blank grid for Generic).
            raster_mesh (tuple of np.ndarray): x and y coordinate grids [m].
            flights (dict[str, dict]): Dictionary of flight trajectory data.
            flight_files (dict[str, str]): Mapping of flight_id -> file path (Generic).
            config (dict): Configuration dictionary.
                - FOOTPRINT_MODE ("Generic" or "SwissDTM")
                - PAIR_MODE ("successive" or "all")
                - LIDAR_* and FLIGHT_DOWNSAMPLING (SwissDTM)
                - POINTCLOUD_DOWNSAMPLING (Generic)
        """
        self.raster_map = raster
        self.flights = flights
        self.flight_files = flight_files
        self.x_mesh, self.y_mesh = raster_mesh

        self.mode = config.get("FOOTPRINT_MODE", "Generic")
        self.mode = "SwissDTM" if self.mode == "SwissDTM" else "Generic"
        self.pair_mode = config["PAIR_MODE"]

        # SwissDTM params
        self.lidar_scan_mode = config.get("LIDAR_SCAN_MODE")
        self.lidar_tilt_angle = config.get("LIDAR_TILT_ANGLE")
        self.lidar_fov = config.get("LIDAR_FOV")
        self.sampling_interval = config.get("FLIGHT_DOWNSAMPLING")

        # Generic params
        self.pc_downsample = config.get("POINTCLOUD_DOWNSAMPLING", 1)

        # Masks
        self.superpos_masks = []
        self.observed_masks = []
        self.superpos_flight_pairs = []

        self.get_superpos_zones()

    def get_superpos_zones(self) -> None:
        """
        Computes all footprint masks and overlaps between flights.
        """
        flight_ids = []
        flight_key_order = list(self.flights.keys())

        tasks = [
            (
                self.mode,
                flight_key,
                self.flights[flight_key],
                self.flight_files,
                self.raster_map,
                self.x_mesh,
                self.y_mesh,
                self.lidar_scan_mode,
                self.lidar_tilt_angle,
                self.lidar_fov,
                self.sampling_interval,
                self.pc_downsample,
            )
            for flight_key in flight_key_order
        ]

        results = []
        for task in tqdm(tasks, total=len(tasks), desc="Génération des footprints"):
            results.append(get_footprint_wrapper(task))

        result_dict = dict(results)
        for flight_key in flight_key_order:
            self.observed_masks.append(result_dict[flight_key])
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

    @staticmethod
    def get_footprint(
        mode: str,
        flight_key: str,
        flight_data: dict,
        flight_files: Dict[str, str],
        raster_map: np.ndarray,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        lidar_scan_mode: str,
        lidar_tilt_angle: float,
        lidar_fov: float,
        sampling_interval: int,
        pc_downsample: int,
    ) -> Tuple[str, np.ndarray]:
        """
        Computes the observed LiDAR footprint for a single flight.
        """
        if mode == "SwissDTM":
            half_fov_rad = np.radians(lidar_fov / 2)
            E = flight_data["lon"][::sampling_interval]
            N = flight_data["lat"][::sampling_interval]
            A = flight_data["alt"][::sampling_interval]

            def get_scan_angles(trajectory_angle, mode, tilt_rad):
                if mode == "left":
                    return trajectory_angle + np.pi / 2 + tilt_rad, trajectory_angle - np.pi / 2 + tilt_rad
                if mode == "right":
                    return trajectory_angle + np.pi / 2 - tilt_rad, trajectory_angle - np.pi / 2 - tilt_rad
                return trajectory_angle + np.pi / 2, trajectory_angle - np.pi / 2

            def angle_diff(a, b):
                return (a - b + np.pi) % (2 * np.pi) - np.pi

            left_mask_total = np.zeros_like(raster_map, dtype=bool)
            right_mask_total = np.zeros_like(raster_map, dtype=bool)

            for e, n, alt in zip(E, N, A):
                trajectory_angle = np.arctan2(N.iloc[-1] - N.iloc[0], E.iloc[-1] - E.iloc[0])
                scanning_angle_1, scanning_angle_2 = get_scan_angles(
                    trajectory_angle, lidar_scan_mode, np.radians(lidar_tilt_angle)
                )

                angle_to_grid = np.arctan2(y_mesh - n, x_mesh - e)
                horizontal_distances = np.sqrt((x_mesh - e) ** 2 + (y_mesh - n) ** 2)
                vertical_distances = alt - raster_map
                line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)
                fov_mask = np.abs(line_of_sight_angles) <= half_fov_rad

                tolerance = np.deg2rad(5)

                for scan_angle in [scanning_angle_1, scanning_angle_2]:
                    direction_mask = np.abs(angle_diff(angle_to_grid, scan_angle)) <= tolerance
                    direction_visible_mask = direction_mask & fov_mask

                    if np.any(direction_visible_mask):
                        idx = np.where(direction_visible_mask)
                        dists = horizontal_distances[idx]
                        max_idx = np.argmax(dists)
                        max_i = idx[0][max_idx]
                        max_j = idx[1][max_idx]

                        if scan_angle == scanning_angle_2:
                            left_mask_total[max_i, max_j] = True
                        else:
                            right_mask_total[max_i, max_j] = True

            polygon = build_polygon_from_masks(left_mask_total, right_mask_total, x_mesh, y_mesh)
            final_mask = contains(polygon, x_mesh, y_mesh)
            return flight_key, final_mask

        # Generic: rasterize point cloud occupancy
        flight_id = flight_key.split("_")[-1]
        input_file = flight_files[flight_id]

        coords = None
        if input_file.endswith(".laz") or input_file.endswith(".las"):
            import laspy
            with laspy.open(input_file) as fh:
                las = fh.read()
                coords = np.column_stack((
                    np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                ))
        elif input_file.endswith(".TXYZS") or input_file.endswith(".txt"):
            import pandas as pd
            df = pd.read_csv(input_file, sep=None, engine="python", header=None)
            coords = df.iloc[:, 1:3].astype(float).values
        else:
            raise ValueError(f"Unsupported file format for footprint: {input_file}")

        coords = downsample_points(coords, pc_downsample)
        coords = np.asarray(coords, dtype=np.float64)
        coords = np.ascontiguousarray(coords)
        coords = coords[np.isfinite(coords).all(axis=1)]

        if coords.shape[0] < 1:
            return flight_key, np.zeros_like(raster_map, dtype=bool)

        x0 = x_mesh[0, 0]
        y0 = y_mesh[0, 0]
        res_x = float(abs(x_mesh[0, 1] - x_mesh[0, 0]))
        res_y = float(abs(y_mesh[1, 0] - y_mesh[0, 0]))

        cols = np.floor((coords[:, 0] - x0) / res_x).astype(int)
        rows = np.floor((y0 - coords[:, 1]) / res_y).astype(int)

        valid = (
            (rows >= 0) & (rows < raster_map.shape[0]) &
            (cols >= 0) & (cols < raster_map.shape[1])
        )

        mask = np.zeros_like(raster_map, dtype=bool)
        mask[rows[valid], cols[valid]] = True

        return flight_key, mask
