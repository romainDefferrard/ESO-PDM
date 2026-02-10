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

        self.pair_mode = config["PAIR_MODE"]

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
                flight_key,
                self.flight_files,
                self.raster_map,
                self.x_mesh,
                self.y_mesh,
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
        flight_key: str,
        flight_files: Dict[str, str],
        raster_map: np.ndarray,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        pc_downsample: int,
    ) -> Tuple[str, np.ndarray]:
        """
        Computes the observed LiDAR footprint for a single flight.
        """
        # Generic: rasterize point cloud occupancy
        flight_id = flight_key.split("_")[-1]
        input_file = flight_files[flight_id]

        coords = None
        if input_file.endswith(".laz") or input_file.endswith(".las"):
            with laspy.open(input_file) as fh:
                las = fh.read()
                coords = np.column_stack((
                    np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                ))
        elif input_file.endswith(".TXYZS") or input_file.endswith(".txt"):
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
