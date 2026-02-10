"""
Filename: flight_data.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This script handles the extraction and organization of flight data directly
    from LAS/LAZ or ASCII file ordering (no trajectory required).

    Main features:
    - Use file order to define flight lines.
    - Compute spatial bounds of all flight data from point clouds.
"""
import pandas as pd
import os
import fnmatch


class FlightData:
    def __init__(self, config):
        """
        Initializes the FlightData object with configuration and loads flight data.

        Input:
            config (dict): Configuration dictionary.
                - LAS_DIR (str): Template path to LAS/LAZ or ASCII files.

        Output:
            - self.flights (dict[str, pd.DataFrame]): Flight data per ID.
            - self.bounds (list[float]): Combined E/N bounds across all flights.
        """
        self.las_dir = config["LAS_DIR"]
        self.pc_downsample = int(config.get("POINTCLOUD_DOWNSAMPLING", 1))

        self.flights = {}  # Store extracted flights
        self.bounds = []  # Store E/N - min/max coordinates for each flight
        self.flight_files = {}  # Mapping flight_id -> file path

        
        self.load_flights_from_las_order()
     
    def load_flights_from_las_order(self) -> None:
        """
        Builds flight IDs based on file order in the LAS directory and computes bounds from points.

        Input:
            None

        Output:
            None, but populates self.flights, self.flight_files, and self.bounds.
        """
        directory = os.path.dirname(self.las_dir)
        pattern = os.path.basename(self.las_dir).replace("{flight_id}", "*")
        prefix, suffix = pattern.split("*", 1)
        files = [
            f for f in os.listdir(directory)
            if fnmatch.fnmatch(f, pattern)
        ]
        files.sort()

        bounds = None
        for filename in files:
            flight_id = self._extract_flight_id(filename, prefix, suffix)
            if flight_id is None:
                continue
            flight_name = f"Flight_{flight_id}"
            self.flights[flight_name] = {}
            self.flight_files[flight_id] = os.path.join(directory, filename)
            min_x, max_x, min_y, max_y = self._read_bounds(self.flight_files[flight_id])
            
            if min_x is None:
                continue
            if bounds is None:
                bounds = [min_x, max_x, min_y, max_y]
            else:
                
                bounds[0] = min(bounds[0], min_x)
                bounds[1] = max(bounds[1], max_x)
                bounds[2] = min(bounds[2], min_y)
                bounds[3] = max(bounds[3], max_y)
        
        if bounds is not None:
            self.bounds = bounds

    def _extract_flight_id(self, filename: str, prefix: str, suffix: str):
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            return None
        return filename[len(prefix) : len(filename) - len(suffix)]


    def _read_bounds(self, input_file: str):
        """
        Return (min_x, max_x, min_y, max_y) without loading full point cloud if possible.
        """
        if input_file.endswith(".laz") or input_file.endswith(".las"):
            import laspy
            with laspy.open(input_file) as fh:
                hdr = fh.header
                return hdr.mins[0], hdr.maxs[0], hdr.mins[1], hdr.maxs[1]
        if input_file.endswith(".TXYZS") or input_file.endswith(".txt"):
            df = pd.read_csv(input_file, sep=None, engine="python", header=None, usecols=[1, 2])
            df = df.astype(float)
            return df.iloc[:, 0].min(), df.iloc[:, 0].max(), df.iloc[:, 1].min(), df.iloc[:, 1].max()
        return None, None, None, None

    
