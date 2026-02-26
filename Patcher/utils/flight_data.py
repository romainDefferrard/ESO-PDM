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
    - Supports huge TXT by streaming (chunksize) + tqdm progress bars.
"""

import os
import fnmatch
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class FlightData:
    # Supported point cloud extensions
    ALLOWED_EXT = (".las", ".laz", ".txt", ".txyzs")

    def __init__(self, config: dict):
        """
        Initializes the FlightData object with configuration and loads flight data.

        Input:
            config (dict): Configuration dictionary.
                - PC_DIR (str): Template path to point cloud files containing '{flight_id}'
                  Example: "/path/Epalinges_{flight_id}_merged.las"
                           "/path/merged_250220_{flight_id}_HA_LR"

                Optional:
                - POINTCLOUD_DOWNSAMPLING (int): kept for compatibility (not used here)
        """
        self.pc_dir = config["PC_DIR"]
        self.pc_downsample = int(config.get("POINTCLOUD_DOWNSAMPLING", 1))

        self.flights = {}        # flight_name -> metadata placeholder dict
        self.bounds = []         # global bounds [minx, maxx, miny, maxy]
        self.flight_files = {}   # flight_id -> full path

        self.load_flights()

    def load_flights(self) -> None:
        """
        Builds flight IDs based on file order in the PC directory and computes global bounds.

        Uses PC_DIR template with '{flight_id}'.

        Examples:
            PC_DIR: ".../Epalinges_{flight_id}_merged.las"
                -> pattern matches ".../Epalinges_*_merged.las"

            PC_DIR: ".../Epalinges_{flight_id}_merged"
                -> pattern matches ".../Epalinges_*_merged.(las|laz|txt|txyzs)" (filtered by ext)
        """
        directory = os.path.dirname(self.pc_dir)
        template = os.path.basename(self.pc_dir)

        if "{flight_id}" not in template:
            raise ValueError("PC_DIR must contain '{flight_id}' (e.g. Epalinges_{flight_id}_merged.las).")

        prefix, suffix = template.split("{flight_id}", 1)

        # If suffix contains an explicit extension, we only accept that extension.
        suffix_root, suffix_ext = os.path.splitext(suffix)
        has_explicit_ext = (suffix_ext.lower() in self.ALLOWED_EXT)

        # Build fnmatch pattern
        if has_explicit_ext:
            pattern = f"{prefix}*{suffix}"
        else:
            pattern = f"{prefix}*{suffix}*"

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"PC_DIR directory does not exist: {directory}")

        files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
        files.sort()

        bounds: Optional[list[float]] = None

        for filename in tqdm(files, desc="[FlightData] Scanning flights", unit="file"):
            # extension filtering
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in self.ALLOWED_EXT:
                continue
            if has_explicit_ext and ext != suffix_ext.lower():
                continue

            flight_id = self._extract_flight_id(filename, prefix, suffix, has_explicit_ext)
            if flight_id is None:
                continue

            flight_name = f"Flight_{flight_id}"
            full_path = os.path.join(directory, filename)

            self.flights[flight_name] = {}
            self.flight_files[flight_id] = full_path

            min_x, max_x, min_y, max_y = self._read_bounds(full_path)
            if min_x is None:
                continue

            if bounds is None:
                bounds = [min_x, max_x, min_y, max_y]
            else:
                bounds[0] = min(bounds[0], min_x)
                bounds[1] = max(bounds[1], max_x)
                bounds[2] = min(bounds[2], min_y)
                bounds[3] = max(bounds[3], max_y)

        if bounds is None:
            raise ValueError(
                f"No point clouds matched PC_DIR template.\n"
                f"PC_DIR={self.pc_dir}\n"
                f"directory={directory}, prefix='{prefix}', suffix='{suffix}'"
            )

        self.bounds = bounds

    @staticmethod
    def _extract_flight_id(filename: str, prefix: str, suffix: str, has_explicit_ext: bool) -> Optional[str]:
        """
        Extract flight_id from filename given prefix and suffix around {flight_id}.

        If PC_DIR had explicit extension in suffix, suffix must match end of filename.
        Otherwise, we strip the file extension and match against suffix without extension.
        """
        if not filename.startswith(prefix):
            return None

        if has_explicit_ext:
            if not filename.endswith(suffix):
                return None
            core = filename[len(prefix): len(filename) - len(suffix)]
            return core if core else None

        base, _ext = os.path.splitext(filename)
        if suffix and not base.endswith(suffix):
            return None

        core = base[len(prefix):]
        if suffix:
            core = core[: len(core) - len(suffix)]

        return core if core else None

    @staticmethod
    def _read_bounds(input_file: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Return (min_x, max_x, min_y, max_y).

        - LAS/LAZ: uses header mins/maxs (fast, no full read).
        - TXT/TXYZS: streams X,Y in chunks + tqdm bytes progress.
          Expected columns for TXT: t, x, y, z, ... (x=col1, y=col2), comma-separated.
        """
        f = input_file.lower()

        # LAS/LAZ: read header bounds
        if f.endswith((".laz", ".las")):
            import laspy
            with laspy.open(input_file) as fh:
                hdr = fh.header
                return float(hdr.mins[0]), float(hdr.maxs[0]), float(hdr.mins[1]), float(hdr.maxs[1])

        # TXT/TXYZS: streaming bounds
        if f.endswith((".txyzs", ".txt")):
            
            min_x = min_y = np.inf
            max_x = max_y = -np.inf

            file_size = os.path.getsize(input_file)
            chunksize = 2_000_000  # tune if needed

            # Use a file handle so we can track exact bytes consumed via tell()
            with open(input_file, "rb") as fb:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"[bounds] {os.path.basename(input_file)}",
                ) as pbar:
                    for chunk in pd.read_csv(
                        fb,
                        header=None,
                        delimiter=",",     # your files are comma-separated
                        usecols=[1, 2],     # x, y
                        dtype=np.float64,
                        chunksize=chunksize,
                    ):
                        # exact bytes consumed since last update
                        pbar.update(fb.tell() - pbar.n)

                        x = chunk.iloc[:, 0].to_numpy()
                        y = chunk.iloc[:, 1].to_numpy()
                        if x.size == 0:
                            continue

                        cxmin = np.nanmin(x); cxmax = np.nanmax(x)
                        cymin = np.nanmin(y); cymax = np.nanmax(y)

                        if np.isfinite(cxmin): min_x = min(min_x, cxmin)
                        if np.isfinite(cxmax): max_x = max(max_x, cxmax)
                        if np.isfinite(cymin): min_y = min(min_y, cymin)
                        if np.isfinite(cymax): max_y = max(max_y, cymax)

            if not np.isfinite([min_x, max_x, min_y, max_y]).all():
                return None, None, None, None

            return float(min_x), float(max_x), float(min_y), float(max_y)

        return None, None, None, None