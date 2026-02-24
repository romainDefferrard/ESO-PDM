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

    ALLOWED_EXT = (".las", ".laz", ".txt", ".txyzs")

    def __init__(self, config):
        """
        Initializes the FlightData object with configuration and loads flight data.

        Input:
            config (dict): Configuration dictionary.
                - PC_DIR (str): Template path to point cloud (LAS/LAZ or ASCII) files.

        Output:
            - self.flights (dict[str, pd.DataFrame]): Flight data per ID.
            - self.bounds (list[float]): Combined E/N bounds across all flights.
        """
        self.pc_dir = config["PC_DIR"]
        self.pc_downsample = int(config.get("POINTCLOUD_DOWNSAMPLING", 1))

        self.flights = {}  # Store extracted flights
        self.bounds = []  # Store E/N - min/max coordinates for each flight
        self.flight_files = {}  # Mapping flight_id -> file path

        
        self.load_flights()
    
    def load_flights(self) -> None: 
        directory = os.path.dirname(self.pc_dir)
        template = os.path.basename(self.pc_dir)

        if "{flight_id}" not in template:
            raise ValueError("PC_DIR must contain '{flight_id}'. The project works following this template")
       
        prefix, suffix = template.split("{flight_id}", 1)
        suffix_root, suffix_ext = os.path.splitext(suffix)
        has_explicit_ext = (suffix_ext.lower() in self.ALLOWED_EXT)

        # Build the fnmatch pattern
        if has_explicit_ext:
            # Example suffix = "_merged.las" -> match exactly .las
            pattern = f"{prefix}*{suffix}"
        else:
            # Example suffix = "_merged" -> match any supported ext
            # We'll filter by ext in code
            pattern = f"{prefix}*{suffix}*"

        files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
        files.sort()

        bounds = None

        for filename in files:
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
import os
import fnmatch
import pandas as pd


class FlightData:
    # Supported point cloud extensions
    ALLOWED_EXT = (".las", ".laz", ".txt", ".txyzs")

    def __init__(self, config):
        """
        Initializes the FlightData object with configuration and loads flight data.

        Input:
            config (dict): Configuration dictionary.
                - PC_DIR (str): Template path to point cloud (LAS/LAZ or ASCII) files.
                  Example: "./data/SAM_Epalinges/Epalinges_{flight_id}_merged.las"
                  or       "./data/SAM_Epalinges/Epalinges_{flight_id}_merged"

        Output:
            - self.flights (dict[str, dict]): Flight data per ID (kept as dict for now).
            - self.bounds (list[float]): Combined E/N bounds across all flights.
            - self.flight_files (dict[str, str]): Mapping flight_id -> file path
        """
        self.pc_dir = config["PC_DIR"]
        self.pc_downsample = int(config.get("POINTCLOUD_DOWNSAMPLING", 1))

        self.flights = {}        # Store extracted flights (metadata placeholder)
        self.bounds = []         # Store global bounds [minx, maxx, miny, maxy]
        self.flight_files = {}   # Mapping flight_id -> file path

        self.load_flights()

    def load_flights(self) -> None:
        """
        Builds flight IDs based on file order in the PC directory and computes bounds from files.

        Uses PC_DIR template with '{flight_id}'.

        Examples:
            PC_DIR: ".../Epalinges_{flight_id}_merged.las"
                -> pattern matches ".../Epalinges_*_merged.las"

            PC_DIR: ".../Epalinges_{flight_id}_merged"
                -> pattern matches ".../Epalinges_*_merged.(las|laz|txt|txyzs)"
        """
        directory = os.path.dirname(self.pc_dir)
        template = os.path.basename(self.pc_dir)

        if "{flight_id}" not in template:
            raise ValueError("PC_DIR must contain '{flight_id}' (e.g. Epalinges_{flight_id}_merged.las).")

        prefix, suffix = template.split("{flight_id}", 1)

        # If suffix already contains an extension, we keep it (single-format matching).
        # Otherwise, we accept any ALLOWED_EXT.
        suffix_root, suffix_ext = os.path.splitext(suffix)
        has_explicit_ext = (suffix_ext.lower() in self.ALLOWED_EXT)

        # Build the fnmatch pattern
        if has_explicit_ext:
            # Example suffix = "_merged.las" -> match exactly .las
            pattern = f"{prefix}*{suffix}"
        else:
            # Example suffix = "_merged" -> match any supported ext
            # We'll filter by ext in code
            pattern = f"{prefix}*{suffix}*"

        files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
        files.sort()

        bounds = None

        for filename in files:
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
    def _extract_flight_id(filename: str, prefix: str, suffix: str, has_explicit_ext: bool):
        """
        Extract flight_id from filename given prefix and suffix around {flight_id}.

        If PC_DIR had explicit extension in suffix, suffix must match exactly.
        Otherwise, we strip the file extension and match against suffix without extension.
        """
        if not filename.startswith(prefix):
            return None

        if has_explicit_ext:
            # suffix includes extension, must match end of filename
            if not filename.endswith(suffix):
                return None
            return filename[len(prefix) : len(filename) - len(suffix)]

        # No explicit ext: remove ext before matching suffix
        base, _ext = os.path.splitext(filename)
        if suffix and not base.endswith(suffix):
            return None

        core = base[len(prefix):]
        if suffix:
            core = core[: len(core) - len(suffix)]

        return core if core else None

    @staticmethod
    def _read_bounds(input_file: str):
        """
        Return (min_x, max_x, min_y, max_y).
        LAS/LAZ uses header mins/maxs (fast).
        TXT/TXYZS reads cols [1,2] (X,Y) fully (could be optimized if needed).
        """
        f = input_file.lower()

        if f.endswith((".laz", ".las")):
            import laspy
            with laspy.open(input_file) as fh:
                hdr = fh.header
                return hdr.mins[0], hdr.maxs[0], hdr.mins[1], hdr.maxs[1]

        if f.endswith((".txyzs", ".txt")):
            df = pd.read_csv(input_file, sep=None, engine="python", header=None, usecols=[1, 2])
            df = df.astype(float)
            return df.iloc[:, 0].min(), df.iloc[:, 0].max(), df.iloc[:, 1].min(), df.iloc[:, 1].max()

        return None, None, None, None


    
