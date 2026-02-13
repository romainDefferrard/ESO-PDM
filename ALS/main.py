"""
Filename: main.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    Main pipeline for Airborne Laser Scanning (ALS) data processing.
    Handles data loading, footprint generation, patch creation, GUI interaction,
    and LAS file extraction based on patch overlap with estimated footprints.
    
    This pipeline uses a TimerLogger utility to benchmark steps in the pipeline.

"""

import argparse
import os
import sys
import subprocess
import shlex
from typing import List, Tuple
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union
from PyQt6.QtWidgets import QApplication
from tqdm import tqdm
import logging
import copy 

from utils.flight_data import FlightData
from utils.footprint_generator import Footprint
from utils.gui import GUIMainWindow
from utils.gui_mls import GUIMainWindowMLS
from utils.las_extractor import LasExtractor
from utils.patch_generator import PatchGenerator
from utils.patch_model import Patch
from utils.raster_loader import RasterLoader
from utils.timer_logger import TimerLogger

import warnings
warnings.filterwarnings("ignore")


class PatcherPipeline():
    def __init__(self, config_path: str):
        """
        Initialize the ALS pipeline using the provided YAML configuration.

        Input:
            config_path (str): Path to the YAML configuration file

        Output:
            None
        """
        self.config = yaml.safe_load(open(config_path, "r"))
        self.las_dir = self.config["LAS_DIR"]
        self.timer = TimerLogger()
    
    def load_data(self) -> None:
        """
        Load flight metadata and build a generic raster grid using the config file.

        Output:
            None
        """
        self.timer.start("Flight & grid loading")
        fd = FlightData(self.config)
        rl = RasterLoader(self.config, flight_bounds=fd.bounds)
        self.raster = rl.raster
        self.raster_mesh = (rl.x_mesh, rl.y_mesh)
        self.flight_data = fd  
        self.timer.stop("Flight & grid loading")


    def generate_footprint(self) -> None:
        """
        Generate the estimated LiDAR footprints for each flight.

        Output:
            None
        """
        self.timer.start("Footprint generation")
        self.footprint = Footprint(
            raster=self.raster,
            raster_mesh=self.raster_mesh,
            flights=self.flight_data.flights,
            flight_files=self.flight_data.flight_files,
            config=self.config
        )
        self.timer.stop("Footprint generation")


    def generate_patches(self):
        """
        Generate rectangular patches from overlapping zones in LiDAR footprints.

        Output:
            None
        """
        self.timer.start("Patch generation")
        self.pg = PatchGenerator(superpos_zones=self.footprint.superpos_masks, raster_mesh=self.raster_mesh, patch_params=self.config["PATCH_DIMS"])
        self.timer.stop("Patch generation")

    def launch_gui(self):
        """
        Launch the PyQt6 GUI for patch visualization and selection.

        Output:
            None
        """

        mode = self.config["SCAN_MODE"]
        app = QApplication(sys.argv)
        
        if mode == "ALS":
            window = GUIMainWindow(
                superpositions=  self.footprint.superpos_masks,
                patches=         self.pg.patches_list,
                centerlines=     self.pg.centerlines_list,
                patch_params=    self.config["PATCH_DIMS"],
                raster_mesh=     self.raster_mesh,
                raster=          self.raster,
                contours=        self.pg.contours_list,
                extraction_state=False,
                flight_pairs=    self.footprint.superpos_flight_pairs,
                output_dir=      self.config["OUTPUT_DIR"],

            )
        elif mode == "MLS":
            crs = self.config.get("CRS", None)
            window = GUIMainWindowMLS(
                superpositions=  self.footprint.temporal_gui_masks,
                raster_mesh=     self.raster_mesh,
                raster=          self.raster,
                time_windows=    self.footprint.superpos_time_windows,
                extraction_state=False,
                flight_pairs=    self.footprint.superpos_flight_pairs,
                output_dir=      self.config["OUTPUT_DIR"],
                crs=crs,
            )   
        window.show()
        app.exec()
        
        self.extraction_state = window.control_panel.extraction_state
        self.output_dir = window.control_panel.output_dir 
        self.execute_limatch = window.control_panel.execute_limatch

        if mode == "ALS":
            self.patch_list = window.control_panel.new_patches_instance

    def extract_als(self) -> None:
        """
        Extract points for each relevant patch across all flights based on the extraction mode.

        Output:
            None
        """
        self.timer.start("ALS Patch extraction (all flights)")

        if self.config["EXTRACTION_MODE"] != "independent":
            raise ValueError(
                f"Unsupported EXTRACTION_MODE: {self.config['EXTRACTION_MODE']}. "
                "Only 'independent' is supported in this minimal pipeline."
            )

        for (flight_i, flight_j), patch_group in zip(self.footprint.superpos_flight_pairs, self.patch_list):
            for flight_id in [flight_i, flight_j]:
                pair_dir = f"{self.output_dir}/Flights_{flight_i}_{flight_j}"
                os.makedirs(pair_dir, exist_ok=True)
                self.process_flight(
                    flight_id, patch_group, self.config["LAS_DIR"], self.output_dir,
                    pair_dir, self.footprint.superpos_flight_pairs, self.pg.contours_list
                )

        self.timer.stop("ALS Patch extraction (all flights)")

    def extract_mls(self) -> None:
        """
        MLS extraction (temporal patch):
        For each spatial overlap, compute time window, then extract ALL points
        from each cloud within that time window.
        """
        self.timer.start("MLS Patch extraction (all pairs)")

        
        for k, (flight_i, flight_j) in enumerate(self.footprint.superpos_flight_pairs):
            t0, t1 = self.footprint.superpos_time_windows[k]  # must be (t0, t1)

            # skip invalid windows
            if not np.isfinite(t0) or not np.isfinite(t1) or t0 >= t1:
                logging.warning(f"[MLS] Invalid time window for pair {flight_i}-{flight_j}: {t0}, {t1}")
                continue

            pair_dir = os.path.join(self.output_dir, f"Flights_{flight_i}_{flight_j}")
            os.makedirs(pair_dir, exist_ok=True)

            for flight_id, other_id in ((flight_i, flight_j), (flight_j, flight_i)):
                input_file = self.flight_data.flight_files[flight_id]
                ext = self._output_extension(input_file)

                extractor = LasExtractor(self.config, input_file, patches=[])
                if not extractor.read_point_cloud():
                    continue

                out_path = os.path.join(
                    pair_dir,
                    f"Patch_from_scan_{flight_id}_with_{other_id}.{ext}"
                )

                ok = extractor.extract_time_window(t0, t1, out_path)
                if not ok:
                    logging.warning(
                        f"[MLS] No points extracted for flight {flight_id} in pair {flight_i}-{flight_j}"
                    )
        self.timer.stop("MLS Patch extraction (all pairs)")


    def process_flight(self, flight_id: str, flight_patch: List[Patch], LAS_DIR: str, OUTPUT_DIR: str,
                        pair_dir: str, superpos_pairs: List[Tuple[str, str]], contours_all: List[np.ndarray]) -> None:
        """
        Processes a single flight by filtering relevant patches, loading the point cloud,
        and applying the configured extraction method. Relevant patches are defined as those whose polygon intersects
        the union polygon of the flight's observed footprint (i.e., merged contours from overlapping flight pairs 
        involving this flight ID).

        extractor = LasExtractor(self.config, input_file, relevant_patches)
        if extractor.read_point_cloud():
        Inputs:
            flight_id (str): Flight identifier.
            flight_patch (List[Patch]): Patches associated with the current flight.
            LAS_DIR (str): Format string to locate LAS file from flight ID.
            OUTPUT_DIR (str): Base output directory.
            pair_dir (str): Output directory for this specific flight pair.
            superpos_pairs (List[Tuple[str, str]]): All overlapping flight pairs.
            contours_all (List[np.ndarray]): Corresponding list of contours.

        Output:
            None
        """
        
        _ = LAS_DIR
        input_file = self.flight_data.flight_files[flight_id]
        flight_polygon = self.get_flight_union_contour(flight_id, superpos_pairs, contours_all)

        relevant_patches = [patch for patch in flight_patch if patch.shapely_polygon.intersects(flight_polygon)]

        extractor = LasExtractor(self.config, input_file, relevant_patches)
        if extractor.read_point_cloud():
            extractor.process_all_patches(relevant_patches, OUTPUT_DIR, flight_id, pair_dir)
            

    def get_flight_union_contour(self, flight_id: str, superpos_pairs: List[Tuple[str, str]], contours_list: List[np.ndarray]) -> Polygon:
        """
        Return the union of all contours (polygons) involving the given flight ID.

        Input:
            flight_id (str): Flight identifier.
            superpos_pairs (List[Tuple[str, str]]): Pairs of overlapping flights.
            contours_list (List[np.ndarray]): Contours corresponding to each pair.

        Output:
            Polygon: Merged polygon of all overlapping contours for the flight.
        """
        polygons = []
        for i, (f1, f2) in enumerate(superpos_pairs):
            if flight_id in (f1, f2):
                contour = contours_list[i]
                if isinstance(contour, np.ndarray):
                    coords = contour.reshape(-1, 2)
                    if len(coords) >= 3:
                        polygons.append(Polygon(coords))
                elif isinstance(contour, Polygon):
                    polygons.append(contour)
        return unary_union(polygons) if polygons else Polygon()

    def run(self):
        """
        Execute the full ALS pipeline from loading data to extraction.

        Output:
            None
        """
        mode = self.config["SCAN_MODE"]

        self.timer.start(f"{mode} total time")

        self.load_data()
        self.generate_footprint()

        if mode == "ALS":
            self.generate_patches()

        self.launch_gui()

        if self.extraction_state:
            if mode == "ALS":
                self.extract_als()
            elif mode == "MLS":
                self.extract_mls()
            if getattr(self, "execute_limatch", False):
                self.run_limatch()
        
        self.timer.stop(f"{mode} total time")
        self.timer.summary()

    def run_limatch(self) -> None:
        from submodules.limatch.main import match_clouds

        mode = self.config["SCAN_MODE"]

        # Load LiMatch config
        limatch_cfg_path = self.config["LIMATCH_CFG"]
        with open(limatch_cfg_path, "r") as f:
            base_cfg = yaml.safe_load(f)

        base_out = base_cfg["prj_folder"] # original base output folder of LiMatch pipeline
        prj_name = self.config.get("PRJ_NAME", "PROJECT")

        def single_run(pair_dir: str, tag: str, c1: str, c2: str) -> None:
            if not (os.path.exists(c1) and os.path.exists(c2)):
                logging.warning(f"LiMatch skipped (missing files): {c1} / {c2}")
                return

            pair_name = os.path.basename(pair_dir)
            out_dir = os.path.join(base_out, prj_name, pair_name, f"{tag}/")
            os.makedirs(out_dir, exist_ok=True)

            cfg = copy.deepcopy(base_cfg)
            cfg["prj_folder"] = out_dir

            print("Running match_clouds:", c1, c2)
            match_clouds(c1, c2, cfg)

        if mode == "ALS":
            for (flight_i, flight_j), patch_group in zip(
                self.footprint.superpos_flight_pairs, self.patch_list
            ):
                pair_dir = os.path.join(self.output_dir, f"Flights_{flight_i}_{flight_j}")
                ext_i = self._output_extension(self.flight_data.flight_files[flight_i])
                ext_j = self._output_extension(self.flight_data.flight_files[flight_j])

                for patch in patch_group:
                    c1 = os.path.join(pair_dir, f"patch_{patch.id}_flight_{flight_i}.{ext_i}")
                    c2 = os.path.join(pair_dir, f"patch_{patch.id}_flight_{flight_j}.{ext_j}")
                    single_run(pair_dir, f"patch_{patch.id}", c1, c2)

        # -------- MLS --------
        elif mode == "MLS":
            for (flight_i, flight_j) in self.footprint.superpos_flight_pairs:
                pair_dir = os.path.join(self.output_dir, f"Flights_{flight_i}_{flight_j}")

                ext_i = self._output_extension(self.flight_data.flight_files[flight_i])
                ext_j = self._output_extension(self.flight_data.flight_files[flight_j])

                c1 = os.path.join(pair_dir, f"Patch_from_scan_{flight_i}_with_{flight_j}.{ext_i}")
                c2 = os.path.join(pair_dir, f"Patch_from_scan_{flight_j}_with_{flight_i}.{ext_j}")

                single_run(pair_dir, "mls_overlap", c1, c2)

        else:
            raise ValueError(f"Unknown SCAN_MODE: {mode}")

    

    @staticmethod
    def _output_extension(input_file: str) -> str:
        """
        Return the output patch extension based on the input file type.
        """
        if input_file.lower().endswith((".laz", ".las")):
            return "laz"
        if input_file.lower().endswith((".txyzs", ".txt")):
            return "TXYZS"
        raise ValueError(f"Unsupported input file format: {input_file}")

            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    pipeline = PatcherPipeline(config_path=args.yml)
    pipeline.run()
