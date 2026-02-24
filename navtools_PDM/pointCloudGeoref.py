import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, Union, Optional

import numpy as np
import yaml
from lib.trajectory import *
from lib.rotations import *
from lib.loaders import *
from lib.lidar import *


# ============================================================
# CONFIG / CLI
# ============================================================

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Georeference LiDAR data using trajectory from YAML config")
    p.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    return p.parse_args()


# ============================================================
# TIME SYNC
# ============================================================

def sync_times_day_shift(t_lasvec: np.ndarray, t_trj: np.ndarray) -> np.ndarray:
    mean_diff = float(np.mean(t_trj) - np.mean(t_lasvec))
    day_shift = int(round(mean_diff / 86400.0))
    offset = day_shift * 86400.0
    print(f"Applying time offset of {offset:.0f} seconds to laser vector time.")
    return t_lasvec + offset


def apply_leapsec(t: np.ndarray, leapsec: Optional[float]) -> np.ndarray:
    if leapsec is None:
        return t
    print(f"Applying leap second correction of {float(leapsec)} seconds.")
    return t + float(leapsec)


# ============================================================
# TRAJECTORY
# ============================================================


def load_and_prepare_trajectory(cfg: Dict[str, Any]):
    trj = load_trajectory(cfg["trj"])

    trj_cfg = cfg["trj"]
    if trj_cfg["type"] == "SBET" or (
        trj_cfg["type"] == "ASCII" and trj_cfg.get("rpy_col") is not None
    ):
        print("Converting orientation info from local NED to ECEF")
        trj.estimate_q_l2e(interp=False)
        for i in range(len(trj.q)):
            trj.q[i] = quatMult(trj.q_l2e[i], trj.q[i])

    return trj


def trajectory_positions_mapping(trj, epsg_out: str = "EPSG:2056") -> np.ndarray:
    """
    Return vehicle positions as Nx3 in mapping CRS (default LV95 EPSG:2056).

    Tries in order:
      1) trj.pos / trj.xyz / trj.enu (already mapping)
      2) trj.ecef (ECEF xyz) -> project to epsg_out
      3) trj.lla (lat,lon,alt) -> project to epsg_out

    Adjust attribute names here if your Trajectory uses different ones.
    """
    # 1) Already in mapping frame?
    for attr in ("pos", "xyz", "enu"):
        if hasattr(trj, attr):
            P = np.asarray(getattr(trj, attr))
            if P.ndim == 2 and P.shape[1] >= 3:
                return P[:, :3].astype(np.float64)

    # Prepare transformers
    ecef_to_map = Transformer.from_crs("EPSG:4978", epsg_out, always_xy=True)
    lla_to_map  = Transformer.from_crs("EPSG:4979", epsg_out, always_xy=True)

    # 2) ECEF available?
    for attr in ("ecef", "xyz_ecef", "p_ecef"):
        if hasattr(trj, attr):
            E = np.asarray(getattr(trj, attr))
            if E.ndim == 2 and E.shape[1] >= 3:
                x, y, z = ecef_to_map.transform(E[:, 0], E[:, 1], E[:, 2])
                return np.column_stack((x, y, z)).astype(np.float64)

    # 3) LLA available?
    for attr in ("lla", "geodetic", "llh"):
        if hasattr(trj, attr):
            L = np.asarray(getattr(trj, attr))
            if L.ndim == 2 and L.shape[1] >= 3:
                # common ambiguity: some store [lat, lon, h] in radians/deg.
                # We'll assume degrees unless cfg says radians. If your loader uses radians, tell me.
                lat = L[:, 0]
                lon = L[:, 1]
                h   = L[:, 2]

                # If you KNOW it's radians, convert here:
                # lat = np.degrees(lat); lon = np.degrees(lon)

                x, y, z = lla_to_map.transform(lon, lat, h)  # lon,lat,h
                return np.column_stack((x, y, z)).astype(np.float64)

    # If nothing found:
    raise AttributeError(
        "Could not find trajectory positions. "
        "Your trj object does not expose pos/xyz/enu/ecef/lla with expected names. "
        "Print(dir(trj)) and add the right attribute name to trajectory_positions_mapping()."
    )

# ============================================================
# FILTER DISTANCE VEHICULE
# ============================================================

def filter_pcd_by_vehicle_distance(
    pcd: np.ndarray,
    trj,
    max_dist_m: float,
    epsg_out: str,
) -> np.ndarray:

    t_pts = pcd[:, 0]
    P = trajectory_positions_mapping(trj, epsg_out = epsg_out)
    t_trj = trj.time

    x_car = np.interp(t_pts, t_trj, P[:, 0])
    y_car = np.interp(t_pts, t_trj, P[:, 1])

    dx = pcd[:, 1] - x_car
    dy = pcd[:, 2] - y_car

    dist = np.sqrt(dx**2 + dy**2)

    mask = dist <= max_dist_m

    print(
        f"Distance filter kept {int(mask.sum())}/{len(pcd)} points "
        f"(<= {max_dist_m} m) | Duration: {float(t_pts.max() - t_pts.min()):.2f} s"
    )
    return pcd[mask]


# ============================================================
# PROCESSING
# ============================================================

def georef_one_file(cfg: Dict[str, Any], trj, path: str):

    print(f"\n[Georef] Processing file: {path}")

    # Load lasvec
    if cfg["lasvec"]["type"] == "SDC":
        lasvec = loadLasVecSDC(path)
    else:
        lasvec = loadLasVecAscii(cfg["lasvec"], cfg["limatch_output"])

    # Time corrections
    lasvec[:, 0] = sync_times_day_shift(lasvec[:, 0], trj.time)
    lasvec[:, 0] = apply_leapsec(lasvec[:, 0], cfg.get("leapsec"))

    # Georef
    pcd = georefLidar(lasvec, trj, cfg)

    # Distance filtering
    dist_cfg = cfg["distance_filtering"]

    if dist_cfg["enable"]:
        pcd = filter_pcd_by_vehicle_distance(
        pcd,
        trj,
        max_dist_m=float(dist_cfg["max_distance_m"]),
        epsg_out=dist_cfg["map_epsg"]

    )


    # Export ASCII
    if cfg["output"]["type"] == "ASCII":
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_dir = cfg["output"]["path"]
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, base_name + "_pcd.txt")

        if cfg["output"]["lasvec"] and not cfg["output"]["lasvec_to_body"]:
            arr = np.column_stack((pcd, lasvec[:, 1:4]))
            np.savetxt(out_path, arr,
                       fmt="%.9f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                       delimiter=",")
        elif cfg["output"]["lasvec"] and cfg["output"]["lasvec_to_body"]:
            np.savetxt(out_path, pcd,
                       fmt="%.9f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                       delimiter=",")
        else:
            np.savetxt(out_path, pcd,
                       fmt="%.9f, %.3f, %.3f, %.3f",
                       delimiter=",")

        print(f"Saved {out_path}")

    return pcd


def run(cfg: Dict[str, Any]):

    trj = load_and_prepare_trajectory(cfg)

    if os.path.isdir(cfg["lasvec"]["path"]):
        paths = glob.glob(os.path.join(cfg["lasvec"]["path"], "*.sdc"))
        paths.sort()
    else:
        paths = [cfg["lasvec"]["path"]]

    print(f"Processing {len(paths)} files")

    for path in paths:
        georef_one_file(cfg, trj, path)


def run_from_yaml(cfg_path: Union[str, Path]):
    cfg = load_config(cfg_path)
    return run(cfg)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)

if __name__ == "__main__":
    main()
