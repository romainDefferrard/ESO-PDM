import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, Union, Optional
import laspy
from laspy import ExtraBytesParams
 
import numpy as np
import yaml
from .lib.trajectory import *
from .lib.rotations import *
from .lib.loaders import *
from .lib.lidar import *
 
 
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
    t_lasvec = np.asarray(t_lasvec, dtype=np.float64)
    t_trj = np.asarray(t_trj, dtype=np.float64)
 
    mask_las = np.isfinite(t_lasvec)
    mask_trj = np.isfinite(t_trj)
 
    if not np.any(mask_las):
        raise ValueError("No finite LiDAR timestamps available for day-shift synchronization.")
    if not np.any(mask_trj):
        raise ValueError("No finite trajectory timestamps available for day-shift synchronization.")
 
    t_las_valid = t_lasvec[mask_las]
    t_trj_valid = t_trj[mask_trj]
 
    # robust central value: median instead of mean
    med_las = float(np.median(t_las_valid))
    med_trj = float(np.median(t_trj_valid))
    diff = med_trj - med_las
 
    print(f"[sync_times_day_shift] median(las)={med_las:.6f}")
    print(f"[sync_times_day_shift] median(trj)={med_trj:.6f}")
    print(f"[sync_times_day_shift] diff={diff:.6f}")
 
    if not np.isfinite(diff):
        raise ValueError(
            f"Computed time difference is not finite.\n"
            f"las valid min/max = [{np.min(t_las_valid)}, {np.max(t_las_valid)}]\n"
            f"trj valid min/max = [{np.min(t_trj_valid)}, {np.max(t_trj_valid)}]"
        )
 
    day_shift = int(round(diff / 86400.0))
    offset = day_shift * 86400.0
 
    print(f"Applying time offset of {offset:.0f} seconds to laser vector time.")
 
    out = t_lasvec.copy()
    out[mask_las] = out[mask_las] + offset
    return out
 
def apply_leapsec(t: np.ndarray, leapsec: Optional[float]) -> np.ndarray:
    if leapsec is None:
        return t
    print(f"Applying leap second correction of {float(leapsec)} seconds.")
    return t + float(leapsec)
 
 
def filter_sdc_paths_by_manifest(
    paths: list,
    manifest_path: Optional[str],
    tw_cfg: Optional[Dict[str, Any]],
) -> list:
    """
    Given a list of SDC file paths, return only those whose time range
    (read from the manifest CSV) overlaps the georef_time_window.
 
    If manifest_path is None or tw_cfg is disabled, all paths are returned as-is.
 
    Manifest CSV format (produced by build_sdc_manifest.ipynb):
        filename, t_min, t_max, n_records
 
    Overlap condition:
        file_t_max >= window_t_lo  AND  file_t_min <= window_t_hi
    """
    import pandas as pd
 
    if manifest_path is None:
        return paths
 
    if tw_cfg is None or not tw_cfg.get("enable", False):
        # Manifest present but no time window: still use manifest to validate paths,
        # but don't drop anything based on time.
        print(f"[manifest] Loaded but georef_time_window disabled — all SDC files kept.")
        return paths
 
    outage   = tw_cfg["outage"]
    margin_s = float(tw_cfg.get("margin_s", 60.0))
    t_lo     = float(outage[0]) - margin_s
    t_hi     = float(outage[0]) + float(outage[1]) + margin_s
 
    print(f"[manifest] Reading {manifest_path}")
    df = pd.read_csv(manifest_path)
 
    if not {"filename", "t_min", "t_max"}.issubset(df.columns):
        raise ValueError(
            f"Manifest {manifest_path} is missing required columns. "
            f"Expected at least: filename, t_min, t_max. Got: {list(df.columns)}"
        )
 
    # Build a set of filenames that overlap the window
    mask = (df["t_max"] >= t_lo) & (df["t_min"] <= t_hi)
    valid_names = set(df.loc[mask, "filename"].values)
 
    print(
        f"[manifest] Window [{t_lo:.3f}, {t_hi:.3f}] — "
        f"{mask.sum()}/{len(df)} files overlap"
    )
 
    filtered = [p for p in paths if os.path.basename(p) in valid_names]
 
    skipped = len(paths) - len(filtered)
    if skipped:
        print(f"[manifest] Skipped {skipped} SDC files outside the time window.")
 
    if not filtered:
        print(
            f"[manifest] WARNING: No SDC files remain after manifest filtering!\n"
            f"  Window: [{t_lo:.3f}, {t_hi:.3f}]\n"
            f"  Check that the manifest filenames match your SDC directory,\n"
            f"  or disable georef_time_window."
        )
 
    return filtered
 
 
def apply_time_window_filter(lasvec: np.ndarray, tw_cfg: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Keep only LiDAR rows whose timestamp falls within:
        [t_start - margin_s,  t_start + duration + margin_s]
 
    Config keys (from pipeline.yml  georef_time_window):
        enable   : bool
        outage   : [t_start, duration]   (GPS seconds, same format as chunk/patcher outages)
        margin_s : float  — extra seconds added on both sides
    """
    if tw_cfg is None or not tw_cfg.get("enable", False):
        return lasvec
 
    outage   = tw_cfg["outage"]          # [t_start, duration]
    margin_s = float(tw_cfg.get("margin_s", 60.0))
 
    t_start  = float(outage[0])
    duration = float(outage[1])
 
    t_lo = t_start - margin_s
    t_hi = t_start + duration + margin_s
 
    print(
        f"[time_window] keeping t in [{t_lo:.3f}, {t_hi:.3f}]  "
        f"(outage [{t_start}, {t_start + duration}] +/- {margin_s} s)"
    )
 
    mask = (lasvec[:, 0] >= t_lo) & (lasvec[:, 0] <= t_hi)
    n_before = len(lasvec)
    lasvec = lasvec[mask]
    print(f"[time_window] {mask.sum()}/{n_before} points kept")
 
    if len(lasvec) == 0:
        raise ValueError(
            f"[time_window] No LiDAR points remain after time-window filter "
            f"[{t_lo:.3f}, {t_hi:.3f}]. Check outage/margin_s or disable the filter."
        )
 
    return lasvec
 
 
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
def georef_one_file(cfg, trj, path):
    import os
    import numpy as np
    import laspy
    from laspy import ExtraBytesParams
 
    print(f"\n[Georef] Processing file: {path}")
    
    # ==================================================
    # BEGINNING KEPT AS-IS
    # ==================================================
    # Load lasvec
    if cfg["lasvec"]["type"] == "SDC":
        lasvec = loadLasVecSDC(path)
    else:
        lasvec = loadLasVecAscii(cfg["lasvec"], cfg["limatch_output"])
 
    # Remove rows with invalid or absurd LiDAR timestamps before time sync
    mask_valid_time = np.isfinite(lasvec[:, 0]) & (np.abs(lasvec[:, 0]) < 1e7)
    n_bad_time = (~mask_valid_time).sum()
 
    if n_bad_time > 0:
        print(f"[Georef] Removing {n_bad_time} rows with invalid/absurd LiDAR timestamps")
        lasvec = lasvec[mask_valid_time]
 
    if len(lasvec) == 0:
        raise ValueError(f"No valid LiDAR timestamps remain after filtering for file: {path}")
 
    # Time corrections
    lasvec[:, 0] = sync_times_day_shift(lasvec[:, 0], trj.time)
    lasvec[:, 0] = apply_leapsec(lasvec[:, 0], cfg.get("leapsec"))
    print(f"[debug] lasvec time min/max: {np.nanmin(lasvec[:,0]):.4f} / {np.nanmax(lasvec[:,0]):.4f}")
    print(f"[debug] trj time min/max:    {np.nanmin(trj.time):.4f} / {np.nanmax(trj.time):.4f}")
 
    # Time-window filter (applied after time corrections so timestamps are in GPS seconds)
    lasvec = apply_time_window_filter(lasvec, cfg.get("georef_time_window"))
    # ==================================================
    # END KEPT AS-IS
    # ==================================================
 
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = cfg["output"]["path"]
    os.makedirs(output_dir, exist_ok=True)
 
    output_type = cfg["output"]["type"]
    include_lasvec = cfg["output"]["lasvec"]
    lasvec_to_body = cfg["output"].get("lasvec_to_body", False)
    dist_cfg = cfg["distance_filtering"]
 
    subchunk_size = 10000000
    n_total = len(lasvec)
 
    # --------------------------------------------------
    # Prepare output
    # --------------------------------------------------
    if output_type == "ASCII":
        out_path = os.path.join(output_dir, base_name + "_pcd.txt")
        if os.path.exists(out_path):
            os.remove(out_path)
 
        def append_ascii(arr):
            if arr.shape[1] == 7:
                fmt = "%.9f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"
            elif arr.shape[1] == 4:
                fmt = "%.9f, %.3f, %.3f, %.3f"
            else:
                raise ValueError(f"Unexpected ASCII output shape: {arr.shape}")
 
            with open(out_path, "ab") as f:
                np.savetxt(f, arr, fmt=fmt, delimiter=",")
 
        append_output = append_ascii
        las_writer = None
        las_header = None
 
    elif output_type == "LAS":
        out_path = os.path.join(output_dir, base_name + "_pcd.las")
        if os.path.exists(out_path):
            os.remove(out_path)
 
        las_writer = None
        las_header = None
 
        def append_las(arr):
            nonlocal las_writer, las_header
 
            if las_writer is None:
                las_header = laspy.LasHeader(point_format=1, version="1.4")
                las_header.scales = np.array([0.001, 0.001, 0.001])
 
                las_header.offsets = np.array([
                    np.floor(arr[:, 1].min()),
                    np.floor(arr[:, 2].min()),
                    np.floor(arr[:, 3].min()),
                ], dtype=np.float64)
 
                if include_lasvec:
                    las_header.add_extra_dim(ExtraBytesParams(name="lasvec_x", type=np.float32))
                    las_header.add_extra_dim(ExtraBytesParams(name="lasvec_y", type=np.float32))
                    las_header.add_extra_dim(ExtraBytesParams(name="lasvec_z", type=np.float32))
 
                las_writer = laspy.open(out_path, mode="w", header=las_header)
 
                print(f"[LAS] header offsets set to {las_header.offsets}")
                print(f"[LAS] header scales  set to {las_header.scales}")
 
            las = laspy.LasData(las_header)
            las.gps_time = arr[:, 0].astype(np.float64)
            las.x = arr[:, 1]
            las.y = arr[:, 2]
            las.z = arr[:, 3]
 
            if include_lasvec:
                if arr.shape[1] < 7:
                    raise ValueError(f"include_lasvec=True but arr has shape {arr.shape}")
                las["lasvec_x"] = arr[:, 4].astype(np.float32)
                las["lasvec_y"] = arr[:, 5].astype(np.float32)
                las["lasvec_z"] = arr[:, 6].astype(np.float32)
 
            las_writer.write_points(las.points)
 
        append_output = append_las
 
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
 
    # --------------------------------------------------
    # Process by subchunks
    # --------------------------------------------------
    total_out = 0
 
    try:
        for i0 in range(0, n_total, subchunk_size):
            i1 = min(i0 + subchunk_size, n_total)
            lasvec_chunk = lasvec[i0:i1]
 
            print(f"\n[Georef] subchunk {i0}:{i1} / {n_total}")
 
            # Georef only this chunk
            pcd_chunk = georefLidar(lasvec_chunk, trj, cfg)
 
            # Distance filtering
            if dist_cfg["enable"]:
                pcd_chunk = filter_pcd_by_vehicle_distance(
                    pcd_chunk,
                    trj,
                    max_dist_m=float(dist_cfg["max_distance_m"]),
                    epsg_out=dist_cfg["map_epsg"],
                )
 
            if len(pcd_chunk) == 0:
                del lasvec_chunk
                del pcd_chunk
                continue
 
            # Build output array
            # pcd_chunk is expected to be:
            #   [t, x, y, z]                  if lasvec_to_body == False
            #   [t, x, y, z, vx, vy, vz]      if lasvec_to_body == True
            if include_lasvec:
                if lasvec_to_body:
                    if pcd_chunk.shape[1] < 7:
                        raise ValueError(
                            f"lasvec_to_body=True but pcd_chunk has shape {pcd_chunk.shape}"
                        )
                    arr = pcd_chunk
                else:
                    # write lasvec in sensor frame from original lasvec_chunk
                    if dist_cfg["enable"]:
                        times_kept = pcd_chunk[:, 0]
                        idx = np.isin(lasvec_chunk[:, 0], times_kept)
                        lasvec_kept = lasvec_chunk[idx]
 
                        if len(lasvec_kept) != len(pcd_chunk):
                            print("[warning] lasvec/pcd mismatch after filtering, fallback by truncation")
                            nmin = min(len(lasvec_kept), len(pcd_chunk))
                            pcd_chunk = pcd_chunk[:nmin]
                            lasvec_kept = lasvec_kept[:nmin]
 
                        arr = np.column_stack((pcd_chunk[:, :4], lasvec_kept[:, 1:4]))
                    else:
                        arr = np.column_stack((pcd_chunk[:, :4], lasvec_chunk[:, 1:4]))
            else:
                arr = pcd_chunk[:, :4]
 
            append_output(arr)
            total_out += len(arr)
 
            print(f"[Georef] written {len(arr)} rows")
 
            del lasvec_chunk
            del pcd_chunk
            del arr
 
    finally:
        if las_writer is not None:
            las_writer.close()
 
    print(f"Saved {out_path}")
    print(f"[Georef] total input rows : {n_total}")
    print(f"[Georef] total output rows: {total_out}")
 
    return out_path
 
 
def run(cfg: Dict[str, Any]):
 
    trj = load_and_prepare_trajectory(cfg)
 
    if os.path.isdir(cfg["lasvec"]["path"]):
        paths = glob.glob(os.path.join(cfg["lasvec"]["path"], "*.sdc"))
        paths.sort()
    else:
        paths = [cfg["lasvec"]["path"]]
 
    # Filter SDC files by manifest + time window before loading anything
    # manifest_path can live under cfg["lasvec"] (scanner YAML) or at top level
    manifest_path = cfg["lasvec"].get("manifest_path") or cfg.get("manifest_path")
    paths = filter_sdc_paths_by_manifest(
        paths,
        manifest_path=manifest_path,
        tw_cfg=cfg.get("georef_time_window"),
    )
 
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