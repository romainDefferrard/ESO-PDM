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
 
    margin_s = float(tw_cfg.get("margin_s", 60.0))
    if "outages" in tw_cfg and tw_cfg["outages"]:
        outages = tw_cfg["outages"]
    else:
        outages = [tw_cfg["outage"]]

    t_lo = min(float(o[0]) - margin_s for o in outages)
    t_hi = max(float(o[0]) + float(o[1]) + margin_s for o in outages)
 
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
    Keep only LiDAR rows whose timestamp falls within one or more windows.

    Config keys (from pipeline.yml  georef_time_window):
        enable   : bool
        outage   : [t_start, duration]        # single window (legacy)
        outages  : [[t_start, duration], ...] # multiple windows
        margin_s : float  — extra seconds added on both sides
    """
    if tw_cfg is None or not tw_cfg.get("enable", False):
        return lasvec

    margin_s = float(tw_cfg.get("margin_s", 60.0))

    # Support both single outage and list of outages
    if "outages" in tw_cfg and tw_cfg["outages"]:
        outages = tw_cfg["outages"]
    else:
        outages = [tw_cfg["outage"]]

    mask = np.zeros(len(lasvec), dtype=bool)
    for outage in outages:
        t_lo = float(outage[0]) - margin_s
        t_hi = float(outage[0]) + float(outage[1]) + margin_s
        print(
            f"[time_window] window [{t_lo:.3f}, {t_hi:.3f}]  "
            f"(outage [{float(outage[0])}, {float(outage[0]) + float(outage[1])}] +/- {margin_s} s)"
        )
        mask |= (lasvec[:, 0] >= t_lo) & (lasvec[:, 0] <= t_hi)

    n_before = len(lasvec)
    lasvec = lasvec[mask]
    print(f"[time_window] {mask.sum()}/{n_before} points kept")

    if len(lasvec) == 0:
        raise ValueError(
            f"[time_window] No LiDAR points remain after time-window filter. "
            f"Check outage/margin_s or disable the filter."
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
 
def filter_pcd_by_vehicle_distance(pcd, trj, max_dist_m, epsg_out, return_mask=False):
    t_pts = pcd[:, 0]
    P = trajectory_positions_mapping(trj, epsg_out=epsg_out)
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

    if return_mask:
        return pcd[mask], mask
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
    # LOAD LASVEC
    # ==================================================
    if cfg["lasvec"]["type"] == "SDC":
        lasvec = loadLasVecSDC(path)
    else:
        lasvec = loadLasVecAscii(cfg["lasvec"], cfg["limatch_output"])

    # --------------------------------------------------
    # 1) Raw sanity filter ONLY (before sync)
    #    -> ne surtout pas comparer à trj.time ici
    # --------------------------------------------------
    mask_valid_raw = (
        np.isfinite(lasvec[:, 0]) &
        np.isfinite(lasvec[:, 1]) &
        np.isfinite(lasvec[:, 2]) &
        np.isfinite(lasvec[:, 3]) &
        (np.abs(lasvec[:, 0]) < 1e8) &
        (np.abs(lasvec[:, 1]) < 1e4) &
        (np.abs(lasvec[:, 2]) < 1e4) &
        (np.abs(lasvec[:, 3]) < 1e4)
    )

    n_bad_raw = int((~mask_valid_raw).sum())
    if n_bad_raw > 0:
        print(f"[Georef] Removing {n_bad_raw} raw rows with invalid values")
        lasvec = lasvec[mask_valid_raw]

    if len(lasvec) == 0:
        raise ValueError(f"No valid raw LiDAR rows remain after sanity filtering for file: {path}")

    # --------------------------------------------------
    # 2) Sync time to trajectory day + leap seconds
    # --------------------------------------------------
    lasvec[:, 0] = sync_times_day_shift(lasvec[:, 0], trj.time)
    lasvec[:, 0] = apply_leapsec(lasvec[:, 0], cfg.get("leapsec"))

    tmin_trj = float(np.nanmin(trj.time))
    tmax_trj = float(np.nanmax(trj.time))

    print(f"[debug] lasvec time min/max: {np.nanmin(lasvec[:,0]):.4f} / {np.nanmax(lasvec[:,0]):.4f}")
    print(f"[debug] trj time min/max:    {tmin_trj:.4f} / {tmax_trj:.4f}")

    # --------------------------------------------------
    # 3) Broad plausibility filter AFTER sync
    #    -> utile pour enlever les gros outliers de temps
    # --------------------------------------------------
    mask_postsync = (
        np.isfinite(lasvec[:, 0]) &
        (lasvec[:, 0] >= tmin_trj - 600.0) &
        (lasvec[:, 0] <= tmax_trj + 600.0)
    )

    n_bad_post = int((~mask_postsync).sum())
    if n_bad_post > 0:
        print(f"[Georef] Removing {n_bad_post} rows outside plausible synced time range")
        lasvec = lasvec[mask_postsync]

    if len(lasvec) == 0:
        raise ValueError(
            f"No LiDAR rows remain after synced-time plausibility filtering for file: {path}"
        )

    # --------------------------------------------------
    # 4) User time window filter
    # --------------------------------------------------
    lasvec = apply_time_window_filter(lasvec, cfg.get("georef_time_window"))

    if len(lasvec) == 0:
        raise ValueError(
            f"No LiDAR rows remain after user time-window filtering for file: {path}"
        )

    # --------------------------------------------------
    # 5) Exact trajectory interpolation range
    #    -> indispensable pour éviter le crash Slerp
    # --------------------------------------------------
    mask_interp_safe = (
        np.isfinite(lasvec[:, 0]) &
        (lasvec[:, 0] >= tmin_trj) &
        (lasvec[:, 0] <= tmax_trj)
    )

    n_bad_interp = int((~mask_interp_safe).sum())
    if n_bad_interp > 0:
        print(f"[Georef] Removing {n_bad_interp} rows outside exact trajectory interpolation range")
        lasvec = lasvec[mask_interp_safe]

    if len(lasvec) == 0:
        raise ValueError(
            f"No LiDAR rows remain inside exact trajectory interpolation range for file: {path}"
        )

    # ==================================================
    # OUTPUT PREP
    # ==================================================
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = cfg["output"]["path"]
    os.makedirs(output_dir, exist_ok=True)

    output_type = cfg["output"]["type"]
    include_lasvec = cfg["output"]["lasvec"]
    lasvec_to_body = cfg["output"].get("lasvec_to_body", False)
    dist_cfg = cfg["distance_filtering"]

    subchunk_size = 10_000_000
    n_total = len(lasvec)

    if output_type == "ASCII":
        out_path = os.path.join(output_dir, base_name + "_pcd.txt")
        if os.path.exists(out_path):
            os.remove(out_path)

        def append_ascii(arr):
            if arr.shape[1] == 8:
                fmt = "%.9f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.0f"
            elif arr.shape[1] == 5:
                fmt = "%.9f, %.3f, %.3f, %.3f, %.0f"
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
                if arr.shape[1] < 8:
                    raise ValueError(f"include_lasvec=True but arr has shape {arr.shape}")
                las["lasvec_x"] = arr[:, 4].astype(np.float32)
                las["lasvec_y"] = arr[:, 5].astype(np.float32)
                las["lasvec_z"] = arr[:, 6].astype(np.float32)
                i_raw = arr[:, 7]
            else:
                i_raw = arr[:, 4]

            p2, p98 = np.percentile(i_raw, [5, 95])
            if p98 > p2:
                i_norm = np.clip((i_raw - p2) / (p98 - p2) * 65535, 0, 65535)
            else:
                i_norm = np.zeros_like(i_raw)
            las.intensity = i_norm.astype(np.uint16)

            las_writer.write_points(las.points)

        append_output = append_las

    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    # ==================================================
    # PROCESS BY SUBCHUNKS
    # ==================================================
    total_out = 0

    try:
        for i0 in range(0, n_total, subchunk_size):
            i1 = min(i0 + subchunk_size, n_total)
            lasvec_chunk = lasvec[i0:i1]

            print(f"\n[Georef] subchunk {i0}:{i1} / {n_total}")

            # sécurité supplémentaire, au cas où
            mask_chunk_interp_safe = (
                np.isfinite(lasvec_chunk[:, 0]) &
                (lasvec_chunk[:, 0] >= tmin_trj) &
                (lasvec_chunk[:, 0] <= tmax_trj)
            )
            if not np.all(mask_chunk_interp_safe):
                n_bad_chunk = int((~mask_chunk_interp_safe).sum())
                print(f"[Georef] subchunk removing {n_bad_chunk} rows outside interpolation range")
                lasvec_chunk = lasvec_chunk[mask_chunk_interp_safe]

            if len(lasvec_chunk) == 0:
                continue

            # pcd_chunk : [t, x, y, z, intensity] (+ vx,vy,vz if lasvec_to_body)
            pcd_chunk = georefLidar(lasvec_chunk, trj, cfg)

            if dist_cfg["enable"]:
                pcd_chunk, mask_keep = filter_pcd_by_vehicle_distance(
                    pcd_chunk,
                    trj,
                    max_dist_m=float(dist_cfg["max_distance_m"]),
                    epsg_out=dist_cfg["map_epsg"],
                    return_mask=True,
                )
            else:
                mask_keep = None

            if len(pcd_chunk) == 0:
                del lasvec_chunk, pcd_chunk
                continue

            if include_lasvec:
                if dist_cfg["enable"]:
                    lasvec_kept = lasvec_chunk[mask_keep]
                    if len(lasvec_kept) != len(pcd_chunk):
                        raise RuntimeError(
                            f"Internal mismatch after distance filtering: "
                            f"len(lasvec_kept)={len(lasvec_kept)} vs len(pcd_chunk)={len(pcd_chunk)}"
                        )
                else:
                    lasvec_kept = lasvec_chunk

                if lasvec_to_body:
                    # pcd_chunk = [t, x, y, z, intensity, vx, vy, vz]
                    # arr       = [t, x, y, z, vx, vy, vz, intensity]
                    arr = np.column_stack((
                        pcd_chunk[:, :4],
                        pcd_chunk[:, 5:8],
                        pcd_chunk[:, 4],
                    ))
                else:
                    # arr = [t, x, y, z, lx, ly, lz, intensity]
                    arr = np.column_stack((
                        pcd_chunk[:, :4],
                        lasvec_kept[:, 1:4],
                        pcd_chunk[:, 4],
                    ))
            else:
                arr = pcd_chunk[:, :5]

            append_output(arr)
            total_out += len(arr)
            print(f"[Georef] written {len(arr)} rows")

            del lasvec_chunk, pcd_chunk, arr

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