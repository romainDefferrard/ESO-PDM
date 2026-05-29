"""
steps/georef.py
===============
Georeferencing step — replaces Georef.py entirely.

Absorbs from old codebase:
  - Georef.py  (loadLasVecSDC, loadLasVecAscii, georefLidar, run, run_from_yaml,
                sync_times_day_shift, apply_leapsec, apply_time_window_filter,
                filter_lasvec_paths_by_manifest, filter_pcd_by_vehicle_distance,
                load_and_prepare_trajectory, trajectory_positions_mapping)
  - pipeline.py (build_georef_cfg, load_scanner_entries, get_scanner_name)

Public API
----------
run(pipe_cfg) -> list[dict]
    Georef all scanners in pipe_cfg["scanners"].
    Returns scanner_entries list.

get_ref_georef_cfg(pipe_cfg, scanner_entries) -> Path
    Returns path to the generated georef config for the reference scanner.
"""

from __future__ import annotations

import argparse
import glob
import math
import multiprocessing
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import laspy
import numpy as np
import yaml
from laspy import ExtraBytesParams
from pyproj import Transformer
from tqdm import tqdm

from pipeline._log import info, sub, subsub, warn
from pipeline.lib.trajectory import (
    Trajectory,
    load_trajectory,
    euler2quat_sequence,
)
from pipeline.lib.rotations import (
    quat2dcm, R_l2e, T, R1, R2, R3,
    quatMult,
)
from pipeline.lib.loaders import loadSBET


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# TIME SYNC
# ═══════════════════════════════════════════════════════════════

def sync_times_day_shift(t_lasvec: np.ndarray, t_trj: np.ndarray) -> np.ndarray:
    t_lasvec = np.asarray(t_lasvec, dtype=np.float64)
    t_trj    = np.asarray(t_trj,    dtype=np.float64)

    mask_las = np.isfinite(t_lasvec)
    mask_trj = np.isfinite(t_trj)
    if not np.any(mask_las): raise ValueError("No finite LiDAR timestamps.")
    if not np.any(mask_trj): raise ValueError("No finite trajectory timestamps.")

    diff      = float(np.median(t_trj[mask_trj])) - float(np.median(t_lasvec[mask_las]))
    day_shift = int(round(diff / 86400.0))
    offset    = day_shift * 86400.0

    out          = t_lasvec.copy()
    out[mask_las] += offset
    return out


def apply_leapsec(t: np.ndarray, leapsec: Optional[float]) -> np.ndarray:
    if leapsec is None:
        return t
    return t + float(leapsec)


# ═══════════════════════════════════════════════════════════════
# TIME WINDOW FILTER
# ═══════════════════════════════════════════════════════════════

def filter_lasvec_paths_by_manifest(
    paths: list,
    manifest_path: Optional[str],
    tw_cfg: Optional[Dict[str, Any]],
) -> list:
    import pandas as pd

    if manifest_path is None:
        return paths
    if tw_cfg is None or not tw_cfg.get("enable", False):
        return paths

    margin_s = float(tw_cfg.get("margin_s", 60.0))
    outages  = tw_cfg["outages"] if tw_cfg.get("outages") else [tw_cfg["outage"]]
    t_lo     = min(float(o[0]) - margin_s for o in outages)
    t_hi     = max(float(o[0]) + float(o[1]) + margin_s for o in outages)

    info(f"[manifest] Reading {manifest_path}")
    df = pandas_read_csv(manifest_path)

    if not {"filename", "t_min", "t_max"}.issubset(df.columns):
        raise ValueError(
            f"Manifest missing required columns (filename, t_min, t_max). "
            f"Got: {list(df.columns)}"
        )

    mask        = (df["t_max"] >= t_lo) & (df["t_min"] <= t_hi)
    valid_names = set(df.loc[mask, "filename"].values)
    info(f"[manifest] Window [{t_lo:.3f}, {t_hi:.3f}] — {mask.sum()}/{len(df)} files overlap")

    filtered = [p for p in paths if os.path.basename(p) in valid_names]
    skipped  = len(paths) - len(filtered)
    if skipped:
        sub(f"Skipped {skipped} files outside the time window.")
    if not filtered:
        warn(f"[manifest] No files remain after filtering — check outage/margin_s.")
    return filtered


def pandas_read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def apply_time_window_filter(
    lasvec: np.ndarray,
    tw_cfg: Optional[Dict[str, Any]],
) -> np.ndarray:
    if tw_cfg is None or not tw_cfg.get("enable", False):
        return lasvec

    margin_s = float(tw_cfg.get("margin_s", 60.0))
    outages  = tw_cfg.get("outages") or [tw_cfg["outage"]]

    mask = np.zeros(len(lasvec), dtype=bool)
    for outage in outages:
        t_lo = float(outage[0]) - margin_s
        t_hi = float(outage[0]) + float(outage[1]) + margin_s
        mask |= (lasvec[:, 0] >= t_lo) & (lasvec[:, 0] <= t_hi)

    n_before = len(lasvec)
    lasvec   = lasvec[mask]
    sub(f"time_window: {mask.sum()}/{n_before} points kept")

    if len(lasvec) == 0:
        raise ValueError(
            "No LiDAR points remain after time-window filter. "
            "Check outage/margin_s or disable the filter."
        )
    return lasvec


# ═══════════════════════════════════════════════════════════════
# LASVEC LOADERS
# ═══════════════════════════════════════════════════════════════

def loadLasVecAscii(path, cfg):
    import pandas as pd
    delimiter = cfg.get("sep", ",")
    skiprows  = cfg.get("skiprows", 1)
    chunks    = []
    file_size = Path(path).stat().st_size
    with tqdm(total=file_size, unit="B", unit_scale=True,
              desc=f"Loading {Path(path).name}") as pbar:
        prev = 0
        for chunk in pd.read_csv(path, delimiter=delimiter, header=None,
                                  skiprows=skiprows, chunksize=10_000_000,
                                  dtype=np.float64):
            chunks.append(chunk.values)
            cur = 0
            pbar.update(cur - prev); prev = cur
    data = np.vstack(chunks)
    return data[:, [3, 0, 1, 2, 4]]


def loadLasVecSDC(sdc_file, chunk_records=2_000_000):
    sdc_file  = Path(sdc_file)
    with open(sdc_file, "rb") as f:
        size_of_header = struct.unpack("<I", f.read(4))[0]

    file_size    = os.path.getsize(sdc_file)
    payload_size = file_size - size_of_header

    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)
        probe = f.read(60 * 30)

    record_size = None
    for rs in range(30, 60):
        timestamps = []
        for i in range(20):
            offset = i * rs
            if offset + 8 > len(probe): break
            timestamps.append(struct.unpack("<d", probe[offset:offset+8])[0])
        valid = [t for t in timestamps if 40000 < t < 700000]
        if len(valid) >= 18 and payload_size % rs == 0:
            record_size = rs; break

    if record_size is None:
        best, best_rs = 0, 40
        for rs in range(30, 60):
            timestamps = [struct.unpack("<d", probe[i*rs:i*rs+8])[0]
                          for i in range(20) if i*rs+8 <= len(probe)]
            valid = sum(1 for t in timestamps if 40000 < t < 700000)
            if valid > best: best, best_rs = valid, rs
        record_size = best_rs
        warn(f"[SDC] no exact record_size found, using {record_size} bytes (fallback)")

    sub(f"SDC record_size = {record_size} bytes")

    record_dtype = np.dtype([
        ("t",   "<f8"), ("f1",  "<f4"), ("f2",  "<f4"),
        ("x",   "<f4"), ("y",   "<f4"), ("z",   "<f4"),
        ("u6",  "<u2"), ("pad", f"V{record_size - 30}"),
    ])
    assert record_dtype.itemsize == record_size

    record_count  = payload_size // record_size
    records       = np.empty((record_count, 5), dtype=np.float64)
    write_pos     = 0
    total_invalid = 0

    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)
        with tqdm(total=record_count, desc=f"Reading SDC {sdc_file.name}",
                  unit="pts", mininterval=0.5) as pbar:
            while True:
                data = np.fromfile(f, dtype=record_dtype, count=chunk_records)
                n0   = len(data)
                if n0 == 0: break
                valid = (np.isfinite(data["t"]) & np.isfinite(data["x"]) &
                         np.isfinite(data["y"]) & np.isfinite(data["z"]))
                n_bad = int(n0 - np.count_nonzero(valid))
                if n_bad > 0: total_invalid += n_bad
                data = data[valid]; n = len(data)
                if n > 0:
                    records[write_pos:write_pos+n, 0] = data["t"]
                    records[write_pos:write_pos+n, 1] = data["x"]
                    records[write_pos:write_pos+n, 2] = data["y"]
                    records[write_pos:write_pos+n, 3] = data["z"]
                    records[write_pos:write_pos+n, 4] = data["u6"].astype(np.float64)
                    write_pos += n
                pbar.update(n0)

    if total_invalid > 0:
        warn(f"[SDC] dropped {total_invalid} records with non-finite values")

    return records[:write_pos]


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY
# ═══════════════════════════════════════════════════════════════

def load_and_prepare_trajectory(cfg: Dict[str, Any]):
    trj     = load_trajectory(cfg["trj"])
    trj_cfg = cfg["trj"]
    if trj_cfg["type"] == "SBET" or (
        trj_cfg["type"] == "ASCII" and trj_cfg.get("rpy_col") is not None
    ):
        sub("Converting orientation: local NED → ECEF")
        trj.estimate_q_l2e(interp=False)
        for i in range(len(trj.q)):
            trj.q[i] = quatMult(trj.q_l2e[i], trj.q[i])
    return trj


def trajectory_positions_mapping(trj, epsg_out: str = "EPSG:2056") -> np.ndarray:
    for attr in ("pos", "xyz", "enu"):
        if hasattr(trj, attr):
            P = np.asarray(getattr(trj, attr))
            if P.ndim == 2 and P.shape[1] >= 3:
                return P[:, :3].astype(np.float64)

    ecef_to_map = Transformer.from_crs("EPSG:4978", epsg_out, always_xy=True)
    lla_to_map  = Transformer.from_crs("EPSG:4979", epsg_out, always_xy=True)

    for attr in ("ecef", "xyz_ecef", "p_ecef"):
        if hasattr(trj, attr):
            E = np.asarray(getattr(trj, attr))
            if E.ndim == 2 and E.shape[1] >= 3:
                x, y, z = ecef_to_map.transform(E[:,0], E[:,1], E[:,2])
                return np.column_stack((x, y, z)).astype(np.float64)

    for attr in ("lla", "geodetic", "llh"):
        if hasattr(trj, attr):
            L = np.asarray(getattr(trj, attr))
            if L.ndim == 2 and L.shape[1] >= 3:
                x, y, z = lla_to_map.transform(L[:,1], L[:,0], L[:,2])
                return np.column_stack((x, y, z)).astype(np.float64)

    raise AttributeError("Could not find trajectory positions in trj object.")


# ═══════════════════════════════════════════════════════════════
# DISTANCE FILTER
# ═══════════════════════════════════════════════════════════════

def filter_pcd_by_vehicle_distance(pcd, trj, max_dist_m, epsg_out, return_mask=False, chunk_i=None, n_chunks=None):
    t_pts = pcd[:, 0]
    P     = trajectory_positions_mapping(trj, epsg_out=epsg_out)
    t_trj = trj.time
    x_car = np.interp(t_pts, t_trj, P[:, 0])
    y_car = np.interp(t_pts, t_trj, P[:, 1])
    dist  = np.sqrt((pcd[:, 1] - x_car)**2 + (pcd[:, 2] - y_car)**2)
    mask  = dist <= max_dist_m
    subsub(f"[{chunk_i}/{n_chunks}] distance filter: {int(mask.sum())}/{len(pcd)} pts ≤ {max_dist_m} m")
    if return_mask:
        return pcd[mask], mask
    return pcd[mask]


# ═══════════════════════════════════════════════════════════════
# GEOREFERENCING CORE
# ═══════════════════════════════════════════════════════════════

def get_R_sensor2body(cfg):
    mount_cfg = cfg["mount"]
    if "R_sensor2body" in mount_cfg:
        return np.array(mount_cfg["R_sensor2body"])
    R_mount   = np.array(mount_cfg["R_mount"])
    bs        = mount_cfg["boresight"]
    R_bore    = (R1(bs["roll"]) @ R2(bs["pitch"]) @ R3(bs["yaw"])).T
    return R_bore @ R_mount


def _georef_chunk(las_chunk, t_chunk, ecef_chunk, q_chunk,
                  R_sensor2body, lever_arm, ltp_origin=None, lasvec_to_body=False):
    lla2ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    xyz_body = R_sensor2body @ las_chunk[:, 1:4].T + lever_arm[:, np.newaxis]
    xyz_ecef = np.zeros_like(xyz_body)
    for i in range(len(las_chunk)):
        xyz_ecef[:, i] = quat2dcm(q_chunk[i]) @ xyz_body[:, i] + ecef_chunk[i]

    if ltp_origin is not None:
        lat, lon, alt = ltp_origin
        ltp_ecef = np.array(lla2ecef.transform(lon, lat, alt))
        R_enu2e  = R_l2e(lat, lon, degrees=True) @ T()
        xyz_georef = (R_enu2e.T @ (xyz_ecef - ltp_ecef.reshape(-1, 1))).T
    else:
        ecef2map = Transformer.from_crs("EPSG:4978", "EPSG:2056")
        xyz_georef = np.array(ecef2map.transform(xyz_ecef[0], xyz_ecef[1], xyz_ecef[2])).T

    range_m      = np.linalg.norm(las_chunk[:, 1:4], axis=1)
    i_corrected  = las_chunk[:, 4] * np.clip(range_m, 3.0, np.inf)**0.5
    out          = np.column_stack((t_chunk, xyz_georef, i_corrected))

    if lasvec_to_body:
        v_body = (R_sensor2body @ las_chunk[:, 1:4].T + lever_arm[:, np.newaxis]).T
        out    = np.column_stack((out, v_body))
    return out


def georefLidar(lasvec, trj, cfg):
    R_sensor2body  = get_R_sensor2body(cfg)
    lever_arm      = np.array(cfg["mount"]["lever_arm"], dtype=np.float64)
    t_interp, ecef_interp, q_interp = trj.interp(lasvec[:, 0], updateSelf=False)
    ltp_origin     = np.array(cfg["ltp_origin"]) if cfg.get("ltp_origin") else None
    lasvec_to_body = cfg["output"]["lasvec_to_body"]

    n         = len(lasvec)
    n_workers = min(4, multiprocessing.cpu_count(), n)

    if n < 80_000 or n_workers <= 1:
        return _georef_chunk(lasvec, t_interp, ecef_interp, q_interp,
                              R_sensor2body, lever_arm, ltp_origin, lasvec_to_body)

    las_chunks  = np.array_split(lasvec,      n_workers)
    t_chunks    = np.array_split(t_interp,    n_workers)
    ecef_chunks = np.array_split(ecef_interp, n_workers)
    q_chunks    = np.array_split(q_interp,    n_workers)
    args = [(lc, tc, ec, qc, R_sensor2body, lever_arm, ltp_origin, lasvec_to_body)
            for lc, tc, ec, qc in zip(las_chunks, t_chunks, ecef_chunks, q_chunks)]

    from multiprocessing import Pool
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(_georef_chunk, args)
    return np.vstack(results)


# ═══════════════════════════════════════════════════════════════
# SINGLE FILE PROCESSING
# ═══════════════════════════════════════════════════════════════

def georef_one_file(cfg, trj, path):
    sub(f"Processing: {Path(path).name}")
    ext = Path(path).suffix.lower()

    if ext == ".sdc":
        lasvec        = loadLasVecSDC(path)
        lasvec[:, 0]  = sync_times_day_shift(lasvec[:, 0], trj.time)
        lasvec[:, 0]  = apply_leapsec(lasvec[:, 0], cfg.get("leapsec"))
    elif ext == ".csv":
        lasvec        = loadLasVecAscii(path, cfg["lasvec"])
        lasvec[:, 0]  = apply_leapsec(lasvec[:, 0], cfg.get("leapsec"))
    else:
        raise ValueError(f"Unsupported lasvec format: {ext}")

    lasvec = apply_time_window_filter(lasvec, cfg.get("georef_time_window"))

    # clip to trajectory range
    tmin_trj = float(trj.time[0])
    tmax_trj = float(trj.time[-1])
    mask     = (np.isfinite(lasvec[:, 0]) &
                (lasvec[:, 0] >= tmin_trj) &
                (lasvec[:, 0] <= tmax_trj))
    lasvec   = lasvec[mask]
    if len(lasvec) == 0:
        raise ValueError(f"No LiDAR rows inside trajectory range for: {path}")

    # output setup
    base_name  = os.path.splitext(os.path.basename(path))[0]
    output_dir = cfg["output"]["path"]
    os.makedirs(output_dir, exist_ok=True)
    output_type    = cfg["output"]["type"]
    include_lasvec = cfg["output"]["lasvec"]
    lasvec_to_body = cfg["output"].get("lasvec_to_body", False)
    dist_cfg       = cfg["distance_filtering"]

    subchunk_size = 10_000_000
    n_total       = len(lasvec)
    n_chunks      = max(1, (n_total + subchunk_size - 1) // subchunk_size)
    total_out     = 0

    if output_type == "ASCII":
        out_path = os.path.join(output_dir, base_name + "_pcd.txt")
        if os.path.exists(out_path): os.remove(out_path)

        def append_output(arr):
            fmts = {8: "%.9f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.0f",
                    5: "%.9f, %.3f, %.3f, %.3f, %.0f",
                    4: "%.9f, %.3f, %.3f, %.3f"}
            fmt = fmts.get(arr.shape[1])
            if fmt is None: raise ValueError(f"Unexpected shape: {arr.shape}")
            with open(out_path, "ab") as f:
                np.savetxt(f, arr, fmt=fmt, delimiter=",")
        las_writer = None

    elif output_type == "LAS":
        out_path   = os.path.join(output_dir, base_name + "_pcd.las")
        if os.path.exists(out_path): os.remove(out_path)
        las_writer = None
        las_header = None

        def append_output(arr):
            nonlocal las_writer, las_header
            if las_writer is None:
                las_header = laspy.LasHeader(point_format=1, version="1.4")
                las_header.scales  = np.array([0.001, 0.001, 0.001])
                las_header.offsets = np.array([
                    np.floor(arr[:, 1].min()),
                    np.floor(arr[:, 2].min()),
                    np.floor(arr[:, 3].min()),
                ], dtype=np.float64)
                if include_lasvec:
                    for dim in ("lasvec_x", "lasvec_y", "lasvec_z"):
                        las_header.add_extra_dim(ExtraBytesParams(name=dim, type=np.float32))
                las_writer = laspy.open(out_path, mode="w", header=las_header)

            las       = laspy.LasData(las_header)
            las.gps_time = arr[:, 0].astype(np.float64)
            las.x = arr[:, 1]; las.y = arr[:, 2]; las.z = arr[:, 3]
            if include_lasvec:
                las["lasvec_x"] = arr[:, 4].astype(np.float32)
                las["lasvec_y"] = arr[:, 5].astype(np.float32)
                las["lasvec_z"] = arr[:, 6].astype(np.float32)
                i_raw = arr[:, 7]
            else:
                i_raw = arr[:, 4]
            p2, p98 = np.percentile(i_raw, [5, 95])
            i_norm  = np.clip((i_raw - p2) / (p98 - p2) * 65535, 0, 65535) if p98 > p2 \
                      else np.zeros_like(i_raw)
            las.intensity = i_norm.astype(np.uint16)
            las_writer.write_points(las.points)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    try:
        n_chunks_total = max(1, (n_total + subchunk_size - 1) // subchunk_size)
        for chunk_i, i0 in enumerate(range(0, n_total, subchunk_size), 1):
            i1           = min(i0 + subchunk_size, n_total)
            lasvec_chunk = lasvec[i0:i1]
            safe         = (np.isfinite(lasvec_chunk[:, 0]) &
                            (lasvec_chunk[:, 0] >= tmin_trj) &
                            (lasvec_chunk[:, 0] <= tmax_trj))
            lasvec_chunk = lasvec_chunk[safe]
            if len(lasvec_chunk) == 0: continue

            pcd_chunk = georefLidar(lasvec_chunk, trj, cfg)

            if dist_cfg.get("enable", False):
                pcd_chunk, mask_keep = filter_pcd_by_vehicle_distance(
                    pcd_chunk, trj,
                    max_dist_m=float(dist_cfg["max_distance_m"]),
                    epsg_out=dist_cfg["map_epsg"],
                    return_mask=True,
                    chunk_i=chunk_i, 
                    n_chunks=n_chunks_total,
                )
            else:
                mask_keep = None

            if len(pcd_chunk) == 0:
                continue

            if include_lasvec:
                lasvec_kept = lasvec_chunk[mask_keep] if dist_cfg.get("enable") else lasvec_chunk
                if lasvec_to_body:
                    arr = np.column_stack((pcd_chunk[:, :4], pcd_chunk[:, 5:8], pcd_chunk[:, 4]))
                else:
                    arr = np.column_stack((pcd_chunk[:, :4], lasvec_kept[:, 1:4], pcd_chunk[:, 4]))
            else:
                arr = pcd_chunk[:, :5]

            append_output(arr)
            total_out += len(arr)
    finally:
        if las_writer is not None:
            las_writer.close()

    sub(f"saved {out_path}  ({total_out} pts)")
    return out_path


# ═══════════════════════════════════════════════════════════════
# RUN (full scanner)
# ═══════════════════════════════════════════════════════════════

def run_scanner(cfg: Dict[str, Any]) -> None:
    """Georef one scanner using a pre-built config dict."""
    trj = load_and_prepare_trajectory(cfg)

    if os.path.isdir(cfg["lasvec"]["path"]):
        paths  = glob.glob(os.path.join(cfg["lasvec"]["path"], "*.sdc"))
        paths += glob.glob(os.path.join(cfg["lasvec"]["path"], "*.csv"))
        paths.sort()
    else:
        paths = [cfg["lasvec"]["path"]]

    manifest_path = cfg["lasvec"].get("manifest_path") or cfg.get("manifest_path")
    paths = filter_lasvec_paths_by_manifest(
        paths, manifest_path=manifest_path, tw_cfg=cfg.get("georef_time_window"),
    )
    sub(f"Processing {len(paths)} file(s)")
    for path in paths:
        georef_one_file(cfg, trj, path)


def run_from_yaml(cfg_path: Union[str, Path]) -> None:
    run_scanner(load_config(cfg_path))


# ═══════════════════════════════════════════════════════════════
# BUILD GEOREF CFG (from pipeline config + scanner YAML)
# ═══════════════════════════════════════════════════════════════

def _scanner_name(cfg_path: Union[str, Path]) -> str:
    d = yaml.safe_load(open(cfg_path, "r"))
    return d.get("scanner_name", Path(cfg_path).stem)


def _build_georef_cfg(scanner_cfg_path: Union[str, Path], pipe_cfg: dict) -> dict:
    sc            = yaml.safe_load(open(scanner_cfg_path, "r"))
    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]
    scanner_name  = sc["scanner_name"]

    outage   = pipe_cfg.get("outage")
    tw_block = pipe_cfg.get("georef_time_window", {})
    if outage and tw_block.get("enable", False):
        tw = {
            "enable":   True,
            "outage":   list(outage),
            "margin_s": float(tw_block.get("margin_s", 30.0)),
        }
    else:
        tw = {"enable": False}

    return {
        "trj":                pipe_cfg["trajectory"],
        "distance_filtering": pipe_cfg["distance_filtering"],
        "lasvec":             {**sc["lasvec"], "manifest_path": sc.get("manifest_path")},
        "leapsec":            sc.get("leapsec"),
        "mount":              sc["mount"],
        "output": {
            **sc["output_defaults"],
            "path": str(root_out_dir / scenario_name / scanner_name),
        },
        "georef_time_window": tw,
    }


def _write_tmp_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ═══════════════════════════════════════════════════════════════
# PIPELINE-FACING API
# ═══════════════════════════════════════════════════════════════

def run(pipe_cfg: dict) -> list[dict]:
    """
    Georef all scanners declared in pipe_cfg["scanners"].
    Returns scanner_entries list of dicts.
    """
    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]
    tmp_dir       = root_out_dir / scenario_name / "tmp" / "generated_configs"
    scanners_raw  = pipe_cfg.get("scanners", {})

    if not scanners_raw:
        raise ValueError("No scanners declared in pipe_cfg['scanners'].")

    scanner_entries: list[dict] = []
    info(f"[georef] {scenario_name} — {len(scanners_raw)} scanner(s)")

    for key, cfg_path_str in scanners_raw.items():
        if cfg_path_str is None:
            continue
        cfg_path = Path(cfg_path_str)
        if not cfg_path.is_absolute():
            pkg_dir = Path(__file__).resolve().parents[1]  # pipeline/
            cfg_path = (pkg_dir / cfg_path).resolve()
        scanner_name = _scanner_name(cfg_path)

        georef_cfg   = _build_georef_cfg(cfg_path, pipe_cfg)
        gen_cfg_path = tmp_dir / f"georef_{scanner_name}.generated.yml"
        _write_tmp_cfg(georef_cfg, gen_cfg_path)

        output_dir = Path(georef_cfg["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        sub(f"{scanner_name} → {output_dir}")
        run_from_yaml(str(gen_cfg_path))
        info(f"[georef] {scanner_name} ✓")

        scanner_entries.append({
            "key":                key,
            "cfg_path":           cfg_path,
            "scanner_name":       scanner_name,
            "generated_cfg_path": gen_cfg_path,
            "output_dir":         output_dir,
        })

    info(f"[georef] done — {len(scanner_entries)} scanner(s)")
    return scanner_entries


def get_ref_georef_cfg(pipe_cfg: dict, scanner_entries: list[dict]) -> Path:
    """Return generated georef config path for the reference scanner."""
    ref_name = pipe_cfg.get("chunk", {}).get("reference_scanner")
    for e in scanner_entries:
        if ref_name is None or e["scanner_name"] == ref_name:
            return e["generated_cfg_path"]
    raise ValueError(
        f"Reference scanner '{ref_name}' not found. "
        f"Available: {[e['scanner_name'] for e in scanner_entries]}"
    )


# ═══════════════════════════════════════════════════════════════
# MINIMAL GEOREF CFG (for chunk step when georef was not run)
# ═══════════════════════════════════════════════════════════════

def write_minimal_georef_cfg(pipe_cfg: dict, scenario_root: Path) -> Path:
    """
    Build and write a minimal georef config that only contains the trajectory
    block — enough for the chunker to load the SBET and compute curvilinear
    distance.  Does NOT contain scanner / lasvec / output fields.

    Written to <scenario_root>/tmp/generated_configs/georef_minimal.yml.
    """
    out_path = (
        scenario_root / "tmp" / "generated_configs" / "georef_minimal.yml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    minimal = {
        "trj":                pipe_cfg["trajectory"],
        "distance_filtering": pipe_cfg.get("distance_filtering", {"enable": False}),
        "georef_time_window": {"enable": False},
        # Dummy mount / lasvec / output so load_config() doesn't crash
        "lasvec":  {"type": "SDC", "cols": [0, 3, 4, 5], "path": "/dev/null"},
        "leapsec": None,
        "mount": {
            "R_mount":   [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            "boresight": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "lever_arm": [0.0, 0.0, 0.0],
        },
        "output": {
            "type": "LAS", "lasvec": False, "lasvec_to_body": False,
            "path": str(scenario_root / "tmp" / "dummy_output"),
        },
    }

    _write_tmp_cfg(minimal, out_path)
    sub(f"minimal georef cfg → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# CLI (standalone use)
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Georeference LiDAR data")
    p.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    args = p.parse_args()
    run_scanner(load_config(args.config))


if __name__ == "__main__":
    main()