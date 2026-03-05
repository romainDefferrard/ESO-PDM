#!/usr/bin/env python3
# rmse_multi_targets_ref_to_las.py
#
# One reference TXT, multiple target TXTs.
# For each target:
#   - read only outage time window (with buffer) from ref and target
#   - assume exact same gps_time array (or allclose if configured)
#   - compute per-point 3D error e3d = ||xyz_tgt - xyz_ref||
#   - compute metrics (RMSE, mean, median, p95, p99, max, threshold rates, etc.)
#   - write LAS of REFERENCE points with ExtraBytes scalar field "e3d" (float32)
# At end: print a clean summary table.
#
# Config format:
# reference:
#   path: /path/ref.txt
# targets:
#   - path: /path/target1.txt
#     outfile: /path/out1.las
#   - path: /path/target2.txt
#     outfile: /path/out2.las
# time:
#   outage_start: 466930.0
#   outage_duration: 5.0
#   buffer: 30.0
# io:
#   delim: ","
# matching:
#   exact_time: true
#   time_atol: 0.0
# las:
#   scale: 0.001

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
import laspy
from tqdm import tqdm


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_window_txt(path: Path, delim: str, tmin: float, tmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming read with early break. Assumes file sorted by gps_time.
    Expected per line (no header):
      gps_time, x, y, z, ...
    Returns (t, xyz) in [tmin, tmax].
    """
    rows_t = []
    rows_xyz = []

    file_size = os.path.getsize(path)
    with path.open("r", encoding="utf-8") as f, tqdm(
        total=file_size, unit="B", unit_scale=True, desc=f"Reading {path.name}"
    ) as pbar:
        for ln in f:
            pbar.update(len(ln.encode("utf-8")))
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue

            parts = ln.split(delim)
            if len(parts) < 4:
                continue

            t = float(parts[0])
            if t < tmin:
                continue
            if t > tmax:
                break

            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            rows_t.append(t)
            rows_xyz.append((x, y, z))

    if not rows_t:
        raise RuntimeError(f"{path}: no points found in window [{tmin}, {tmax}]")

    return np.asarray(rows_t, dtype=np.float64), np.asarray(rows_xyz, dtype=np.float64)


def validate_times(t_ref: np.ndarray, t_tgt: np.ndarray, exact: bool, atol: float):
    if len(t_ref) != len(t_tgt):
        raise RuntimeError(f"Different number of points in window: ref={len(t_ref)} tgt={len(t_tgt)}")

    if exact:
        if not np.array_equal(t_ref, t_tgt):
            raise RuntimeError("gps_time arrays are not exactly identical in the selected window")
    else:
        if not np.allclose(t_ref, t_tgt, atol=atol, rtol=0.0):
            raise RuntimeError(f"gps_time arrays differ more than atol={atol}")


def compute_metrics(e3d: np.ndarray, dxyz: np.ndarray) -> Dict[str, float]:
    # dxyz = xyz_tgt - xyz_ref (N,3)
    rmse = float(np.sqrt(np.mean(e3d * e3d)))
    mean = float(np.mean(e3d))
    med = float(np.median(e3d))
    p95 = float(np.percentile(e3d, 95))
    p99 = float(np.percentile(e3d, 99))
    mx = float(np.max(e3d))

    # threshold rates (easy to report)
    thr_01 = float(np.mean(e3d > 0.1)) * 100.0
    thr_05 = float(np.mean(e3d > 0.5)) * 100.0
    thr_10 = float(np.mean(e3d > 1.0)) * 100.0

    # bias / drift direction indicators (not written in LAS, but great for analysis)
    dx_mean, dy_mean, dz_mean = map(float, np.mean(dxyz, axis=0))
    dx_std, dy_std, dz_std = map(float, np.std(dxyz, axis=0))

    return {
        "N": float(len(e3d)),
        "RMSE": rmse,
        "mean": mean,
        "median": med,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "%>0.1m": thr_01,
        "%>0.5m": thr_05,
        "%>1.0m": thr_10,
        "dx_mean": dx_mean,
        "dy_mean": dy_mean,
        "dz_mean": dz_mean,
        "dx_std": dx_std,
        "dy_std": dy_std,
        "dz_std": dz_std,
    }


def write_ref_las_with_e3d(out_las: Path, t: np.ndarray, xyz_ref: np.ndarray, e3d: np.ndarray, scale: float):
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([scale, scale, scale], dtype=np.float64)
    hdr.offsets = np.array(
        [float(xyz_ref[:, 0].min()), float(xyz_ref[:, 1].min()), float(xyz_ref[:, 2].min())],
        dtype=np.float64,
    )

    las = laspy.LasData(hdr)
    las.x = xyz_ref[:, 0]
    las.y = xyz_ref[:, 1]
    las.z = xyz_ref[:, 2]
    las.gps_time = t

    las.add_extra_dim(laspy.ExtraBytesParams(name="e3d", type=np.float32))
    las["e3d"] = e3d.astype(np.float32)

    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_las))


def format_summary(rows: List[Dict[str, object]]) -> str:
    # simple pretty columns without pandas/tabulate
    cols = [
        ("target", 34),
        ("N", 10),
        ("RMSE", 10),
        ("p95", 10),
        ("p99", 10),
        ("max", 10),
        ("%>0.5m", 10),
        ("%>1.0m", 10),
        ("dx_mean", 10),
        ("dy_mean", 10),
        ("dz_mean", 10),
    ]

    def fnum(x, w):
        if x is None:
            return " " * w
        if isinstance(x, (int, float)):
            # integers for N, else 4 decimals
            if abs(x - int(x)) < 1e-9 and w >= 6:
                s = f"{int(x)}"
            else:
                s = f"{x:.4f}"
        else:
            s = str(x)
        return s[:w].ljust(w)

    lines = []
    header = " ".join([c[0].ljust(c[1]) for c in cols])
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = " ".join([fnum(r.get(name), w) for name, w in cols])
        lines.append(line)
    return "\n".join(lines)


def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ref_txt = Path(cfg["reference"]["path"])
    targets = cfg["targets"]

    t0 = float(cfg["time"]["outage_start"])
    dur = float(cfg["time"]["outage_duration"])
    buf = float(cfg["time"]["buffer"])
    tmin = t0 - buf
    tmax = t0 + dur + buf

    delim = cfg.get("io", {}).get("delim", ",")
    scale = float(cfg.get("las", {}).get("scale", 0.001))

    exact_time = bool(cfg.get("matching", {}).get("exact_time", True))
    time_atol = float(cfg.get("matching", {}).get("time_atol", 0.0))

    logging.info("Reference: %s", ref_txt)
    logging.info("Window: [%.3f , %.3f] (start=%.3f dur=%.1f buffer=%.1f)", tmin, tmax, t0, dur, buf)
    logging.info("LAS scale: %.6f m (%.1f mm)", scale, scale * 1000.0)

    # Read reference once
    tR, xyzR = read_window_txt(ref_txt, delim, tmin, tmax)
    logging.info("Reference points in window: %d", len(tR))

    summary_rows: List[Dict[str, object]] = []

    for i, tgt in enumerate(targets, start=1):
        tgt_path = Path(tgt["path"])
        out_las = Path(tgt["outfile"])

        logging.info("")
        logging.info("Target %d/%d: %s", i, len(targets), tgt_path)
        logging.info("Output LAS: %s", out_las)

        tT, xyzT = read_window_txt(tgt_path, delim, tmin, tmax)
        logging.info("Target points in window: %d", len(tT))

        validate_times(tR, tT, exact=exact_time, atol=time_atol)

        dxyz = xyzT - xyzR
        e3d = np.sqrt(np.sum(dxyz * dxyz, axis=1))

        metrics = compute_metrics(e3d, dxyz)
        write_ref_las_with_e3d(out_las, tR, xyzR, e3d, scale)

        logging.info("RMSE_3D: %.4f m | p95: %.4f m | max: %.4f m", metrics["RMSE"], metrics["p95"], metrics["max"])
        logging.info("Wrote: %s", out_las)

        summary_rows.append({
            "target": str(tgt_path),
            "outfile": str(out_las),
            **metrics,
        })

    logging.info("")
    logging.info("========== RMSE SUMMARY ==========")
    print(format_summary(summary_rows))
    logging.info("==================================")

    # (Optional) write a CSV summary next to the config, uncomment if you want
    out_csv = Path(args.config).with_suffix(".summary.csv")
    with out_csv.open("w", encoding="utf-8") as f:
        keys = list(summary_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    logging.info("Wrote summary CSV: %s", out_csv)


if __name__ == "__main__":
    main()