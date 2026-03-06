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
    rows_t = []
    rows_xyz = []

    file_size = os.path.getsize(path)
    acc = 0

    with path.open("r", encoding="utf-8") as f, tqdm(
        total=file_size, unit="B", unit_scale=True, desc=f"Reading {path.name}"
    ) as pbar:
        for ln in f:
            acc += len(ln)
            if acc >= 1_000_000:
                pbar.update(acc)
                acc = 0

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

            rows_t.append(t)
            rows_xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))

        if acc > 0:
            pbar.update(acc)

    if not rows_t:
        raise RuntimeError(f"{path}: no points found in window [{tmin}, {tmax}]")

    return np.asarray(rows_t, dtype=np.float64), np.asarray(rows_xyz, dtype=np.float64)

# -------------------------------------------------------------
# REMOVE DUPLICATED TIMES AND KEEP ONLY MUTUAL UNIQUE TIMES
# -------------------------------------------------------------
def keep_only_mutual_unique_times(
    t_ref,
    xyz_ref,
    t_tgt,
    xyz_tgt,
):

    u_ref, c_ref = np.unique(t_ref, return_counts=True)
    u_tgt, c_tgt = np.unique(t_tgt, return_counts=True)

    unique_ref = set(u_ref[c_ref == 1])
    unique_tgt = set(u_tgt[c_tgt == 1])

    keep_times = unique_ref & unique_tgt

    mask_ref = np.isin(t_ref, list(keep_times))
    mask_tgt = np.isin(t_tgt, list(keep_times))

    stats = {
        "ref_raw": len(t_ref),
        "tgt_raw": len(t_tgt),
        "ref_removed": int((~mask_ref).sum()),
        "tgt_removed": int((~mask_tgt).sum()),
        "kept_times": len(keep_times),
    }

    return (
        t_ref[mask_ref],
        xyz_ref[mask_ref],
        t_tgt[mask_tgt],
        xyz_tgt[mask_tgt],
        stats,
    )


def match_by_time(t_ref, xyz_ref, t_tgt, xyz_tgt):

    i = 0
    j = 0

    idx_ref = []
    idx_tgt = []

    n_ref = len(t_ref)
    n_tgt = len(t_tgt)

    while i < n_ref and j < n_tgt:

        tr = t_ref[i]
        tt = t_tgt[j]

        if tr == tt:

            idx_ref.append(i)
            idx_tgt.append(j)

            i += 1
            j += 1

        elif tr < tt:
            i += 1

        else:
            j += 1

    idx_ref = np.asarray(idx_ref)
    idx_tgt = np.asarray(idx_tgt)

    return t_ref[idx_ref], xyz_ref[idx_ref], xyz_tgt[idx_tgt]


def compute_metrics(dxyz: np.ndarray) -> Dict[str, float]:

    e3d = np.linalg.norm(dxyz, axis=1)

    rmse = float(np.sqrt(np.mean(e3d**2)))
    mean = float(np.mean(e3d))
    med = float(np.median(e3d))
    p95 = float(np.percentile(e3d, 95))
    p99 = float(np.percentile(e3d, 99))
    mx = float(np.max(e3d))

    thr_01 = float(np.mean(e3d > 0.1)) * 100.0
    thr_05 = float(np.mean(e3d > 0.5)) * 100.0
    thr_10 = float(np.mean(e3d > 1.0)) * 100.0

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
    }, e3d


def write_ref_las_with_e3d(out_las: Path, t, xyz_ref, e3d, scale):

    hdr = laspy.LasHeader(point_format=6, version="1.4")

    hdr.scales = np.array([scale, scale, scale])
    hdr.offsets = np.array(
        [
            float(xyz_ref[:, 0].min()),
            float(xyz_ref[:, 1].min()),
            float(xyz_ref[:, 2].min()),
        ]
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


def main():

    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

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

    logging.info("Reference: %s", ref_txt)
    logging.info("Window: [%.3f , %.3f]", tmin, tmax)

    tR, xyzR = read_window_txt(ref_txt, delim, tmin, tmax)

    logging.info("Reference points in window: %d", len(tR))

    for i, tgt in enumerate(targets, start=1):

        tgt_path = Path(tgt["path"])
        out_las = Path(tgt["outfile"]).with_suffix(".las")

        logging.info("")
        logging.info("Target %d/%d: %s", i, len(targets), tgt_path)

        tT, xyzT = read_window_txt(tgt_path, delim, tmin, tmax)

        logging.info("Target points in window: %d", len(tT))

        # -----------------------------------------
        # REMOVE DUPLICATE TIMES
        # -----------------------------------------

        tR_f, xyzR_f, tT_f, xyzT_f, stats = keep_only_mutual_unique_times(
            tR, xyzR, tT, xyzT
        )

        logging.info(
            "Removed ambiguous timestamps: ref_removed=%d tgt_removed=%d",
            stats["ref_removed"],
            stats["tgt_removed"],
        )

        # -----------------------------------------
        # MATCH
        # -----------------------------------------

        tM, xyzR_m, xyzT_m = match_by_time(
            tR_f,
            xyzR_f,
            tT_f,
            xyzT_f,
        )

        logging.info("Matched points: %d", len(tM))

        dxyz = xyzT_m - xyzR_m

        metrics, e3d = compute_metrics(dxyz)

        write_ref_las_with_e3d(out_las, tM, xyzR_m, e3d, scale)

        logging.info(
            "RMSE=%.4f  p95=%.4f  max=%.4f",
            metrics["RMSE"],
            metrics["p95"],
            metrics["max"],
        )

        logging.info("Wrote LAS: %s", out_las)


if __name__ == "__main__":
    main()