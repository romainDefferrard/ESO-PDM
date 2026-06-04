"""
gsd_analysis.py
===============
Measures GSD (nearest-neighbour distance) on multiple MLS point clouds.

- Processes all merged_*.las in a directory sequentially
- Each file is loaded, processed, then freed before the next (memory safe)
- Intermediate results saved to CSV after each file
- Final aggregation across all files

Usage:
    python gsd_analysis.py --dir /path/to/merged_dir [--every 1000] [--out results.csv]
    python gsd_analysis.py --las /path/to/single.las  [--every 1000]
"""

import argparse
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree


def now():
    return time.strftime("%H:%M:%S")

def print_step(msg):
    print(f"[{now()}] {msg}", flush=True)

def metrics(vals):
    if len(vals) == 0:
        return {"N": 0, "RMSE": np.nan, "Q50": np.nan, "Q90": np.nan, "STD": np.nan}
    return {
        "N":    len(vals),
        "RMSE": float(np.sqrt(np.mean(vals**2))),
        "Q50":  float(np.percentile(vals, 50)),
        "Q90":  float(np.percentile(vals, 90)),
        "STD":  float(np.std(vals)),
    }


def process_subset(xyz, range_vals, label, sample_idx, range_bins, filename):
    """
    Build KDTree, run NN query, compute metrics.
    Returns a list of dicts (one row per range bin + one global row).
    """
    rows = []

    if len(xyz) < 2:
        print_step(f"  [{label}] Not enough points — skip")
        return rows

    print_step(f"  [{label}] KDTree on {len(xyz):,} pts ...")
    t0 = time.time()
    tree = cKDTree(xyz)
    print_step(f"  [{label}] KDTree in {time.time()-t0:.1f}s")

    valid_idx = sample_idx[sample_idx < len(xyz)]
    if len(valid_idx) == 0:
        print_step(f"  [{label}] No sampled points — skip")
        return rows

    print_step(f"  [{label}] NN query on {len(valid_idx):,} pts ...")
    t0 = time.time()
    dists, _ = tree.query(xyz[valid_idx], k=2, workers=-1)
    nn_dists = dists[:, 1]
    range_q  = range_vals[valid_idx]
    print_step(f"  [{label}] NN query in {time.time()-t0:.1f}s")

    del tree  # free KDTree memory

    # Globale
    m = metrics(nn_dists)
    print(f"    [global] N={m['N']:,}  RMSE={m['RMSE']*100:.3f} cm  "
          f"Q50={m['Q50']*100:.3f} cm  Q90={m['Q90']*100:.3f} cm  "
          f"STD={m['STD']*100:.3f} cm")
    rows.append({"file": filename, "scanner": label, "range_bin": "global", **m})

    # Par tranche
    for r_lo, r_hi in range_bins:
        mask = (range_q >= r_lo) & (range_q < r_hi)
        m = metrics(nn_dists[mask])
        label_bin = f"{r_lo:.0f}-{r_hi:.0f}m"
        print(f"    [{label_bin}] N={m['N']:,}  RMSE={m['RMSE']*100:.3f} cm  "
              f"Q50={m['Q50']*100:.3f} cm  Q90={m['Q90']*100:.3f} cm  "
              f"STD={m['STD']*100:.3f} cm")
        rows.append({"file": filename, "scanner": label, "range_bin": label_bin, **m})

    return rows


def process_file(las_path, every, range_bins):
    """
    Load a LAS file, process the 3 subsets, free memory.
    Returns a list of result dicts.
    """
    print_step(f"{'='*60}")
    print_step(f"File: {las_path.name}  ({las_path.stat().st_size/1e9:.2f} GB)")

    t0 = time.time()
    las = laspy.read(las_path)
    n_total = len(las.x)
    print_step(f"Loaded: {n_total:,} pts in {time.time()-t0:.1f}s")

    # Extract fields
    print_step("Extracting xyz / lasvec / scanner_src ...")
    xyz = np.column_stack([
        np.asarray(las.x,           dtype=np.float64),
        np.asarray(las.y,           dtype=np.float64),
        np.asarray(las.z,           dtype=np.float64),
    ])
    lasvec = np.column_stack([
        np.asarray(las['lasvec_x'], dtype=np.float32),
        np.asarray(las['lasvec_y'], dtype=np.float32),
        np.asarray(las['lasvec_z'], dtype=np.float32),
    ])
    scanner_src = np.asarray(las['scanner_src'], dtype=np.uint8)
    del las  # free LAS from memory
    gc.collect()

    range_vals = np.linalg.norm(lasvec, axis=1).astype(np.float32)
    del lasvec
    print_step(f"Range: min={range_vals.min():.2f} m  max={range_vals.max():.2f} m  "
               f"mean={range_vals.mean():.2f} m")

    # Sampling
    sample_global = np.arange(0, n_total, every)
    print_step(f"Sampling: 1/{every} → {len(sample_global):,} pts")

    subsets = {
        "ALL":  np.ones(n_total, dtype=bool),
        "VUX":  scanner_src == 2,
        "PUCK": scanner_src == 1,
    }

    all_rows = []
    global_to_local = np.full(n_total, -1, dtype=np.int64)  # réutilisé par subset

    for label, mask_subset in subsets.items():
        idx_subset = np.where(mask_subset)[0]
        n_sub = len(idx_subset)
        print_step(f"  [{label}] {n_sub:,} pts ({100*n_sub/n_total:.1f}%)")

        if n_sub < 2:
            continue

        # Indices locaux des points échantillonnés dans ce sous-ensemble
        in_subset        = mask_subset[sample_global]
        sample_in_global = sample_global[in_subset]
        global_to_local[:] = -1
        global_to_local[idx_subset] = np.arange(n_sub)
        sample_local = global_to_local[sample_in_global]
        sample_local = sample_local[sample_local >= 0]

        rows = process_subset(
            xyz=xyz[idx_subset],
            range_vals=range_vals[idx_subset],
            label=label,
            sample_idx=sample_local,
            range_bins=range_bins,
            filename=las_path.name,
        )
        all_rows.extend(rows)

    del xyz, range_vals, scanner_src, global_to_local
    gc.collect()

    return all_rows


def print_aggregated(df, title="Aggregated results"):
    """Print aggregated metrics (N-weighted average) by scanner × range bin."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Scanner':8s}  {'Range bin':12s}  {'Files':>5}  {'N total':>10}  "
          f"{'RMSE(cm)':>9}  {'Q50(cm)':>8}  {'Q90(cm)':>8}  {'STD(cm)':>8}")
    print(f"  {'─'*80}")

    for (scanner, rbin), grp in df.groupby(["scanner", "range_bin"]):
        n_files = grp["file"].nunique()
        n_total = grp["N"].sum()
        # Moyenne pondérée par N
        w = grp["N"].values.astype(float)
        def wavg(col):
            vals = grp[col].values
            mask = np.isfinite(vals) & np.isfinite(w)
            if mask.sum() == 0: return np.nan
            return float(np.average(vals[mask], weights=w[mask]))

        rmse = wavg("RMSE")
        q50  = wavg("Q50")
        q90  = wavg("Q90")
        std  = wavg("STD")

        print(f"  {scanner:8s}  {rbin:12s}  {n_files:>5}  {n_total:>10,}  "
              f"{rmse*100:>9.3f}  {q50*100:>8.3f}  {q90*100:>8.3f}  {std*100:>8.3f}")

    print(f"{'='*70}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las",   help="Single LAS file")
    ap.add_argument("--dir",   help="Directory containing merged_*.las files")
    ap.add_argument("--glob",  default="merged_*.las",
                    help="Glob pattern (default: merged_*.las)")
    ap.add_argument("--every", type=int, default=1000,
                    help="Sample 1 point every N (default: 1000)")
    ap.add_argument("--out",   default="gsd_results.csv",
                    help="Output CSV (default: gsd_results.csv)")
    args = ap.parse_args()

    RANGE_BINS = [(0, 10), (10, 20), (20, 30)]

    # Liste des fichiers à traiter
    if args.las:
        files = [Path(args.las)]
    elif args.dir:
        files = sorted(Path(args.dir).glob(args.glob))
        if not files:
            print(f"No files found matching {args.glob} in {args.dir}")
            return
    else:
        print("Specify --las or --dir")
        return

    out_csv = Path(args.out)
    print_step(f"{len(files)} file(s) to process")
    print_step(f"Output CSV: {out_csv}")

    all_rows = []
    t_total = time.time()

    for i, las_path in enumerate(files):
        print_step(f"File {i+1}/{len(files)}: {las_path.name}")
        try:
            rows = process_file(las_path, args.every, RANGE_BINS)
            all_rows.extend(rows)

            # Sauvegarde incrémentale après chaque fichier
            df_tmp = pd.DataFrame(all_rows)
            df_tmp.to_csv(out_csv, index=False, float_format="%.6f")
            print_step(f"CSV updated ({len(df_tmp)} rows)")

        except Exception as e:
            print_step(f"  ERROR on {las_path.name}: {e}")
            continue

    if not all_rows:
        print_step("No results.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False, float_format="%.6f")

    print_step(f"Total processing time: {(time.time()-t_total)/60:.1f} min")
    print_step(f"Final CSV: {out_csv}  ({len(df)} rows)")

    print_aggregated(df, title=f"Aggregated results — {len(files)} files")


if __name__ == "__main__":
    main()