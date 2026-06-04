"""
steps/s2s_chunks.py
===================
Scan-to-Scan spatial chunking + LiMatch for forward/backward pass pairs.

Receives a pair of LAS files (forward and backward scan) produced by the
Patcher and splits them into spatially co-located chunks perpendicular to
the direction of travel.

Chunking strategy:
  - Trajectory is projected to local CRS (LV95/EPSG:2056 by default).
  - The FORWARD pass timestamps define the trajectory slice and spatial
    reference frame (origin + travel tangent + chunk edges).
  - Both fwd and bwd files are chunked on the SAME spatial grid so that
    matching chunk indices always cover the same ground region.
  - No timestamps are used for chunk assignment — only XY projection.

CLI usage
---------
    python -m pipeline.steps.s2s_chunks \\
        --pairs_dirs  /path/to/patcher_output \\
        --sbet        /path/to/ODyN.out \\
        --limatch_cfg /path/to/MLS.yml \\
        --out_root    /path/to/output \\
        --L           20.0
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import laspy
from pyproj import Transformer
from tqdm import tqdm


# === Trajectory helpers ========================================

def load_sbet(sbet_path: Path, epsg_out: str = "EPSG:2056"):
    """Load binary SBET (17 float64 cols), project to local CRS. Returns t,x,y sorted by time."""
    data = np.fromfile(sbet_path, dtype=np.float64).reshape(-1, 17)
    t   = data[:, 0]
    lat = np.degrees(data[:, 1])
    lon = np.degrees(data[:, 2])
    alt = data[:, 3]

    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x, y, _ = transformer.transform(lon, lat, alt)

    order = np.argsort(t)
    return t[order].copy(), x[order].copy(), y[order].copy()


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    return np.concatenate(([0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))))


def trajectory_slice_for_time(t_trj, x_trj, y_trj, t0, t1):
    """Return trajectory segment covering [t0, t1]."""
    mask = (t_trj >= t0) & (t_trj <= t1)
    if mask.sum() < 2:
        raise RuntimeError(f"Trajectory has <2 poses in [{t0:.1f}, {t1:.1f}]")
    return t_trj[mask], x_trj[mask], y_trj[mask]


# === Spatial chunking helpers ==================================

def build_chunk_edges_s(s_start: float, s_end: float,
                         L: float, min_last: float) -> np.ndarray:
    total = s_end - s_start
    if total <= 0:
        return np.array([s_start, s_end])
    if total <= L:
        return np.array([s_start, s_end])

    edges = list(np.arange(s_start, s_end, L))
    if edges[-1] != s_end:
        edges.append(s_end)

    if len(edges) >= 3 and (edges[-1] - edges[-2]) < min_last:
        edges.pop(-2)

    return np.array(edges, dtype=np.float64)


def compute_mean_tangent(x_trj: np.ndarray, y_trj: np.ndarray) -> np.ndarray:
    """Mean unit tangent = direction from first to last trajectory point."""
    dx = x_trj[-1] - x_trj[0]
    dy = y_trj[-1] - y_trj[0]
    norm = np.sqrt(dx*dx + dy*dy)
    if norm < 1e-6:
        raise RuntimeError("Trajectory segment too short or degenerate.")
    return np.array([dx / norm, dy / norm])


def assign_chunks_spatial(
    px: np.ndarray,
    py: np.ndarray,
    origin: np.ndarray,
    tangent: np.ndarray,
    s_edges: np.ndarray,
) -> np.ndarray:
    """
    Project each point onto the tangent axis and return its chunk index.
    Returns -1 for points outside the [s_edges[0], s_edges[-1]] range.
    """
    proj = (px - origin[0]) * tangent[0] + (py - origin[1]) * tangent[1]
    k = np.searchsorted(s_edges, proj, side="right") - 1
    k[(k < 0) | (k >= len(s_edges) - 1)] = -1
    return k


def las_time_bounds(las_path: Path, time_field: str = "gps_time",
                    chunk_size: int = 2_000_000) -> Tuple[float, float]:
    t_first = t_last = None
    with laspy.open(las_path) as r:
        for pts in r.chunk_iterator(chunk_size):
            if not len(pts):
                continue
            t = np.asarray(pts[time_field], dtype=np.float64)
            if t_first is None:
                t_first = float(t[0])
            t_last = float(t[-1])
    if t_first is None:
        raise RuntimeError(f"No timestamps in {las_path}")
    return t_first, t_last


# === Core: spatially chunk one LAS file ========================

def chunk_las_spatial(
    las_path: Path,
    out_dir: Path,
    origin: np.ndarray,
    tangent: np.ndarray,
    s_edges: np.ndarray,
    out_prefix: str = "chunk_",
    chunk_size: int = 2_000_000,
) -> List[int]:
    """
    Split las_path into spatial slabs perpendicular to `tangent`.
    Returns list of chunk indices that were written (all chunks kept).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = len(s_edges) - 1

    writers: dict[int, laspy.LasWriter] = {}
    counts:  dict[int, int] = {}

    with laspy.open(las_path) as reader:
        header_template = reader.header

        def open_chunk(k: int):
            if k in writers:
                return
            p = out_dir / f"{out_prefix}{k:04d}.las"
            hdr = copy.deepcopy(header_template)
            writers[k] = laspy.open(p, mode="w", header=hdr)
            counts[k] = 0

        def close_chunk(k: int):
            if k not in writers:
                return
            writers[k].close()
            del writers[k]

        with tqdm(total=reader.header.point_count, unit="pts",
                  desc=f"  chunk {las_path.name}", leave=False) as pbar:
            for points in reader.chunk_iterator(chunk_size):
                pbar.update(len(points))
                if not len(points):
                    continue

                px = np.asarray(points.x, dtype=np.float64)
                py = np.asarray(points.y, dtype=np.float64)

                k_arr = assign_chunks_spatial(px, py, origin, tangent, s_edges)

                valid = k_arr >= 0
                if not np.any(valid):
                    continue

                pts_v = points[valid]
                k_v   = k_arr[valid]

                for kk in np.unique(k_v):
                    sel = k_v == kk
                    open_chunk(int(kk))
                    writers[int(kk)].write_points(pts_v[sel])
                    counts[int(kk)] += int(sel.sum())

        for k in list(writers.keys()):
            close_chunk(k)

    kept = sorted(counts.keys())
    print(f"  → {len(kept)}/{n_chunks} chunks written in {out_dir}")
    return kept


# === Pair discovery ============================================

PAIR_RE = re.compile(r"Patch_from_scan_(\d+)_with_(\d+)\.las$", re.IGNORECASE)


def find_pairs(pair_dir: Path) -> List[Tuple[Path, Path]]:
    files = list(pair_dir.rglob("Patch_from_scan_*_with_*.las"))
    parsed = {}
    for f in files:
        m = PAIR_RE.match(f.name)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        parsed[(a, b)] = f

    pairs = []
    seen = set()
    for (a, b), fa in sorted(parsed.items()):
        if (a, b) in seen or (b, a) in seen:
            continue
        if (b, a) in parsed:
            fb = parsed[(b, a)]
            pairs.append((fa, fb) if a < b else (fb, fa))
            seen.add((a, b))
            seen.add((b, a))
    return pairs


# === Process one pair ==========================================

def process_pair(
    fwd_path: Path,
    bwd_path: Path,
    out_dir: Path,
    t_trj: np.ndarray,
    x_trj: np.ndarray,
    y_trj: np.ndarray,
    L: float = 20.0,
    min_last_chunk_m: float = 10.0,
    time_field: str = "gps_time",
    limatch_cfg_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    min_time_sep: float = 0.0,
):
    print(f"\n{'='*60}")
    print(f"[pair] fwd: {fwd_path.name}")
    print(f"[pair] bwd: {bwd_path.name}")
    print(f"[pair] out: {out_dir}")
    print(f"{'='*60}")

    # === 1. Read time bounds + temporal separation filter ========
    print("[pair] Reading fwd time bounds...")
    t0_fwd, t1_fwd = las_time_bounds(fwd_path, time_field)
    print(f"  fwd: [{t0_fwd:.3f}, {t1_fwd:.3f}]")

    if min_time_sep > 0.0:
        t0_bwd, t1_bwd = las_time_bounds(bwd_path, time_field)
        t_mean_fwd = 0.5 * (t0_fwd + t1_fwd)
        t_mean_bwd = 0.5 * (t0_bwd + t1_bwd)
        dt = abs(t_mean_fwd - t_mean_bwd)
        if dt < min_time_sep:
            print(f"[pair] SKIPPED — time sep {dt:.1f}s < min_time_sep {min_time_sep:.1f}s")
            return
        print(f"[pair] time sep: {dt:.1f}s >= {min_time_sep:.1f}s  OK")

    out_dir.mkdir(parents=True, exist_ok=True)

    _, x_f, y_f = trajectory_slice_for_time(t_trj, x_trj, y_trj, t0_fwd, t1_fwd)

    # === 2. Define the spatial grid from the forward trajectory ==
    tangent = compute_mean_tangent(x_f, y_f)
    origin  = np.array([x_f[0], y_f[0]])

    proj_f  = (x_f - origin[0]) * tangent[0] + (y_f - origin[1]) * tangent[1]
    s_start = float(proj_f.min())
    s_end   = float(proj_f.max())

    s_edges  = build_chunk_edges_s(s_start, s_end, L, min_last_chunk_m)
    n_chunks = len(s_edges) - 1

    print(f"[pair] tangent: ({tangent[0]:.4f}, {tangent[1]:.4f})")
    print(f"[pair] strip:   {s_end - s_start:.1f} m  →  {n_chunks} chunks × ~{L} m")

    # === 3. Spatial chunking — same grid for both files ==========
    fwd_out = out_dir / "fwd"
    bwd_out = out_dir / "bwd"

    print("[pair] Chunking FORWARD...")
    kept_fwd = chunk_las_spatial(
        las_path=fwd_path, out_dir=fwd_out,
        origin=origin, tangent=tangent, s_edges=s_edges,
    )

    print("[pair] Chunking BACKWARD...")
    kept_bwd = chunk_las_spatial(
        las_path=bwd_path, out_dir=bwd_out,
        origin=origin, tangent=tangent, s_edges=s_edges,
    )

    # === 4. LiMatch on matching chunk pairs ======================
    if limatch_cfg_path is None or repo_root is None:
        print("[pair] LiMatch skipped (no cfg provided)")
        return

    common = sorted(set(kept_fwd) & set(kept_bwd))
    print(f"\n[pair] Running LiMatch on {len(common)} chunk pairs...")

    import yaml

    limatch_parent = repo_root / "Patcher" / "submodules"
    if str(limatch_parent) not in sys.path:
        sys.path.insert(0, str(limatch_parent))
    from limatch.main import match_clouds

    lim_cfg_base = yaml.safe_load(open(str(limatch_cfg_path), "r"))

    for k in common:
        c_fwd = fwd_out / f"chunk_{k:04d}.las"
        c_bwd = bwd_out / f"chunk_{k:04d}.las"

        lm_out = out_dir / "limatch" / f"chunk_{k:04d}"
        lm_out.mkdir(parents=True, exist_ok=True)
        (lm_out / "plots").mkdir(exist_ok=True)
        (lm_out / "tiles").mkdir(exist_ok=True)
        (lm_out / "cor_outputs").mkdir(exist_ok=True)

        cfg = copy.deepcopy(lim_cfg_base)
        cfg["prj_folder"] = str(lm_out) + os.sep

        print(f"  [limatch] chunk_{k:04d}: {c_fwd.name} ↔ {c_bwd.name}")
        try:
            match_clouds(str(c_fwd), str(c_bwd), cfg)
        except Exception as e:
            print(f"  [limatch] chunk_{k:04d} FAILED: {e}")


# === Resume: run LiMatch on already-chunked directories ========

def resume_limatch_from_chunks(
    chunk_pair_dir: Path,
    limatch_cfg_path: Path,
    repo_root: Path,
):
    """
    Given a directory that already contains fwd/ and bwd/ sub-folders with
    chunk_XXXX.las files, run LiMatch on every matching pair that has not
    been processed yet (i.e. whose limatch/chunk_XXXX/ output folder does
    not already exist or is empty).
    """
    fwd_dir = chunk_pair_dir / "fwd"
    bwd_dir = chunk_pair_dir / "bwd"

    if not fwd_dir.is_dir() or not bwd_dir.is_dir():
        print(f"  [resume] SKIP — missing fwd/ or bwd/ in {chunk_pair_dir}")
        return

    CHUNK_RE = re.compile(r"chunk_(\d{4})\.las$", re.IGNORECASE)

    def index_set(d: Path) -> set:
        return {int(CHUNK_RE.match(f.name).group(1))
                for f in d.glob("chunk_*.las") if CHUNK_RE.match(f.name)}

    kept_fwd = index_set(fwd_dir)
    kept_bwd = index_set(bwd_dir)
    common   = sorted(kept_fwd & kept_bwd)

    print(f"\n{'='*60}")
    print(f"[resume] {chunk_pair_dir.name}")
    print(f"  fwd chunks: {len(kept_fwd)}  bwd chunks: {len(kept_bwd)}  common: {len(common)}")
    print(f"{'='*60}")

    if not common:
        print("  [resume] No common chunks — nothing to do.")
        return

    import yaml

    limatch_parent = repo_root / "Patcher" / "submodules"
    if str(limatch_parent) not in sys.path:
        sys.path.insert(0, str(limatch_parent))
    from limatch.main import match_clouds

    lim_cfg_base = yaml.safe_load(open(str(limatch_cfg_path), "r"))

    skipped = 0
    for k in common:
        lm_out = chunk_pair_dir / "limatch" / f"chunk_{k:04d}"
        if lm_out.exists() and any(lm_out.iterdir()):
            skipped += 1
            continue

        c_fwd = fwd_dir / f"chunk_{k:04d}.las"
        c_bwd = bwd_dir / f"chunk_{k:04d}.las"

        lm_out.mkdir(parents=True, exist_ok=True)
        (lm_out / "plots").mkdir(exist_ok=True)
        (lm_out / "tiles").mkdir(exist_ok=True)
        (lm_out / "cor_outputs").mkdir(exist_ok=True)

        cfg = copy.deepcopy(lim_cfg_base)
        cfg["prj_folder"] = str(lm_out) + os.sep

        print(f"  [limatch] chunk_{k:04d}: {c_fwd.name} ↔ {c_bwd.name}")
        try:
            match_clouds(str(c_fwd), str(c_bwd), cfg)
        except Exception as e:
            print(f"  [limatch] chunk_{k:04d} FAILED: {e}")

    if skipped:
        print(f"  [resume] {skipped}/{len(common)} chunks already done — skipped.")


# === Entry point ===============================================

def main():
    parser = argparse.ArgumentParser(
        description="S2S spatial chunking + LiMatch for forward/backward pass pairs"
    )
    parser.add_argument("--pairs_dirs", nargs="+", default=[],
                        help="Directories containing Patch_from_scan_*_with_*.las pairs")
    parser.add_argument("--sbet",      default=None,
                        help="SBET trajectory file (required with --pairs_dirs)")
    parser.add_argument("--out_root",  default=None,
                        help="Root output directory (required with --pairs_dirs)")
    parser.add_argument("--resume_dirs", nargs="+", default=[],
                        help="Already-chunked pair directories — run LiMatch only")
    parser.add_argument("--limatch_cfg", default=None)
    parser.add_argument("--L",           type=float, default=20.0,
                        help="Chunk length in metres")
    parser.add_argument("--min_last_m",  type=float, default=10.0)
    parser.add_argument("--epsg",        default="EPSG:2056")
    parser.add_argument("--time_field",  default="gps_time")
    parser.add_argument("--min_time_sep", type=float, default=0.0,
                        help="Min temporal separation (s) between fwd/bwd scan means")
    parser.add_argument("--repo_root",   default=None)
    args = parser.parse_args()

    if not args.pairs_dirs and not args.resume_dirs:
        parser.error("Provide at least --pairs_dirs or --resume_dirs")

    limatch_cfg_path = Path(args.limatch_cfg) if args.limatch_cfg else None
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[1]

    # === Resume mode =============================================
    if args.resume_dirs:
        if limatch_cfg_path is None:
            parser.error("--limatch_cfg is required with --resume_dirs")
        for rd_str in args.resume_dirs:
            resume_limatch_from_chunks(
                chunk_pair_dir=Path(rd_str),
                limatch_cfg_path=limatch_cfg_path,
                repo_root=repo_root,
            )

    # === Normal mode =============================================
    if args.pairs_dirs:
        if args.sbet is None:
            parser.error("--sbet is required with --pairs_dirs")
        if args.out_root is None:
            parser.error("--out_root is required with --pairs_dirs")

        print(f"[main] Loading trajectory: {args.sbet}")
        t_trj, x_trj, y_trj = load_sbet(Path(args.sbet), epsg_out=args.epsg)
        print(f"[main] {len(t_trj)} poses, t=[{t_trj[0]:.1f}, {t_trj[-1]:.1f}]")

        out_root = Path(args.out_root)

        for pairs_dir_str in args.pairs_dirs:
            pairs_dir = Path(pairs_dir_str)
            print(f"\n[main] Scanning: {pairs_dir}")
            pairs = find_pairs(pairs_dir)
            if not pairs:
                print(f"  [warn] No pairs found in {pairs_dir}")
                continue
            print(f"  Found {len(pairs)} pair(s)")

            for fwd_path, bwd_path in pairs:
                rel = fwd_path.parent.relative_to(pairs_dir.parent) \
                      if fwd_path.parent != pairs_dir else Path(pairs_dir.name)
                pair_name = f"{fwd_path.stem}_vs_{bwd_path.stem}"
                out_dir   = out_root / rel / pair_name

                process_pair(
                    fwd_path=fwd_path,
                    bwd_path=bwd_path,
                    out_dir=out_dir,
                    t_trj=t_trj,
                    x_trj=x_trj,
                    y_trj=y_trj,
                    L=args.L,
                    min_last_chunk_m=args.min_last_m,
                    time_field=args.time_field,
                    limatch_cfg_path=limatch_cfg_path,
                    repo_root=repo_root,
                    min_time_sep=args.min_time_sep,
                )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()