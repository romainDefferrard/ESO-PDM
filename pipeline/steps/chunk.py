"""
steps/chunk.py
==============
Chunking step — replaces Chunker.py entirely.

Absorbs:
  - Chunker.py  (all functions: chunk_las_by_distance_streaming_intervals,
                 chunk_txt_by_distance_streaming_intervals, build_chunk_edges_s,
                 write_gps_multi_outage, merge_intervals, file_time_bounds_fast,
                 file_time_bounds_fast_las, build_trajectory_bbox, etc.)
  - pipeline.py (combined_multi_outage_scenario logic)

Design changes vs old code
--------------------------
- chunk_variant removed: chunks always around the outage window.
- outage is pipe_cfg["outage"] = [t_start, duration_s].
- margin_s from georef_time_window.margin_s is reused as pre/post buffer.
- min_last_chunk_m = 2/3 * length_m (enforced here, not in config).
- min_points removed from config (hardcoded to 0 — keep all chunks).
- cloud_fmt auto-detected from merged_dir.
- delimiter, skiprows, time_field hardcoded (CSV col0 = GPS time).

Public API
----------
run(pipe_cfg, merged_dir, cfg_georef_path) -> Path
    Returns chunks_root directory.

Utility functions also exported for standalone use:
  write_gps_multi_outage, merge_intervals, file_time_bounds_fast,
  file_time_bounds_fast_las, extract_scan_id, build_trajectory_bbox
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional, Union

import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline._log import info, sub, warn
from pipeline.steps.georef import load_config, load_and_prepare_trajectory, trajectory_positions_mapping


# ═══════════════════════════════════════════════════════════════
# SCAN ID HELPERS
# ═══════════════════════════════════════════════════════════════

MERGED_SCAN_RE = re.compile(r"merged_(\d+)")

def extract_scan_id(p: Path) -> int:
    m = MERGED_SCAN_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse scan id from: {p.name}")
    return int(m.group(1))


# ═══════════════════════════════════════════════════════════════
# TIME BOUNDS (fast, head+tail only)
# ═══════════════════════════════════════════════════════════════

def _parse_time_from_line(line: str, delimiter: str) -> float:
    parts = line.strip().split() if delimiter == "whitespace" else line.strip().split(delimiter)
    return float(parts[0])


def file_time_bounds_fast(txt_path, delimiter: str = ",", max_head_lines: int = 5000):
    txt_path = Path(txt_path)
    t_first  = None
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_head_lines: break
            try:
                t_first = _parse_time_from_line(line, delimiter); break
            except Exception:
                continue
    if t_first is None:
        raise RuntimeError(f"No valid time at start of {txt_path}")

    block_size = 64 * 1024
    with txt_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        start = max(0, f.tell() - block_size)
        f.seek(start)
        data = f.read().decode("utf-8", errors="ignore")

    t_last = None
    for line in reversed(data.splitlines()):
        try:
            t_last = _parse_time_from_line(line, delimiter); break
        except Exception:
            continue

    if t_last is None:
        raise RuntimeError(f"No valid time at end of {txt_path}")

    return float(t_first), float(t_last)


def file_time_bounds_fast_las(las_path, time_field: str = "gps_time",
                               chunk_size: int = 2_000_000):
    t_min, t_max = np.inf, -np.inf
    with laspy.open(las_path) as r:
        for pts in r.chunk_iterator(chunk_size):
            t = np.asarray(pts[time_field], dtype=np.float64)
            if t.size:
                t_min = min(t_min, float(t.min()))
                t_max = max(t_max, float(t.max()))
    if t_min == np.inf:
        raise RuntimeError(f"No valid {time_field} in {las_path}")
    return t_min, t_max


# ═══════════════════════════════════════════════════════════════
# INTERVAL UTILITIES
# ═══════════════════════════════════════════════════════════════

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals: return []
    intervals = sorted((float(a), float(b)) for a, b in intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        out[-1] = (la, max(lb, b)) if a <= lb else out.append((a, b)) or out[-1]
    return out


def write_gps_multi_outage(
    gps_in:    Union[str, Path],
    gps_out:   Union[str, Path],
    outages:   list[tuple[float, float]],
    delimiter: str = ",",
) -> tuple[int, int]:
    """Remove GNSS samples during outage windows. Returns (kept, removed)."""
    gps_in, gps_out = Path(gps_in), Path(gps_out)
    gps_out.parent.mkdir(parents=True, exist_ok=True)
    intervals = sorted((float(s), float(s) + float(d)) for s, d in outages)
    kept = removed = j = 0
    with gps_in.open("r", encoding="utf-8", errors="ignore") as fin, \
         gps_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s: continue
            parts = s.split(delimiter) if delimiter != "whitespace" else s.split()
            try:
                t = float(parts[0])
            except Exception:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            while j < len(intervals) and t > intervals[j][1]:
                j += 1
            if j < len(intervals) and intervals[j][0] <= t <= intervals[j][1]:
                removed += 1
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    return kept, removed


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY → CURVILINEAR DISTANCE
# ═══════════════════════════════════════════════════════════════

def _cumulative_distance_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x); dy = np.diff(y)
    return np.concatenate(([0.0], np.cumsum(np.sqrt(dx*dx + dy*dy))))


def _build_s_of_t(cfg_georef_path: Union[str, Path], epsg_out: str = "EPSG:2056"):
    cfg = load_config(cfg_georef_path)
    trj = load_and_prepare_trajectory(cfg)
    P   = trajectory_positions_mapping(trj, epsg_out=epsg_out)
    t   = np.asarray(trj.time, dtype=np.float64)
    order = np.argsort(t)
    t, P  = t[order], P[order]
    s     = _cumulative_distance_xy(P[:, 0], P[:, 1])
    return t, s


def _load_traj_xy(cfg_georef_path: Union[str, Path], epsg_out: str = "EPSG:2056"):
    from pyproj import Transformer
    cfg  = load_config(cfg_georef_path)
    data = np.fromfile(cfg["trj"]["path"], dtype=np.float64).reshape(-1, 17)
    t    = data[:, 0]
    lat  = np.degrees(data[:, 1]); lon = np.degrees(data[:, 2]); alt = data[:, 3]
    tr   = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x, y, _ = tr.transform(lon, lat, alt)
    order = np.argsort(t)
    return t[order], x[order], y[order]


# ═══════════════════════════════════════════════════════════════
# CHUNK EDGE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_chunk_edges_s(s_start: float, s_end: float,
                         chunk_length: float, min_last_chunk: float) -> np.ndarray:
    total = s_end - s_start
    if total <= 0 or total <= chunk_length:
        return np.array([s_start, s_end], dtype=np.float64)
    edges = list(np.arange(s_start, s_end, chunk_length))
    if edges[-1] != s_end:
        edges.append(s_end)
    if len(edges) >= 3 and (edges[-1] - edges[-2]) < min_last_chunk:
        edges.pop(-2)
    return np.array(edges, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# BBOX INDEX
# ═══════════════════════════════════════════════════════════════

def _write_bbox_for_dir(
    out_dir:   Path,
    out_prefix: str = "chunk_",
    time_field: str = "gps_time",
    t_trj: Optional[np.ndarray] = None,
    x_trj: Optional[np.ndarray] = None,
    y_trj: Optional[np.ndarray] = None,
) -> None:
    use_traj = t_trj is not None and x_trj is not None and y_trj is not None

    bbox_rows = []
    for chunk_file in sorted(out_dir.glob(f"{out_prefix}*.las")):
        try:
            all_t = []
            with laspy.open(chunk_file) as r:
                for pts in r.chunk_iterator(500_000):
                    if len(pts):
                        all_t.append(np.asarray(pts[time_field], dtype=np.float64))
            if not all_t: continue
            t_all   = np.concatenate(all_t)
            t_start = float(t_all.min()); t_end = float(t_all.max())

            if use_traj:
                order   = np.argsort(t_trj)
                t_q     = np.clip(np.linspace(t_start, t_end, 20), t_trj[order[0]], t_trj[order[-1]])
                x_v     = np.interp(t_q, t_trj[order], x_trj[order])
                y_v     = np.interp(t_q, t_trj[order], y_trj[order])
                x_min, x_max = float(x_v.min()), float(x_v.max())
                y_min, y_max = float(y_v.min()), float(y_v.max())
            else:
                all_x, all_y = [], []
                with laspy.open(chunk_file) as r:
                    for pts in r.chunk_iterator(500_000):
                        if len(pts):
                            all_x.append(np.asarray(pts.x, dtype=np.float64))
                            all_y.append(np.asarray(pts.y, dtype=np.float64))
                xa = np.concatenate(all_x); ya = np.concatenate(all_y)
                x_min, x_max = float(xa.min()), float(xa.max())
                y_min, y_max = float(ya.min()), float(ya.max())

            bbox_rows.append({"chunk_file": chunk_file.name,
                               "t_start": t_start, "t_end": t_end,
                               "x_min": x_min, "x_max": x_max,
                               "y_min": y_min, "y_max": y_max})
        except Exception as e:
            warn(f"[bbox] {chunk_file.name}: {e}")

    if bbox_rows:
        pd.DataFrame(bbox_rows).sort_values("t_start") \
          .to_csv(out_dir / "chunk_bbox.csv", index=False)


def build_trajectory_bbox(
    chunks_root: Union[str, Path],
    t_trj: np.ndarray, x_trj: np.ndarray, y_trj: np.ndarray,
    time_field: str = "gps_time", out_prefix: str = "chunk_",
) -> None:
    chunks_root = Path(chunks_root)
    for scan_dir in sorted(d for d in chunks_root.iterdir() if d.is_dir()):
        if any(scan_dir.glob(f"{out_prefix}*.las")):
            _write_bbox_for_dir(scan_dir, out_prefix=out_prefix,
                                 time_field=time_field,
                                 t_trj=t_trj, x_trj=x_trj, y_trj=y_trj)


# ═══════════════════════════════════════════════════════════════
# LAS CHUNKER
# ═══════════════════════════════════════════════════════════════

def chunk_las_by_distance_streaming_intervals(
    las_path:        Union[str, Path],
    cfg_georef_path: Union[str, Path],
    out_dir:         Union[str, Path],
    L:               float,
    intervals:       list[tuple[float, float]],
    epsg_out:        str = "EPSG:2056",
    min_points:      int = 0,
    out_prefix:      str = "chunk_",
    time_field:      str = "gps_time",
    chunk_size:      int = 2_000_000,
    min_last_chunk_m: float = 10.0,
    t_trj: Optional[np.ndarray] = None,
    x_trj: Optional[np.ndarray] = None,
    y_trj: Optional[np.ndarray] = None,
) -> int:
    import copy
    las_path = Path(las_path); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not intervals: return 0

    intervals     = sorted((float(a), float(b)) for a, b in intervals)
    first_start   = intervals[0][0]; last_end = intervals[-1][1]
    base_scan     = extract_scan_id(las_path)
    tmin_pts, tmax_pts = file_time_bounds_fast_las(las_path, time_field=time_field)

    # build curvilinear distance from trajectory
    if t_trj is None:
        t_trj2, s_trj = _build_s_of_t(cfg_georef_path, epsg_out)
    else:
        dx = np.diff(x_trj); dy = np.diff(y_trj)
        s_trj  = np.concatenate(([0.0], np.cumsum(np.sqrt(dx*dx + dy*dy))))
        t_trj2 = t_trj

    mask  = (t_trj2 >= tmin_pts) & (t_trj2 <= tmax_pts)
    t_m   = t_trj2[mask]
    x_m   = x_trj[mask] if x_trj is not None else None
    y_m   = y_trj[mask] if y_trj is not None else None

    dx2 = np.diff(x_m if x_m is not None else np.zeros_like(t_m))
    dy2 = np.diff(y_m if y_m is not None else np.zeros_like(t_m))
    s_m = np.concatenate(([0.0], np.cumsum(np.sqrt(dx2*dx2 + dy2*dy2))))

    s_start = float(np.interp(first_start, t_m, s_m))
    s_end   = float(np.interp(last_end,   t_m, s_m))
    s_edges = build_chunk_edges_s(s_start, s_end, L, min_last_chunk_m)
    t_edges = np.interp(s_edges, s_m, t_m)

    writers: dict[int, laspy.LasWriter] = {}
    counts:  dict[int, int]             = {}
    header_template = None

    def open_chunk(k):
        if k in writers: return
        chunk_id = base_scan + k
        p = out_dir / f"{out_prefix}{chunk_id:04d}.las"
        hdr = copy.deepcopy(header_template)
        writers[k] = laspy.open(p, mode="w", header=hdr)
        counts[k]  = 0

    def close_chunk(k):
        if k not in writers: return
        writers[k].close(); del writers[k]
        p = out_dir / f"{out_prefix}{(base_scan + k):04d}.las"
        if counts.get(k, 0) < min_points:
            p.unlink(missing_ok=True)

    with laspy.open(las_path) as reader:
        header_template = reader.header
        with tqdm(total=reader.header.point_count, unit="pts",
                  desc=f"Chunk {las_path.name}", leave=False) as pbar:
            for points in reader.chunk_iterator(chunk_size):
                pbar.update(len(points))
                t = np.asarray(points[time_field], dtype=np.float64)
                mask = (t >= first_start) & (t <= last_end)
                if not np.any(mask): continue
                pts, t = points[mask], t[mask]
                k = np.searchsorted(t_edges, t, side="right") - 1
                valid = (k >= 0) & (k < len(t_edges) - 1)
                pts, k = pts[valid], k[valid]
                if not len(pts): continue
                for kk in np.unique(k):
                    sel = (k == kk)
                    open_chunk(int(kk))
                    writers[int(kk)].write_points(pts[sel])
                    counts[int(kk)] += int(sel.sum())

    for k in list(writers): close_chunk(k)

    _write_bbox_for_dir(out_dir, out_prefix=out_prefix, time_field=time_field,
                         t_trj=t_trj, x_trj=x_trj, y_trj=y_trj)
    return len(counts)


# ═══════════════════════════════════════════════════════════════
# TXT CHUNKER
# ═══════════════════════════════════════════════════════════════

def chunk_txt_by_distance_streaming_intervals(
    txt_path:        Union[str, Path],
    cfg_georef_path: Union[str, Path],
    out_dir:         Union[str, Path],
    L:               float,
    intervals:       list[tuple[float, float]],
    epsg_out:        str = "EPSG:2056",
    delimiter:       str = ",",
    skiprows:        int = 0,
    min_points:      int = 0,
    out_prefix:      str = "chunk_",
) -> int:
    txt_path = Path(txt_path); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not intervals: return 0

    intervals   = sorted((float(a), float(b)) for a, b in intervals)
    last_end    = intervals[-1][1]; first_start = intervals[0][0]
    base_scan   = extract_scan_id(txt_path)
    tmin_pts, tmax_pts = file_time_bounds_fast(txt_path, delimiter=delimiter)

    t_trj, s_trj = _build_s_of_t(cfg_georef_path, epsg_out)
    keep  = (t_trj >= tmin_pts) & (t_trj <= tmax_pts)
    if keep.sum() < 2: raise RuntimeError("Trajectory range does not overlap TXT time range.")
    t_trj, s_trj = t_trj[keep], s_trj[keep]

    s_start = float(np.interp(first_start, t_trj, s_trj))
    s_end   = float(np.interp(last_end,    t_trj, s_trj))
    starts  = np.arange(s_start, s_end, L)
    t_edges = np.interp(np.clip(starts, s_start, s_end), s_trj, t_trj)
    t_last  = float(np.interp(min(s_end, starts[-1] + L), s_trj, t_trj))
    t_edges = np.concatenate([t_edges, [t_last]])

    writers: dict[int, object] = {}
    counts:  dict[int, int]    = {}

    def open_chunk(k):
        if k in writers: return
        p = out_dir / f"{out_prefix}{(base_scan + k):04d}.txt"
        writers[k] = p.open("w", encoding="utf-8"); counts[k] = 0

    def close_chunk(k):
        if k not in writers: return
        writers[k].close(); del writers[k]
        if counts.get(k, 0) < min_points:
            (out_dir / f"{out_prefix}{(base_scan + k):04d}.txt").unlink(missing_ok=True)

    chunk_idx = 0
    while chunk_idx + 1 < len(t_edges) and t_edges[chunk_idx + 1] <= intervals[0][0]:
        chunk_idx += 1

    total_bytes = txt_path.stat().st_size
    j = 0
    open_chunk(chunk_idx)

    with txt_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         tqdm(total=total_bytes, unit="B", unit_scale=True,
              desc=f"Chunk {txt_path.name}", leave=False) as pbar:
        for _ in range(skiprows): fin.readline()
        while True:
            pos0 = fin.tell(); line = fin.readline()
            if not line: break
            pbar.update(fin.tell() - pos0)
            s = line.strip()
            if not s: continue
            parts = s.split(delimiter) if delimiter != "whitespace" else s.split()
            try: t = float(parts[0])
            except Exception: continue
            if t > last_end: break
            while j < len(intervals) and t > intervals[j][1]: j += 1
            if j >= len(intervals): break
            if not (intervals[j][0] <= t <= intervals[j][1]): continue
            while chunk_idx + 1 < len(t_edges) and t >= t_edges[chunk_idx + 1]:
                close_chunk(chunk_idx); chunk_idx += 1; open_chunk(chunk_idx)
            if chunk_idx >= len(t_edges) - 1: break
            if t_edges[chunk_idx] <= t < t_edges[chunk_idx + 1]:
                writers[chunk_idx].write(line); counts[chunk_idx] += 1

    for k in list(writers): close_chunk(k)
    return len(counts)


# ═══════════════════════════════════════════════════════════════
# FILE SELECTION BY WINDOW
# ═══════════════════════════════════════════════════════════════

def _detect_fmt(folder: Path) -> str:
    for ext in ("las", "laz", "txt"):
        if any(folder.glob(f"*.{ext}")): return ext
    raise FileNotFoundError(f"No .las/.laz/.txt in {folder}")


def _select_files_by_window(
    merged_dir: Path, cloud_fmt: str, t_lo: float, t_hi: float,
) -> list[Path]:
    manifest = merged_dir / "merged_manifest.csv"
    if manifest.exists():
        rows = []
        with manifest.open("r") as f:
            for r in csv.DictReader(f):
                rows.append({"t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
                              "path": merged_dir / r["filename"]})
        rows.sort(key=lambda r: r["t_start"])
        selected = [r["path"] for r in rows if r["t_end"] >= t_lo and r["t_start"] <= t_hi]
        info(f"[chunk] manifest: {len(selected)} file(s) in window")
        return selected

    files = sorted(merged_dir.glob(f"*.{cloud_fmt}"), key=extract_scan_id)
    selected = []
    for f in files:
        try:
            t0, t1 = (file_time_bounds_fast(f) if cloud_fmt == "txt"
                      else file_time_bounds_fast_las(f))
            if t1 >= t_lo and t0 <= t_hi:
                selected.append(f)
        except Exception as e:
            warn(f"[chunk] {f.name}: {e} — including anyway")
            selected.append(f)
    info(f"[chunk] fallback: {len(selected)} file(s) in window")
    return selected


# ═══════════════════════════════════════════════════════════════
# PIPELINE-FACING API
# ═══════════════════════════════════════════════════════════════

def run(
    pipe_cfg:        dict,
    merged_dir:      Union[str, Path],
    cfg_georef_path: Union[str, Path],
) -> Path:
    """
    Generate chunks around the outage window.
    Returns chunks_root Path.
    """
    merged_dir      = Path(merged_dir)
    cfg_georef_path = Path(cfg_georef_path)

    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]
    chunk_cfg     = pipe_cfg.get("chunk", {})

    # ── Outage window ──────────────────────────────────────────
    outage = pipe_cfg.get("outage")
    if outage is None:
        raise ValueError("pipe_cfg['outage'] = [t_start, duration_s] is required.")
    t_start  = float(outage[0]); duration = float(outage[1])
    margin_s = float(pipe_cfg.get("georef_time_window", {}).get("margin_s", 30.0))
    t_lo     = t_start - margin_s
    t_hi     = t_start + duration + margin_s
    intervals = [(t_lo, t_hi)]
    info(f"[chunk] window: [{t_lo:.1f}, {t_hi:.1f}] s  ({duration:.0f}s ± {margin_s:.0f}s)")

    # ── Parameters ─────────────────────────────────────────────
    chunk_source = chunk_cfg.get("source", "generate")
    L            = float(chunk_cfg.get("length_m", 15.0))
    min_last_m   = round(L * 2.0 / 3.0, 2)   # 2/3 rule, not in config
    epsg_out     = chunk_cfg.get("epsg_out", "EPSG:2056")

    out_root    = chunk_cfg.get("output_root") or root_out_dir / scenario_name / "scenario_combined"
    chunks_root = Path(out_root) / f"chunks_{int(L)}m"
    chunks_root.mkdir(parents=True, exist_ok=True)

    if chunk_source == "existing":
        existing = Path(chunk_cfg.get("existing_root", chunks_root))
        if not existing.exists():
            raise FileNotFoundError(f"chunk.source='existing' but not found: {existing}")
        info(f"[chunk] reusing existing → {existing}")
        return existing

    cloud_fmt = _detect_fmt(merged_dir)
    sub(f"cloud format: {cloud_fmt}")

    selected = _select_files_by_window(merged_dir, cloud_fmt, t_lo, t_hi)
    if not selected:
        raise FileNotFoundError(f"No merged files in window [{t_lo:.1f}, {t_hi:.1f}]")

    # Load trajectory once for bbox
    t_trj = x_trj = y_trj = None
    try:
        t_trj, x_trj, y_trj = _load_traj_xy(cfg_georef_path, epsg_out)
        sub(f"trajectory: {len(t_trj)} poses")
    except Exception as e:
        warn(f"[chunk] trajectory load failed: {e} — bbox disabled")

    for f in selected:
        out_sub = chunks_root / f.stem
        out_sub.mkdir(parents=True, exist_ok=True)
        sub(f"chunk {L:.1f}m: {f.name}")
        if cloud_fmt == "txt":
            chunk_txt_by_distance_streaming_intervals(
                txt_path=f, cfg_georef_path=cfg_georef_path,
                out_dir=out_sub, L=L, intervals=intervals,
                epsg_out=epsg_out, delimiter=",", skiprows=0, min_points=0,
            )
        else:
            chunk_las_by_distance_streaming_intervals(
                las_path=f, cfg_georef_path=cfg_georef_path,
                out_dir=out_sub, L=L, intervals=intervals,
                epsg_out=epsg_out, min_points=0,
                time_field="gps_time", min_last_chunk_m=min_last_m,
                t_trj=t_trj, x_trj=x_trj, y_trj=y_trj,
            )

    info(f"[chunk] done — {len(selected)} file(s) → {chunks_root}")
    return chunks_root
