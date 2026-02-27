import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
from tqdm import tqdm

import numpy as np

from .pointCloudGeoref import load_config, load_and_prepare_trajectory, trajectory_positions_mapping

MERGED_SCAN_RE = re.compile(r"merged_(\d+)")  # merged_0100...


def extract_scan_id(p: Path) -> int:
    m = MERGED_SCAN_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot read scan id from merged filename: {p.name}")
    return int(m.group(1))


# ============================
# Distance <-> time helpers (trajectory)
# ============================

def cumulative_distance_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)
    return np.concatenate(([0.0], np.cumsum(ds)))


def invert_s_to_t(s: np.ndarray, t: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    return np.interp(s_query, s, t)


def build_s_of_t(cfg: Dict[str, Any], epsg_out: str = "EPSG:2056"):
    trj = load_and_prepare_trajectory(cfg)
    P = trajectory_positions_mapping(trj, epsg_out=epsg_out)  # Nx3 in map
    t = np.asarray(trj.time, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    P = P[order]

    s = cumulative_distance_xy(P[:, 0], P[:, 1])
    return t, s

def _parse_time_from_line(line: str, delimiter: str):
    """
    Extract GPS time from a line.
    Assumes time is column 0.
    """
    line = line.strip()
    if not line:
        raise ValueError("empty")

    parts = line.split() if delimiter == "whitespace" else line.split(delimiter)
    return float(parts[0])


def file_time_bounds_fast(
    txt_path,
    delimiter: str = ",",
    max_head_lines: int = 5000,
):
    """
    FAST estimation of GPS time range of a large TXT point cloud.

    Returns:
        (t_first, t_last)

    Reads only:
        - first valid lines
        - last ~64KB of file

    Does NOT load full file.
    """

    txt_path = Path(txt_path)

    # =====================
    # 1) FIRST VALID TIME
    # =====================
    t_first = None

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_head_lines:
                break
            try:
                t_first = _parse_time_from_line(line, delimiter)
                break
            except Exception:
                continue

    if t_first is None:
        raise RuntimeError(f"No valid time found at beginning of {txt_path}")

    # =====================
    # 2) LAST VALID TIME
    # =====================
    t_last = None

    block_size = 64 * 1024  # 64 KB

    with txt_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        filesize = f.tell()

        start = max(0, filesize - block_size)
        f.seek(start)

        data = f.read().decode("utf-8", errors="ignore")

    for line in reversed(data.splitlines()):
        try:
            t_last = _parse_time_from_line(line, delimiter)
            break
        except Exception:
            continue

    # fallback (rare)
    if t_last is None:
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    t_last = _parse_time_from_line(line, delimiter)
                except Exception:
                    continue

    if t_last is None:
        raise RuntimeError(f"No valid time found at end of {txt_path}")

    return float(t_first), float(t_last)

# ============================
# Chunker: distance-based 
# ============================

def chunk_txt_by_distance_streaming_intervals(
    txt_path: Union[str, Path],
    cfg_georef_path: Union[str, Path],
    out_dir: Union[str, Path],
    L: float,
    intervals: list[tuple[float, float]],   # merged/sorted intervals, time-based
    epsg_out: str = "EPSG:2056",
    delimiter: str = ",",
    skiprows: int = 0,
    min_points: int = 1000,
    out_prefix: str = "chunk_",
) -> int:
    """
    One-pass streaming chunker for HUGE TXT:
    - no intermediate multiwindow file
    - reads txt_path once
    - writes chunks (15m) only for points whose time is inside ANY intervals

    Returns number of chunks kept (>= min_points).
    """
    txt_path = Path(txt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not intervals:
        print(f"[chunk_intervals] No intervals -> nothing to do for {txt_path.name}")
        return 0

    # ensure sorted/merged
    intervals = sorted([(float(a), float(b)) for a, b in intervals], key=lambda x: x[0])
    last_end = intervals[-1][1]

    base_scan = extract_scan_id(txt_path)

    # Build s(t) from trajectory, clipped to txt time range (fast bounds)
    tmin_pts, tmax_pts = file_time_bounds_fast(txt_path, delimiter=delimiter)

    cfg = load_config(cfg_georef_path)
    t_trj, s_trj = build_s_of_t(cfg, epsg_out=epsg_out)

    keep = (t_trj >= tmin_pts) & (t_trj <= tmax_pts)
    if int(keep.sum()) < 2:
        raise RuntimeError("Trajectory time range does not overlap TXT gps_time range.")
    t_trj = t_trj[keep]
    s_trj = s_trj[keep]

    # Chunk boundaries in distance -> time edges
    s_start = float(s_trj[0])
    s_end = float(s_trj[-1])
    starts = np.arange(s_start, s_end, L, dtype=np.float64)

    t_edges = invert_s_to_t(s_trj, t_trj, np.clip(starts, s_start, s_end))
    t_last_edge = float(invert_s_to_t(s_trj, t_trj, np.array([min(s_end, starts[-1] + L)], dtype=np.float64))[0])
    t_edges = np.concatenate([t_edges, [t_last_edge]])

    # streaming writers
    writers: dict[int, any] = {}
    counts: dict[int, int] = {}
    kept_chunks: set[int] = set()

    def open_chunk(k: int):
        if k in writers:
            return
        chunk_id = base_scan + k
        p = out_dir / f"{out_prefix}{chunk_id:04d}.txt"
        writers[k] = p.open("w", encoding="utf-8")
        counts[k] = 0

    def close_chunk(k: int):
        if k not in writers:
            return
        writers[k].close()
        del writers[k]
        if counts.get(k, 0) < min_points:
            p = out_dir / f"{out_prefix}{(base_scan + k):04d}.txt"
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            kept_chunks.add(k)

    # interval pointer + chunk pointer
    j = 0
    chunk_idx = 0

    # fast-forward chunk_idx to first interval start (optional)
    # find first chunk where edge end > intervals[0][0]
    while chunk_idx + 1 < len(t_edges) and t_edges[chunk_idx + 1] <= intervals[0][0]:
        chunk_idx += 1

    # progress by bytes
    total_bytes = txt_path.stat().st_size
    open_chunk(chunk_idx)

    with txt_path.open("r", encoding="utf-8", errors="ignore") as fin, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc=f"Chunk {txt_path.name}"
    ) as pbar:
        for _ in range(skiprows):
            _ = fin.readline()

        while True:
            pos0 = fin.tell()
            line = fin.readline()
            if not line:
                break
            pos1 = fin.tell()
            pbar.update(pos1 - pos0)

            s = line.strip()
            if not s:
                continue

            parts = s.split(delimiter) if delimiter != "whitespace" else s.split()
            try:
                t = float(parts[0])
            except Exception:
                continue

            # stop early when beyond last interval and time is monotonic
            if t > last_end:
                break

            # advance interval pointer
            while j < len(intervals) and t > intervals[j][1]:
                j += 1
            if j >= len(intervals):
                break

            # if not inside interval -> skip point
            if not (intervals[j][0] <= t <= intervals[j][1]):
                continue

            # advance chunk pointer
            while chunk_idx + 1 < len(t_edges) and t >= t_edges[chunk_idx + 1]:
                close_chunk(chunk_idx)
                chunk_idx += 1
                open_chunk(chunk_idx)

            if chunk_idx >= len(t_edges) - 1:
                break

            # write point
            if t_edges[chunk_idx] <= t < t_edges[chunk_idx + 1]:
                writers[chunk_idx].write(line)
                counts[chunk_idx] += 1

    # close remaining
    for k in list(writers.keys()):
        close_chunk(k)

    n_kept = len(kept_chunks)
    print(f"[chunk_intervals] kept {n_kept} chunks in {out_dir}")
    return n_kept

# ============================
# GNSS outage + time-window extraction
# ============================

def write_gps_multi_outage(
    gps_in: Union[str, Path],
    gps_out: Union[str, Path],
    outages: list[tuple[float, float]],  # (start, duration)
    delimiter: str = ",",
) -> tuple[int, int]:
    """
    Remove GNSS samples during multiple outages.
    outages: [(start, duration), ...]
    Single pass, efficient with pointer.
    """
    gps_in = Path(gps_in)
    gps_out = Path(gps_out)

    intervals = sorted([(float(s), float(s) + float(d)) for s, d in outages], key=lambda x: x[0])

    kept = removed = 0
    j = 0

    with gps_in.open("r", encoding="utf-8", errors="ignore") as fin, gps_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue

            parts = s.split(delimiter) if delimiter != "whitespace" else s.split()
            try:
                t = float(parts[0])
            except Exception:
                # keep headers/comments
                fout.write(line if line.endswith("\n") else line + "\n")
                continue

            while j < len(intervals) and t > intervals[j][1]:
                j += 1

            in_outage = (j < len(intervals) and intervals[j][0] <= t <= intervals[j][1])
            if in_outage:
                removed += 1
                continue

            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1

    return kept, removed

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent intervals."""
    if not intervals:
        return []
    intervals = sorted([(float(a), float(b)) for a, b in intervals], key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:  # overlap/adjacent
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def extract_time_windows_txt(
    txt_in: Union[str, Path],
    txt_out: Union[str, Path],
    intervals: list[tuple[float, float]],
    delimiter: str = ",",
    skiprows: int = 0,
) -> int:
    """
    Streaming extraction with tqdm progress bar.
    Works safely for very large TXT (30GB+).
    """

    txt_in = Path(txt_in)
    txt_out = Path(txt_out)

    filesize = txt_in.stat().st_size

    intervals = sorted(intervals)
    last_end = intervals[-1][1]

    written = 0
    j = 0

    with txt_in.open("r", encoding="utf-8", errors="ignore") as fin, \
         txt_out.open("w", encoding="utf-8") as fout:

        with tqdm(
            total=filesize,
            unit="B",
            unit_scale=True,
            desc=f"Extract {txt_in.name}",
        ) as pbar:

            for _ in range(skiprows):
                next(fin, None)

            while True:
                pos_before = fin.tell()
                line = fin.readline()
                if not line:
                    break

                pbar.update(fin.tell() - pos_before)

                s = line.strip()
                if not s:
                    continue

                parts = s.split(delimiter) if delimiter != "whitespace" else s.split()

                try:
                    t = float(parts[0])
                except Exception:
                    continue

                while j < len(intervals) and t > intervals[j][1]:
                    j += 1
                if j >= len(intervals):
                    break

                if t > last_end:
                    break

                if intervals[j][0] <= t <= intervals[j][1]:
                    fout.write(line)
                    written += 1

    print(f"[extract] wrote {written:,} points → {txt_out}")
    return written