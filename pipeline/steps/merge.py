"""
steps/merge.py
==============
Merge step — combines georeferenced point clouds from multiple scanners
into unified spatial groups.

Three presets are supported:
  vux_only        : pairwise merge of VUX scanner outputs (HA + LR).
  all             : vux_only, then overlays PUCK points time-sliced onto
                    each VUX scan.
  puck_on_existing: overlays PUCK onto an existing VUX merge, skipping
                    the VUX merge step.

Writes a merged_manifest.csv index in each output directory for
downstream time-windowed file selection.

Public API
----------
run(pipe_cfg, scanner_entries) -> dict
    Returns {group_name: Path} for each merged group produced.
"""

from __future__ import annotations

import csv
import gc
import re
import shutil
from pathlib import Path
from typing import Optional, Union
import copy 

import laspy as lp
import numpy as np
import pandas as pd

from pipeline._log import info, sub, warn

# === Low-level file utilities ==================================

SCAN_RE = re.compile(r"^(\d{6})_(\d{6})")


def _extract_scan_id(p: Path) -> Optional[int]:
    m = SCAN_RE.match(p.stem)
    return int(m.group(2)) if m else None


def _list_clouds(dir_path: Path) -> list[Path]:
    exts = {".txt", ".las", ".laz"}
    return sorted(p for p in dir_path.iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


def _index_by_scan(files: list[Path]) -> tuple[dict[int, Path], list[tuple[str, str]]]:
    idx, skipped = {}, []
    for p in files:
        sid = _extract_scan_id(p)
        if sid is None:
            skipped.append((p.name, "no scan id"))
        elif sid in idx:
            skipped.append((p.name, f"duplicate scan={sid}"))
        else:
            idx[sid] = p
    return idx, skipped


def merge_two_txt(
    a: Path, b: Path, out: Path,
    *, delimiter: str = ",", skiprows: int = 0,
    sort_by_time: bool = False, float_fmt: str = "%.10f",
) -> None:
    def _load(p):
        arr = np.loadtxt(p, delimiter=delimiter, skiprows=skiprows)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    A, B = _load(a), _load(b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Column mismatch: {a.name} has {A.shape[1]}, {b.name} has {B.shape[1]}")
    M = np.vstack([A, B])
    if sort_by_time:
        M = M[np.argsort(M[:, 0])]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, M, delimiter=delimiter, fmt=float_fmt)
    sub(f"{a.name} + {b.name} → {out.name}  ({len(A)} + {len(B)} rows)")


def merge_two_las(a: Path, b: Path, out: Path) -> None:
    with lp.open(a) as fa, lp.open(b) as fb:
        ha, hb = fa.header, fb.header

        dim_a = list(ha.point_format.dimension_names)
        dim_b = list(hb.point_format.dimension_names)
        if dim_a != dim_b:
            raise ValueError(f"LAS dimension mismatch:\n  {a.name}: {dim_a}\n  {b.name}: {dim_b}")

        has_lasvec = all(d in dim_a for d in ("lasvec_x", "lasvec_y", "lasvec_z"))

        header = lp.LasHeader(point_format=ha.point_format, version=ha.version)
        header.scales  = np.minimum(np.array(ha.scales, dtype=np.float64),
                                     np.array(hb.scales, dtype=np.float64))
        header.offsets = np.floor(np.minimum(
            np.array(ha.mins, dtype=np.float64),
            np.array(hb.mins, dtype=np.float64),
        ))
        if has_lasvec and "lasvec_x" not in list(header.point_format.dimension_names):
            for dim in ("lasvec_x", "lasvec_y", "lasvec_z"):
                header.add_extra_dim(lp.ExtraBytesParams(name=dim, type=np.float32))

        out.parent.mkdir(parents=True, exist_ok=True)
        n_a, n_b = ha.point_count, hb.point_count

        def _reencode(points, hdr):
            rec = lp.ScaleAwarePointRecord(
                points.array, point_format=points.point_format,
                scales=points.scales, offsets=points.offsets,
            )
            las = lp.LasData(hdr)
            las.x, las.y, las.z = rec.x, rec.y, rec.z
            for dim in hdr.point_format.dimension_names:
                if dim in ("X","Y","Z","x","y","z"):
                    continue
                if dim in points.array.dtype.names:
                    las[dim] = points[dim]
            return las

        with lp.open(out, mode="w", header=header) as writer:
            for pts in fa.chunk_iterator(1_000_000):
                writer.write_points(_reencode(pts, header).points)
            for pts in fb.chunk_iterator(1_000_000):
                writer.write_points(_reencode(pts, header).points)

    sub(f"{a.name} + {b.name} → {out.name}  ({n_a} + {n_b} pts)")


def merge_cloud_pairs(
    dir_a: Path, dir_b: Path, out_dir: Path,
    delimiter: str = ",", sort_by_time: bool = False,
    out_prefix: str = "merged_", out_suffix: str = "",
    start_idx: Optional[int] = None, step: int = 1000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_a, _ = _index_by_scan(_list_clouds(dir_a))
    idx_b, _ = _index_by_scan(_list_clouds(dir_b))
    scans     = sorted(set(idx_a) & set(idx_b))

    if not scans:
        warn("[merge] No matching scan pairs found.")
        return

    for i, scan in enumerate(scans):
        a, b       = idx_a[scan], idx_b[scan]
        scan_label = start_idx + i * step if start_idx is not None else scan
        ext        = a.suffix.lower()
        if ext != b.suffix.lower():
            raise ValueError(f"Extension mismatch for scan {scan}: {a.name} vs {b.name}")
        out = out_dir / f"{out_prefix}{scan_label}{out_suffix}{ext}"
        if ext == ".txt":
            merge_two_txt(a, b, out, delimiter=delimiter, sort_by_time=sort_by_time)
        elif ext in (".las", ".laz"):
            merge_two_las(a, b, out)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    info(f"[merge] {len(scans)} pairs merged → {out_dir}")


# === Time manifest =============================================

def _build_time_manifest(merged_dir: Path, cloud_fmt: str) -> Path:
    rows = []
    files = sorted(merged_dir.glob(f"*.{cloud_fmt}"))
    for f in files:
        try:
            if cloud_fmt in ("las", "laz"):
                t_min, t_max = _las_time_bounds(f)
            else:
                t_min, t_max = _txt_time_bounds(f)
            # Extract scan_id from filename (e.g. "merged_1000_HA_LR.las" → 1000)
            sid = _extract_scan_id(f)
            if sid is None:
                # Fallback: parse first numeric group in stem
                m = re.search(r'(\d+)', f.stem)
                sid = int(m.group(1)) if m else 0
            rows.append({"scan_id": sid, "filename": f.name,
                          "t_start": t_min, "t_end": t_max})
        except Exception as e:
            warn(f"[merge] manifest: {f.name}: {e}")
    out = merged_dir / "merged_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _las_time_bounds(path: Path, chunk_size: int = 2_000_000) -> tuple[float, float]:
    t_min, t_max = np.inf, -np.inf
    with lp.open(path) as r:
        for pts in r.chunk_iterator(chunk_size):
            t = np.asarray(pts["gps_time"], dtype=np.float64)
            if t.size:
                t_min = min(t_min, float(t.min()))
                t_max = max(t_max, float(t.max()))
    return t_min, t_max


def _txt_time_bounds(path: Path) -> tuple[float, float]:
    data = np.loadtxt(path, delimiter=",", usecols=0)
    return float(data.min()), float(data.max())


def _detect_fmt(folder: Path) -> str:
    for ext in ("las", "laz", "txt"):
        if any(folder.glob(f"*.{ext}")):
            return ext
    raise FileNotFoundError(f"No .las/.laz/.txt found in {folder}")

def get_all_dim_names(header_or_record):
    dims = list(header_or_record.point_format.dimension_names)
    try:
        extra_dims = list(header_or_record.point_format.extra_dimension_names)
    except Exception:
        extra_dims = []

    for d in extra_dims:
        if d not in dims:
            dims.append(d)
    return dims

# === VUX merge (HA + LR pairwise) ==============================

def _merge_vux_scanners(
    scanner_entries: list[dict],
    vux_keys:        list[str],
    out_dir:         Path,
) -> Path:
    key_map = {e["key"]: e for e in scanner_entries}
    group   = [key_map[k] for k in vux_keys if k in key_map]

    if len(group) < len(vux_keys):
        missing = [k for k in vux_keys if k not in key_map]
        raise ValueError(f"[merge] Unknown vux_scanners keys: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(group) == 1:
        info("[merge] VUX: single scanner, skipping merge")
        return Path(group[0]["output_dir"])

    current_dir = Path(group[0]["output_dir"])
    for i in range(1, len(group)):
        next_dir = Path(group[i]["output_dir"])
        target   = out_dir if i == len(group) - 1 \
                   else out_dir.parent / f"tmp_vux_merge_{i}"
        target.mkdir(parents=True, exist_ok=True)
        info(f"[merge] VUX: {current_dir.name} + {next_dir.name} → {target.name}")
        merge_cloud_pairs(
            dir_a=current_dir, dir_b=next_dir, out_dir=target,
            out_prefix="merged_", out_suffix="_HA_LR",
            start_idx=1000, step=1000,
        )
        current_dir = target

    fmt = _detect_fmt(current_dir)
    _build_time_manifest(current_dir, fmt)
    info(f"[merge] VUX done → {current_dir}")
    return current_dir


# === VUX + PUCK merge (time-based) =============================

def make_output_header_from_vux_header(vux_header, add_scanner_src: bool = True):
    out_header = copy.deepcopy(vux_header)

    if add_scanner_src:
        dim_names = list(out_header.point_format.dimension_names)
        try:
            extra_names = list(out_header.point_format.extra_dimension_names)
        except Exception:
            extra_names = []

        if "scanner_src" not in dim_names and "scanner_src" not in extra_names:
            out_header.add_extra_dim(
                lp.ExtraBytesParams(name="scanner_src", type=np.uint8)
            )

    return out_header


def _global_offsets(paths: list[Path]) -> np.ndarray:
    """Compute safe LAS offsets = floor(global xyz min) across all files."""
    x_min = y_min = z_min = np.inf
    for p in paths:
        if not p.exists():
            continue
        with lp.open(p) as r:
            h = r.header
            # header.mins may be [0,0,0] if unset — fall back to reading first chunk
            hm = np.array(h.mins, dtype=np.float64)
            if np.all(hm == 0):
                try:
                    chunk = next(r.chunk_iterator(500_000))
                    rec = lp.ScaleAwarePointRecord(
                        chunk.array, point_format=chunk.point_format,
                        scales=chunk.scales, offsets=chunk.offsets,
                    )
                    hm = np.array([rec.x.min(), rec.y.min(), rec.z.min()], dtype=np.float64)
                except StopIteration:
                    pass
            x_min = min(x_min, float(hm[0]))
            y_min = min(y_min, float(hm[1]))
            z_min = min(z_min, float(hm[2]))
    return np.floor(np.array([x_min, y_min, z_min], dtype=np.float64))



def convert_points_to_output(points, out_header, scanner_src_value: int):
    n = len(points)

    las_in = lp.ScaleAwarePointRecord(
        points.array,
        point_format=points.point_format,
        scales=points.scales,
        offsets=points.offsets,
    )

    out_points = lp.ScaleAwarePointRecord.zeros(
        n,
        point_format=out_header.point_format,
        scales=out_header.scales,
        offsets=out_header.offsets,
    )

    out_points.x = las_in.x
    out_points.y = las_in.y
    out_points.z = las_in.z

    src_dims = set(points.array.dtype.names)
    out_dims = get_all_dim_names(out_header)

    for dim in out_dims:
        if dim in ("X", "Y", "Z", "x", "y", "z", "scanner_src"):
            continue
        if dim in src_dims:
            out_points[dim] = points[dim]

    if "gps_time" in src_dims and "gps_time" in out_dims:
        out_points["gps_time"] = points["gps_time"]

    if "scanner_src" in out_dims:
        out_points["scanner_src"] = np.full(n, scanner_src_value, dtype=np.uint8)

    return out_points


def _merge_puck_on_vux(
    vux_dir:         Path,
    puck_dir:        Path,
    out_dir:         Path,
    manifest_path:   Path,
    output_suffix:   str = "_VUX_PUCK",
    scanner_src_vux: int = 2,
    scanner_src_puck:int = 1,
    chunk_size:      int = 10_000_000,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with manifest_path.open("r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "scan_id": int(r["scan_id"]),
                "t_start": float(r["t_start"]),
                "t_end":   float(r["t_end"]),
                "path":    vux_dir / r["filename"],
            })
    rows.sort(key=lambda r: r["t_start"])
    puck_files  = sorted(puck_dir.glob("*.las"))
    manifest_rows = []
    prev_end    = -np.inf

    info(f"[merge] VUX+PUCK: {len(rows)} VUX files  {len(puck_files)} PUCK files")
    offsets = _global_offsets(
        [r["path"] for r in rows] + puck_files
    )

    for i, row in enumerate(rows):
        vux_file    = row["path"]
        is_last     = (i == len(rows) - 1)
        current_end = row["t_end"]

        stem_base = vux_file.stem.split("_HA_LR")[0]
        out_file  = out_dir / f"{stem_base}{output_suffix}.las"

        n_vux = n_puck = 0
        t_min = np.inf
        t_max = -np.inf

        with lp.open(vux_file) as vux_r:
            header_out = make_output_header_from_vux_header(
                vux_r.header,
                add_scanner_src=True,
            )
            header_out.offsets = offsets
            with lp.open(out_file, mode="w", header=header_out) as writer:
                for pts in vux_r.chunk_iterator(chunk_size):
                    out_pts = convert_points_to_output(
                        pts,
                        header_out,
                        scanner_src_value=scanner_src_vux,
                    )
                    writer.write_points(out_pts)
                    t_arr = np.asarray(pts["gps_time"], dtype=np.float64)
                    if t_arr.size:
                        t_min = min(t_min, float(t_arr.min()))
                        t_max = max(t_max, float(t_arr.max()))
                    n_vux += len(pts)

                for pf in puck_files:
                    with lp.open(pf) as puck_r:
                        for pts in puck_r.chunk_iterator(chunk_size):
                            pt = np.asarray(pts["gps_time"], dtype=np.float64)
                            mask = (pt > prev_end) & (pt <= current_end) \
                                   if not is_last else pt > prev_end
                            if np.any(mask):
                                sel     = pts[mask]
                                out_pts = convert_points_to_output(
                                    sel,
                                    header_out,
                                    scanner_src_value=scanner_src_puck,
                                )
                                writer.write_points(out_pts)
                                t_s  = np.asarray(sel["gps_time"], dtype=np.float64)
                                t_min = min(t_min, float(t_s.min()))
                                t_max = max(t_max, float(t_s.max()))
                                n_puck += len(sel)
                            del pts, pt, mask
                            gc.collect()

        sub(f"{out_file.name} | VUX={n_vux}  PUCK={n_puck}")
        manifest_rows.append({
            "scan_id": row["scan_id"], "filename": out_file.name,
            "t_start": t_min if np.isfinite(t_min) else 0.0,
            "t_end":   t_max if np.isfinite(t_max) else 0.0,
            "n_vux": n_vux, "n_puck": n_puck, "n_total": n_vux + n_puck,
        })
        prev_end = current_end
        gc.collect()

    pd.DataFrame(manifest_rows).sort_values("scan_id") \
      .to_csv(out_dir / "merged_manifest.csv", index=False)
    info(f"[merge] VUX+PUCK done → {out_dir}")
    return out_dir



# === Pipeline-facing API =======================================

def run(pipe_cfg: dict, scanner_entries: list[dict]) -> dict:
    """
    Execute merge according to pipe_cfg["merge"].
    Returns {group_name: Path}.
    """
    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]
    scenario_root = root_out_dir / scenario_name
    merge_cfg     = pipe_cfg.get("merge", {})

    preset          = merge_cfg.get("preset", "all")
    vux_keys        = merge_cfg.get("vux_scanners", ["ha_cfg", "lr_cfg"])
    puck_key        = merge_cfg.get("puck_scanner", "puck_cfg")
    output_suffix   = merge_cfg.get("output_suffix", "_VUX_PUCK")
    scanner_src_vux = int(merge_cfg.get("scanner_src_vux", 2))
    scanner_src_puck= int(merge_cfg.get("scanner_src_puck", 1))
    chunk_size      = int(merge_cfg.get("chunk_size", 10_000_000))
    cleanup_all     = bool(merge_cfg.get("cleanup", False))

    merged_groups: dict[str, Path] = {}
    entry_map     = {e["key"]: e for e in scanner_entries}

    info(f"[merge] preset={preset}")

    if preset in ("vux_only", "all"):
        vux_out  = scenario_root / "merged" / "HA_LR"
        vux_dir  = _merge_vux_scanners(scanner_entries, vux_keys, vux_out)
        merged_groups["HA_LR"] = vux_dir

    if preset in ("all", "puck_on_existing"):
        if preset == "puck_on_existing":
            vux_dir = Path(merge_cfg.get("vux_input_dir") or
                           scenario_root / "merged" / "HA_LR")
            if not vux_dir.exists():
                raise FileNotFoundError(
                    f"[merge] puck_on_existing: vux_input_dir not found: {vux_dir}"
                )

        puck_entry = entry_map.get(puck_key)
        if puck_entry is None:
            raise ValueError(
                f"[merge] puck_scanner='{puck_key}' not found. "
                f"Available: {list(entry_map)}"
            )
        all_out = scenario_root / "merged" / "ALL"
        _merge_puck_on_vux(
            vux_dir=vux_dir,
            puck_dir=Path(puck_entry["output_dir"]),
            out_dir=all_out,
            manifest_path=vux_dir / "merged_manifest.csv",
            output_suffix=output_suffix,
            scanner_src_vux=scanner_src_vux,
            scanner_src_puck=scanner_src_puck,
            chunk_size=chunk_size,
        )
        merged_groups["ALL"] = all_out

        if cleanup_all:
            dirs = [vux_dir] + [Path(e["output_dir"]) for e in scanner_entries]
            for d in dirs:
                if d.exists():
                    sub(f"cleanup: {d}")
                    shutil.rmtree(d, ignore_errors=True)

    elif preset not in ("vux_only",):
        raise ValueError(f"[merge] Unknown preset '{preset}'.")

    return merged_groups