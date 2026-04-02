"""
Script de pipeline orchestrant:
- la lecture des .sdc
- le géoréférencement
- le merging des deux singles beam
- chunking ou patching
- limatch
- OutageChunk (GNSS outages + chunks de 15m autour)
"""

from pathlib import Path
from .gnss_scenarios import write_gps_cycle_slips
import yaml
from typing import List, Tuple, Union, Optional
import subprocess
import sys
import os
import re
import csv
from tqdm import tqdm
import copy
import gc
import numpy as np
import pandas as pd
import laspy
import argparse

from .pointCloudGeoref import run_from_yaml
from .singleBeamMerging import merge_txt_pairs
from .Chunker import (
    chunk_txt_by_distance_streaming_intervals,
    chunk_las_by_distance_streaming_intervals,
    write_gps_multi_outage,
    merge_intervals,
    file_time_bounds_fast,
    file_time_bounds_fast_las,
    extract_scan_id,
    build_trajectory_bbox,
)
from .gnss_scenarios import write_gps_cycle_slips
import logging
# ============================================================================
# 0) Utils
# ============================================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def get_repo_root() -> Path:
    """
    navtools_PDM/pipeline.py -> parents[1] = repo root (ESO-PDM)
    """
    return Path(__file__).resolve().parents[1]
def detect_cloud_format_in_dir(folder: Union[str, Path]) -> str:
    folder = Path(folder)

    n_las = len(list(folder.glob("*.las")))
    n_laz = len(list(folder.glob("*.laz")))
    n_txt = len(list(folder.glob("*.txt")))

    if n_las > 0:
        return "las"
    if n_laz > 0:
        return "laz"
    if n_txt > 0:
        return "txt"

    raise FileNotFoundError(
        f"No supported cloud files found in {folder} "
        "(expected .las, .laz or .txt)"
    )

def get_scanner_cfg_paths(pipe_cfg: dict) -> list[Path]:
    scanners = pipe_cfg.get("scanners", {})
    cfg_paths = []

    for _, scanner_cfg_path in scanners.items():
        if scanner_cfg_path is None:
            continue
        cfg_paths.append(Path(scanner_cfg_path))

    if not cfg_paths:
        raise ValueError("No scanner config paths found in pipe_cfg['scanners']")

    return cfg_paths

def get_scanner_name(scanner_cfg_path: Union[str, Path]) -> str:
    scanner_cfg = yaml.safe_load(open(scanner_cfg_path, "r"))
    return scanner_cfg.get("scanner_name", Path(scanner_cfg_path).stem)

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def build_georef_cfg(scanner_cfg_path: Union[str, Path], pipe_cfg: dict) -> dict:
    scanner_cfg = yaml.safe_load(open(scanner_cfg_path, "r"))

    scenario_name = pipe_cfg["scenario_name"]
    root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
    scanner_name = scanner_cfg["scanner_name"]

    cfg = {}

    # pipeline-driven fields
    cfg["trj"] = pipe_cfg["trajectory"]
    cfg["distance_filtering"] = pipe_cfg["distance_filtering"]

    # scanner-driven fields
    cfg["lasvec"] = scanner_cfg["lasvec"]
    cfg["leapsec"] = scanner_cfg["leapsec"]
    cfg["mount"] = scanner_cfg["mount"]

    # output
    out_defaults = scanner_cfg["output_defaults"]
    cfg["output"] = {
        **out_defaults,
        "path": str(root_out_dir / scenario_name / scanner_name),
    }

    cfg["limatch_output"] = None

    # Propagate time-window filter if defined at pipeline level
    tw = pipe_cfg.get("georef_time_window")
    if tw is not None:
        cfg["georef_time_window"] = tw

    # Propagate manifest_path from scanner cfg into lasvec block
    manifest_path = scanner_cfg.get("manifest_path")
    if manifest_path is not None:
        cfg["lasvec"]["manifest_path"] = manifest_path

    return cfg

def write_temp_yaml(cfg: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def _pick_pair_files(pair_dir: Path) -> tuple[Path, Path]:
    # Exemple attendu:
    # Patch_from_scan_3000_with_4000.txt
    # Patch_from_scan_4000_with_3000.txt
    files = list(pair_dir.glob("Patch_from_scan_*_with_*.txt"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need 2 patch files in {pair_dir}, found {len(files)}")

    # parse ids
    pat = re.compile(r"Patch_from_scan_(\d+)_with_(\d+)\.txt$")
    parsed = []
    for f in files:
        m = pat.match(f.name)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            parsed.append((a, b, f))

    if len(parsed) < 2:
        # fallback
        files = sorted(files)
        return files[0], files[1]

    # On cherche une paire (a,b) et (b,a)
    for a, b, f_ab in parsed:
        for c, d, f_ba in parsed:
            if a == d and b == c:   # (a,b) et (b,a)
                # ordre stable: plus petit id en c1
                if a < b:
                    return f_ab, f_ba
                else:
                    return f_ba, f_ab

    # fallback: deux premiers
    parsed.sort(key=lambda x: (x[0], x[1], x[2].name))
    return parsed[0][2], parsed[1][2]


def list_las_files_sorted(folder: Path) -> list[Path]:
    """
    Return all .las files in folder, sorted by filename.
    """
    return sorted(folder.glob("*.las"))

def extract_time_prefix(path: Path) -> str:
    """
    Extract the first two underscore-separated blocks from filename stem.
    Example:
      260225_124306_VUX-HA1_pcd.las -> 260225_124306
    """
    parts = path.stem.split("_")
    if len(parts) < 2:
        return path.stem
    return "_".join(parts[:2])
# ============================================================================
# 1) Georef + merge
# ============================================================================
def load_scanner_entries(pipe_cfg: dict) -> list[dict]:
    """
    Read scanner configs declared in pipeline.yml.

    Returns a list of dicts:
      {
        "key": "ha_cfg",
        "cfg_path": Path(...),
        "scanner_name": "HA",
        "output_dir": Path(...)
      }
    """
    scanners_cfg = pipe_cfg.get("scanners", {})
    if not scanners_cfg:
        raise ValueError("No scanners declared in pipeline config under 'scanners'.")

    entries = []

    for scanner_key, scanner_cfg_path in scanners_cfg.items():
        scanner_cfg_path = Path(scanner_cfg_path)
        scanner_cfg = yaml.safe_load(open(scanner_cfg_path, "r"))

        scanner_name = scanner_cfg.get("scanner_name", scanner_key)

        # same logic as build_georef_cfg()
        scenario_name = pipe_cfg["scenario_name"]
        root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
        output_dir = root_out_dir / scenario_name / scanner_name

        entries.append({
            "key": scanner_key,
            "cfg_path": scanner_cfg_path,
            "scanner_name": scanner_name,
            "scanner_cfg": scanner_cfg,
            "output_dir": output_dir,
        })

    return entries

def georef_scanners(scanner_cfg_paths: list[Union[str, Path]], pipe_cfg: dict) -> list[dict]:
    """
    Build temporary georef cfgs for all scanners and run georeferencing.

    Returns a list of metadata dicts:
      {
        "scanner_name": "HA",
        "scanner_cfg_path": Path(...),
        "generated_cfg_path": Path(...),
        "output_dir": Path(...),
      }
    """
    scenario_name = pipe_cfg["scenario_name"]
    root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
    tmp_cfg_dir = root_out_dir / scenario_name / "tmp" / "generated_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    scanners_meta = []

    print("\n======================================")
    print("[georef] Starting georeferencing")
    print(f"[georef] scenario:       {scenario_name}")
    print(f"[georef] total scanners: {len(scanner_cfg_paths)}")
    print("======================================\n")

    for scanner_cfg_path in scanner_cfg_paths:
        scanner_cfg_path = Path(scanner_cfg_path)
        scanner_name = get_scanner_name(scanner_cfg_path)

        georef_cfg = build_georef_cfg(str(scanner_cfg_path), pipe_cfg)
        generated_cfg_path = tmp_cfg_dir / f"georef_{scanner_name}.generated.yml"
        write_temp_yaml(georef_cfg, generated_cfg_path)

        output_dir = Path(georef_cfg["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        print("--------------------------------------")
        print(f"[georef] scanner:        {scanner_name}")
        print(f"[georef] scanner cfg:    {scanner_cfg_path}")
        print(f"[georef] generated cfg:  {generated_cfg_path}")
        print(f"[georef] output dir:     {output_dir}")
        print("--------------------------------------")
        output_dir = Path(georef_cfg["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_from_yaml(str(generated_cfg_path))

        scanners_meta.append({
            "scanner_name": scanner_name,
            "scanner_cfg_path": scanner_cfg_path,
            "generated_cfg_path": generated_cfg_path,
            "output_dir": output_dir,
        })

    print("\n======================================")
    print("[georef] All scanners processed")
    print("======================================\n")

    return scanners_meta
def get_merged_manifest_path(
    merged_dir: Union[str, Path],
    manifest_name: str = "merged_manifest.csv",
) -> Path:
    merged_dir = Path(merged_dir)
    return merged_dir / manifest_name


def build_merged_time_manifest(
    merged_dir: Union[str, Path],
    cloud_fmt: str,
    manifest_name: str = "merged_manifest.csv",
    delimiter: str = ",",
    time_field: str = "gps_time",
) -> Path:
    """
    Scan merged_dir once and write:
      scan_id,filename,t_start,t_end

    Supports txt / las / laz.
    """
    merged_dir = Path(merged_dir)
    manifest_path = get_merged_manifest_path(merged_dir, manifest_name)

    cloud_fmt = str(cloud_fmt).lower()
    if cloud_fmt not in {"txt", "las", "laz"}:
        raise ValueError(
            f"Unsupported cloud_fmt='{cloud_fmt}' for manifest. "
            "Expected txt, las or laz."
        )

    files = sorted(merged_dir.glob(f"*.{cloud_fmt}"), key=extract_scan_id)
    if not files:
        raise FileNotFoundError(f"No .{cloud_fmt} files found in {merged_dir}")

    rows = []
    for f in files:
        scan_id = extract_scan_id(f)

        if cloud_fmt == "txt":
            t0, t1 = file_time_bounds_fast(f, delimiter=delimiter)
        else:
            t0, t1 = file_time_bounds_fast_las(f, time_field=time_field)

        rows.append(
            {
                "scan_id": scan_id,
                "filename": f.name,
                "t_start": float(t0),
                "t_end": float(t1),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["scan_id", "filename", "t_start", "t_end"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[manifest] written: {manifest_path} ({len(rows)} rows)")
    return manifest_path

def make_vux_puck_output_name(vux_file: Path, output_suffix: str = "_VUX_PUCK") -> str:
    """
    Example:
      merged_1000_HA_LR.las -> merged_1000_VUX_PUCK.las
    If the input does not end with _HA_LR, append the suffix before extension.
    """
    name = vux_file.name
    stem = vux_file.stem
    suffix = vux_file.suffix

    if stem.endswith("_HA_LR"):
        stem = stem[:-len("_HA_LR")] + output_suffix
    else:
        stem = stem + output_suffix

    return stem + suffix


def get_las_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.las"))
    if not files:
        raise FileNotFoundError(f"No .las files found in {folder}")
    return files


def get_time_bounds_chunked_las(las_path: Path, chunk_size: int = 10_000_000):
    t_min = np.inf
    t_max = -np.inf
    n_total = 0

    with laspy.open(las_path) as reader:
        dims = list(reader.header.point_format.dimension_names)
        if "gps_time" not in dims:
            raise ValueError(f"'gps_time' absent in {las_path}")

        for points in reader.chunk_iterator(chunk_size):
            t = np.asarray(points["gps_time"], dtype=np.float64)
            if t.size == 0:
                continue
            t_min = min(t_min, float(t.min()))
            t_max = max(t_max, float(t.max()))
            n_total += len(t)

    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError(f"Impossible to read gps_time in {las_path}")

    return t_min, t_max, n_total


def load_vux_infos_from_manifest(vux_dir: Path, manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = {"scan_id", "filename", "t_start", "t_end"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manifest: {missing}")

    df = df.sort_values("t_end").reset_index(drop=True)

    vux_infos = []
    print("\n======================================")
    print("[VUX+PUCK] VUX temporal bounds (manifest)")
    print("======================================")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyse temps VUX", unit="file"):
        f = vux_dir / row["filename"]
        if not f.exists():
            raise FileNotFoundError(f"Manifest references missing file: {f}")

        info = {
            "file": f,
            "t_min": float(row["t_start"]),
            "t_max": float(row["t_end"]),
            "scan_id": int(row["scan_id"]),
        }
        vux_infos.append(info)
        print(f"{f.name} | {info['t_min']:.6f} -> {info['t_max']:.6f}")

    return vux_infos


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
                laspy.ExtraBytesParams(name="scanner_src", type=np.uint8)
            )

    return out_header


def convert_points_to_output(points, out_header, scanner_src_value: int):
    n = len(points)

    las_in = laspy.ScaleAwarePointRecord(
        points.array,
        point_format=points.point_format,
        scales=points.scales,
        offsets=points.offsets,
    )

    out_points = laspy.ScaleAwarePointRecord.zeros(
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


def merge_vux_group_with_puck_by_time(
    vux_dir: Union[str, Path],
    puck_dir: Union[str, Path],
    out_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    output_suffix: str = "_VUX_PUCK",
    scanner_src_vux: int = 2,
    scanner_src_puck: int = 1,
    chunk_size: int = 10_000_000,
) -> Path:
    """
    Merge a VUX group (typically HA_LR) with PUCK using the temporal policy:
      cloud i gets PUCK points such that prev_end < t <= current_end
      last cloud gets all remaining points with t > prev_end

    Output naming follows the VUX merged cloud names.
    A new merged_manifest.csv is written in out_dir.
    """
    vux_dir = Path(vux_dir)
    puck_dir = Path(puck_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = vux_dir / "merged_manifest.csv"
    manifest_path = Path(manifest_path)

    vux_infos = load_vux_infos_from_manifest(vux_dir, manifest_path)
    puck_files = get_las_files(puck_dir)

    print("\n======================================")
    print("[VUX+PUCK] PUCK temporal bounds")
    print("======================================")
    for f in tqdm(puck_files, desc="Analyse temps PUCK", unit="file"):
        t_min, t_max, n = get_time_bounds_chunked_las(f, chunk_size=chunk_size)
        print(f"{f.name} | n={n} | {t_min:.6f} -> {t_max:.6f}")

    print("\n======================================")
    print("[VUX+PUCK] Merge VUX group + PUCK")
    print("======================================")

    prev_end = -np.inf
    manifest_rows = []

    for i, vux_info in enumerate(tqdm(vux_infos, desc="Merge VUX->ALL", unit="cloud")):
        vux_file = vux_info["file"]
        current_end = vux_info["t_max"]
        is_last = (i == len(vux_infos) - 1)
        out_file = out_dir / make_vux_puck_output_name(vux_file, output_suffix=output_suffix)

        print("\n--------------------------------------")
        print(f"[{i+1}/{len(vux_infos)}] VUX file: {vux_file.name}")
        print(f"Window: ({prev_end:.6f}, {'+inf' if is_last else f'{current_end:.6f}'}]")
        print(f"Output: {out_file}")
        print("--------------------------------------")

        n_vux_written = 0
        n_puck_written = 0
        merged_t_min = np.inf
        merged_t_max = -np.inf

        with laspy.open(vux_file) as vux_reader:
            dims = list(vux_reader.header.point_format.dimension_names)
            if "gps_time" not in dims:
                raise ValueError(f"'gps_time' absent in {vux_file}")

            out_header = make_output_header_from_vux_header(vux_reader.header, add_scanner_src=True)

            with laspy.open(out_file, mode="w", header=out_header) as writer:
                for vux_points in tqdm(
                    vux_reader.chunk_iterator(chunk_size),
                    total=max(1, int(np.ceil(vux_reader.header.point_count / chunk_size))),
                    desc=f"Write VUX {vux_file.stem}",
                    unit="chunk",
                    leave=False,
                ):
                    vux_t = np.asarray(vux_points["gps_time"], dtype=np.float64)
                    if vux_t.size > 0:
                        merged_t_min = min(merged_t_min, float(vux_t.min()))
                        merged_t_max = max(merged_t_max, float(vux_t.max()))

                    out_points = convert_points_to_output(vux_points, out_header, scanner_src_value=scanner_src_vux)
                    writer.write_points(out_points)
                    n_vux_written += len(vux_points)

                    del vux_points, vux_t, out_points
                    gc.collect()

                for puck_file in tqdm(puck_files, desc=f"PUCK for {vux_file.stem}", unit="file", leave=False):
                    with laspy.open(puck_file) as puck_reader:
                        dims = list(puck_reader.header.point_format.dimension_names)
                        if "gps_time" not in dims:
                            raise ValueError(f"'gps_time' absent in {puck_file}")

                        selected_this_file = 0

                        for puck_points in puck_reader.chunk_iterator(chunk_size):
                            pt = np.asarray(puck_points["gps_time"], dtype=np.float64)

                            if is_last:
                                mask = pt > prev_end
                            else:
                                mask = (pt > prev_end) & (pt <= current_end)

                            if np.any(mask):
                                selected = puck_points[mask]
                                selected_t = np.asarray(selected["gps_time"], dtype=np.float64)

                                if selected_t.size > 0:
                                    merged_t_min = min(merged_t_min, float(selected_t.min()))
                                    merged_t_max = max(merged_t_max, float(selected_t.max()))

                                out_points = convert_points_to_output(selected, out_header, scanner_src_value=scanner_src_puck)
                                writer.write_points(out_points)

                                n_sel = len(selected)
                                selected_this_file += n_sel
                                n_puck_written += n_sel

                                del selected, selected_t, out_points

                            del puck_points, pt, mask
                            gc.collect()

                        if selected_this_file > 0:
                            print(f"[PUCK] {puck_file.name}: {selected_this_file} selected point(s)")

        print(
            f"[OK] {out_file.name} | "
            f"VUX={n_vux_written} pts | "
            f"PUCK added={n_puck_written} pts | "
            f"TOTAL={n_vux_written + n_puck_written} pts"
        )

        manifest_rows.append({
            "scan_id": int(vux_info["scan_id"]),
            "filename": out_file.name,
            "t_start": float(merged_t_min),
            "t_end": float(merged_t_max),
            "n_vux_points": int(n_vux_written),
            "n_puck_points": int(n_puck_written),
            "n_total_points": int(n_vux_written + n_puck_written),
        })

        prev_end = current_end
        gc.collect()

    manifest_df = pd.DataFrame(manifest_rows).sort_values("scan_id").reset_index(drop=True)
    manifest_out = out_dir / "merged_manifest.csv"
    manifest_df.to_csv(manifest_out, index=False)

    print(f"[VUX+PUCK] Manifest written: {manifest_out}")
    print("[VUX+PUCK] scanner_src: 2=VUX, 1=PUCK")

    return out_dir

def merge_scanner_group(
    group_name: str,
    group_entries: list[dict],
    out_dir: Path,
    delimiter: str = ",",
) -> Path:
    """
    Merge 1..N scanner output dirs into out_dir.
    Current implementation performs sequential pairwise merges.

    Returns final merged directory.
    """
    import shutil
    import tempfile

    out_dir.mkdir(parents=True, exist_ok=True)

    scanner_names = [e["scanner_name"] for e in group_entries]

    print("\n======================================")
    print(f"[merge] Starting merge group: {group_name}")
    print(f"[merge] scanners: {' + '.join(scanner_names)}")
    for e in group_entries:
        print(f"[merge] {e['scanner_name']} dir: {e['output_dir']}")
    print(f"[merge] OUT dir: {out_dir}")
    print("======================================\n")

    if len(group_entries) == 1:
        print(f"[merge] Group {group_name} has only one scanner -> no merge needed.")
        current_dir = Path(group_entries[0]["output_dir"])
    else:
        current_dir = Path(group_entries[0]["output_dir"])

        for i in range(1, len(group_entries)):
            next_dir = Path(group_entries[i]["output_dir"])

            if i == len(group_entries) - 1:
                target_dir = out_dir
            else:
                target_dir = out_dir.parent / f"tmp_merge_{group_name}_{i}"
                target_dir.mkdir(parents=True, exist_ok=True)

            print("--------------------------------------")
            print(f"[merge] step {i}/{len(group_entries)-1}")
            print(f"[merge] A: {current_dir}")
            print(f"[merge] B: {next_dir}")
            print(f"[merge] target: {target_dir}")
            print("--------------------------------------")

            merge_txt_pairs(
                current_dir,
                next_dir,
                target_dir,
                delimiter=delimiter,
                sort_by_time=False,
                out_prefix="merged_",
                out_suffix=f"_{group_name}",
                start_idx=1000,
                step=1000,
            )
            current_dir = target_dir

    print("\n======================================")
    print(f"[merge] Merge group completed: {group_name}")
    print(f"[merge] final dir: {current_dir}")
    print("======================================\n")

    
    detected_fmt = detect_cloud_format_in_dir(current_dir)

    print(f"[merge] detected merged format: {detected_fmt}")

    build_kwargs = dict(
        merged_dir=current_dir,
        cloud_fmt=detected_fmt,
        manifest_name="merged_manifest.csv",
    )

    if detected_fmt == "txt":
        build_kwargs["delimiter"] = delimiter
    else:
        build_kwargs["time_field"] = "gps_time"

    build_merged_time_manifest(**build_kwargs)

    return current_dir

def georef_and_merge(pipe_cfg: dict) -> dict:
    """
    Run georeferencing for all scanners declared in pipe_cfg['scanners'].
    Then optionally merge scanner groups declared in pipe_cfg['merge_groups'].

    Returns a dict with:
      {
        "scanner_entries": ...,
        "merged_groups": {group_name: merged_dir}
      }
    """
    scenario_name = pipe_cfg["scenario_name"]
    root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_root = root_out_dir / scenario_name

    scanner_entries = load_scanner_entries(pipe_cfg)

    print("\n======================================")
    print("[georef_and_merge] scanner_entries after load_scanner_entries:")
    for i, e in enumerate(scanner_entries):
        print(f"  [{i}] {e}")
    print("======================================\n")

    # Preserve original YAML keys by cfg path
    original_key_by_cfg = {
        str(e["cfg_path"]): e["key"]
        for e in scanner_entries
        if e.get("cfg_path") is not None and e.get("key") is not None
    }

    if pipe_cfg["steps"].get("georef", False):
        scanner_cfg_paths = [e["cfg_path"] for e in scanner_entries]
        scanner_entries = georef_scanners(scanner_cfg_paths, pipe_cfg)

        # Re-attach original YAML key if possible
        for e in scanner_entries:
            cfg_ref = e.get("scanner_cfg_path") or e.get("cfg_path")
            if cfg_ref is not None:
                cfg_ref = str(cfg_ref)
                if cfg_ref in original_key_by_cfg:
                    e["key"] = original_key_by_cfg[cfg_ref]

        print("\n======================================")
        print("[georef_and_merge] scanner_entries after georef_scanners:")
        for i, e in enumerate(scanner_entries):
            print(f"  [{i}] {e}")
        print("======================================\n")

    merged_groups = {}

    if pipe_cfg["steps"].get("merge", False):
        merge_groups = pipe_cfg.get("merge_groups", [])
        if not merge_groups:
            print("[merge] No merge_groups defined in pipeline config. Skipping merge.")
        else:
            entry_map = {}

            for e in scanner_entries:
                candidates = [
                    e.get("key"),
                    e.get("scanner_name"),
                    e.get("name"),
                    e.get("scanner"),
                ]

                candidates = [c for c in candidates if c is not None]

                if not candidates:
                    raise ValueError(
                        "[merge] scanner entry has no usable identifier "
                        f"(expected one of: key, scanner_name, name, scanner). Entry: {e}"
                    )

                # Keep a primary key for display/debug
                if e.get("key") is None:
                    e["key"] = candidates[0]

                # Register ALL aliases
                for c in candidates:
                    entry_map[c] = e

            print("\n======================================")
            print("[merge] available scanner keys:")
            for k in entry_map:
                print(f"  - {k}")
            print("======================================\n")

            for group in merge_groups:
                group_name = group["name"]
                group_keys = group["scanners"]

                group_entries = []
                for k in group_keys:
                    if k not in entry_map:
                        raise ValueError(
                            f"Merge group '{group_name}' references unknown scanner key '{k}'. "
                            f"Available keys: {list(entry_map.keys())}"
                        )
                    group_entries.append(entry_map[k])

                group_out_dir = scenario_root / "merged" / group_name

                merged_dir = merge_scanner_group(
                    group_name=group_name,
                    group_entries=group_entries,
                    out_dir=group_out_dir,
                    delimiter=",",
                )
                merged_groups[group_name] = merged_dir

            vux_puck_cfg = pipe_cfg.get("vux_puck_merge", {})
            if vux_puck_cfg.get("enabled", False):
                vux_group_name = vux_puck_cfg["vux_group"]
                puck_key = vux_puck_cfg["puck_scanner"]

                if vux_group_name not in merged_groups:
                    raise ValueError(
                        f"vux_puck_merge.vux_group='{vux_group_name}' not found in merged_groups: "
                        f"{list(merged_groups.keys())}"
                    )

                puck_entry = entry_map.get(puck_key)
                if puck_entry is None:
                    raise ValueError(
                        f"vux_puck_merge.puck_scanner='{puck_key}' not found. "
                        f"Available keys: {list(entry_map.keys())}"
                    )

                final_group_name = vux_puck_cfg.get("name", "ALL")
                final_out_dir = vux_puck_cfg.get("output_dir", None)
                if final_out_dir is None:
                    final_out_dir = scenario_root / "merged" / final_group_name
                else:
                    final_out_dir = Path(final_out_dir)

                manifest_path = vux_puck_cfg.get("manifest_path", None)
                if manifest_path is None:
                    manifest_path = Path(merged_groups[vux_group_name]) / "merged_manifest.csv"
                else:
                    manifest_path = Path(manifest_path)

                merged_dir = merge_vux_group_with_puck_by_time(
                    vux_dir=merged_groups[vux_group_name],
                    puck_dir=puck_entry["output_dir"],
                    out_dir=final_out_dir,
                    manifest_path=manifest_path,
                    output_suffix=vux_puck_cfg.get("output_suffix", "_VUX_PUCK"),
                    scanner_src_vux=int(vux_puck_cfg.get("scanner_src_vux", 2)),
                    scanner_src_puck=int(vux_puck_cfg.get("scanner_src_puck", 1)),
                    chunk_size=int(vux_puck_cfg.get("chunk_size", 10_000_000)),
                )
                merged_groups[final_group_name] = merged_dir
                build_merged_time_manifest(
                    merged_dir=merged_dir,
                    cloud_fmt="las",
                    manifest_name="merged_manifest.csv",
                    time_field="gps_time",
                )
                    

    return {
        "scanner_entries": scanner_entries,
        "merged_groups": merged_groups,
    }
# ============================================================================
# 2) Chunking
# ============================================================================


def chunk_clouds(
    merged_dir,
    cfg_georef_path,
    chunks_out=None,
    L: float = 36.0,
    epsg_out: str = "EPSG:2056",
    delimiter: str = ",",
    skiprows: int = 0,
    min_points: int = 2000,
    cloud_fmt: str = "las",
    time_field: str = "gps_time",
):
    """
    Chunk all merged .txt files in merged_dir.
    Writes results in chunks_out/<merged_file_stem>/chunk_xxxx.txt
    Returns chunks_root directory.
    """
    merged_dir = Path(merged_dir)

    if chunks_out is None:
        chunks_root = merged_dir.parent / "chunks"
    else:
        chunks_root = Path(chunks_out)

    chunks_root.mkdir(parents=True, exist_ok=True)

    if cloud_fmt.lower() == "las":
        merged_files = sorted(merged_dir.glob("*.las"))
    else:
        merged_files = sorted(merged_dir.glob("*.txt"))

    if not merged_files:
        raise FileNotFoundError(f"No .{cloud_fmt} found in merged directory: {merged_dir}")

    for f in merged_files:
        out_sub = chunks_root / f.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        if cloud_fmt.lower() == "las":
            chunk_las_by_distance_streaming_intervals(
                las_path=f,
                cfg_georef_path=str(cfg_georef_path),
                out_dir=str(out_sub),
                L=L,
                intervals=intervals,
                epsg_out=epsg_out,
                min_points=min_points_chunk,
                time_field=time_field,
                min_last_chunk_m=chunk_cfg.get("min_last_chunk_m", 10.0),
            )
        else:
            chunk_txt_by_distance_streaming_intervals(
                txt_path=str(f),
                cfg_georef_path=str(cfg_georef_path),
                out_dir=str(out_sub),
                L=L,
                intervals=[(-np.inf, np.inf)],
                epsg_out=epsg_out,
                delimiter=delimiter,
                skiprows=skiprows,
                min_points=min_points,
            )

    return chunks_root

def chunk_pairs_neighbors(
    chunk_files: List[Path],
    neighbor_k: int = 1,
) -> List[Tuple[Path, Path]]:
    """
    Generate chunk pairs using k forward neighbors.

    neighbor_k = 1 → (i,i+1)
    neighbor_k = 2 → (i,i+1),(i,i+2)
    neighbor_k = 3 → ...
    """

    pairs = []
    n = len(chunk_files)

    for i in range(n):
        for d in range(1, neighbor_k + 1):
            j = i + d
            if j < n:
                pairs.append((chunk_files[i], chunk_files[j]))

    return pairs

def compute_bbox_from_traj(t_start, t_end, t_trj, x_trj, y_trj, n=20):
    t_query = np.linspace(t_start, t_end, n)
    t_query = np.clip(t_query, t_trj[0], t_trj[-1])

    x = np.interp(t_query, t_trj, x_trj)
    y = np.interp(t_query, t_trj, y_trj)

    return float(x.min()), float(x.max()), float(y.min()), float(y.max())

def load_sbet_xy(sbet_path: Path, epsg_out: str):
    data = np.fromfile(sbet_path, dtype=np.float64).reshape(-1, 17)

    t   = data[:, 0]
    lat = np.degrees(data[:, 1])
    lon = np.degrees(data[:, 2])
    alt = data[:, 3]

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x_map, y_map, _ = transformer.transform(lon, lat, alt)

    order = np.argsort(t)
    return t[order], x_map[order], y_map[order]

def combined_multi_outage_scenario(
    merged_dir: Union[str, Path],
    cfg_georef_path: Union[str, Path],
    gps_in: Union[str, Path],
    outages: list[tuple[float, float]],
    pre: float = 30.0,
    post: float = 30.0,
    out_root: Union[str, Path] = "scenario_combined",
    delimiter: str = ",",
    min_points_chunk: int = 2000,
    epsg_out: str = "EPSG:2056",
    do_chunks: bool = True,
    reuse_chunks: bool = True,
    force: bool = False,
    chunk_cfg: Optional[dict] = None,
) -> tuple[Path, Path]:
    import csv

    merged_dir = Path(merged_dir)
    cfg_georef_path = Path(cfg_georef_path)
    gps_in = Path(gps_in)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    chunk_cfg = chunk_cfg or {}

    cloud_fmt = str(chunk_cfg.get("cloud_fmt", "txt")).lower()
    time_field = str(chunk_cfg.get("time_field", "gps_time"))
    delimiter = chunk_cfg.get("delimiter", delimiter)
    skiprows = int(chunk_cfg.get("skiprows", 0))
    min_points_chunk = int(chunk_cfg.get("min_points", min_points_chunk))
    epsg_out = str(chunk_cfg.get("epsg_out", epsg_out))
    L = float(chunk_cfg.get("length_m", 15.0))

    manifest_name = str(chunk_cfg.get("manifest_name", "merged_manifest.csv"))
    use_manifest = bool(chunk_cfg.get("use_manifest", True))

    if cloud_fmt not in {"txt", "las", "laz"}:
        raise ValueError(
            f"Unsupported chunk.cloud_fmt='{cloud_fmt}'. "
            "Expected one of: txt, las, laz."
        )

    gps_out = out_root / "GPS_outage.txt"
    kept, removed = write_gps_multi_outage(
        gps_in, gps_out, outages, delimiter=delimiter
    )

    print(
        f"[combined] GPS_outage written: {gps_out} "
        f"(kept={kept}, removed={removed})"
    )

    expanded_intervals = [
        (float(s) - pre, float(s) + float(d) + post)
        for (s, d) in outages
    ]
    intervals = merge_intervals(expanded_intervals)

    print("\n==============================")
    print("[combined scenario]")
    print(f"Outages: {outages}")
    print(f"Expanded intervals: {intervals}")
    print(f"cloud_fmt: {cloud_fmt}")
    if cloud_fmt in {"las", "laz"}:
        print(f"time_field: {time_field}")
    print("==============================\n")

    first_interval_start = intervals[0][0]
    last_interval_end = intervals[-1][1]

    chunks_root = out_root / f"chunks_{int(L)}m"
    chunks_root.mkdir(parents=True, exist_ok=True)

    if not do_chunks:
        print(f"[combined] reuse existing chunks -> {chunks_root}")
        return chunks_root, gps_out

    selected_files = []
    manifest_path = merged_dir / manifest_name

    if use_manifest and manifest_path.exists():
        print(f"[combined] Using manifest: {manifest_path}")

        rows = []
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                row_path = merged_dir / r["filename"]
                rows.append(
                    {
                        "scan_id": int(r["scan_id"]),
                        "filename": r["filename"],
                        "t_start": float(r["t_start"]),
                        "t_end": float(r["t_end"]),
                        "path": row_path,
                    }
                )

        rows.sort(key=lambda r: r["t_start"])

        total_rows = len(rows)

        for r in rows:
            t0 = r["t_start"]
            t1 = r["t_end"]
            f = r["path"]

            if t1 < first_interval_start:
                print(
                    f"[outage selection] {f.name:<45} "
                    f"time=[{t0:.3f},{t1:.3f}] -> skipped (before)"
                )
                continue

            if t0 > last_interval_end:
                print(
                    f"[outage selection] {f.name:<45} "
                    f"time=[{t0:.3f},{t1:.3f}] -> stop (after)"
                )
                break

            involved = not (t1 < first_interval_start or t0 > last_interval_end)
            status = "involved" if involved else "skipped"

            print(
                f"[outage selection] {f.name:<45} "
                f"time=[{t0:.3f},{t1:.3f}] -> {status}"
            )

            if involved:
                selected_files.append(f)

        print(f"\n[outage summary] {len(selected_files)} / {total_rows} involved\n")

    else:
        if use_manifest:
            print(f"[combined] Manifest not found: {manifest_path}")
        print("[combined] Fallback to direct cloud scan")

        merged_pattern = f"*.{cloud_fmt}"
        merged_files = sorted(
            merged_dir.glob(merged_pattern),
            key=extract_scan_id
        )

        if not merged_files:
            raise FileNotFoundError(f"No {merged_pattern} found in {merged_dir}")

        for f in merged_files:
            try:
                if cloud_fmt == "txt":
                    t0, t1 = file_time_bounds_fast(f, delimiter=delimiter)
                else:
                    t0, t1 = file_time_bounds_fast_las(f, time_field=time_field)

                if t1 < first_interval_start:
                    print(
                        f"[outage selection] {f.name:<45} "
                        f"time=[{t0:.3f},{t1:.3f}] -> skipped (before)"
                    )
                    continue

                if t0 > last_interval_end:
                    print(
                        f"[outage selection] {f.name:<45} "
                        f"time=[{t0:.3f},{t1:.3f}] -> stop (after)"
                    )
                    break

                involved = not (t1 < first_interval_start or t0 > last_interval_end)
                status = "involved" if involved else "skipped"

                print(
                    f"[outage selection] {f.name:<45} "
                    f"time=[{t0:.3f},{t1:.3f}] -> {status}"
                )

                if involved:
                    selected_files.append(f)

            except Exception as e:
                print(
                    f"[outage selection] {f.name} bounds FAIL "
                    f"({type(e).__name__}: {e}) -> process"
                )
                selected_files.append(f)

        print(f"\n[outage summary] {len(selected_files)} / {len(merged_files)} involved\n")

    processed = 0

    # Load trajectory once for vehicle-based bbox
    _t_trj = _x_trj = _y_trj = None

    try:
        from pyproj import Transformer
        from .pointCloudGeoref import load_config

        _georef_cfg = load_config(str(cfg_georef_path))
        _sbet_path  = _georef_cfg["trj"]["path"]

        data = np.fromfile(_sbet_path, dtype=np.float64).reshape(-1, 17)

        t   = data[:, 0]
        lat = np.degrees(data[:, 1])
        lon = np.degrees(data[:, 2])
        alt = data[:, 3]

        transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
        x_map, y_map, _ = transformer.transform(lon, lat, alt)

        order = np.argsort(t)

        _t_trj = t[order]
        _x_trj = x_map[order]
        _y_trj = y_map[order]

        print(f"[combined] Trajectory loaded ({len(_t_trj)} poses)")

    except Exception as e:
        print(f"[combined] WARNING: no trajectory ({e})")

    for f in selected_files:
        out_sub = chunks_root / f.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        print(
            f"[combined] CHUNK DIRECT {L:.1f}m "
            f"{f.name} -> {out_sub}",
            flush=True
        )

        if cloud_fmt == "txt":
            chunk_txt_by_distance_streaming_intervals(
                txt_path=f,
                cfg_georef_path=str(cfg_georef_path),
                out_dir=str(out_sub),
                L=L,
                intervals=intervals,
                epsg_out=epsg_out,
                delimiter=delimiter,
                skiprows=skiprows,
                min_points=min_points_chunk,
            )
        else:
            chunk_las_by_distance_streaming_intervals(
                las_path=f,
                cfg_georef_path=str(cfg_georef_path),
                out_dir=str(out_sub),
                L=L,
                intervals=intervals,
                epsg_out=epsg_out,
                min_points=min_points_chunk,
                time_field=time_field,
                min_last_chunk_m=chunk_cfg.get("min_last_chunk_m", 10.0),
                t_trj=_t_trj,
                x_trj=_x_trj,
                y_trj=_y_trj,
            )

        processed += 1

    # Rebuild bbox on existing chunks too (e.g. source="existing")
    if _t_trj is not None:
        bbox_rows = []

        las_files = sorted(out_sub.glob("chunk_*.las"))

        for las_path in las_files:
            try:
                with laspy.open(las_path) as reader:
                    all_t = []

                    for pts in reader.chunk_iterator(500_000):
                        if len(pts):
                            all_t.append(np.asarray(pts[time_field], dtype=np.float64))

                if not all_t:
                    continue

                t_all = np.concatenate(all_t)

                t_start = float(t_all.min())
                t_end   = float(t_all.max())

                x_min, x_max, y_min, y_max = compute_bbox_from_traj(
                    t_start,
                    t_end,
                    _t_trj,
                    _x_trj,
                    _y_trj,
                )

                bbox_rows.append({
                    "chunk_file": las_path.name,
                    "t_start": t_start,
                    "t_end": t_end,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                })

            except Exception as e:
                print(f"[bbox] Warning: {las_path.name}: {e}")

        if bbox_rows:
            df = pd.DataFrame(bbox_rows).sort_values("t_start")
            df.to_csv(out_sub / "chunk_bbox.csv", index=False)
            

    print(f"\n[combined] processed files: {processed}/{len(selected_files)}")
    print(f"[combined] chunks root: {chunks_root}")

    return chunks_root, gps_out

# ============================================================================
# 3) LiMatch 
# ============================================================================

def get_limatch_match_clouds(repo_root: Path):
    """
    Import limatch as a package, so relative imports work.
    Requires: Patcher/submodules/limatch/__init__.py (empty file).
    """
    limatch_parent = repo_root / "Patcher" / "submodules"  # parent of limatch/
    if str(limatch_parent) not in sys.path:
        sys.path.insert(0, str(limatch_parent))

    from limatch.main import match_clouds
    return match_clouds


def run_limatch_api(
    repo_root: Path,
    limatch_cfg_path: Union[str, Path],
    cloud1: Union[str, Path],
    cloud2: Union[str, Path],
    out_dir: Union[str, Path],
):
    match_clouds = get_limatch_match_clouds(repo_root)
    cfg = yaml.safe_load(open(str(limatch_cfg_path), "r"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Important: LiMatch writes to prj_folder
    cfg["prj_folder"] = str(out_dir) + os.sep
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (out_dir / "cor_outputs").mkdir(parents=True, exist_ok=True)

    print(f"[limatch-api] prj_folder={cfg['prj_folder']}")

    return match_clouds(str(cloud1), str(cloud2), cfg)


def run_limatch_on_chunks_per_scan(
    chunks_root: Union[str, Path],
    limatch_cfg: Union[str, Path],
    out_root: Union[str, Path],
    do_cross_scan: bool = True,
    neighbor_k: int = 2,
) -> None:
    """
    chunks_root:
      .../subset/chunks/
        merged_100_HA_LR/
          chunk_0100.txt / .las / .laz
          chunk_0101.txt / .las / .laz
          ...
        merged_200_HA_LR/
          chunk_0200.txt / .las / .laz
          ...

    neighbor_k:
      1 -> match chunk_i with chunk_(i+1)
      2 -> match chunk_i with chunk_(i+1) and chunk_(i+2)
      ...
    """
    repo_root = get_repo_root()

    chunks_root = Path(chunks_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    limatch_cfg = Path(limatch_cfg)
    lim_cfg = yaml.safe_load(open(str(limatch_cfg), "r"))
    cloud_fmt = str(lim_cfg.get("cloud_fmt", "txt")).lower()

    if cloud_fmt not in {"txt", "las", "laz"}:
        raise ValueError(
            f"Unsupported LiMatch cloud_fmt='{cloud_fmt}' in {limatch_cfg}. "
            "Expected one of: txt, las, laz."
        )

    chunk_pattern = f"chunk_*.{cloud_fmt}"

    scan_dirs = sorted(
        [p for p in chunks_root.iterdir() if p.is_dir()],
        key=extract_scan_id
    )
    if not scan_dirs:
        raise FileNotFoundError(f"No scan dirs found in {chunks_root}")

    print("\n======================================")
    print("[limatch] Running LiMatch on chunks")
    print(f"[limatch] chunks_root: {chunks_root}")
    print(f"[limatch] out_root:    {out_root}")
    print(f"[limatch] cloud_fmt:   {cloud_fmt}")
    print(f"[limatch] pattern:     {chunk_pattern}")
    print(f"[limatch] neighbor_k = {neighbor_k}")
    print(f"[limatch] intra-scan: chunk_i ↔ chunk_(i+1 ... i+{neighbor_k})")
    print(f"[limatch] cross-scan: {do_cross_scan}")
    print(f"[limatch] total scans: {len(scan_dirs)}")
    print("======================================\n")

    prev_last_chunk: Optional[Path] = None

    for scan_dir in scan_dirs:
        chunk_files = sorted(scan_dir.glob(chunk_pattern))
        if len(chunk_files) < 1:
            print(f"[limatch] skip {scan_dir.name}: no chunks matching {chunk_pattern}")
            continue

        pairs = []
        if len(chunk_files) >= 2:
            pairs += chunk_pairs_neighbors(chunk_files, neighbor_k)

        if do_cross_scan and (prev_last_chunk is not None):
            first_chunk = chunk_files[0]
            pairs.append((prev_last_chunk, first_chunk))

        prev_last_chunk = chunk_files[-1]

        if not pairs:
            print(f"[limatch] skip {scan_dir.name}: not enough pairs")
            continue

        scan_out = out_root / scan_dir.name
        scan_out.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[limatch] scan={scan_dir.name} | "
            f"chunks={len(chunk_files)} | "
            f"neighbor_k={neighbor_k} | "
            f"pairs={len(pairs)}"
        )

        for a, b in pairs:
            pair_name = f"{a.stem}__{b.stem}"
            pair_out = scan_out / pair_name

            try:
                run_limatch_api(
                    repo_root=repo_root,
                    limatch_cfg_path=limatch_cfg,
                    cloud1=a,
                    cloud2=b,
                    out_dir=pair_out,
                )
            except Exception as e:
                print(
                    f"[limatch] FAIL {scan_dir.name}/{pair_name}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

def find_spatial_crossing_pairs(
    chunks_root: Path,
    cloud_fmt: str,
    min_chunk_separation: int = 10,
    overlap_margin_m: float = 3.0,
) -> List[Tuple[Path, Path, str]]:
    """
    Trouve toutes les paires (chunk_i, chunk_j) qui se croisent
    spatialement avec |id_i - id_j| > min_chunk_separation.

    Lit les chunk_bbox.csv dans chaque sous-dossier merged_*.
    Retourne [(path_a, path_b, merged_parent_name), ...]
    """

    # Charge tous les bbox de tous les scans
    all_chunks = []
    scan_dirs = sorted(
        [p for p in chunks_root.iterdir() if p.is_dir()],
        key=extract_scan_id
    )

    for scan_dir in scan_dirs:
        bbox_path = scan_dir / "chunk_bbox.csv"
        if not bbox_path.exists():
            print(f"[crossings] No chunk_bbox.csv in {scan_dir.name} — skip")
            continue
        df = pd.read_csv(bbox_path)
        df["scan_dir"]  = str(scan_dir)
        df["chunk_path"] = df["chunk_file"].apply(
            lambda f: str(scan_dir / f)
        )
        all_chunks.append(df)

    if not all_chunks:
        print("[crossings] No bbox index found — cannot detect crossings")
        return []

    df_all = pd.concat(all_chunks, ignore_index=True)

    # Extrait les chunk_id numériques pour calculer la séparation
    def get_chunk_id(fname):
        m = re.search(r"chunk_(\d+)", fname)
        return int(m.group(1)) if m else -1

    df_all = df_all.sort_values("t_start").reset_index(drop=True)
    df_all["seq_idx"] = df_all.index

    # Détecte les croisements par overlap de bounding box avec marge
    pairs = []
    rows = df_all.to_dict("records")
    n = len(rows)

    print(f"[crossings] Checking {n} chunks for spatial overlaps "
          f"(min_separation={min_chunk_separation}, margin={overlap_margin_m}m)...")

    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]

            # Filtre séparation temporelle minimale
            if abs(a["seq_idx"] - b["seq_idx"]) <= min_chunk_separation:
                continue

            # Test overlap bounding box avec marge
            x_overlap = (a["x_min"] - overlap_margin_m <= b["x_max"] and
                         b["x_min"] - overlap_margin_m <= a["x_max"])
            y_overlap = (a["y_min"] - overlap_margin_m <= b["y_max"] and
                         b["y_min"] - overlap_margin_m <= a["y_max"])

            if x_overlap and y_overlap:
                path_a = Path(a["chunk_path"])
                path_b = Path(b["chunk_path"])
                # Le merged_parent est celui du chunk le plus ancien (a)
                merged_parent = Path(a["scan_dir"]).name
                pairs.append((path_a, path_b, merged_parent))

    print(f"[crossings] Found {len(pairs)} crossing pairs")
    return pairs


def run_limatch_on_spatial_crossings(
    chunks_root: Union[str, Path],
    limatch_cfg: Union[str, Path],
    out_root: Union[str, Path],
    min_chunk_separation: int = 10,
    overlap_margin_m: float = 3.0,
) -> None:
    """
    Lance LiMatch sur toutes les paires de chunks qui se croisent
    spatialement (hors paires consécutives).

    Structure de sortie identique à run_limatch_on_chunks_per_scan :
      out_root/{merged_parent}/{chunk_A}__{chunk_B}/
    """
    repo_root   = get_repo_root()
    chunks_root = Path(chunks_root)
    out_root    = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    limatch_cfg = Path(limatch_cfg)
    lim_cfg     = yaml.safe_load(open(str(limatch_cfg), "r"))
    cloud_fmt   = str(lim_cfg.get("cloud_fmt", "las")).lower()

    crossing_pairs = find_spatial_crossing_pairs(
        chunks_root=chunks_root,
        cloud_fmt=cloud_fmt,
        min_chunk_separation=min_chunk_separation,
        overlap_margin_m=overlap_margin_m,
    )

    if not crossing_pairs:
        print("[crossings] No crossing pairs found — nothing to do")
        return

    print(f"\n[crossings] Running LiMatch on {len(crossing_pairs)} crossing pairs")

    for a, b, merged_parent in crossing_pairs:
        pair_name = f"{a.stem}__{b.stem}"
        pair_out  = out_root / merged_parent / pair_name

        print(f"  {merged_parent}/{pair_name}")
        try:
            run_limatch_api(
                repo_root=repo_root,
                limatch_cfg_path=limatch_cfg,
                cloud1=a,
                cloud2=b,
                out_dir=pair_out,
            )
        except Exception as e:
            print(f"  [FAIL] {pair_name}: {type(e).__name__}: {e}")
            continue

    print("[crossings] Done")

def _pick_pair_files_any(pair_dir: Path, file_glob: str = "*.las") -> Tuple[Path, Path]:
    """
    Récupère 2 nuages dans pair_dir.
    Priorité aux noms Patch_from_scan_*_with_*.las
    Sinon, si exactement 2 fichiers matchent file_glob, on les prend.
    """
    clouds = sorted(pair_dir.glob(file_glob))

    if len(clouds) < 2:
        raise FileNotFoundError(f"Need 2 files in {pair_dir}, found {len(clouds)}")

    # Priorité aux fichiers Patch_from_scan_..._with_...
    pat = re.compile(r"^Patch_from_scan_(\d+)_with_(\d+)\.(las|txt)$", re.IGNORECASE)
    patch_clouds = [p for p in clouds if pat.match(p.name)]

    if len(patch_clouds) >= 2:
        if len(patch_clouds) == 2:
            return patch_clouds[0], patch_clouds[1]

        # si > 2, on essaie de trouver une vraie paire réciproque
        parsed = []
        for p in patch_clouds:
            m = pat.match(p.name)
            s1 = m.group(1)
            s2 = m.group(2)
            parsed.append((p, s1, s2))

        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                p1, a1, b1 = parsed[i]
                p2, a2, b2 = parsed[j]
                if a1 == b2 and b1 == a2:
                    return p1, p2

        return patch_clouds[0], patch_clouds[1]

    # fallback : si exactement 2 fichiers, on prend les 2
    if len(clouds) == 2:
        return clouds[0], clouds[1]

    raise RuntimeError(f"Ambiguous files in {pair_dir}: {[p.name for p in clouds]}")


def _collect_pair_dirs(
    patcher_out_root: Path,
    pattern_dir: str = r"^Flights_\d+_\d+$",
    file_glob: str = "*.las",
    descend_one_level: bool = True,
) -> List[Path]:
    """
    Cas gérés:
      A) patcher_out_root/Flights_1_2/*.las
      B) patcher_out_root/Flights_1_2/part_1/*.las
    """
    dir_re = re.compile(pattern_dir)
    top_dirs = sorted([p for p in patcher_out_root.iterdir() if p.is_dir() and dir_re.match(p.name)])

    if not top_dirs:
        raise FileNotFoundError(f"No Flights_*_* dirs found in {patcher_out_root}")

    pair_dirs = []

    for top in top_dirs:
        direct_files = sorted(top.glob(file_glob))
        if len(direct_files) >= 2:
            pair_dirs.append(top)
            continue

        if descend_one_level:
            subdirs = sorted([d for d in top.iterdir() if d.is_dir()])
            for sub in subdirs:
                sub_files = sorted(sub.glob(file_glob))
                if len(sub_files) >= 2:
                    pair_dirs.append(sub)

    return pair_dirs


def run_limatch_on_patcher_outputs(
    patcher_out_root: Union[str, Path],
    limatch_cfg: Union[str, Path],
    out_root: Optional[Union[str, Path]] = None,
    pattern_dir: str = r"^Flights_\d+_\d+$",
    file_glob: str = "*.las",
    descend_one_level: bool = True,
) -> None:
    """
    Gère :
      - paires directement dans Flights_*_*
      - ou paires un niveau plus bas
      - .las ou .txt selon file_glob
    """
    repo_root = get_repo_root()

    patcher_out_root = Path(patcher_out_root)
    if out_root is None:
        out_root = patcher_out_root.parent / "output_limatch"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_dirs = _collect_pair_dirs(
        patcher_out_root=patcher_out_root,
        pattern_dir=pattern_dir,
        file_glob=file_glob,
        descend_one_level=descend_one_level,
    )

    if not pair_dirs:
        raise FileNotFoundError(
            f"No pair dirs containing at least 2 files matching {file_glob} found in {patcher_out_root}"
        )

    print("\n======================================")
    print("[limatch] Running LiMatch on Patcher outputs")
    print(f"[limatch] patcher_out_root: {patcher_out_root}")
    print(f"[limatch] out_root:         {out_root}")
    print(f"[limatch] total pairs:      {len(pair_dirs)}")
    print("======================================\n")

    for pair_dir in pair_dirs:
        clouds = sorted(pair_dir.glob(file_glob))

        if len(clouds) < 2:
            print(f"[limatch] SKIP {pair_dir.name}: found {len(clouds)} file(s) (need 2).")
            continue

        try:
            cloud1, cloud2 = _pick_pair_files_any(pair_dir, file_glob=file_glob)
        except Exception as e:
            print(f"[limatch] SKIP {pair_dir}: {e}")
            continue

        # nom de sortie
        # si on est dans Flights_1_2/part_1 -> output = Flights_1_2_part_1
        parent_name = pair_dir.parent.name
        if re.match(pattern_dir, parent_name):
            pair_name = f"{parent_name}_{pair_dir.name}"
        else:
            pair_name = pair_dir.name

        pair_out = out_root / pair_name
        pair_out.mkdir(parents=True, exist_ok=True)

        print(f"[limatch] pair={pair_name} | cloud1={cloud1.name} | cloud2={cloud2.name}")

        try:
            run_limatch_api(
                repo_root=repo_root,
                limatch_cfg_path=limatch_cfg,
                cloud1=cloud1,
                cloud2=cloud2,
                out_dir=pair_out,
            )
        except Exception as e:
            print(f"[limatch] FAIL {pair_name}: {type(e).__name__}: {e}")
            continue

def merge_limatch_correspondences(
    limatch_root: Union[str, Path],
    workflow_type: str,
    output_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Merge all LiDAR_p2p files produced by LiMatch into one file.

    Parameters
    ----------
    limatch_root : path to the root LiMatch output folder
    workflow_type : "chunk" or "patcher"
    output_file : optional explicit output file path

    Returns
    -------
    Path to merged LiDAR_p2p file
    """
    limatch_root = Path(limatch_root)

    if not limatch_root.exists():
        raise FileNotFoundError(f"LiMatch root does not exist: {limatch_root}")

    if workflow_type == "chunk":
        pattern = "LiDAR_p2p_chunk*"
    elif workflow_type == "patcher":
        pattern = "LiDAR_p2p_Patch*"
    else:
        raise ValueError(f"Unknown workflow_type: {workflow_type}")

    cor_dirs = sorted(p for p in limatch_root.rglob("cor_outputs") if p.is_dir())

    files_to_merge = []
    for d in cor_dirs:
        files_to_merge.extend(sorted(d.glob(pattern)))

    files_to_merge = [f for f in files_to_merge if f.is_file()]

    if not files_to_merge:
        raise FileNotFoundError(
            f"No correspondence files found in {limatch_root} with pattern {pattern}"
        )

    if output_file is None:
        output_file = limatch_root / "LiDAR_p2p.txt"
    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print("[merge_correspondences] Merging LiMatch correspondences")
    print(f"[merge_correspondences] workflow_type: {workflow_type}")
    print(f"[merge_correspondences] limatch_root:  {limatch_root}")
    print(f"[merge_correspondences] pattern:      {pattern}")
    print(f"[merge_correspondences] n files:      {len(files_to_merge)}")
    print(f"[merge_correspondences] output_file:  {output_file}")
    print("======================================\n")

    with output_file.open("w", encoding="utf-8") as fout:
        for f in files_to_merge:
            with f.open("r", encoding="utf-8", errors="replace") as fin:
                content = fin.read()
                fout.write(content)
                if not content.endswith("\n"):
                    fout.write("\n")

    print(f"[merge_correspondences] Done: {output_file}")
    return output_file
# ============================================================================
# 4) Patcher as CLI 
# ============================================================================

def run_patcher_cli(patcher_cfg: Union[str,Path]) -> None:
    """
    Run Patcher as a standalone CLI (no Python imports from Patcher),
    so Patcher stays runnable independently and its internal imports stay unchanged.
    """
    repo_root = Path(__file__).resolve().parents[1]   # .../ESO-PDM
    patcher_dir = repo_root / "Patcher"

    patcher_cfg = Path(patcher_cfg)
    if not patcher_cfg.is_absolute():
        patcher_cfg = (repo_root / patcher_cfg).resolve()

    cmd = [sys.executable, "main.py", "-y", str(patcher_cfg)]
    print(f"\n[patcher] cwd={patcher_dir}")
    print(f"[patcher] cmd={' '.join(cmd)}\n")

    subprocess.run(cmd, cwd=str(patcher_dir), check=True)

# ============================================================================
# 5) Pipeline
# ============================================================================

def run_pipeline(pipe_cfg: dict) -> None:

    repo_root = get_repo_root()

    mode = pipe_cfg.get("mode")

    paths_cfg = pipe_cfg.get("paths", {})
    steps_cfg = pipe_cfg.get("steps", {})
    chunk_cfg = pipe_cfg.get("chunk", {})
    chunk_variant_cfg = pipe_cfg.get("chunk_variant", {})
    lim_cfg = pipe_cfg.get("limatch", {})
    patcher_block = pipe_cfg.get("patcher", {})
    merge_cor_cfg = pipe_cfg.get("merge_correspondences", {})

    limatch_cfg = repo_root / paths_cfg["limatch_cfg"]
    root_out_dir = Path(paths_cfg["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]

    # default outputs
    default_chunks_out = root_out_dir / scenario_name / "chunks"
    default_limatch_out = root_out_dir / scenario_name / "limatch"
    default_merged_out = root_out_dir / scenario_name / "merged"

    patcher_cfg = paths_cfg.get("patcher_cfg", None)

    print("\n======================================")
    print("[pipeline] Starting pipeline")
    print(f"[pipeline] mode:            {mode}")
    print(f"[pipeline] scenario:        {scenario_name}")
    print(f"[pipeline] root_out_dir:    {root_out_dir}")
    print(f"[pipeline] limatch_cfg:     {limatch_cfg}")
    print(f"[pipeline] patcher_cfg:     {patcher_cfg}")
    print("======================================\n")

    # ------------------------------------------------------------
    # Optional georef / merge
    # ------------------------------------------------------------
    pipeline_state = {
        "scanner_entries": [],
        "merged_groups": {},
    }

    if pipe_cfg["steps"].get("georef", False) or steps_cfg.get("merge", False):
        pipeline_state = georef_and_merge(pipe_cfg)

    # ------------------------------------------------------------
    # Standalone VUX+PUCK merge — runs even if steps.merge=false,
    # e.g. when working on already-georeferenced clouds
    # ------------------------------------------------------------
    elif steps_cfg.get("vux_puck_merge", False):
        vux_puck_cfg = pipe_cfg.get("vux_puck_merge", {})
        if not vux_puck_cfg.get("enabled", False):
            print("[pipeline] steps.vux_puck_merge=true but vux_puck_merge.enabled=false — skipping")
        else:
            scenario_root = root_out_dir / scenario_name
            vux_group_name = vux_puck_cfg["vux_group"]
            puck_key       = vux_puck_cfg["puck_scanner"]

            # Resolve VUX merged dir — explicit override or standard structure
            vux_dir = vux_puck_cfg.get("vux_input_dir", None)
            if vux_dir is None:
                vux_dir = scenario_root / "merged" / vux_group_name
            vux_dir = Path(vux_dir)

            # Resolve PUCK georef output dir from scanner cfg
            scanner_entries = load_scanner_entries(pipe_cfg)
            puck_entry = next(
                (e for e in scanner_entries
                 if e.get("key") == puck_key or e.get("scanner_name") == puck_key),
                None,
            )
            if puck_entry is None:
                raise ValueError(
                    f"vux_puck_merge.puck_scanner='{puck_key}' not found in scanners. "
                    f"Available: {[e.get('key') for e in scanner_entries]}"
                )
            puck_dir = Path(puck_entry["output_dir"])

            # Resolve output dir
            final_group_name = vux_puck_cfg.get("name", "ALL")
            final_out_dir    = vux_puck_cfg.get("output_dir", None)
            if final_out_dir is None:
                final_out_dir = scenario_root / "merged" / final_group_name
            final_out_dir = Path(final_out_dir)

            # Resolve manifest
            manifest_path = vux_puck_cfg.get("manifest_path", None)
            if manifest_path is None:
                manifest_path = vux_dir / "merged_manifest.csv"
            manifest_path = Path(manifest_path)

            print("\n======================================")
            print("[pipeline] Standalone VUX+PUCK merge")
            print(f"[pipeline] vux_dir:   {vux_dir}")
            print(f"[pipeline] puck_dir:  {puck_dir}")
            print(f"[pipeline] manifest:  {manifest_path}")
            print(f"[pipeline] out_dir:   {final_out_dir}")
            print("======================================\n")

            merged_dir = merge_vux_group_with_puck_by_time(
                vux_dir=vux_dir,
                puck_dir=puck_dir,
                out_dir=final_out_dir,
                manifest_path=manifest_path,
                output_suffix=vux_puck_cfg.get("output_suffix", "_VUX_PUCK"),
                scanner_src_vux=int(vux_puck_cfg.get("scanner_src_vux", 2)),
                scanner_src_puck=int(vux_puck_cfg.get("scanner_src_puck", 1)),
                chunk_size=int(vux_puck_cfg.get("chunk_size", 10_000_000)),
            )
            build_merged_time_manifest(
                merged_dir=merged_dir,
                cloud_fmt="las",
                manifest_name="merged_manifest.csv",
                time_field="gps_time",
            )
            pipeline_state["merged_groups"][final_group_name] = merged_dir

    if mode == "georef_only":
        print("[pipeline] georef_only done")
        return

    # ------------------------------------------------------------
    # Resolve merged input for downstream workflows
    # ------------------------------------------------------------
    merged_out = None

    # 1) explicit override
    if "merged_input_root" in chunk_cfg and chunk_cfg["merged_input_root"] is not None:
        merged_out = Path(chunk_cfg["merged_input_root"])

    # 2) from merge_groups outputs
    if merged_out is None:
        merged_group_name = chunk_cfg.get("merged_group", None)
        if merged_group_name is not None:
            if merged_group_name not in pipeline_state["merged_groups"]:
                raise ValueError(
                    f"chunk.merged_group='{merged_group_name}' not found in merged_groups outputs: "
                    f"{list(pipeline_state['merged_groups'].keys())}"
                )
            merged_out = Path(pipeline_state["merged_groups"][merged_group_name])

    # 3) fallback default merged folder
    if merged_out is None:
        merged_out = default_merged_out

    # ------------------------------------------------------------
    # Resolve georef reference cfg for chunking
    # ------------------------------------------------------------
    scanner_cfg_paths = get_scanner_cfg_paths(pipe_cfg)
    if not scanner_cfg_paths:
        raise ValueError("No scanners defined in pipe_cfg['scanners']")

    scanner_cfg_paths = get_scanner_cfg_paths(pipe_cfg)
    if not scanner_cfg_paths:
        raise ValueError("No scanners defined in pipe_cfg['scanners']")

    ref_scanner_name = chunk_cfg.get("reference_scanner", None)
    if ref_scanner_name is None:
        ref_scanner_name = get_scanner_name(scanner_cfg_paths[0])

    available_scanner_names = [get_scanner_name(p) for p in scanner_cfg_paths]
    if ref_scanner_name not in available_scanner_names:
        raise ValueError(
            f"chunk.reference_scanner='{ref_scanner_name}' not found among "
            f"{available_scanner_names}"
        )

    tmp_cfg_dir = root_out_dir / scenario_name / "tmp" / "generated_configs"
    cfg_georef_path = tmp_cfg_dir / f"georef_{ref_scanner_name}.generated.yml"

    if not cfg_georef_path.exists():
        # fallback: build it now
        ref_scanner_cfg_path = None
        for p in scanner_cfg_paths:
            if get_scanner_name(p) == ref_scanner_name:
                ref_scanner_cfg_path = p
                break

        if ref_scanner_cfg_path is None:
            raise ValueError(f"Could not resolve scanner cfg for {ref_scanner_name}")

        georef_cfg = build_georef_cfg(str(ref_scanner_cfg_path), pipe_cfg)
        write_temp_yaml(georef_cfg, cfg_georef_path)

    cfg_georef_path = str(cfg_georef_path)

    print("\n======================================")
    print("[pipeline] Resolved inputs")
    print(f"[pipeline] merged_out:      {merged_out}")
    print(f"[pipeline] georef ref cfg:  {cfg_georef_path}")
    print("======================================\n")

    # ------------------------------------------------------------
    # CHUNK WORKFLOW
    # ------------------------------------------------------------
    if mode == "chunk":

        chunk_source = chunk_cfg.get("source", "generate")
        chunk_variant_type = chunk_variant_cfg.get("type", "standard")
        limatch_out = None

        # 1) Resolve chunk source
        if chunk_source == "existing":
            chunks_root = Path(chunk_cfg["existing_root"])
            if not chunks_root.exists():
                raise FileNotFoundError(f"Existing chunks root does not exist: {chunks_root}")

            print("\n======================================")
            print("[chunk] Reusing existing chunks")
            print(f"[chunk] chunks_root: {chunks_root}")
            print("======================================\n")

        elif chunk_source == "generate":

            if chunk_variant_type == "standard":
                chunks_out = chunk_cfg.get("output_root", None)
                if chunks_out is None:
                    chunks_out = default_chunks_out

                chunks_root = chunk_txt(
                    merged_dir=merged_out,
                    cfg_georef_path=cfg_georef_path,
                    chunks_out=chunks_out,
                    L=chunk_cfg.get("length_m", 15.0),
                    min_points=chunk_cfg.get("min_points", 2000),
                    epsg_out=chunk_cfg.get("epsg_out", "EPSG:2056"),
                    delimiter=chunk_cfg.get("delimiter", ","),
                    skiprows=chunk_cfg.get("skiprows", 0),
                )

            elif chunk_variant_type == "outage_window":
                ow = chunk_variant_cfg.get("outage_window", {})
                if not ow.get("enabled", False):
                    raise ValueError(
                        "chunk_variant.type='outage_window' but chunk_variant.outage_window.enabled is False"
                    )

                out_root = ow.get("output_root", None)
                if out_root is None:
                    out_root = root_out_dir / scenario_name / "scenario_combined"

                chunks_root, gps_outage = combined_multi_outage_scenario(
                    merged_dir=merged_out,
                    cfg_georef_path=cfg_georef_path,
                    gps_in=paths_cfg["gps_input"],
                    outages=ow["outages"],
                    pre=ow.get("pre_s", 30.0),
                    post=ow.get("post_s", 30.0),
                    out_root=out_root,
                    delimiter=chunk_cfg.get("delimiter", ","),
                    min_points_chunk=chunk_cfg.get("min_points", 2000),
                    epsg_out=chunk_cfg.get("epsg_out", "EPSG:2056"),
                    do_chunks=steps_cfg.get("chunk", True),
                    chunk_cfg=chunk_cfg,
                )

                print(f"[chunk] outage-window chunks root: {chunks_root}")
                print(f"[chunk] outage GPS file: {gps_outage}")

            else:
                raise ValueError(f"Unknown chunk_variant.type: {chunk_variant_type}")

        else:
            raise ValueError(f"Unknown chunk.source: {chunk_source}")

        # 2) Run LiMatch
        if steps_cfg.get("limatch", False) and lim_cfg.get("run", True):
            limatch_out = lim_cfg.get("output_root", None)
            if limatch_out is None:
                limatch_out = default_limatch_out

            # 2a) Paires consécutives — seulement si demandé
            if lim_cfg.get("neighbor_k", 1) > 0 or lim_cfg.get("do_cross_scan", True):
                run_limatch_on_chunks_per_scan(
                    chunks_root=chunks_root,
                    limatch_cfg=limatch_cfg,
                    out_root=limatch_out,
                    do_cross_scan=lim_cfg.get("do_cross_scan", True),
                    neighbor_k=lim_cfg.get("neighbor_k", 1),
                )

            # 2b) Paires croisements spatiaux (optionnel)
            if lim_cfg.get("do_spatial_crossings", False):
                crossings_out = lim_cfg.get("crossings_output_root", None)
                if crossings_out is None:
                    crossings_out = Path(str(limatch_out) + "_crossings")
                print(f"[debug] crossing_min_separation = {lim_cfg.get('crossing_min_separation')}")
                print(f"[debug] crossing_overlap_margin_m = {lim_cfg.get('crossing_overlap_margin_m')}")
                run_limatch_on_spatial_crossings(
                    chunks_root=chunks_root,
                    limatch_cfg=limatch_cfg,
                    out_root=crossings_out,
                    min_chunk_separation=lim_cfg.get("crossing_min_separation", 10),
                    overlap_margin_m=lim_cfg.get("crossing_overlap_margin_m", 3.0),
                )

        # 3) Optional merge correspondences
        if merge_cor_cfg.get("enabled", False):
            if limatch_out is None:
                raise ValueError(
                    "merge_correspondences.enabled=True but LiMatch output is not available. "
                    "Enable steps.limatch or provide a valid limatch output root."
                )

            merged_p2p_out = merge_cor_cfg.get("output_file", None)
            merge_limatch_correspondences(
                limatch_root=limatch_out,
                workflow_type="chunk",
                output_file=merged_p2p_out,
            )

        return

    # ------------------------------------------------------------
    # PATCHER WORKFLOW
    # ------------------------------------------------------------
    elif mode == "patcher":

        patcher_source = patcher_block.get("source", "run")
        patcher_out_root = Path(patcher_block["output_root"])
        limatch_out = None

        # 1) Run patcher only if requested
        if patcher_source == "run":
            if patcher_cfg is None:
                raise ValueError("paths.patcher_cfg is required when patcher.source='run'")
            if patcher_block.get("run", False):
                run_patcher_cli(patcher_cfg)

        elif patcher_source == "existing":
            if not patcher_out_root.exists():
                raise FileNotFoundError(f"Patcher output root does not exist: {patcher_out_root}")

        else:
            raise ValueError(f"Unknown patcher.source: {patcher_source}")

        # 2) Optional GPS outage file generation
        gps_outage_block = patcher_block.get("gps_outage_file", {})
        if steps_cfg.get("gps_outage_file", False) and gps_outage_block.get("enabled", False):
            scenario_dir = Path(gps_outage_block["output_root"])
            scenario_dir.mkdir(parents=True, exist_ok=True)
            gps_out = scenario_dir / "GPS.txt"

            kept, removed = write_gps_multi_outage(
                Path(paths_cfg["gps_input"]),
                gps_out,
                gps_outage_block["outages"],
                delimiter=",",
            )

            print(f"[patcher] GPS outage generated: {gps_out}")
            print(f"[patcher] kept={kept}, removed={removed}")

        # 3) Run LiMatch on patcher outputs
        if steps_cfg.get("limatch", False) and lim_cfg.get("run", True):
            limatch_out = lim_cfg.get("output_root", None)
            if limatch_out is None:
                limatch_out = root_out_dir / scenario_name / "output_limatch"

            run_limatch_on_patcher_outputs(
                patcher_out_root=patcher_out_root,
                limatch_cfg=limatch_cfg,
                out_root=limatch_out,
            )

        # 4) Optional merge correspondences
        if merge_cor_cfg.get("enabled", False):
            if limatch_out is None:
                raise ValueError(
                    "merge_correspondences.enabled=True but LiMatch output is not available. "
                    "Enable steps.limatch or provide a valid limatch output root."
                )

            merged_p2p_out = merge_cor_cfg.get("output_file", None)
            merge_limatch_correspondences(
                limatch_root=limatch_out,
                workflow_type="patcher",
                output_file=merged_p2p_out,
            )

        return

    elif mode is None:
        return

    else:
        raise ValueError("mode must be one of: 'chunk', 'patcher', 'georef_only'")

def log_pipeline_config(cfg: dict):
    logging.info("")
    logging.info("============================================================")
    logging.info("                 PIPELINE CONFIGURATION")
    logging.info("============================================================")

    logging.info("Mode: %s", cfg.get("mode"))
    logging.info("Scenario name: %s", cfg.get("scenario_name"))

    logging.info("")
    logging.info("Trajectory:")
    traj = cfg.get("trajectory", {})
    logging.info("   Path: %s", traj.get("path"))
    logging.info("   Type: %s", traj.get("type"))

    logging.info("")
    logging.info("Distance filtering:")
    dist = cfg.get("distance_filtering", {})
    if dist:
        for k, v in dist.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Scanners:")
    scanners = cfg.get("scanners", {})
    if scanners:
        for key, scanner_cfg_path in scanners.items():
            if scanner_cfg_path is None:
                continue
            try:
                scanner_name = get_scanner_name(scanner_cfg_path)
            except Exception:
                scanner_name = "<unknown>"
            logging.info("   %s: %s  (scanner_name=%s)", key, scanner_cfg_path, scanner_name)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Paths:")
    paths = cfg.get("paths", {})
    if paths:
        for k, v in paths.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Steps:")
    steps = cfg.get("steps", {})
    if steps:
        for k, v in steps.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Merge:")
    merge = cfg.get("merge", {})
    if merge:
        for k, v in merge.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("VUX+PUCK merge:")
    vux_puck = cfg.get("vux_puck_merge", {})
    if vux_puck:
        for k, v in vux_puck.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Chunk:")
    chunk = cfg.get("chunk", {})
    if chunk:
        for k, v in chunk.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Chunk variant:")
    chunk_variant = cfg.get("chunk_variant", {})
    if chunk_variant:
        logging.info("   type: %s", chunk_variant.get("type"))
        if "outage_window" in chunk_variant:
            for k, v in chunk_variant["outage_window"].items():
                logging.info("   outage_window.%s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("LiMatch:")
    limatch = cfg.get("limatch", {})
    if limatch:
        for k, v in limatch.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Merge correspondences:")
    merge_cor = cfg.get("merge_correspondences", {})
    if merge_cor:
        for k, v in merge_cor.items():
            logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("")
    logging.info("Patcher:")
    patcher = cfg.get("patcher", {})
    if patcher:
        for k, v in patcher.items():
            if k == "gps_outage_file" and isinstance(v, dict):
                logging.info("   gps_outage_file:")
                for kk, vv in v.items():
                    logging.info("      %s: %s", kk, vv)
            else:
                logging.info("   %s: %s", k, v)
    else:
        logging.info("   <none>")

    logging.info("============================================================")
    logging.info("")

if __name__ == "__main__":

    #pipe_cfg = yaml.safe_load(open("navtools_PDM/PDM_configs/pipeline_patcher.yml", "r"))
    parser = argparse.ArgumentParser(
        description="LiDAR processing pipeline"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to pipeline YAML configuration file",
    )
    args = parser.parse_args()
 
    pipe_cfg = yaml.safe_load(open(args.config, "r"))

    setup_logger()
    log_pipeline_config(pipe_cfg)

    # Optional: generate all georef cfgs once for inspection
    scenario_name = pipe_cfg["scenario_name"]
    root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
    tmp_dir = root_out_dir / scenario_name / "tmp" / "generated_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scanner_cfg_paths = get_scanner_cfg_paths(pipe_cfg)

    logging.info("[pipeline] Generating temporary georef configs...")
    for scanner_cfg_path in scanner_cfg_paths:
        scanner_name = get_scanner_name(scanner_cfg_path)
        georef_cfg = build_georef_cfg(str(scanner_cfg_path), pipe_cfg)
        tmp_cfg_path = tmp_dir / f"georef_{scanner_name}.generated.yml"
        write_temp_yaml(georef_cfg, tmp_cfg_path)

        logging.info(
            "   scanner=%s | source=%s | generated=%s",
            scanner_name,
            scanner_cfg_path,
            tmp_cfg_path,
        )

    logging.info("[pipeline] Temporary config generation done.")

    # New pipeline entry point: everything comes from pipe_cfg
    run_pipeline(pipe_cfg=pipe_cfg)