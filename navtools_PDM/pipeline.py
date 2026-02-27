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
import yaml
from typing import List, Tuple, Union, Optional
import subprocess
import sys
import os

from .pointCloudGeoref import run_from_yaml
from .singleBeamMerging import merge_txt_pairs
from .Chunker import (
    chunk_txt_by_distance_streaming_intervals,
    write_gps_multi_outage,
    extract_time_windows_txt,
    merge_intervals,
    file_time_bounds_fast, 
)

# ============================================================================
# 0) Utils paths
# ============================================================================

def get_repo_root() -> Path:
    """
    navtools_PDM/pipeline.py -> parents[1] = repo root (ESO-PDM)
    """
    return Path(__file__).resolve().parents[1]

# ============================================================================
# 1) Georef + merge
# ============================================================================

def georef_and_merge(cfg_ha_path: str, cfg_lr_path: str, merged_out: str):
    # 1. Run georef both for HA and LR scanner
    run_from_yaml(cfg_ha_path)
    run_from_yaml(cfg_lr_path)

    # 2. Merge the point clouds of both single beam scanner  
    cfg_ha = yaml.safe_load(open(cfg_ha_path, "r"))
    cfg_lr = yaml.safe_load(open(cfg_lr_path, "r"))

    dir_ha = Path(cfg_ha["output"]["path"])
    dir_lr = Path(cfg_lr["output"]["path"])
    out_dir = Path(merged_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    merge_txt_pairs(
        dir_ha,
        dir_lr,
        out_dir,
        delimiter=",",
        out_prefix="merged_",
        out_suffix="_HA_LR",
    )

# ============================================================================
# 2) Chunking
# ============================================================================


def chunk_txt(
    merged_dir: Union[str, Path],
    cfg_georef_path: str,
    chunks_out: Optional[Union[str, Path]] = None,
    L: float = 36.0,
    epsg_out: str = "EPSG:2056",
    delimiter: str = ",",
    skiprows: int = 0,
    min_points: int = 2000,
) -> Path:
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

    merged_files = sorted(merged_dir.glob("*.txt"))
    if not merged_files:
        raise FileNotFoundError(f"No .txt found in merged directory: {merged_dir}")

    print(f"\n[chunk] Found {len(merged_files)} merged txt files in: {merged_dir}")
    print(f"[chunk] Output chunks dir: {chunks_root}")
    print(f"[chunk] Params: L={L} m, min_points={min_points}")

    for f in merged_files:
        out_sub = chunks_root / f.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        print(f"\n[chunk] Chunking: {f.name}")
        chunk_txt_by_distance_streaming(
            txt_path=str(f),
            cfg_georef_path=str(cfg_georef_path),  # reuse HA config for trajectory
            out_dir=str(out_sub),
            L=L,
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
) -> tuple[Path, Path]:
    """
    Combined GNSS outage scenario

    - creates ONE degraded GPS file
    - selects only involved merged point clouds
    - chunks directly inside outage time windows
    - optionally reuses existing chunks
    """

    merged_dir = Path(merged_dir)
    cfg_georef_path = Path(cfg_georef_path)
    gps_in = Path(gps_in)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1) GPS outage file
    # ============================================================

    gps_out = out_root / "GPS_outage.txt"
    kept, removed = write_gps_multi_outage(
        gps_in, gps_out, outages, delimiter=delimiter
    )

    print(
        f"[combined] GPS_outage written: {gps_out} "
        f"(kept={kept}, removed={removed})"
    )

    # ============================================================
    # 2) Time windows
    # ============================================================

    intervals = [
        (float(s) - pre, float(s) + float(d) + post)
        for (s, d) in outages
    ]
    intervals = merge_intervals(intervals)

    print("\n==============================")
    print("[combined scenario]")
    print(f"Outages: {outages}")
    print(f"Expanded intervals: {intervals}")
    print("==============================\n")

    chunks_root = out_root / "chunks_15m"
    chunks_root.mkdir(parents=True, exist_ok=True)

    if not do_chunks:
        print(f"[combined] reuse existing chunks → {chunks_root}")
        return chunks_root, gps_out

    # ============================================================
    # 3) Select involved merged clouds (FAST)
    # ============================================================

    merged_files = sorted(merged_dir.glob("*.txt"))
    if not merged_files:
        raise FileNotFoundError(f"No .txt found in {merged_dir}")

    def overlaps_any_interval(t0, t1, ints):
        return any(not (t1 < a or t0 > b) for a, b in ints)

    selected_files = []

    for f in merged_files:
        try:
            t0, t1 = file_time_bounds_fast(f, delimiter=delimiter)
            involved = overlaps_any_interval(t0, t1, intervals)

            status = "✅ involved" if involved else "❌ skipped"
            print(
                f"[outage selection] {f.name:<45} "
                f"time=[{t0:.3f},{t1:.3f}] → {status}"
            )

            if involved:
                selected_files.append(f)

        except Exception as e:
            print(f"[outage selection] {f.name} bounds FAIL → process")
            selected_files.append(f)

    print(
        f"\n[outage summary] "
        f"{len(selected_files)} / {len(merged_files)} involved\n"
    )

    # ============================================================
    # 4) DIRECT CHUNKING (NO EXTRACT STEP)
    # ============================================================

    processed = 0

    for f in selected_files:

        out_sub = chunks_root / f.stem

        # reuse existing chunks
        if (
            reuse_chunks
            and not force
            and out_sub.exists()
            and any(out_sub.glob("chunk_*.txt"))
        ):
            print(f"[combined] REUSE chunks → {out_sub}")
            processed += 1
            continue

        out_sub.mkdir(parents=True, exist_ok=True)

        print(
            f"[combined] CHUNK DIRECT 15m "
            f"{f.name} → {out_sub}",
            flush=True
        )

        chunk_txt_by_distance_streaming_intervals(
            txt_path=f,
            cfg_georef_path=str(cfg_georef_path),
            out_dir=str(out_sub),
            L=15.0,
            intervals=intervals,
            epsg_out=epsg_out,
            delimiter=delimiter,
            skiprows=0,
            min_points=min_points_chunk,
        )

        processed += 1

    print(
        f"\n[combined] processed files: "
        f"{processed}/{len(selected_files)}"
    )
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
    neighbor_k: int=2, 
) -> None:
    """
    chunks_root:
      .../subset/chunks/
        merged_100_HA_LR/
          chunk_0100.txt
          chunk_0101.txt
          ...
        merged_200_HA_LR/
          chunk_0200.txt
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
 
    scan_dirs = sorted([p for p in chunks_root.iterdir() if p.is_dir()])
    if not scan_dirs:
        raise FileNotFoundError(f"No scan dirs found in {chunks_root}")
    
    print("\n======================================")
    print("[limatch] Running LiMatch on chunks")
    print(f"[limatch] chunks_root: {chunks_root}")
    print(f"[limatch] out_root:    {out_root}")
    print(f"[limatch] neighbor_k = {neighbor_k}")
    print(f"[limatch] intra-scan: chunk_i ↔ chunk_(i+1 ... i+{neighbor_k})")
    print(f"[limatch] cross-scan: {do_cross_scan}")
    print(f"[limatch] total scans: {len(scan_dirs)}")
    print("======================================\n")
    
    prev_last_chunk: Optional[Path] = None # In order to run limatch on last chunk of a scan line with first chunk of the next scan

    for scan_dir in scan_dirs:
        chunk_files = sorted(scan_dir.glob("chunk_*.txt"))
        if len(chunk_files) < 1:
            print(f"[limatch] skip {scan_dir.name}: no chunks")
            continue

        # intra-scan pairs (comme avant)
        pairs = []
        if len(chunk_files) >= 2:
            pairs += chunk_pairs_neighbors(chunk_files, neighbor_k)

        # cross-scan pair: last(prev_scan) vs first(current_scan)
        if do_cross_scan and (prev_last_chunk is not None):
            first_chunk = chunk_files[0]
            pairs.append((prev_last_chunk, first_chunk))

        # update prev_last_chunk for next iteration
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
                print(f"[limatch] FAIL {scan_dir.name}/{pair_name}: {type(e).__name__}: {e}")
                continue
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


def run_pipeline(
    mode: Optional[str],
    cfg_ha: str,
    cfg_lr: str,
    merged_out: str,
    patcher_cfg: Union[str, None],
    do_georef_merge: bool = True,
) -> None:

    repo_root = get_repo_root()
    limatch_cfg = repo_root / "Patcher" / "submodules" / "limatch" / "configs" / "MLS.yml"

    if do_georef_merge:
        georef_and_merge(cfg_ha, cfg_lr, merged_out)

    if mode == "Chunk":
        # standard chunking 
        chunks_root = chunk_txt(
            merged_dir=merged_out,
            cfg_georef_path=cfg_ha,  # trajectory source
            chunks_out=os.path.join(os.path.dirname(merged_out) + "/chunks"),
            L=15.0,
            min_points=2000,
            epsg_out="EPSG:2056",
            delimiter=",",
            skiprows=0,
        )

        limatch_out = Path(merged_out).parent / "limatch_chunks"

        run_limatch_on_chunks_per_scan(
            chunks_root=chunks_root,
            limatch_cfg=limatch_cfg,
            out_root=limatch_out,
            do_cross_scan=True,
            neighbor_k=2,
        )

    elif mode == "OutageChunk":
        # Outage-driven chunking around a time interval, chunks remaining 15m
        # --- parameters to tune ---
        gps_in = "/media/b085164/Elements/Gobet_ODyN_v1/v1_base_AB/in/GPS.txt" 
        outages = [                 # Define GNSS outages as the GPS start time and the duration [s]
            (466590.0, 60.0),
            (466750.0, 90.0),
        ]
        
        pre = 30.0
        post = 30.0             # Chunking before and after the outages pre and post in [s]
        # --------------------------

        scenario_root = Path(merged_out).parent / "scenario_combined"

        chunks_root, gps_outage = combined_multi_outage_scenario(
            merged_dir=merged_out,
            cfg_georef_path=cfg_ha,
            gps_in=gps_in,
            outages=outages,
            pre=pre,
            post=post,
            out_root=scenario_root,
            delimiter=",",
            min_points_window=2000,
            min_points_chunk=2000,
            epsg_out="EPSG:2056",
            do_chunks=True,            # Set to True if you want to create chunk (normal case)
        )

        # run limatch on ALL consecutive chunks (same as simple chunk scenario)
        limatch_out = scenario_root / "limatch"
        run_limatch_on_chunks_per_scan(
            chunks_root=chunks_root,
            limatch_cfg=limatch_cfg,
            out_root=limatch_out,
            do_cross_scan=True,
            neighbor_k = 1,  # number to define the consecutive chunks on which limatch is run. 1: k <-> k+1, 2: k <-> k+1 and k <-> k+2
        )

        print(f"[combined] Ready for solver: GPS={gps_outage} limatch={limatch_out}")

    elif mode == "Patcher":
        if patcher_cfg is None:
            raise ValueError("patcher_cfg is required when mode='Patcher'")
        run_patcher_cli(patcher_cfg)

    elif mode is None:
        pass

    else:
        raise ValueError("mode must be 'Chunk' or 'OutageChunk' or 'Patcher'")


if __name__ == "__main__":

    cfg_ha = "navtools_PDM/PDM_configs/georef_SAM-HA.yml"
    cfg_lr = "navtools_PDM/PDM_configs/georef_SAM-LR.yml"
    merged_out = "/media/b085164/Elements/PCD_SAM/All_pcd/merged"
    patcher_cfg = "Patcher/config/MLS_Epalinges_config.yml"  # modify if needed

    mode = "OutageChunk"  # None / "Chunk" / "OutageChunk" / "Patcher"

    run_pipeline(
        mode,
        cfg_ha,
        cfg_lr,
        merged_out,
        patcher_cfg=patcher_cfg,
        do_georef_merge=False   # To set to False if files already Georeferenced 
    )