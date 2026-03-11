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

from .pointCloudGeoref import run_from_yaml
from .singleBeamMerging import merge_txt_pairs
from .Chunker import (
    chunk_txt_by_distance_streaming_intervals,
    write_gps_multi_outage,
    extract_time_windows_txt,
    merge_intervals,
    file_time_bounds_fast, 
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

# ============================================================================
# 1) Georef + merge
# ============================================================================

def georef_and_merge(cfg_ha_path: str, cfg_lr_path: str, merged_out: str):
    import shutil
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

    print("\n======================================")
    print("[merge] Starting HA + LR merging")
    print(f"[merge] HA dir: {dir_ha}")
    print(f"[merge] LR dir: {dir_lr}")
    print(f"[merge] OUT dir: {out_dir}")
    print("======================================\n")
    try:
        merge_txt_pairs(
            dir_ha,
            dir_lr,
            out_dir,
            delimiter=",",
            out_prefix="merged_",
            out_suffix="_HA_LR",
        )

        print("\n======================================")
        print("[merge] Merge completed successfully")
        print("[merge] Removing HA and LR folders")
        print("======================================\n")

        # delete HA and LR folders
        if dir_ha.exists():
            shutil.rmtree(dir_ha)
            print(f"[merge] Deleted {dir_ha}")

        if dir_lr.exists():
            shutil.rmtree(dir_lr)
            print(f"[merge] Deleted {dir_lr}")

    except Exception as e:
        print("\n======================================")
        print("[merge] ERROR during merging")
        print("[merge] HA/LR folders kept for debugging")
        print("======================================\n")
        raise
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
        chunk_txt_by_distance_streaming_intervals(
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


def run_limatch_on_patcher_outputs(
    patcher_out_root: Union[str, Path],
    limatch_cfg: Union[str, Path],
    out_root: Optional[Union[str, Path]] = None,
    pattern_dir: str = r"^Flights_\d+_\d+$",
    file_glob: str = "*.txt",
) -> None:
    """
    patcher_out_root:
      .../output_Patcher/
        Flights_3000_4000/
          <one or multiple .txt>
        Flights_3000_5000/
          ...

    out_root (default):
      sibling folder next to output_Patcher: .../output_limatch/
        Flights_3000_4000/
          (LiMatch outputs)
    """
    repo_root = get_repo_root()

    patcher_out_root = Path(patcher_out_root)
    if out_root is None:
        out_root = patcher_out_root.parent / "output_limatch"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dir_re = re.compile(pattern_dir)

    pair_dirs = sorted([p for p in patcher_out_root.iterdir() if p.is_dir() and dir_re.match(p.name)])
    if not pair_dirs:
        raise FileNotFoundError(f"No Flights_*_* dirs found in {patcher_out_root}")

    print("\n======================================")
    print("[limatch] Running LiMatch on Patcher outputs")
    print(f"[limatch] patcher_out_root: {patcher_out_root}")
    print(f"[limatch] out_root:         {out_root}")
    print(f"[limatch] total pairs:      {len(pair_dirs)}")
    print("======================================\n")

    for pair_dir in pair_dirs:
        clouds = sorted(pair_dir.glob(file_glob))

        # IMPORTANT:
        # LiMatch attend 2 nuages en entrée.
        # Si ton Patcher ne produit qu'UN seul .txt par paire,
        # on ne peut pas lancer LiMatch sans une stratégie de split (à définir).
        if len(clouds) < 2:
            print(f"[limatch] SKIP {pair_dir.name}: found {len(clouds)} file(s) (need 2).")
            continue

        # Si tu as exactement 2 fichiers -> parfait.
        # Si tu en as plus, on prend les 2 premiers (tu peux changer la logique ici).
        try:
            cloud1, cloud2 = _pick_pair_files(pair_dir)
        except Exception as e:
            print(f"[limatch] SKIP {pair_dir.name}: {e}")
            continue

        pair_out = out_root / pair_dir.name
        pair_out.mkdir(parents=True, exist_ok=True)

        print(f"[limatch] pair={pair_dir.name} | cloud1={cloud1.name} | cloud2={cloud2.name}")

        try:
            run_limatch_api(
                repo_root=repo_root,
                limatch_cfg_path=limatch_cfg,
                cloud1=cloud1,
                cloud2=cloud2,
                out_dir=pair_out,
            )
        except Exception as e:
            print(f"[limatch] FAIL {pair_dir.name}: {type(e).__name__}: {e}")
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

def run_pipeline(
    mode: Optional[str],
    cfg_ha: str,
    cfg_lr: str,
    merged_out: str,
    patcher_cfg: Union[str, None],
    pipe_cfg: dict,
) -> None:

    repo_root = get_repo_root()

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


    # Default outputs
    default_chunks_out = root_out_dir / scenario_name / "chunks"
    default_limatch_out = root_out_dir / scenario_name / "limatch"
    default_merged_out = root_out_dir / scenario_name / "merged"

    # ------------------------------------------------------------
    # Optional georef / merge
    # ------------------------------------------------------------
    if steps_cfg.get("georef", False) or steps_cfg.get("merge", False):
        georef_and_merge(cfg_ha, cfg_lr, merged_out)

    if mode == "georef_only":
        print("[pipeline] georef_only done")
        return

    # ------------------------------------------------------------
    # CHUNK WORKFLOW
    # ------------------------------------------------------------
    if mode == "chunk":

        chunk_source = chunk_cfg.get("source", "generate")
        chunk_variant_type = chunk_variant_cfg.get("type", "standard")

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
                    cfg_georef_path=cfg_ha,
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
                    cfg_georef_path=cfg_ha,
                    gps_in=paths_cfg["gps_input"],
                    outages=ow["outages"],
                    pre=ow.get("pre_s", 30.0),
                    post=ow.get("post_s", 30.0),
                    out_root=out_root,
                    delimiter=chunk_cfg.get("delimiter", ","),
                    min_points_chunk=chunk_cfg.get("min_points", 2000),
                    epsg_out=chunk_cfg.get("epsg_out", "EPSG:2056"),
                    do_chunks=steps_cfg.get("chunk", True),
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

            run_limatch_on_chunks_per_scan(
                chunks_root=chunks_root,
                limatch_cfg=limatch_cfg,
                out_root=limatch_out,
                do_cross_scan=lim_cfg.get("do_cross_scan", True),
                neighbor_k=lim_cfg.get("neighbor_k", 1),
            )
        if merge_cor_cfg.get("enabled", False):
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

        # 1) Run patcher only if requested
        if patcher_source == "run":
            if patcher_cfg is None:
                raise ValueError("patcher_cfg is required when patcher.source='run'")
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
        if merge_cor_cfg.get("enabled", False):
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
    logging.info("   Path: %s", cfg["trajectory"]["path"])
    logging.info("   Type: %s", cfg["trajectory"]["type"])

    logging.info("")
    logging.info("Distance filtering:")
    dist = cfg.get("distance_filtering", {})
    for k, v in dist.items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Scanners:")
    logging.info("   HA config: %s", cfg["scanners"]["ha_cfg"])
    logging.info("   LR config: %s", cfg["scanners"]["lr_cfg"])

    logging.info("")
    logging.info("Paths:")
    for k, v in cfg.get("paths", {}).items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Steps:")
    for k, v in cfg.get("steps", {}).items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Merge:")
    for k, v in cfg.get("merge", {}).items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Chunk:")
    for k, v in cfg.get("chunk", {}).items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Chunk variant:")
    logging.info("   type: %s", cfg.get("chunk_variant", {}).get("type"))
    if "outage_window" in cfg.get("chunk_variant", {}):
        for k, v in cfg["chunk_variant"]["outage_window"].items():
            logging.info("   outage_window.%s: %s", k, v)

    logging.info("")
    logging.info("LiMatch:")
    for k, v in cfg.get("limatch", {}).items():
        logging.info("   %s: %s", k, v)
        
    logging.info("")
    logging.info("Merge correspondences:")
    for k, v in cfg.get("merge_correspondences", {}).items():
        logging.info("   %s: %s", k, v)

    logging.info("")
    logging.info("Patcher:")
    for k, v in cfg.get("patcher", {}).items():
        if k == "gps_outage_file":
            logging.info("   gps_outage_file:")
            for kk, vv in v.items():
                logging.info("      %s: %s", kk, vv)
        else:
            logging.info("   %s: %s", k, v)


    logging.info("============================================================")
    logging.info("")

if __name__ == "__main__":

    pipe_cfg = yaml.safe_load(open("navtools_PDM/PDM_configs/pipeline.yml", "r"))

    setup_logger()
    log_pipeline_config(pipe_cfg)

    ha_cfg_path = pipe_cfg["scanners"]["ha_cfg"]
    lr_cfg_path = pipe_cfg["scanners"]["lr_cfg"]

    ha_cfg = build_georef_cfg(ha_cfg_path, pipe_cfg)
    lr_cfg = build_georef_cfg(lr_cfg_path, pipe_cfg)

    tmp_dir = Path(pipe_cfg["paths"]["root_out_dir"]) / "_generated_cfgs" / pipe_cfg["scenario_name"]
    tmp_ha = tmp_dir / "georef_HA.generated.yml"
    tmp_lr = tmp_dir / "georef_LR.generated.yml"

    write_temp_yaml(ha_cfg, tmp_ha)
    write_temp_yaml(lr_cfg, tmp_lr)

    merged_out = str(Path(pipe_cfg["paths"]["root_out_dir"]) / pipe_cfg["scenario_name"] / "merged")
    patcher_cfg = pipe_cfg["paths"]["patcher_cfg"]
    mode = pipe_cfg["mode"]

    run_pipeline(
        mode=mode,
        cfg_ha=str(tmp_ha),
        cfg_lr=str(tmp_lr),
        merged_out=merged_out,
        patcher_cfg=patcher_cfg,
        pipe_cfg=pipe_cfg,
    )