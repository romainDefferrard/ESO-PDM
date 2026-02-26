"""
Script de pipeline orchestrant:
- la lecture des .sdc
- le géoréférencement
- le merging des deux singles beam
- +++
"""

from pathlib import Path
import yaml
from typing import List, Tuple, Union, Optional
import subprocess
import sys
import os

from .pointCloudGeoref import run_from_yaml
from .singleBeamMerging import merge_txt_pairs
from .Chunker import chunk_txt_by_distance

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
        chunk_txt_by_distance(
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

def chunk_pairs_neighbor2(chunk_files: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Given sorted chunk files [c0, c1, c2, ...], return pairs:
    (c0,c1),(c0,c2),(c1,c2),(c1,c3),...
    i.e. (i,i+1) and (i,i+2).
    """
    pairs = []
    n = len(chunk_files)
    for i in range(n):
        if i + 1 < n:
            pairs.append((chunk_files[i], chunk_files[i+1]))
        if i + 2 < n:
            pairs.append((chunk_files[i], chunk_files[i+2]))
    return pairs

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
    """
    repo_root = get_repo_root()

    chunks_root = Path(chunks_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scan_dirs = sorted([p for p in chunks_root.iterdir() if p.is_dir()])
    if not scan_dirs:
        raise FileNotFoundError(f"No scan dirs found in {chunks_root}")

    for scan_dir in scan_dirs:
        chunk_files = sorted(scan_dir.glob("chunk_*.txt"))
        if len(chunk_files) < 2:
            print(f"[limatch] skip {scan_dir.name}: not enough chunks ({len(chunk_files)})")
            continue

        pairs = chunk_pairs_neighbor2(chunk_files)

        scan_out = out_root / scan_dir.name
        scan_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[limatch] scan={scan_dir.name} chunks={len(chunk_files)} pairs={len(pairs)}")

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

def run_pipeline(mode: str, cfg_ha: str, cfg_lr: str, merged_out: str, patcher_cfg: Union[str,None], do_georef_merge: bool = True) -> None:

    repo_root = get_repo_root()

    if do_georef_merge:
        georef_and_merge(cfg_ha, cfg_lr, merged_out)

    if mode.lower() == "chunk":
        # 1. chunk merged clouds
        
        chunks_root = chunk_txt(
            merged_dir=merged_out,
            cfg_georef_path=cfg_ha,  # trajectory source
            chunks_out=os.path.join(os.path.dirname(merged_out) +"/chunks"), 
            L=15.0,
            min_points=2000,
            epsg_out="EPSG:2056",
            delimiter=",",
            skiprows=0, 
            )
        
        # 2. LiMatch on chunks 
        limatch_cfg = repo_root / "Patcher" / "submodules" / "limatch" / "configs" / "MLS.yml"
        limatch_out = Path(merged_out).parent / "limatch_chunks"

        run_limatch_on_chunks_per_scan(
            chunks_root=chunks_root,
            limatch_cfg=limatch_cfg,
            out_root=limatch_out,
        )

    elif mode.lower() == "patcher":
        if patcher_cfg is None:
            raise ValueError("patcher_cfg is required when mode='Patcher'")
        run_patcher_cli(patcher_cfg)

    else:
        raise ValueError("mode must be 'Chunk' or 'Patcher'")


if __name__ == "__main__":

    cfg_ha = "navtools_PDM/PDM_configs/georef_SAM-HA.yml"
    cfg_lr = "navtools_PDM/PDM_configs/georef_SAM-LR.yml"
    merged_out = "/media/b085164/Elements/PCD_SAM/village/merged"
    patcher_cfg = "Patcher/config/MLS_Epalinges_config.yml"  # à modifier le dossier ou sont les nuages géoref dans la cfg Patcher

    mode = "Patcher" # "Chunk" or "Patcher" 

    run_pipeline(mode, cfg_ha, cfg_lr, merged_out, patcher_cfg=patcher_cfg, do_georef_merge=False) # set go_georef_merge to False is set of .sdc have already be georeferenced 