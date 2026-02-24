"""
Script de pipeline orchestrant:
- la lecture des .sdc
- le géoréférencement
- le merging des deux singles beam
- +++
"""

from pathlib import Path
import yaml

from pointCloudGeoref import run_from_yaml
from singleBeamMerging import merge_txt_pairs
from Chunker import chunk_txt_by_distance

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


def chunk_txt(
    merged_dir: Path,
    cfg_georef_path: str,
    chunks_out: str = None,
    L: float = 36.0,
    S: float = 16.0,
    epsg_out: str = "EPSG:2056",
    delimiter: str = ",",
    skiprows: int = 0,
    min_points: int = 2000,
):
    """
    Chunk all merged .txt files in merged_dir.
    Writes results in chunks_out/<merged_file_stem>/
    """
    merged_dir = Path(merged_dir)
    if chunks_out is None:
        chunks_dir = merged_dir / "chunks"
    else:
        chunks_dir = Path(chunks_out)

    chunks_dir.mkdir(parents=True, exist_ok=True)

    merged_files = sorted(merged_dir.glob("*.txt"))
    if not merged_files:
        raise FileNotFoundError(f"No .txt found in merged directory: {merged_dir}")

    print(f"\n[chunk] Found {len(merged_files)} merged txt files in: {merged_dir}")
    print(f"[chunk] Output chunks dir: {chunks_dir}")
    print(f"[chunk] Params: L={L} m, S={S} m, min_points={min_points}")

    for f in merged_files:
        out_sub = chunks_dir / f.stem
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


if __name__ == "__main__":

    cfg_ha = "PDM_configs/georef_SAM-HA.yml"
    cfg_lr = "PDM_configs/georef_SAM-LR.yml"
    merged_out = "/media/b085164/Elements/PCD_SAM/test_distance/merged"

    mode = "Chunk" # "Chunk" or "Patcher" 

#    georef_and_merge(cfg_ha, cfg_lr, merged_out)

    if mode == "Chunk":
        chunk_txt(
            merged_dir=merged_out,
            cfg_georef_path=cfg_ha,  # trajectory source
            chunks_out="/media/b085164/Elements/PCD_SAM/test_distance/chunks", ## MODIF -> recuperer basename de merged_out et rajouter "chunks"
            L=15.0,
            min_points=2000,
            epsg_out="EPSG:2056",
            delimiter=",",
            skiprows=0,
        )
    elif mode == "Patcher":
        # voir plus tard.. faire appel à la pipeline Patcher. 
        print("[patcher] Not implemented yet.")
        pass

    else:
        raise ValueError("mode must be 'Chunk' or 'Patcher'")