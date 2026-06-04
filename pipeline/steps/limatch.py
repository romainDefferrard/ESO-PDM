"""
steps/limatch.py
================
LiMatch step — runs point cloud correspondence matching on spatial chunks.

Two matching modes:
  F2B (forward-to-backward): matches consecutive chunk pairs within and
      across scan passes, controlled by neighbor_k and do_cross_scan.
  Crossings: detects spatially overlapping chunks from different passes
      via the chunk_bbox.csv index and runs LiMatch on those pairs.

uncertainty_r injection (from the limatch: block in pipe_cfg):
  - uncertainty_r_min + uncertainty_r_max present → range mode
  - uncertainty_r present                         → scalar mode
  - neither                                       → keep LiMatch yml value

Public API
----------
run(pipe_cfg, chunks_root, lm_block) -> Path | None
    F2B + spatial crossings on pre-generated chunks.

run_s2s(pipe_cfg) -> Path | None
    Patcher (headless) + S2S spatial chunking + LiMatch.

merge_correspondences(root, output) -> Path
    Collect all cor_outputs/ files into a single correspondence file.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import yaml

from pipeline._log import info, sub, warn
from .chunk import extract_scan_id


# === Repo root =================================================

def _repo_root() -> Path:
    # pipeline/steps/limatch.py → pipeline/ → ESO-PDM/
    return Path(__file__).resolve().parents[2]


# === LiMatch import ============================================

def _get_match_clouds() -> callable:
    parent = _repo_root() / "Patcher" / "submodules"
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from limatch.main import match_clouds
    return match_clouds


# === uncertainty_r injection ===================================

def _inject_uncertainty(cfg: dict, lim_pipeline_cfg: dict) -> dict:
    r_min = lim_pipeline_cfg.get("uncertainty_r_min")
    r_max = lim_pipeline_cfg.get("uncertainty_r_max")
    r     = lim_pipeline_cfg.get("uncertainty_r")

    if r_min is not None and r_max is not None:
        cfg["uncertainty_r_min"] = float(r_min)
        cfg["uncertainty_r_max"] = float(r_max)
        cfg.pop("uncertainty_r", None)
    elif r is not None:
        cfg["uncertainty_r"] = float(r)
        cfg.pop("uncertainty_r_min", None)
        cfg.pop("uncertainty_r_max", None)

    # max_kpts override
    max_kpts = lim_pipeline_cfg.get("max_kpts")
    if max_kpts is not None:
        cfg["max_kpts"] = int(max_kpts)

    return cfg


# === Run one pair ==============================================

def _run_pair(
    limatch_cfg_path: Path,
    cloud1:           Path,
    cloud2:           Path,
    out_dir:          Path,
    lim_pipeline_cfg: dict,
) -> None:
    """Load the LiMatch YAML, inject uncertainty_r, run match_clouds."""
    match_clouds = _get_match_clouds()
    cfg = yaml.safe_load(open(str(limatch_cfg_path), "r"))
    _inject_uncertainty(cfg, lim_pipeline_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["prj_folder"] = str(out_dir) + os.sep
    for d in ("plots", "tiles", "cor_outputs"):
        (out_dir / d).mkdir(exist_ok=True)
    match_clouds(str(cloud1), str(cloud2), cfg)


# === F2B — consecutive pairs ===================================

def _run_f2b(
    chunks_root:      Path,
    limatch_cfg_path: Path,
    out_root:         Path,
    cloud_fmt:        str,
    neighbor_k:       int,
    do_cross_scan:    bool,
    lim_pipeline_cfg: dict,
) -> None:
    chunk_pattern = f"chunk_*.{cloud_fmt}"
    scan_dirs = sorted(
        (p for p in chunks_root.iterdir() if p.is_dir()),
        key=extract_scan_id,
    )
    if not scan_dirs:
        raise FileNotFoundError(f"No scan dirs in {chunks_root}")

    info(f"[limatch/f2b] {len(scan_dirs)} scan(s)  k={neighbor_k}  cross={do_cross_scan}")
    prev_last: Optional[Path] = None

    for scan_dir in scan_dirs:
        files = sorted(scan_dir.glob(chunk_pattern))
        if not files:
            warn(f"[limatch/f2b] no {chunk_pattern} in {scan_dir.name} — skip")
            continue

        pairs: list[tuple[Path, Path]] = []
        if neighbor_k > 0:
            n = len(files)
            pairs += [(files[i], files[i + d])
                      for i in range(n) for d in range(1, neighbor_k + 1)
                      if i + d < n]
        if do_cross_scan and prev_last is not None:
            pairs.append((prev_last, files[0]))
        prev_last = files[-1]

        if not pairs:
            warn(f"[limatch/f2b] {scan_dir.name}: no pairs")
            continue

        sub(f"{scan_dir.name}: {len(files)} chunks  {len(pairs)} pair(s)")

        for a, b in pairs:
            pair_name = f"{a.stem}__{b.stem}"
            try:
                _run_pair(limatch_cfg_path, a, b,
                          out_root / scan_dir.name / pair_name,
                          lim_pipeline_cfg)
            except Exception as e:
                warn(f"[limatch/f2b] FAIL {scan_dir.name}/{pair_name}: "
                     f"{type(e).__name__}: {e}")


# === Spatial crossings =========================================

def _find_crossing_pairs(
    chunks_root: Path, cloud_fmt: str, min_sep: int, margin_m: float,
) -> list[tuple[Path, Path, str]]:
    import pandas as pd

    scan_dirs = sorted(
        (p for p in chunks_root.iterdir() if p.is_dir()), key=extract_scan_id
    )
    all_chunks = []
    for sd in scan_dirs:
        bbox = sd / "chunk_bbox.csv"
        if not bbox.exists():
            warn(f"[limatch/crossings] no chunk_bbox.csv in {sd.name} — skip")
            continue
        df = pd.read_csv(bbox)
        df["scan_dir"]   = str(sd)
        df["chunk_path"] = df["chunk_file"].apply(lambda f: str(sd / f))
        all_chunks.append(df)

    if not all_chunks:
        warn("[limatch/crossings] no bbox index — cannot detect crossings")
        return []

    df_all = (pd.concat(all_chunks, ignore_index=True)
                .sort_values("t_start")
                .reset_index(drop=True))
    df_all["seq_idx"] = df_all.index
    rows  = df_all.to_dict("records")
    n     = len(rows)
    pairs = []
    info(f"[limatch/crossings] {n} chunks  min_sep={min_sep}  margin={margin_m}m")

    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]
            if abs(a["seq_idx"] - b["seq_idx"]) <= min_sep:
                continue
            x_ok = (a["x_min"] - margin_m <= b["x_max"] and
                    b["x_min"] - margin_m <= a["x_max"])
            y_ok = (a["y_min"] - margin_m <= b["y_max"] and
                    b["y_min"] - margin_m <= a["y_max"])
            if x_ok and y_ok:
                pairs.append((Path(a["chunk_path"]), Path(b["chunk_path"]),
                               Path(a["scan_dir"]).name))

    info(f"[limatch/crossings] {len(pairs)} crossing pair(s)")
    return pairs


def _run_crossings(
    chunks_root:      Path,
    limatch_cfg_path: Path,
    out_root:         Path,
    cloud_fmt:        str,
    min_sep:          int,
    margin_m:         float,
    lim_pipeline_cfg: dict,
) -> None:
    crossing_pairs = _find_crossing_pairs(chunks_root, cloud_fmt, min_sep, margin_m)
    if not crossing_pairs:
        return
    for a, b, merged_parent in crossing_pairs:
        pair_name = f"{a.stem}__{b.stem}"
        sub(f"{merged_parent}/{pair_name}")
        try:
            _run_pair(limatch_cfg_path, a, b,
                      out_root / merged_parent / pair_name,
                      lim_pipeline_cfg)
        except Exception as e:
            warn(f"[limatch/crossings] FAIL {pair_name}: {type(e).__name__}: {e}")
    info("[limatch/crossings] done")


# === Merge correspondences =====================================

def merge_correspondences(
    limatch_root: Union[str, Path],
    output_file:  Optional[Union[str, Path]] = None,
    pattern:      str = "LiDAR_p2p_chunk*",
) -> Path:
    limatch_root = Path(limatch_root)
    cor_dirs = sorted(p for p in limatch_root.rglob("cor_outputs") if p.is_dir())
    files    = [f for d in cor_dirs for f in sorted(d.glob(pattern)) if f.is_file()]

    if not files:
        raise FileNotFoundError(
            f"No correspondence files (pattern={pattern}) under {limatch_root}"
        )

    output_file = Path(output_file) if output_file else limatch_root / "LiDAR_p2p.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    info(f"[limatch] merging {len(files)} file(s) → {output_file.name}")
    with output_file.open("w", encoding="utf-8") as fout:
        for f in files:
            content = f.read_text(encoding="utf-8", errors="replace")
            fout.write(content)
            if not content.endswith("\n"):
                fout.write("\n")

    return output_file


# === S2S — Patcher + spatial chunking ==========================
def run_s2s(pipe_cfg: dict) -> Optional[Path]:
    """
    Scan-to-Scan step: Patcher (headless) → s2s_chunks → LiMatch.

    Flow
    ----
    1. Build a Patcher config from pipe_cfg, overriding PC_DIR and OUTPUT_DIR.
       Skipped automatically if patcher_out_root already contains Flights_* dirs.
    2. Run Patcher in headless mode (no GUI) — skips launch_gui(), calls extract_mls() directly.
    3. Pass the patcher output (Flights_*/  dirs) to s2s_chunks.process_pair().
    4. Optionally merge correspondences.

    PC_DIR format expected by FlightData / Patcher
    -----------------------------------------------
    Template with {flight_id}, e.g.:
        /root/scenario/merged/ALL/merged_{flight_id}_VUX_PUCK.las

    Config keys (under pipe_cfg)
    ----------------------------
    paths.patcher_cfg       : base Patcher YAML to inherit non-overridden fields
    paths.limatch_cfg       : LiMatch YAML for s2s chunking
    trajectory.path         : SBET file

    s2s:
      patcher_out_root: null      # null = <root_out_dir>/<scenario>/s2s/patcher_output
      output_root:      null      # null = <root_out_dir>/<scenario>/s2s
      pc_dir_override:  null
      pc_dir_suffix:    "_VUX_PUCK.las"
      L:            20.0
      min_last_m:   null
      min_points:   500
      min_time_sep: 0.0
      epsg:         EPSG:2056
      limatch:
        enabled:       true
        uncertainty_r: 2.0        # optional — falls back to LiMatch yml value
        max_kpts:      10000      # optional — falls back to LiMatch yml value
    """
    repo_root = _repo_root()
    s2s_cfg   = pipe_cfg.get("s2s", {})
    paths_cfg = pipe_cfg.get("paths", {})
    merge_cor = pipe_cfg.get("merge_correspondences", {})

    if not s2s_cfg:
        warn("[s2s] no s2s: block in config — nothing to do")
        return None

    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]

    # === Resolve output dirs =====================================
    out_root = Path(
        s2s_cfg.get("output_root") or
        root_out_dir / scenario_name / "s2s"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    patcher_out_root = Path(
        s2s_cfg.get("patcher_out_root") or
        out_root / "patcher_output"
    )
    patcher_out_root.mkdir(parents=True, exist_ok=True)

    # === Build PC_DIR from merged/ALL ============================
    if s2s_cfg.get("pc_dir_override"):
        pc_dir = str(s2s_cfg["pc_dir_override"])
    else:
        merged_dir = (
            pipe_cfg.get("_merged_dir")
            or root_out_dir / scenario_name / "merged" / "ALL"
        )
        pc_suffix = s2s_cfg.get("pc_dir_suffix", "_VUX_PUCK.las")
        pc_dir = str(Path(merged_dir) / f"merged_{{flight_id}}{pc_suffix}")

    sub(f"PC_DIR: {pc_dir}")
    sub(f"patcher_out: {patcher_out_root}")

    # === Run Patcher — skip if output already exists =============
    existing_flights = list(patcher_out_root.glob("Flights_*"))
    if existing_flights:
        info(f"[s2s/patcher] {len(existing_flights)} existing Flights_* dir(s) found "
             f"— skipping Patcher")
    else:
        patcher_cfg_rel = paths_cfg.get("patcher_cfg")
        if not patcher_cfg_rel:
            raise ValueError("[s2s] paths.patcher_cfg is required when Patcher output "
                             "does not already exist.")

        patcher_cfg_abs = (repo_root / patcher_cfg_rel).resolve()
        if not patcher_cfg_abs.exists():
            raise FileNotFoundError(
                f"[s2s] Patcher config not found: {patcher_cfg_abs}")

        import yaml as _yaml
        patcher_config = _yaml.safe_load(open(str(patcher_cfg_abs), "r"))
        patcher_config["PC_DIR"]     = pc_dir
        patcher_config["OUTPUT_DIR"] = str(patcher_out_root)
        patcher_config["SCAN_MODE"]  = "MLS"

        patcher_dir = repo_root / "Patcher"
        if str(patcher_dir) not in sys.path:
            sys.path.insert(0, str(patcher_dir))

        info("[s2s] running Patcher (headless, no GUI)…")
        from utils.Patcher import PatcherPipeline
        pipeline = PatcherPipeline(patcher_config)
        pipeline.load_data()
        pipeline.generate_footprint()
        pipeline.extraction_state = True
        pipeline.output_dir       = str(patcher_out_root)
        pipeline.execute_limatch  = False
        pipeline.extract_mls()
        info(f"[s2s/patcher] done → {patcher_out_root}")

    # === S2S spatial chunking + LiMatch ==========================
    import yaml as _yaml   # may not have been imported above if patcher was skipped

    limatch_cfg_rel = paths_cfg.get("limatch_cfg", "")
    limatch_cfg_path = (repo_root / limatch_cfg_rel).resolve()
    if not limatch_cfg_path.exists():
        raise FileNotFoundError(f"[s2s] LiMatch config not found: {limatch_cfg_path}")

    sbet_path = Path(pipe_cfg["trajectory"]["path"])
    if not sbet_path.exists():
        raise FileNotFoundError(f"[s2s] SBET not found: {sbet_path}")

    # Import s2s_chunks — search in pipeline/ then ESO-PDM root
    _s2s_candidates = [
        Path(__file__).resolve().parent,              # pipeline/steps/  ← ajouter en premier
        Path(__file__).resolve().parents[1],          # pipeline/
        Path(__file__).resolve().parents[2],          # ESO-PDM/
        Path(__file__).resolve().parents[2] / "navtools_PDM",
    ]
    _s2s_found = False
    for _cand in _s2s_candidates:
        if (_cand / "s2s_chunks.py").exists():
            if str(_cand) not in sys.path:
                sys.path.insert(0, str(_cand))
            _s2s_found = True
            break
    if not _s2s_found:
        raise ImportError(
            "s2s_chunks.py not found. Place it in pipeline/ or ESO-PDM/."
        )
    from s2s_chunks import load_sbet, find_pairs, process_pair

    info(f"[s2s] loading trajectory: {sbet_path.name}")
    epsg = s2s_cfg.get("epsg", "EPSG:2056")
    t_trj, x_trj, y_trj = load_sbet(sbet_path, epsg_out=epsg)
    sub(f"{len(t_trj)} poses  t=[{t_trj[0]:.1f}, {t_trj[-1]:.1f}]")

    pairs = find_pairs(patcher_out_root)
    if not pairs:
        warn(f"[s2s] no Patch_from_scan_*_with_*.las pairs in {patcher_out_root}")
        return None

    info(f"[s2s] {len(pairs)} pair(s) found")

    L          = float(s2s_cfg.get("L",          20.0))
    min_last_m = L * 2.0 / 3.0
    min_t_sep  = float(s2s_cfg.get("min_time_sep", 0.0))

    # === Inject LiMatch overrides into a temp config =============
    # All fields are optional — if absent, the LiMatch yml value is kept.
    s2s_lm_cfg = s2s_cfg.get("limatch", {})
    if s2s_lm_cfg:
        import tempfile
        _lm_base = _yaml.safe_load(open(str(limatch_cfg_path), "r"))
        _inject_uncertainty(_lm_base, s2s_lm_cfg)
        _tmp_lm = Path(tempfile.mktemp(suffix=".yml"))
        _yaml.safe_dump(_lm_base, open(str(_tmp_lm), "w"), sort_keys=False)
        limatch_cfg_path = _tmp_lm

    for fwd_path, bwd_path in pairs:
        pair_name = f"{fwd_path.stem}_vs_{bwd_path.stem}"
        sub(f"{pair_name}")
        try:
            process_pair(
                fwd_path=fwd_path,
                bwd_path=bwd_path,
                out_dir=out_root / pair_name,
                t_trj=t_trj, x_trj=x_trj, y_trj=y_trj,
                L=L,
                min_last_chunk_m=min_last_m,
                limatch_cfg_path=limatch_cfg_path,
                repo_root=repo_root,
                min_time_sep=min_t_sep,
            )
        except Exception as e:
            warn(f"[s2s] FAIL {pair_name}: {type(e).__name__}: {e}")

    # === Merge correspondences ===================================
    if merge_cor.get("enabled", False):
        mc_out = merge_cor.get("output_file")
        try:
            merge_correspondences(
                limatch_root=out_root,
                output_file=Path(mc_out) if mc_out else None,
                pattern="LiDAR_p2p_Patch*",
            )
        except FileNotFoundError as e:
            warn(f"[s2s] merge_correspondences: {e}")

    info(f"[s2s] done → {out_root}")
    return out_root


def run(pipe_cfg: dict, chunks_root: Union[str, Path], lm_block: dict | None = None) -> Optional[Path]:
    """
    F2B + spatial crossings on pre-generated chunks.

    lm_block : the limatch sub-block from the calling step (chunk.limatch or s2s.limatch).
               If None, falls back to pipe_cfg['limatch'] for backward compatibility.
    """
    chunks_root  = Path(chunks_root)
    repo_root    = _repo_root()
    root_out_dir = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario     = pipe_cfg["scenario_name"]
    lim_cfg      = lm_block if lm_block is not None else pipe_cfg.get("limatch", {})
    merge_cor    = pipe_cfg.get("merge_correspondences", {})

    limatch_cfg_path = (repo_root / pipe_cfg["paths"]["limatch_cfg"]).resolve()
    if not limatch_cfg_path.exists():
        raise FileNotFoundError(f"LiMatch config not found: {limatch_cfg_path}")

    limatch_out   = Path(lim_cfg.get("output_root") or
                         root_out_dir / scenario / "limatch")
    crossings_out = Path(lim_cfg.get("crossings_output_root") or
                         str(limatch_out) + "_crossings")

    lim_base  = yaml.safe_load(open(str(limatch_cfg_path), "r"))
    cloud_fmt = str(lim_base.get("cloud_fmt", "las")).lower()

    neighbor_k   = int( lim_cfg.get("neighbor_k",           0))
    do_cross     = bool(lim_cfg.get("do_cross_scan",         False))
    do_crossings = bool(lim_cfg.get("do_spatial_crossings",  False))
    ran          = False

    if neighbor_k > 0 or do_cross:
        _run_f2b(
            chunks_root=chunks_root,
            limatch_cfg_path=limatch_cfg_path,
            out_root=limatch_out,
            cloud_fmt=cloud_fmt,
            neighbor_k=neighbor_k,
            do_cross_scan=do_cross,
            lim_pipeline_cfg=lim_cfg,
        )
        ran = True

    if do_crossings:
        _run_crossings(
            chunks_root=chunks_root,
            limatch_cfg_path=limatch_cfg_path,
            out_root=crossings_out,
            cloud_fmt=cloud_fmt,
            min_sep=int(  lim_cfg.get("crossing_min_separation",   10)),
            margin_m=float(lim_cfg.get("crossing_overlap_margin_m", 3.0)),
            lim_pipeline_cfg=lim_cfg,
        )
        ran = True

    if not ran:
        warn("[limatch] nothing to run "
             "(neighbor_k=0, do_cross_scan=false, do_spatial_crossings=false)")
        return None

    if merge_cor.get("enabled", False):
        mc_out = merge_cor.get("output_file")
        merge_correspondences(limatch_out, Path(mc_out) if mc_out else None)

    info(f"[limatch] done → {limatch_out}")
    return limatch_out