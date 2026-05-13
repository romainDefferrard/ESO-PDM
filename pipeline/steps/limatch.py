"""
steps/limatch.py
================
LiMatch step.

Changes vs previous version
----------------------------
cfg_overrides removed from public API.
uncertainty_r / uncertainty_r_min / uncertainty_r_max are now first-class
fields in the pipeline config under the limatch: block.

  Rule:
    - if both uncertainty_r_min AND uncertainty_r_max are present → use those,
      ignore uncertainty_r from the LiMatch yml
    - elif uncertainty_r is present → use that scalar
    - else → use whatever is in the LiMatch yml unchanged

New step: s2s (Scan-to-Scan via Patcher + s2s_chunks.py)
  Activated by steps.s2s: true in the pipeline config.
  Requires: paths.patcher_cfg and limatch.s2s block.

Public API
----------
run(pipe_cfg, chunks_root)          -> Path | None   (F2B + crossings)
run_s2s(pipe_cfg)                   -> Path | None   (Patcher + s2s_chunks)
merge_correspondences(root, output) -> Path
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


# ═══════════════════════════════════════════════════════════════
# REPO ROOT
# ═══════════════════════════════════════════════════════════════

def _repo_root() -> Path:
    # pipeline/steps/limatch.py → pipeline/ → ESO-PDM/
    return Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════
# LIMATCH IMPORT
# ═══════════════════════════════════════════════════════════════

def _get_match_clouds() -> callable:
    parent = _repo_root() / "Patcher" / "submodules"
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from limatch.main import match_clouds
    return match_clouds


# ═══════════════════════════════════════════════════════════════
# UNCERTAINTY_R INJECTION
# ═══════════════════════════════════════════════════════════════

def _inject_uncertainty(cfg: dict, lim_pipeline_cfg: dict) -> dict:
    """
    Inject uncertainty values from the pipeline config into the LiMatch cfg.

    If uncertainty_r_min + uncertainty_r_max are both given:
        → set both, remove uncertainty_r
    Elif uncertainty_r is given:
        → set it, remove _min/_max
    Else:
        → leave the LiMatch yml unchanged (use whatever is in the file)
    """
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

    return cfg


# ═══════════════════════════════════════════════════════════════
# RUN ONE PAIR
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# F2B — CONSECUTIVE PAIRS
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# SPATIAL CROSSINGS
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# MERGE CORRESPONDENCES
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# S2S — PATCHER + SPATIAL CHUNKING
# ═══════════════════════════════════════════════════════════════

def run_s2s(pipe_cfg: dict) -> Optional[Path]:
    """
    Scan-to-Scan step: Patcher → s2s_chunks → LiMatch.

    Flow
    ----
    1. Optionally run Patcher CLI to generate Patch_from_scan_*_with_*.las
    2. Load SBET trajectory
    3. For each pair: chunk spatially (forward-pass tangent grid) + LiMatch
    4. Optionally merge correspondences

    Config (under pipe_cfg)
    -----------------------
    paths.patcher_cfg          : Patcher YAML (relative to ESO-PDM root)
    paths.limatch_cfg          : LiMatch YAML
    trajectory.path            : SBET file

    limatch.s2s:
      run_patcher:   false      # run Patcher first?
      patcher_out_root:         # where Patcher put its outputs
      output_root:              # where to write s2s results (null = auto)
      L:             20.0       # chunk length (m)
      min_last_m:    null       # min last-chunk length (null = 2/3 * L)
      min_time_sep:  0.0        # skip pairs with mean-time separation < this (s)
      epsg:          EPSG:2056

    limatch.uncertainty_r / uncertainty_r_min / uncertainty_r_max
      → injected into LiMatch cfg as usual
    """
    repo_root = _repo_root()
    s2s_cfg   = pipe_cfg.get("s2s", {})
    paths_cfg = pipe_cfg.get("paths", {})
    merge_cor = pipe_cfg.get("merge_correspondences", {})

    if not s2s_cfg:
        warn("[s2s] limatch.s2s block missing — nothing to do")
        return None

    # ── Resolve paths ──────────────────────────────────────────
    limatch_cfg_path = (repo_root / paths_cfg["limatch_cfg"]).resolve()
    if not limatch_cfg_path.exists():
        raise FileNotFoundError(f"[s2s] LiMatch config not found: {limatch_cfg_path}")

    sbet_path = Path(pipe_cfg["trajectory"]["path"])
    if not sbet_path.exists():
        raise FileNotFoundError(f"[s2s] SBET not found: {sbet_path}")

    out_root = Path(
        s2s_cfg.get("output_root") or
        Path(paths_cfg["root_out_dir"]) / pipe_cfg["scenario_name"] / "s2s"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    patcher_out_root = Path(s2s_cfg["patcher_out_root"]) \
        if s2s_cfg.get("patcher_out_root") else out_root / "patcher_output"

    # ── 1. Optionally run Patcher ──────────────────────────────
    if bool(s2s_cfg.get("run_patcher", False)):
        patcher_cfg_rel = paths_cfg.get("patcher_cfg")
        if not patcher_cfg_rel:
            raise ValueError("[s2s] paths.patcher_cfg required when s2s.run_patcher=true")
        patcher_cfg_abs = (repo_root / patcher_cfg_rel).resolve()
        cmd = [sys.executable, "main.py", "-y", str(patcher_cfg_abs)]
        info(f"[s2s/patcher] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root / "Patcher"), check=True)
    else:
        if not patcher_out_root.exists():
            raise FileNotFoundError(
                f"[s2s] patcher_out_root not found: {patcher_out_root}. "
                "Set run_patcher: true or point patcher_out_root to an existing directory."
            )

    # ── 2. Load trajectory ─────────────────────────────────────
    # s2s_chunks.py lives alongside the pipeline package
    pkg_dir = Path(__file__).resolve().parents[1]   # pipeline/
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    from s2s_chunks import load_sbet, find_pairs, process_pair

    info(f"[s2s] loading trajectory: {sbet_path.name}")
    epsg = s2s_cfg.get("epsg", "EPSG:2056")
    t_trj, x_trj, y_trj = load_sbet(sbet_path, epsg_out=epsg)
    sub(f"{len(t_trj)} poses  t=[{t_trj[0]:.1f}, {t_trj[-1]:.1f}]")

    # ── 3. Find pairs and process ──────────────────────────────
    pairs = find_pairs(patcher_out_root)
    if not pairs:
        warn(f"[s2s] no Patch_from_scan_*_with_*.las pairs in {patcher_out_root}")
        return None

    info(f"[s2s] {len(pairs)} pair(s) found")

    L          = float(s2s_cfg.get("L",          20.0))
    min_last_m = float(s2s_cfg.get("min_last_m") or L * 2.0 / 3.0)
    min_t_sep  = float(s2s_cfg.get("min_time_sep", 0.0))

    # Inject uncertainty_r into a temp LiMatch cfg so process_pair picks it up
    s2s_lm_cfg = s2s_cfg.get("limatch", {})
    if s2s_lm_cfg:
        import tempfile
        _lm_base = yaml.safe_load(open(str(limatch_cfg_path), "r"))
        _inject_uncertainty(_lm_base, s2s_lm_cfg)
        _tmp_lm = Path(tempfile.mktemp(suffix=".yml"))
        yaml.safe_dump(_lm_base, open(str(_tmp_lm), "w"), sort_keys=False)
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

    # ── 4. Merge correspondences ───────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
# PIPELINE-FACING API
# ═══════════════════════════════════════════════════════════════

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