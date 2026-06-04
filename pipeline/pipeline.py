"""
pipeline.py — MLS processing pipeline.

Reads a YAML configuration file and executes a sequence of steps on
Mobile Laser Scanning (MLS) data. Each step is individually toggled via
the `steps` block in the config and carries its own parameter sub-block.

Pipeline steps (in order)
--------------------------
1. Georef  : Georeferences raw scanner data using trajectory + calibration.
             Produces one output directory per scanner.
2. Merge   : Merges georeferenced point clouds into unified spatial groups
             (e.g. ALL, HA_LR).
3. Chunk   : Splits the merged cloud into overlapping spatial chunks.
             Optionally runs LiMatch F2B / crossings on the resulting chunks.
4. S2S     : Runs LiMatch scan-to-scan matching on the merged cloud.
             Optionally applies a patcher before matching.

Config — steps block
---------------------
steps:
  georef: true
  merge:  true
  chunk:  true   # has its own limatch sub-block (F2B matching)
  s2s:    false  # has its own limatch sub-block (S2S matching)

Config — limatch sub-block (inside chunk or s2s)
-------------------------------------------------
  limatch:
    enabled:           true
    neighbor_k:        1
    do_crossings:      true
    uncertainty_r_min: 0.0   # optional — overrides uncertainty_r in LiMatch yml
    uncertainty_r_max: 2.0   # must provide both _min and _max together
    # uncertainty_r: 2.0     # alternative: single scalar (convenient for s2s)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from pipeline._log import setup as _setup_log, info, sub
from pipeline.steps import georef  as _georef
from pipeline.steps import merge   as _merge
from pipeline.steps import chunk   as _chunk
from pipeline.steps import limatch as _limatch


def _log_summary(cfg: dict) -> None:
    traj   = cfg.get("trajectory", {})
    steps  = cfg.get("steps", {})
    outage = cfg.get("outage", [])
    active = [k for k, v in steps.items() if v]

    info("─" * 60)
    info(f"[pipeline] {cfg.get('scenario_name')}  |  out={cfg['paths']['root_out_dir']}")
    info(f"[pipeline] steps : {' → '.join(active) if active else '—'}")
    sub(f"traj   : {traj.get('path', '—')}")
    if outage:
        t0, dur = float(outage[0]), float(outage[1])
        margin  = cfg.get("georef_time_window", {}).get("margin_s", 30.0)
        sub(f"outage : {t0}  +{dur}s  (±{margin}s georef buffer)")
    info("─" * 60)


def run_pipeline(pipe_cfg: dict) -> None:
    _setup_log()
    _log_summary(pipe_cfg)

    steps         = pipe_cfg.get("steps", {})
    root_out_dir  = Path(pipe_cfg["paths"]["root_out_dir"])
    scenario_name = pipe_cfg["scenario_name"]
    scenario_root = root_out_dir / scenario_name

    # Build scanner_entries from config — needed by merge even when georef=false
    pkg_dir      = Path(__file__).resolve().parent
    scanners_raw = pipe_cfg.get("scanners", {})
    scanner_entries: list[dict] = []
    for key, cfg_path_str in scanners_raw.items():
        if cfg_path_str is None:
            continue
        p = Path(cfg_path_str)
        if not p.is_absolute():
            p = (pkg_dir / p).resolve()
        name = _georef._scanner_name(p)
        scanner_entries.append({
            "key":                key,
            "cfg_path":           p,
            "scanner_name":       name,
            "generated_cfg_path": root_out_dir / scenario_name / "tmp" /
                                  "generated_configs" / f"georef_{name}.generated.yml",
            "output_dir":         root_out_dir / scenario_name / name,
        })

    # === 1. Georef ======================================================
    if steps.get("georef", False):
        scanner_entries = _georef.run(pipe_cfg)

    # === 2. Merge =======================================================
    merged_groups: dict = {}
    if steps.get("merge", False):
        merged_groups = _merge.run(pipe_cfg, scanner_entries)

    # Resolve merged_dir for chunk step
    merged_dir = (
        merged_groups.get("ALL")
        or merged_groups.get("HA_LR")
        or pipe_cfg.get("chunk", {}).get("merged_input_root")
        or scenario_root / "merged" / "ALL"
    )

    # Resolve reference georef config path for chunk step.
    # If georef was not run (no generated cfg on disk), build a minimal one
    # on-the-fly from the pipeline config so the chunker can load the trajectory.
    cfg_georef_path = _georef.get_ref_georef_cfg(pipe_cfg, scanner_entries) \
                      if scanner_entries else None

    if steps.get("chunk", False) and (
        cfg_georef_path is None or not Path(cfg_georef_path).exists()
    ):
        # Build and write a minimal georef config so the chunker can load the traj
        cfg_georef_path = _georef.write_minimal_georef_cfg(pipe_cfg, scenario_root)

    # === 3. Chunk + optional LiMatch F2B/crossings ======================
    if steps.get("chunk", False):
        chunks_root = _chunk.run(pipe_cfg, merged_dir, cfg_georef_path)

        chunk_lm = pipe_cfg.get("chunk", {}).get("limatch", {})
        if chunk_lm.get("enabled", False):
            _limatch.run(pipe_cfg, chunks_root, lm_block=chunk_lm)

    # === 4. S2S + optional LiMatch ======================================
    if steps.get("s2s", False):
        pipe_cfg["_merged_dir"] = str(merged_dir)
        _limatch.run_s2s(pipe_cfg)

    info("[pipeline] ✓ done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLS processing pipeline")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        pipe_cfg = yaml.safe_load(f)
    run_pipeline(pipe_cfg)