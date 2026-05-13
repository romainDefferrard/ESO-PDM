"""
project_io.py
=============
All disk I/O and pipeline-YAML construction.
No Streamlit imports here — pure data layer.

Key design decisions
---------------------
- "mode" is derived from active steps (never stored in UI dict).
- The outage window is a single global field [t_start, duration] shared
  by georef_time_window AND chunk_variant (always enabled when chunking).
- min_last_chunk_m is enforced as 2/3 * length_m in build_pipeline_yaml.
- chunk_variant is always active (hardcoded); no UI toggle needed.
- lm_run follows step_limatch (not a separate toggle).
"""

from __future__ import annotations
from pathlib import Path
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
STREAMLIT_DIR   = Path(__file__).parent
REPO_ROOT       = STREAMLIT_DIR.parent
PROJECTS_DIR    = STREAMLIT_DIR / "projects"
DEFAULT_CFGS    = STREAMLIT_DIR / "default_configs"
PIPELINE_SCRIPT = REPO_ROOT / "navtools_PDM" / "pipeline.py"

PROJECTS_DIR.mkdir(exist_ok=True)

# ── Low-level helpers ──────────────────────────────────────────────────────

def project_dir(p: str)    -> Path: return PROJECTS_DIR / p
def scanners_dir(p: str)   -> Path: return project_dir(p) / "scanners"
def scenarios_dir(p: str)  -> Path: return project_dir(p) / "scenarios"
def project_meta_path(p: str) -> Path: return project_dir(p) / "project.yml"
def scanner_path(p: str, sc: str)  -> Path: return scanners_dir(p)  / f"{sc}.yml"
def scenario_path(p: str, sc: str) -> Path: return scenarios_dir(p) / f"{sc}.yml"

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}

def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

# ── Listing ────────────────────────────────────────────────────────────────

def list_projects() -> list[str]:
    return sorted(
        p.name for p in PROJECTS_DIR.iterdir()
        if p.is_dir() and project_meta_path(p.name).exists()
    )

def list_scenarios(proj: str) -> list[str]:
    d = scenarios_dir(proj)
    return sorted(p.stem for p in d.glob("*.yml")) if d.exists() else []

def list_scanner_files(proj: str) -> list[str]:
    d = scanners_dir(proj)
    return sorted(p.stem for p in d.glob("*.yml")) if d.exists() else []

# ── Project / scanner / scenario I/O ──────────────────────────────────────

def load_project_meta(proj: str) -> dict:  return load_yaml(project_meta_path(proj))
def save_project_meta(proj: str, meta: dict): save_yaml(project_meta_path(proj), meta)
def load_scanner(proj: str, sc: str) -> dict:  return load_yaml(scanner_path(proj, sc))
def save_scanner(proj: str, sc: str, data: dict): save_yaml(scanner_path(proj, sc), data)
def load_scenario(proj: str, sc: str) -> dict:  return load_yaml(scenario_path(proj, sc))
def save_scenario(proj: str, sc: str, data: dict): save_yaml(scenario_path(proj, sc), data)

# ── Default structures ─────────────────────────────────────────────────────

def default_project_meta(name: str) -> dict:
    return {
        "name":         name,
        "description":  "",
        "root_out_dir": "",
        "gps_input":    "",
        "patcher_cfg":  "Patcher/config/MLS_Epalinges_config.yml",
        "scanners":     ["HA", "LR", "PUCK"],
    }

def default_scanner(name: str) -> dict:
    return {
        "scanner_name": name,
        "lasvec": {"type": "SDC", "cols": [0, 3, 4, 5], "path": f"/path/to/SDC/{name}"},
        "leapsec": 18,
        "mount": {
            "R_mount":   [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            "boresight": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "lever_arm": [0.0, 0.0, 0.0],
        },
        "output_defaults": {"type": "LAS", "lasvec": True, "lasvec_to_body": True},
        "manifest_path": f"/path/to/manifests/manifest_{name}.csv",
    }

def default_scenario_ui(name: str = "") -> dict:
    """
    Flat UI dict for one scenario.

    Outage is a single global definition used by:
      - georef_time_window (when tw_enable=True)
      - chunk_variant (always, when chunking is active)
      - GPS outage file generation (when step_gps_outage=True)

    Fields removed vs previous version:
      - chunk_min_points    (not used)
      - chunk_time_field    (hardcoded to gps_time in YAML builder)
      - cv_*                (chunk variant is always on; uses outage_* + buf_*)
      - lm_run              (derived from step_limatch)
      - tw_use_list / tw_outages_json  (single outage only)
    """
    return {
        "name":          name,
        "scenario_name": f"{name}/georef_F2B",
        "root_out_dir":  "",          # per-scenario override (empty = use project meta)
        "traj_path":     "",
        "traj_type":     "SBET",
        "limatch_cfg":   "Patcher/submodules/limatch/configs/MLS_F2B_1.yml",
        # ── steps ──────────────────────────────────────────────────────────
        "step_georef":     True,
        "step_merge":      True,
        "step_chunk":      False,
        "step_limatch":    False,
        "step_gps_outage": False,
        # ── outage (global) ────────────────────────────────────────────────
        "outage_start":  305120.0,   # GPS seconds
        "outage_dur":    580.0,      # seconds
        "buf_pre_s":     30.0,       # buffer before outage for georef window
        "buf_post_s":    30.0,       # buffer after  outage for georef window
        # ── georef / time window ───────────────────────────────────────────
        "tw_enable":     False,      # restrict georef to outage window
        # ── distance filtering ─────────────────────────────────────────────
        "dist_enable":   True,
        "dist_max_m":    30.0,
        "dist_epsg":     "EPSG:2056",
        # ── merge ──────────────────────────────────────────────────────────
        "merge_preset":               "all",
        "merge_out_prefix":           "merged_",
        "merge_out_suffix":           "_HA_LR",
        "merge_output_suffix":        "_VUX_PUCK",
        "merge_src_vux":              2,
        "merge_src_puck":             1,
        "merge_chunk_size":           10_000_000,
        "merge_cleanup":              True,
        "merge_cleanup_scanner_dirs": True,
        "merge_vux_input_dir":        "",
        # ── chunk ──────────────────────────────────────────────────────────
        "chunk_source":            "generate",
        "chunk_existing_root":     "",
        "chunk_merged_input_root": "",
        "chunk_length_m":          15.0,
        "chunk_epsg":              "EPSG:2056",
        "chunk_fmt":               "las",
        # ── limatch ────────────────────────────────────────────────────────
        "lm_output_root":           "",
        "lm_neighbor_k":            0,
        "lm_do_cross_scan":         False,
        "lm_do_spatial_crossings":  True,
        "lm_crossings_output_root": "",
        "lm_crossing_min_sep":      30,
        "lm_crossing_overlap_m":    3.0,
        "lm_use_overrides":         False,
        "lm_f2b_uncertainty_r_min":   0.0,
        "lm_f2b_uncertainty_r_max":   2.0,
        "lm_cross_uncertainty_r_min": 0.0,
        "lm_cross_uncertainty_r_max": 2.0,
        # ── merge correspondences (now in LiMatch section) ─────────────────
        "mc_enabled": True,
    }

# ── Build pipeline YAML ────────────────────────────────────────────────────

def build_pipeline_yaml(project: str, ui: dict) -> dict:
    meta     = load_project_meta(project)
    scanners = list_scanner_files(project)

    # Scanner entries
    scanner_entries = {f"{s.lower()}_cfg": str(scanner_path(project, s)) for s in scanners}
    vux_keys  = [f"{s.lower()}_cfg" for s in scanners if "PUCK" not in s.upper()]
    puck_keys = [f"{s.lower()}_cfg" for s in scanners if "PUCK"     in s.upper()]
    puck_key  = puck_keys[0] if puck_keys else "puck_cfg"

    # ── Outage window (global) ─────────────────────────────────────────────
    t_start  = float(ui["outage_start"])
    duration = float(ui["outage_dur"])
    buf_pre  = float(ui["buf_pre_s"])
    buf_post = float(ui["buf_post_s"])

    # ── Georef time window ─────────────────────────────────────────────────
    tw: dict = {"enable": bool(ui["tw_enable"])}
    if ui["tw_enable"]:
        tw["outage"]   = [t_start, duration]
        tw["margin_s"] = max(buf_pre, buf_post)   # conservative: largest buffer

    # ── Chunk: min_last_chunk_m = 2/3 * length_m (enforced here, not in UI) ─
    length_m       = float(ui["chunk_length_m"])
    min_last_chunk = round(length_m * 2.0 / 3.0, 2)

    # ── Chunk variant (always active when chunking) ─────────────────────────
    chunk_variant = {
        "type": "outage_window",
        "outage_window": {
            "enabled":     True,
            "output_root": None,
            "outages":     [[t_start, duration]],
            "pre_s":       buf_pre,
            "post_s":      buf_post,
        },
    }

    # ── Merge ──────────────────────────────────────────────────────────────
    merge: dict = {
        "preset":           ui["merge_preset"],
        "vux_scanners":     vux_keys,
        "puck_scanner":     puck_key,
        "out_prefix":       ui["merge_out_prefix"],
        "out_suffix":       ui["merge_out_suffix"],
        "output_suffix":    ui["merge_output_suffix"],
        "scanner_src_vux":  ui["merge_src_vux"],
        "scanner_src_puck": ui["merge_src_puck"],
        "chunk_size":       ui["merge_chunk_size"],
        "cleanup": {
            "enabled":      ui["merge_cleanup"],
            "scanner_dirs": ui["merge_cleanup_scanner_dirs"],
        },
    }
    if ui["merge_preset"] == "puck_on_existing" and ui.get("merge_vux_input_dir","").strip():
        merge["vux_input_dir"] = ui["merge_vux_input_dir"].strip()

    # ── LiMatch — run follows step_limatch ─────────────────────────────────
    limatch: dict = {
        "run":                       bool(ui.get("step_limatch", False)),
        "output_root":               ui["lm_output_root"] or None,
        "neighbor_k":                ui["lm_neighbor_k"],
        "do_cross_scan":             ui["lm_do_cross_scan"],
        "do_spatial_crossings":      ui["lm_do_spatial_crossings"],
        "crossings_output_root":     ui["lm_crossings_output_root"] or None,
        "crossing_min_separation":   ui["lm_crossing_min_sep"],
        "crossing_overlap_margin_m": ui["lm_crossing_overlap_m"],
    }
    if ui.get("lm_use_overrides"):
        limatch["cfg_overrides_f2b"] = {
            "uncertainty_r_min": ui["lm_f2b_uncertainty_r_min"],
            "uncertainty_r_max": ui["lm_f2b_uncertainty_r_max"],
        }
        limatch["cfg_overrides_crossings"] = {
            "uncertainty_r_min": ui["lm_cross_uncertainty_r_min"],
            "uncertainty_r_max": ui["lm_cross_uncertainty_r_max"],
        }

    # ── Mode derived from steps ─────────────────────────────────────────────
    mode = "chunk" if (ui.get("step_chunk") or ui.get("step_limatch")) else "georef_only"

    # ── root_out_dir: scenario override > project meta ─────────────────────
    root_out = ui.get("root_out_dir","").strip() or meta.get("root_out_dir","")

    return {
        "mode":          mode,
        "scenario_name": ui["scenario_name"],
        "scanners":      scanner_entries,
        "merge_groups":  [{"name": "HA_LR", "scanners": vux_keys}],
        "trajectory": {
            "type": ui["traj_type"],
            "path": ui["traj_path"],
            "cfg":  {"type": ui["traj_type"]},
        },
        "georef_time_window": tw,
        "distance_filtering": {
            "enable":         ui["dist_enable"],
            "max_distance_m": ui["dist_max_m"],
            "map_epsg":       ui["dist_epsg"],
            "filter_trj":     None,
        },
        "paths": {
            "root_out_dir": root_out,
            "patcher_cfg":  meta.get("patcher_cfg", "Patcher/config/MLS_Epalinges_config.yml"),
            "limatch_cfg":  ui.get("limatch_cfg", ""),
            "gps_input":    meta.get("gps_input", ""),
        },
        "steps": {
            "georef":          ui["step_georef"],
            "merge":           ui["step_merge"],
            "chunk":           ui["step_chunk"],
            "limatch":         ui["step_limatch"],
            "gps_outage_file": ui["step_gps_outage"],
        },
        "merge":                 merge,
        "merge_correspondences": {"enabled": ui["mc_enabled"], "output_file": None},
        "chunk": {
            "source":            ui["chunk_source"],
            "existing_root":     ui["chunk_existing_root"] or None,
            "merged_input_root": ui["chunk_merged_input_root"] or None,
            "merged_group":      None,
            "reference_scanner": "HA",
            "output_root":       None,
            "length_m":          length_m,
            "min_last_chunk_m":  min_last_chunk,
            "epsg_out":          ui["chunk_epsg"],
            "delimiter":         ",",
            "skiprows":          0,
            "cloud_fmt":         ui["chunk_fmt"],
            "time_field":        "gps_time",
            "manifest_name":     "merged_manifest.csv",
            "use_manifest":      True,
        },
        "chunk_variant": chunk_variant,
        "limatch":  limatch,
        "patcher": {
            "run": False, "source": "existing", "output_root": None,
            "gps_outage_file": {"enabled": False, "output_root": None, "outages": []},
        },
    }


def ui_from_pipeline_yaml(yd: dict, name: str = "") -> dict:
    """Import an existing pipeline YAML → UI dict (best-effort)."""
    ui = default_scenario_ui(name)
    ui["scenario_name"] = yd.get("scenario_name", "")

    paths = yd.get("paths", {})
    ui["root_out_dir"] = paths.get("root_out_dir", "")
    ui["limatch_cfg"]  = paths.get("limatch_cfg", "")

    traj = yd.get("trajectory", {})
    ui["traj_path"] = traj.get("path", "")
    ui["traj_type"] = traj.get("type", "SBET")

    # Outage — read from georef_time_window or chunk_variant
    tw = yd.get("georef_time_window", {})
    ui["tw_enable"] = tw.get("enable", False)
    if "outage" in tw:
        ui["outage_start"] = float(tw["outage"][0])
        ui["outage_dur"]   = float(tw["outage"][1])
        ui["buf_pre_s"]    = float(tw.get("margin_s", 30.0))
        ui["buf_post_s"]   = float(tw.get("margin_s", 30.0))
    elif "outages" in tw and tw["outages"]:
        ui["outage_start"] = float(tw["outages"][0][0])
        ui["outage_dur"]   = float(tw["outages"][0][1])
        ui["buf_pre_s"]    = float(tw.get("margin_s", 30.0))
        ui["buf_post_s"]   = float(tw.get("margin_s", 30.0))
    # Fallback: read from chunk_variant
    cv = yd.get("chunk_variant", {}).get("outage_window", {})
    if cv.get("outages") and ui["outage_start"] == 305120.0:
        ui["outage_start"] = float(cv["outages"][0][0])
        ui["outage_dur"]   = float(cv["outages"][0][1])
    if cv.get("pre_s"):  ui["buf_pre_s"]  = float(cv["pre_s"])
    if cv.get("post_s"): ui["buf_post_s"] = float(cv["post_s"])

    dist = yd.get("distance_filtering", {})
    ui["dist_enable"] = dist.get("enable", True)
    ui["dist_max_m"]  = float(dist.get("max_distance_m", 30.0))
    ui["dist_epsg"]   = dist.get("map_epsg", "EPSG:2056")

    steps = yd.get("steps", {})
    ui["step_georef"]     = steps.get("georef", True)
    ui["step_merge"]      = steps.get("merge", True)
    ui["step_chunk"]      = steps.get("chunk", False)
    ui["step_limatch"]    = steps.get("limatch", False)
    ui["step_gps_outage"] = steps.get("gps_outage_file", False)

    merge = yd.get("merge", {})
    ui["merge_preset"]               = merge.get("preset", "all")
    ui["merge_out_prefix"]           = merge.get("out_prefix", "merged_")
    ui["merge_out_suffix"]           = merge.get("out_suffix", "_HA_LR")
    ui["merge_output_suffix"]        = merge.get("output_suffix", "_VUX_PUCK")
    ui["merge_src_vux"]              = int(merge.get("scanner_src_vux", 2))
    ui["merge_src_puck"]             = int(merge.get("scanner_src_puck", 1))
    ui["merge_chunk_size"]           = int(merge.get("chunk_size", 10_000_000))
    cl = merge.get("cleanup", {})
    ui["merge_cleanup"]              = cl.get("enabled", True)
    ui["merge_cleanup_scanner_dirs"] = cl.get("scanner_dirs", True)
    ui["merge_vux_input_dir"]        = merge.get("vux_input_dir", "")

    chunk = yd.get("chunk", {})
    ui["chunk_source"]            = chunk.get("source", "generate")
    ui["chunk_existing_root"]     = chunk.get("existing_root", "") or ""
    ui["chunk_merged_input_root"] = chunk.get("merged_input_root", "") or ""
    ui["chunk_length_m"]          = float(chunk.get("length_m", 15.0))
    ui["chunk_epsg"]              = chunk.get("epsg_out", "EPSG:2056")
    ui["chunk_fmt"]               = chunk.get("cloud_fmt", "las")

    lm = yd.get("limatch", {})
    ui["lm_output_root"]          = lm.get("output_root", "") or ""
    ui["lm_neighbor_k"]           = int(lm.get("neighbor_k", 0))
    ui["lm_do_cross_scan"]        = lm.get("do_cross_scan", False)
    ui["lm_do_spatial_crossings"] = lm.get("do_spatial_crossings", True)
    ui["lm_crossings_output_root"]= lm.get("crossings_output_root", "") or ""
    ui["lm_crossing_min_sep"]     = int(lm.get("crossing_min_separation", 30))
    ui["lm_crossing_overlap_m"]   = float(lm.get("crossing_overlap_margin_m", 3.0))
    if "cfg_overrides_f2b" in lm or "cfg_overrides_crossings" in lm:
        ui["lm_use_overrides"] = True
        f2b = lm.get("cfg_overrides_f2b", {})
        ui["lm_f2b_uncertainty_r_min"] = float(f2b.get("uncertainty_r_min", 0.0))
        ui["lm_f2b_uncertainty_r_max"] = float(f2b.get("uncertainty_r_max", 2.0))
        cx  = lm.get("cfg_overrides_crossings", {})
        ui["lm_cross_uncertainty_r_min"] = float(cx.get("uncertainty_r_min", 0.0))
        ui["lm_cross_uncertainty_r_max"] = float(cx.get("uncertainty_r_max", 2.0))

    ui["mc_enabled"] = yd.get("merge_correspondences", {}).get("enabled", True)
    return ui

# ── Project creation ───────────────────────────────────────────────────────

def create_project(
    name: str,
    description: str = "",
    root_out_dir: str = "",
    gps_input: str = "",
    scanner_names: list[str] | None = None,
):
    scanner_names = scanner_names or ["HA", "LR", "PUCK"]
    meta = default_project_meta(name)
    meta.update({"description": description, "root_out_dir": root_out_dir,
                 "gps_input": gps_input, "scanners": scanner_names})
    save_project_meta(name, meta)
    for sc_name in scanner_names:
        candidates = list(DEFAULT_CFGS.glob(f"scanner_{sc_name}*.yml"))
        sc_data = load_yaml(candidates[0]) if candidates else default_scanner(sc_name)
        save_scanner(name, sc_name, sc_data)
    scenarios_dir(name).mkdir(parents=True, exist_ok=True)


def create_project_from_template(template_path: Path) -> str | None:
    """Returns project name, or None if it already existed."""
    yd   = load_yaml(template_path)
    name = template_path.stem
    if name in list_projects():
        return None
    paths = yd.get("paths", {})
    create_project(name, root_out_dir=paths.get("root_out_dir",""),
                   gps_input=paths.get("gps_input",""))
    for key, cfg_path_str in yd.get("scanners", {}).items():
        sc_name = key.replace("_cfg","").upper()
        p       = Path(str(cfg_path_str))
        abs_p   = REPO_ROOT / p if not p.is_absolute() else p
        sc_data = load_yaml(abs_p) if abs_p.exists() else default_scanner(sc_name)
        save_scanner(name, sc_name, sc_data)
    ui = ui_from_pipeline_yaml(yd, name=template_path.stem)
    save_scenario(name, template_path.stem, {"_ui": ui})
    return name
