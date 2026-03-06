from pathlib import Path


def get_effective_scenario_root(cfg):
    project_root = Path(cfg["project_root"])
    method = cfg["scenario"]["method"]
    run_name = cfg["scenario"]["run_name"]

    if cfg["scenario"].get("manual_scenario_root", False):
        return Path(cfg["outputs"]["scenario_root"])

    return project_root / method / run_name


def scenario_paths(cfg):
    scenario_root = get_effective_scenario_root(cfg)
    tmp_root = scenario_root / cfg["outputs"].get("tmp_dirname", "tmp")
    georef_tmp = tmp_root / "georef_outputs"
    generated_cfg_dir = tmp_root / "generated_configs"

    return {
        "scenario_root": scenario_root,
        "merged_dir": scenario_root / cfg["outputs"].get("merged_dirname", "merged"),
        "chunks_dir": scenario_root / cfg["outputs"].get("chunks_dirname", "chunks"),
        "limatch_dir": scenario_root / cfg["outputs"].get("limatch_dirname", "output_limatch"),
        "patcher_dir": scenario_root / cfg["outputs"].get("patcher_dirname", "output_patcher"),
        "results_dir": scenario_root / cfg["outputs"].get("results_dirname", "results"),
        "logs_dir": scenario_root / cfg["outputs"].get("logs_dirname", "logs"),
        "tmp_root": tmp_root,
        "georef_tmp_root": georef_tmp,
        "ha_tmp_dir": georef_tmp / "HA",
        "lr_tmp_dir": georef_tmp / "LR",
        "puck_tmp_dir": georef_tmp / "PUCK",
        "generated_cfg_dir": generated_cfg_dir,
        "scenario_combined_dir": scenario_root / "scenario_combined",
        "scenario_gps_outage_dir": scenario_root / "scenario_gps_outage",
    }


def scenario_status(cfg):
    paths = scenario_paths(cfg)

    return {
        "merged_exists": paths["merged_dir"].exists(),
        "chunks_exists": paths["chunks_dir"].exists(),
        "limatch_exists": paths["limatch_dir"].exists(),
        "patcher_exists": paths["patcher_dir"].exists(),
        "tmp_exists": paths["tmp_root"].exists(),
        "merged_n_txt": len(list(paths["merged_dir"].glob("*.txt"))) if paths["merged_dir"].exists() else 0,
        "ha_tmp_n_txt": len(list(paths["ha_tmp_dir"].glob("*.txt"))) if paths["ha_tmp_dir"].exists() else 0,
        "lr_tmp_n_txt": len(list(paths["lr_tmp_dir"].glob("*.txt"))) if paths["lr_tmp_dir"].exists() else 0,
    }