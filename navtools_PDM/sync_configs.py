from pathlib import Path
import yaml
from copy import deepcopy


def _safe_load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _safe_write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_nested(data, keys, value):
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _ensure_exists(path, label):
    if not Path(path).exists():
        raise FileNotFoundError("{0} does not exist: {1}".format(label, path))


def build_derived_values(app_cfg):
    if app_cfg["scenario"].get("manual_scenario_root", False):
        scenario_root = Path(app_cfg["outputs"]["scenario_root"])
    else:
        scenario_root = Path(app_cfg["project_root"]) / app_cfg["scenario"]["method"] / app_cfg["scenario"]["run_name"]

    tmp_root = scenario_root / app_cfg["outputs"].get("tmp_dirname", "tmp")
    georef_tmp = tmp_root / "georef_outputs"
    generated_cfg_dir = tmp_root / "generated_configs"

    values = {
        "scenario_root": scenario_root,
        "tmp_root": tmp_root,
        "georef_tmp_root": georef_tmp,
        "ha_tmp_dir": georef_tmp / "HA",
        "lr_tmp_dir": georef_tmp / "LR",
        "puck_tmp_dir": georef_tmp / "PUCK",
        "generated_cfg_dir": generated_cfg_dir,
        "merged_dir": scenario_root / app_cfg["outputs"].get("merged_dirname", "merged"),
        "chunks_dir": scenario_root / app_cfg["outputs"].get("chunks_dirname", "chunks"),
        "limatch_dir": scenario_root / app_cfg["outputs"].get("limatch_dirname", "output_limatch"),
        "patcher_dir": scenario_root / app_cfg["outputs"].get("patcher_dirname", "output_patcher"),
        "results_dir": scenario_root / app_cfg["outputs"].get("results_dirname", "results"),
        "logs_dir": scenario_root / app_cfg["outputs"].get("logs_dirname", "logs"),
        "scenario_combined_dir": scenario_root / "scenario_combined",
        "scenario_gps_outage_dir": scenario_root / "scenario_gps_outage",
    }

    app_cfg["outputs"]["scenario_root"] = str(scenario_root)
    return values


def sync_georef_configs(app_cfg, derived):
    repo_root = Path(app_cfg["repo_root"])

    template_ha = repo_root / "navtools_PDM" / "PDM_configs" / "georef_SAM-HA.yml"
    template_lr = repo_root / "navtools_PDM" / "PDM_configs" / "georef_SAM-LR.yml"

    _ensure_exists(template_ha, "HA georef template")
    _ensure_exists(template_lr, "LR georef template")
    _ensure_exists(app_cfg["paths"]["traj_path"], "Trajectory")
    _ensure_exists(app_cfg["paths"]["sdc_ha"], "SDC HA")
    _ensure_exists(app_cfg["paths"]["sdc_lr"], "SDC LR")

    data_ha = _safe_load_yaml(template_ha)
    data_lr = _safe_load_yaml(template_lr)

    _set_nested(data_ha, ["trj", "path"], app_cfg["paths"]["traj_path"])
    _set_nested(data_ha, ["lasvec", "path"], app_cfg["paths"]["sdc_ha"])
    _set_nested(data_ha, ["output", "path"], str(derived["ha_tmp_dir"]))

    _set_nested(data_lr, ["trj", "path"], app_cfg["paths"]["traj_path"])
    _set_nested(data_lr, ["lasvec", "path"], app_cfg["paths"]["sdc_lr"])
    _set_nested(data_lr, ["output", "path"], str(derived["lr_tmp_dir"]))

    ha_out = derived["generated_cfg_dir"] / "georef_SAM-HA.generated.yml"
    lr_out = derived["generated_cfg_dir"] / "georef_SAM-LR.generated.yml"

    _safe_write_yaml(ha_out, data_ha)
    _safe_write_yaml(lr_out, data_lr)

    return {
        "ha_yaml": ha_out,
        "lr_yaml": lr_out,
    }


def sync_patcher_config(app_cfg, derived):
    template = Path(app_cfg["paths"]["patcher_cfg_template"])
    _ensure_exists(template, "Patcher template")

    data = _safe_load_yaml(template)

    if "OUTPUT_DIR" in data:
        data["OUTPUT_DIR"] = str(derived["patcher_dir"])

    if "PC_DIR" in data:
        data["PC_DIR"] = str(derived["merged_dir"])

    if "LIMATCH_CFG" in data:
        data["LIMATCH_CFG"] = str(derived["generated_cfg_dir"] / "limatch.generated.yml")

    out = derived["generated_cfg_dir"] / "patcher.generated.yml"
    _safe_write_yaml(out, data)

    return {
        "patcher_yaml": out,
    }


def sync_limatch_config(app_cfg, derived):
    template = Path(app_cfg["paths"]["limatch_cfg_template"])
    _ensure_exists(template, "LiMatch template")

    data = _safe_load_yaml(template)

    if "prj_folder" in data:
        data["prj_folder"] = str(derived["limatch_dir"]) + "/"

    possible_traj_keys = [
        ("trajectory", "path"),
        ("traj", "path"),
        ("trj", "path"),
    ]

    for keys in possible_traj_keys:
        cur = data
        ok = True
        for key in keys[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                ok = False
                break
            cur = cur[key]
        if ok and keys[-1] in cur:
            cur[keys[-1]] = app_cfg["paths"]["traj_path"]

    out = derived["generated_cfg_dir"] / "limatch.generated.yml"
    _safe_write_yaml(out, data)

    return {
        "limatch_yaml": out,
    }


def sync_all_configs(app_cfg):
    derived = build_derived_values(app_cfg)

    georef_cfgs = sync_georef_configs(app_cfg, derived)
    limatch_cfg = sync_limatch_config(app_cfg, derived)
    patcher_cfg = sync_patcher_config(app_cfg, derived)

    result = {}
    result.update(derived)
    result.update(georef_cfgs)
    result.update(limatch_cfg)
    result.update(patcher_cfg)
    return result