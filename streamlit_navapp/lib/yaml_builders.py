from pathlib import Path
from copy import deepcopy
from lib.config_io import read_yaml, write_yaml

def _set_nested(d: dict, keys: list[str], value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def generate_georef_yaml(base_yaml_path: Path, output_yaml_path: Path, *, traj_path: str, lasvec_path: str, output_path: str):
    data = read_yaml(base_yaml_path)
    _set_nested(data, ["trj", "path"], traj_path)
    _set_nested(data, ["lasvec", "path"], lasvec_path)
    _set_nested(data, ["output", "path"], output_path)
    write_yaml(output_yaml_path, data)
    return output_yaml_path

def generate_scenario_georef_pair(cfg: dict):
    repo_root = Path(cfg["repo_root"])
    scenario_root = Path(cfg["outputs"]["scenario_root"])
    generated_dir = scenario_root / "_generated_configs"

    ha_base = repo_root / "navtools_PDM" / "PDM_configs" / "georef_SAM-HA.yml"
    lr_base = repo_root / "navtools_PDM" / "PDM_configs" / "georef_SAM-LR.yml"

    ha_out = generated_dir / "georef_SAM-HA.generated.yml"
    lr_out = generated_dir / "georef_SAM-LR.generated.yml"

    generate_georef_yaml(
        ha_base,
        ha_out,
        traj_path=cfg["paths"]["traj_path"],
        lasvec_path=cfg["paths"]["sdc_ha"],
        output_path=str(scenario_root / cfg["outputs"]["ha_dirname"]),
    )
    generate_georef_yaml(
        lr_base,
        lr_out,
        traj_path=cfg["paths"]["traj_path"],
        lasvec_path=cfg["paths"]["sdc_lr"],
        output_path=str(scenario_root / cfg["outputs"]["lr_dirname"]),
    )
    return {"ha_yaml": ha_out, "lr_yaml": lr_out}