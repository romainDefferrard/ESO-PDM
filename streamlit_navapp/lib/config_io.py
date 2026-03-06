from pathlib import Path
from copy import deepcopy
import yaml


DEFAULT_CONFIG = {
    "project_root": "/media/b085164/Elements/PCD_SAM/Georef_v6",
    "repo_root": "/home/b085164/PDM_Romain_Defferrard/ESO-PDM",
    "python_exec": "python3",
    "scenario": {
        "method": "outage_only",
        "run_name": "base",
        "mode": "GeorefOnly",
        "manual_scenario_root": False,
    },
    "paths": {
        "traj_path": "/media/b085164/Elements/Gobet_ODyN_v1/v6/out_outage_only/ODyN_outageOnly.out",
        "sdc_ha": "/media/b085164/Elements/RD_SAM_SDC/subset_HA",
        "sdc_lr": "/media/b085164/Elements/RD_SAM_SDC/subset_LR",
        "gps_in": "/media/b085164/Elements/Gobet_ODyN_v1/v1_base_AB/in/GPS.txt",
        "patcher_cfg_template": "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/Patcher/config/MLS_Epalinges_config.yml",
        "limatch_cfg_template": "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/Patcher/submodules/limatch/configs/MLS.yml",
        "patcher_out_root": "/media/b085164/Elements/PCD_SAM/parking/output_Patcher",
    },
    "outputs": {
        "scenario_root": "/media/b085164/Elements/PCD_SAM/Georef_v6/outage_only/base",
        "merged_dirname": "merged",
        "tmp_dirname": "tmp",
        "limatch_dirname": "output_limatch",
        "chunks_dirname": "chunks",
        "patcher_dirname": "output_patcher",
        "results_dirname": "results",
        "logs_dirname": "logs",
    },
    "execution": {
        "do_georef_merge": True,
        "delete_tmp_after_success": True,
        "do_chunks": True,
        "reuse_chunks": True,
        "force_rebuild_chunks": False,
        "do_gps_outage": False,
        "show_intermediate_outputs": False,
    },
    "merge": {
        "delimiter": ",",
        "skiprows": 0,
        "sort_by_time": True,
        "out_prefix": "merged_",
        "out_suffix": "_HA_LR",
    },
    "chunk": {
        "length_m": 15.0,
        "min_points": 2000,
        "neighbor_k": 1,
        "do_cross_scan": True,
        "epsg_out": "EPSG:2056",
    },
    "outage": {
        "start": 466930.0,
        "duration": 200.0,
        "pre": 30.0,
        "post": 30.0,
        "outages": [
            [466930.0, 200.0]
        ],
    },
    "cycleslip": {
        "scenarios": [
            {
                "name": "slip_default",
                "slips": [
                    {
                        "t0": 466930.0,
                        "duration": 60.0,
                        "dx": 1.0,
                        "dy": 0.0,
                        "dz": 0.0,
                        "shape": "step",
                    }
                ],
            }
        ]
    },
}


def deep_update(base, update):
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_app_config(path):
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)

    with open(path, "r") as f:
        loaded = yaml.safe_load(f) or {}

    return deep_update(DEFAULT_CONFIG, loaded)


def save_app_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)