#!/usr/bin/env python3
"""
generate_georef_batch.py
------------------------
Generates 6 pipeline YAMLs (3 outages x 2 methods: F2B / COMBINED)
with correct output structure:

    georef_ALL_traj_outage_X/
        georef_F2B/         <- root_out_dir=.../georef_ALL_traj_outage_X, scenario_name=georef_F2B
            HA/
            LR/
            PUCK/
            merged/
        georef_COMBINED/
            ...

Usage:
    python generate_georef_batch.py
"""

from pathlib import Path
import yaml
import copy

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR     = Path("/media/b085164/Elements/CALIB_26_02_25")
CONFIGS_DIR  = Path("/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs")

BASE_CONFIGS = {
    1: CONFIGS_DIR / "pipeline_outage_1.yml",
    2: CONFIGS_DIR / "pipeline_outage_2.yml",
    3: CONFIGS_DIR / "pipeline_outage_3.yml",
}

OUTAGE_DIRS = {
    1: BASE_DIR / "georef_ALL_traj_outage_1",
    2: BASE_DIR / "georef_ALL_traj_outage_2",
    3: BASE_DIR / "georef_ALL_traj_outage_3",
}

TRAJECTORIES = {
    1: {
        "F2B":      str(BASE_DIR / "ODyN_calib/Outage_1_305120_305700/F2B/out/traj_F2B_1.out"),
        "COMBINED": str(BASE_DIR / "ODyN_calib/Outage_1_305120_305700/COMBINED/out/traj_COMBINED_1.out"),
    },
    2: {
        "F2B":      str(BASE_DIR / "ODyN_calib/Outage_2_305645_306120/F2B/out/v2/traj_F2B_2.out"),
        "COMBINED": str(BASE_DIR / "ODyN_calib/Outage_2_305645_306120/COMBINED/out/v2/traj_COMBINED_2.out"),
    },
    3: {
        "F2B":      str(BASE_DIR / "ODyN_calib/Outage_3_306295_306640/F2B/out/v2/traj_F2B_3.out"),
        "COMBINED": str(BASE_DIR / "ODyN_calib/Outage_3_306295_306640/COMBINED/out/v2/traj_COMBINED_3.out"),
    },
}

OUTPUT_DIR = CONFIGS_DIR / "georef_batch"

# ---------------------------------------------------------------------------

STEPS_GEOREF_ONLY = {
    "georef":          True,
    "merge":           True,
    "vux_puck_merge":  True,
    "chunk":           False,
    "limatch":         False,
    "gps_outage_file": False,
}


def generate_config(base_cfg: dict, outage_id: int, method: str, traj_path: str) -> dict:
    cfg = copy.deepcopy(base_cfg)

    # root_out_dir = .../georef_ALL_traj_outage_X
    # scenario_name = georef_F2B  or  georef_COMBINED
    # -> output lands in .../georef_ALL_traj_outage_X/georef_F2B/
    cfg["paths"]["root_out_dir"] = str(OUTAGE_DIRS[outage_id])
    cfg["scenario_name"]         = f"georef_{method}"

    cfg["trajectory"]["path"] = traj_path
    cfg["steps"]              = STEPS_GEOREF_ONLY

    return cfg


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []  # [(label, yaml_path)]

    for outage_id in [1, 2, 3]:
        base_cfg = yaml.safe_load(BASE_CONFIGS[outage_id].read_text())

        for method, traj_path in TRAJECTORIES[outage_id].items():
            label    = f"outage_{outage_id}_{method}"
            cfg      = generate_config(base_cfg, outage_id, method, traj_path)
            out_path = OUTPUT_DIR / f"pipeline_{label}.yml"

            out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            generated.append((label, out_path))

            print(f"  [{label}]")
            print(f"    root_out_dir  = {cfg['paths']['root_out_dir']}")
            print(f"    scenario_name = {cfg['scenario_name']}")
            print(f"    trajectory    = {traj_path}")
            print(f"    -> output dir = {OUTAGE_DIRS[outage_id]}/georef_{method}/")
            print()

    # Shell script
    shell_path = OUTPUT_DIR / "run_georef_batch.sh"
    lines = [
        "#!/bin/bash",
        "# Auto-generated — georef + merge batch",
        "# 6 runs: 3 outages x 2 methods (F2B / COMBINED)",
        "#",
        "# Output structure:",
        "#   georef_ALL_traj_outage_X/georef_F2B/",
        "#   georef_ALL_traj_outage_X/georef_COMBINED/",
        "",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
        "# Adjust REPO_ROOT if needed — should be the root of your ESO-PDM repo",
        'REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"',
        'cd "${REPO_ROOT}"',
        "",
        'PYTHON="${PYTHON:-python3}"',
        'LOG_DIR="/media/b085164/Elements/CALIB_26_02_25/logs/georef_batch"',
        'mkdir -p "${LOG_DIR}"',
        'TIMESTAMP=$(date +"%Y%m%d_%H%M%S")',
        "",
        'echo "=============================="',
        'echo " Georef batch — $(date)"',
        'echo " Repo root : ${REPO_ROOT}"',
        'echo "=============================="',
        "",
    ]

    for label, yaml_path in generated:
        lines += [
            f'echo ""',
            f'echo "[$(date +%H:%M:%S)] ===== {label} ====="',
            f'"${{PYTHON}}" -m navtools_PDM.pipeline -c "{yaml_path}" \\',
            f'    2>&1 | tee "${{LOG_DIR}}/pipeline_{label}_${{TIMESTAMP}}.log"',
            f'[ "${{PIPESTATUS[0]}}" -eq 0 ] || {{ echo "[ERROR] {label} failed"; exit 1; }}',
            f'echo "[$(date +%H:%M:%S)] {label} DONE"',
            "",
        ]

    lines += [
        'echo ""',
        'echo "=============================="',
        'echo " ALL DONE — $(date)"',
        'echo "=============================="',
    ]

    shell_path.write_text("\n".join(lines))
    shell_path.chmod(0o755)

    print(f"Shell script : {shell_path}")
    print(f"\nLaunch with :")
    print(f"  bash {shell_path}")
    print(f"\nOr set REPO_ROOT explicitly:")
    print(f"  REPO_ROOT=/your/repo/root bash {shell_path}")


if __name__ == "__main__":
    main()