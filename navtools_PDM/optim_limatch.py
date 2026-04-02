from pathlib import Path
from datetime import datetime
import os
import sys
import yaml
import traceback


BATCH_CONFIG_PATH = Path("/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/limatch_scenario.yml")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def import_limatch_match_clouds(repo_root: Path):
    limatch_parent = repo_root / "Patcher" / "submodules"
    if str(limatch_parent) not in sys.path:
        sys.path.insert(0, str(limatch_parent))

    from limatch.main import match_clouds
    return match_clouds


def build_effective_mls(base_cfg_path: Path, scenario: dict, out_dir: Path):
    cfg = load_yaml(base_cfg_path)

    for k in ["icp_patch_r", "icp_max_n", "icp_thresh", "icp_conv", "icp_vox_s"]:
        if k in scenario:
            cfg[k] = scenario[k]

    cfg["prj_folder"] = str(out_dir) + os.sep
    return cfg


def print_batch_header(config_path: Path, scenarios: list, pairs: list):
    line = "=" * 100
    print("\n" + line)
    print("[BATCH] LiMatch scenarios from explicit scan pairs")
    print(line)
    print(f"[BATCH] config file : {config_path}")
    print(f"[BATCH] scenarios   : {len(scenarios)}")
    print(f"[BATCH] pairs       : {len(pairs)}")
    print(f"[BATCH] total runs  : {len(scenarios) * len(pairs)}")
    print(line + "\n")


def print_scenario_banner(scenario: dict, pair: dict, out_dir: Path):
    line = "=" * 100
    print("\n" + line)
    print(f"[RUN] scenario={scenario['name']} | pair={pair['name']}")
    print(line)
    print("ICP parameters")
    print(f"  - icp_patch_r : {scenario.get('icp_patch_r')}")
    print(f"  - icp_max_n   : {scenario.get('icp_max_n')}")
    print(f"  - icp_thresh  : {scenario.get('icp_thresh')}")
    print(f"  - icp_conv    : {scenario.get('icp_conv')}")
    print(f"  - icp_vox_s   : {scenario.get('icp_vox_s')}")
    print("")
    print("Inputs")
    print(f"  - cloud1      : {pair['cloud1']}")
    print(f"  - cloud2      : {pair['cloud2']}")
    print("")
    print("Output")
    print(f"  - out_dir     : {out_dir}")
    print(line + "\n")


def write_log(log_path: Path, scenario: dict, pair: dict, effective_cfg_path: Path, out_dir: Path):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"date={datetime.now().isoformat()}\n")
        f.write(f"scenario_name={scenario['name']}\n")
        f.write(f"pair_name={pair['name']}\n")
        f.write(f"cloud1={pair['cloud1']}\n")
        f.write(f"cloud2={pair['cloud2']}\n")
        f.write(f"effective_cfg={effective_cfg_path}\n")
        f.write(f"out_dir={out_dir}\n")
        for k, v in scenario.items():
            f.write(f"{k}={v}\n")


def run_one_pair(match_clouds, limatch_cfg_base: Path, batch_root: Path, scenario: dict, pair: dict):
    scenario_name = scenario["name"]
    pair_name = pair["name"]

    scenario_root = batch_root / scenario_name
    pair_root = scenario_root / pair_name
    cfg_dir = pair_root / "configs"
    out_dir = pair_root / "results"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (out_dir / "cor_outputs").mkdir(parents=True, exist_ok=True)

    effective_cfg = build_effective_mls(limatch_cfg_base, scenario, out_dir)
    effective_cfg_path = cfg_dir / "MLS_effective.yml"
    save_yaml(effective_cfg, effective_cfg_path)

    log_path = pair_root / "log.txt"

    print_scenario_banner(scenario, pair, out_dir)
    write_log(log_path, scenario, pair, effective_cfg_path, out_dir)

    try:
        match_clouds(pair["cloud1"], pair["cloud2"], effective_cfg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("status=SUCCESS\n")

        print(f"[DONE] scenario={scenario_name} | pair={pair_name}\n")

    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("status=FAILED\n")
            f.write(f"error_type={type(e).__name__}\n")
            f.write(f"error={e}\n")
            f.write(traceback.format_exc())

        print(f"[FAIL] scenario={scenario_name} | pair={pair_name} | {type(e).__name__}: {e}\n")


def find_lidar_p2p_files_for_scenario(scenario_root: Path):
    """
    Ne prend que les fichiers du type:
      results/cor_outputs/LiDAR_p2p_output_*.txt
    """
    files = sorted(scenario_root.glob("**/results/cor_outputs/LiDAR_p2p_output_*.txt"))
    return files


def merge_text_files(files, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"[merge_p2p] Aucun fichier à merger pour {out_file}")
        return None

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as fin:
                content = fin.read()

            if not content.strip():
                continue

            if i > 0 and not content.startswith("\n"):
                fout.write("\n")

            fout.write(content)

            if not content.endswith("\n"):
                fout.write("\n")

    print(f"[merge_p2p] Merged {len(files)} files -> {out_file}")
    return out_file


def merge_scenario_lidar_p2p_files(batch_root: Path, scenario_name: str):
    scenario_root = batch_root / scenario_name
    merged_dir = scenario_root / "merged"
    merged_out = merged_dir / f"{scenario_name}_LiDAR_p2p_merged.txt"

    files = find_lidar_p2p_files_for_scenario(scenario_root)

    print("\n" + "=" * 100)
    print(f"[merge_p2p] scenario={scenario_name}")
    print(f"[merge_p2p] LiDAR_p2p files found: {len(files)}")
    for f in files:
        print(f"  - {f}")
    print("=" * 100 + "\n")

    return merge_text_files(files, merged_out)


def main():
    batch_cfg = load_yaml(BATCH_CONFIG_PATH)

    repo_root = Path(batch_cfg["repo_root"]).resolve()
    limatch_cfg_base = Path(batch_cfg["limatch_cfg_base"]).resolve()
    batch_root = Path(batch_cfg["batch_root"]).resolve()

    run_mode = batch_cfg.get("run_mode", "all")
    selected_name = batch_cfg.get("selected_scenario")
    scenarios = batch_cfg.get("scenarios", [])
    pairs = batch_cfg.get("pairs", [])

    if not scenarios:
        raise ValueError("No scenarios found in batch config.")
    if not pairs:
        raise ValueError("No pairs found in batch config.")

    if run_mode == "one":
        scenarios = [s for s in scenarios if s["name"] == selected_name]
        if not scenarios:
            raise ValueError(f"Scenario not found: {selected_name}")

    match_clouds = import_limatch_match_clouds(repo_root)

    print_batch_header(BATCH_CONFIG_PATH, scenarios, pairs)

    for scenario in scenarios:
        for pair in pairs:
            run_one_pair(
                match_clouds=match_clouds,
                limatch_cfg_base=limatch_cfg_base,
                batch_root=batch_root,
                scenario=scenario,
                pair=pair,
            )

        merge_scenario_lidar_p2p_files(
            batch_root=batch_root,
            scenario_name=scenario["name"],
        )


if __name__ == "__main__":
    main()