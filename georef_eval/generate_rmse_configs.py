"""
generate_rmse_configs.py
------------------------
Generates one shell script + one YAML per (method, outage_scan) for rmse_streaming.py.

Matching strategy: temporal overlap between outage scan windows and ODyN scan windows.
For each outage scan, find the ODyN scan whose [t_start, t_end] has the largest overlap
with the outage scan window.

Usage:
    python generate_rmse_configs.py --config gen_config_outage_1.yml
"""

import argparse
import csv
import logging
from pathlib import Path

import yaml


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_manifest(path: Path) -> list:
    """Return list of dicts: {scan_id, filename, t_start, t_end}"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "scan_id":  int(row["scan_id"]),
                "filename": row["filename"],
                "t_start":  float(row["t_start"]),
                "t_end":    float(row["t_end"]),
            })
    return rows


def find_best_ref_scan(outage_scan: dict, ref_manifest: list):
    """
    Find the ODyN ref scan with the largest temporal overlap with outage_scan.
    Returns (ref_scan_dict, overlap_seconds) or (None, 0).
    """
    best      = None
    best_ovlp = 0.0

    ot0, ot1 = outage_scan["t_start"], outage_scan["t_end"]

    for ref in ref_manifest:
        rt0, rt1 = ref["t_start"], ref["t_end"]
        ovlp = max(0.0, min(ot1, rt1) - max(ot0, rt0))
        if ovlp > best_ovlp:
            best_ovlp = ovlp
            best      = ref

    return best, best_ovlp


def generate_configs_for_method(
    method_name: str,
    method_dir: Path,
    method_manifest: list,
    ref_dir: Path,
    ref_manifest: list,
    output_dir: Path,
    outage_id: str,
    buffer: float,
    scale: float,
) -> Path:

    out_dir_method = output_dir / method_name
    out_dir_method.mkdir(parents=True, exist_ok=True)

    per_scan_yamls = []

    for tgt_scan in method_manifest:
        sid      = tgt_scan["scan_id"]
        tgt_file = method_dir / tgt_scan["filename"]

        if not tgt_file.exists():
            logging.warning("  [%s] File not found, skipping scan %d: %s",
                            method_name, sid, tgt_file)
            continue

        ref_scan, ovlp = find_best_ref_scan(tgt_scan, ref_manifest)

        if ref_scan is None:
            logging.warning("  [%s] No ODyN overlap for scan %d [%.3f, %.3f] — skipped",
                            method_name, sid, tgt_scan["t_start"], tgt_scan["t_end"])
            continue

        ref_file = ref_dir / ref_scan["filename"]
        if not ref_file.exists():
            logging.warning("  [%s] Ref file missing for scan %d: %s",
                            method_name, sid, ref_file)
            continue

        # Time window: outage scan bounds + buffer, used to filter the (large) ref file
        t_start = tgt_scan["t_start"] - buffer
        t_end   = tgt_scan["t_end"]   + buffer

        out_las   = out_dir_method / f"rmse_{outage_id}_{method_name}_scan{sid}.las"
        yaml_path = out_dir_method / f"rmse_{outage_id}_{method_name}_scan{sid}.yml"

        rmse_cfg = {
            "reference": {"path": str(ref_file)},
            "targets": [{
                "path":    str(tgt_file),
                "outfile": str(out_las),
            }],
            "time": {
                "outage_start":    round(t_start, 6),
                "outage_duration": round(t_end - t_start, 6),
                "buffer":          0.0,
            },
            "io":  {"delim": ","},
            "las": {"scale": scale},
        }

        yaml_path.write_text(yaml.dump(rmse_cfg, default_flow_style=False, sort_keys=False))
        per_scan_yamls.append(yaml_path)

        logging.info(
            "  scan %4d [%.3f–%.3f] -> ODyN scan %4d [%.3f–%.3f]  overlap=%.1fs",
            sid, tgt_scan["t_start"], tgt_scan["t_end"],
            ref_scan["scan_id"], ref_scan["t_start"], ref_scan["t_end"], ovlp,
        )

    if not per_scan_yamls:
        logging.warning("  No valid pairs for method '%s'.", method_name)
        return None

    # Shell script: runs all per-scan YAMLs sequentially
    shell_path = out_dir_method / f"run_rmse_{outage_id}_{method_name}.sh"
    lines = [
        "#!/bin/bash",
        f"# Auto-generated — RMSE for method={method_name}  outage={outage_id}",
        f"# {len(per_scan_yamls)} scan(s)",
        "",
    ]
    for yp in per_scan_yamls:
        lines.append(f"python rmse_streaming.py --config '{yp}' || exit 1")
    lines += ["", f"echo '=== Done: {method_name} / {outage_id} ==='"]
    shell_path.write_text("\n".join(lines))
    shell_path.chmod(0o755)

    logging.info("  -> %d YAMLs + shell: %s", len(per_scan_yamls), shell_path)
    return shell_path


def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Generator config YAML")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    outage_id    = cfg["outage_id"]
    ref_dir      = Path(cfg["ref_dir"])
    ref_manifest = read_manifest(Path(cfg["ref_manifest"]))
    output_dir   = Path(cfg["output_dir"])
    buffer       = float(cfg.get("time", {}).get("buffer", 2.0))
    scale        = float(cfg.get("las",  {}).get("scale",  0.001))

    logging.info("Outage : %s", outage_id)
    logging.info("Ref    : %s  (%d scans)", ref_dir, len(ref_manifest))
    logging.info("Output : %s", output_dir)

    methods = list(cfg.get("methods", []))

    for method_cfg in methods:
        name      = method_cfg["name"]
        mdir      = Path(method_cfg["dir"])
        mmanifest = read_manifest(Path(method_cfg["manifest"]))

        logging.info("")
        logging.info("Method: %s  (%d scans)", name, len(mmanifest))

        generate_configs_for_method(
            method_name=name,
            method_dir=mdir,
            method_manifest=mmanifest,
            ref_dir=ref_dir,
            ref_manifest=ref_manifest,
            output_dir=output_dir,
            outage_id=outage_id,
            buffer=buffer,
            scale=scale,
        )

    logging.info("")
    logging.info("Run each method with:")
    for method_cfg in methods:
        name = method_cfg["name"]
        logging.info("  bash %s/%s/run_rmse_%s_%s.sh",
                     output_dir, name, outage_id, name)


if __name__ == "__main__":
    main()