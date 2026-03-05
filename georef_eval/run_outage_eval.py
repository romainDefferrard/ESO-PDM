#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml
import logging

def cmd_to_str(cmd):
    return " ".join(map(str, cmd))

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def first_last_time(path: Path, delim=","):
    """Ultra fast read of first and last gps_time in file."""
    # first line
    with open(path, "r", encoding="utf-8") as f:
        first_line = None
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                first_line = ln
                break
        if first_line is None:
            raise ValueError(f"{path}: no data lines found")
        t_first = float(first_line.split(delim)[0])

    # last line
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        last_line = f.readline().decode().strip()
        t_last = float(last_line.split(delim)[0])

    return t_first, t_last

def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML configuration file")
    args = ap.parse_args()

    logging.info("Loading config: %s", args.config)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref = Path(cfg["data"]["ref"])
    est = Path(cfg["data"]["est"])
    out = Path(cfg["data"]["out_dir"])

    t0 = cfg["outage"]["t0"]
    duration = cfg["outage"]["duration"]
    pad = cfg["outage"]["padding"]

    time_tol = cfg["matching"]["time_tol"]

    field = cfg["visualization"]["field"]
    vmin = cfg["visualization"]["vmin"]
    vmax = cfg["visualization"]["vmax"]
    o3d_only = cfg["visualization"]["o3d_only"]

    tmin = t0 - pad
    tmax = t0 + duration + pad

    delim = cfg.get("io", {}).get("delim", ",")
    no_header = cfg.get("io", {}).get("no_header", False)
    schema = cfg.get("io", {}).get("schema", "txyzlxyz")
    

    logging.info("Checking time range inside files...")

    t_first, t_last = first_last_time(ref, delim)

    logging.info("File time range: [%.3f , %.3f]", t_first, t_last)

    if tmax < t_first or tmin > t_last:
        logging.error("Requested window NOT inside file range")
        logging.error("Window: [%.3f , %.3f]", tmin, tmax)
        logging.error("File:   [%.3f , %.3f]", t_first, t_last)
        logging.error("Use another time or another file.")
        return
    else:
        logging.info("Requested window overlaps file time range ✓")

    logging.info("Reference cloud: %s", ref)
    logging.info("Estimated cloud: %s", est)

    logging.info("Outage center: %.3f", t0)
    logging.info("Outage duration: %.1f s", duration)
    logging.info("Padding: %.1f s", pad)

    logging.info("Time window used: [%.3f , %.3f]", tmin, tmax)

    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Step 1: compute georef errors
    # ------------------------------
    logging.info("Running georef_error.py ...")

    cmd_err = [
        "python", "georef_error.py",
        "--ref", str(ref),
        "--est", str(est),
        "--out", str(out),
        "--time_tol", str(time_tol),
        "--vmax", str(vmax),
        "--tmin", str(tmin),
        "--tmax", str(tmax),
        "--delim", str(delim),
        "--schema", str(schema),
    ]
    if o3d_only:
        cmd_err.append("--o3d_only")

    if no_header:
        cmd_err.append("--no_header")

    logging.info("Command: %s", " ".join(map(str, cmd_err)))    
    subprocess.run(cmd_err, check=True)

    logging.info("Georef error computation finished")

    # ------------------------------
    # Step 2: Open3D visualization
    # ------------------------------
    in_txt = out / "est_with_error.txt"
    out_ply = out / f"colored_{field}.ply"

    logging.info("Launching Open3D visualization")

    cmd_o3d = [
        "python",
        "georef_colorize_o3d.py",
        "--in_txt", str(in_txt),
        "--field", field,
        "--out_ply", str(out_ply),
        "--vmin", str(vmin),
        "--vmax", str(vmax),
        "--show",
    ]

    logging.info("Command: %s", " ".join(cmd_o3d))
    subprocess.run(cmd_o3d, check=True)

    logging.info("Open3D visualization finished")

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()