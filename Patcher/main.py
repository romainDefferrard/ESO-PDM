"""
Filename: main.py
Author: Romain Defferrard
Date: 04-06-2025

Description:

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from utils.Patcher import PatcherPipeline



def run_patcher_from_config(config_path: str | Path) -> None:
    """
    Importable entrypoint: call this from other scripts.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline = PatcherPipeline(cfg)
    pipeline.run_patcher()


def main():
    """
    CLI wrapper
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
    args = ap.parse_args()

    run_patcher_from_config(args.yml)


if __name__ == "__main__":
    main()