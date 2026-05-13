#!/usr/bin/env python3
"""
run_limatch.py
--------------
Lance LiMatch entre deux nuages de points.

Usage:
    python run_limatch.py <cloud1> <cloud2> [--out-dir <path>] [--uncertainty-r <float>] [--cfg <path>]
"""

import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="LiMatch entre deux nuages")
parser.add_argument("cloud1", help="Chemin vers le premier nuage (.las/.laz)")
parser.add_argument("cloud2", help="Chemin vers le deuxième nuage (.las/.laz)")
parser.add_argument("--out-dir",       default=None,  help="Dossier de sortie (défaut : ./limatch_out)")
#parser.add_argument("--uncertainty-r", default=40.0,  type=float, help="uncertainty_r (défaut : 25)")
parser.add_argument("--cfg",           default=None,  help="Chemin vers le fichier de config LiMatch")
parser.add_argument("--repo-root",     default=None,  help="Chemin vers la racine ESO-PDM")
args = parser.parse_args()

cloud1 = Path(args.cloud1)
cloud2 = Path(args.cloud2)

for c, label in [(cloud1, "cloud1"), (cloud2, "cloud2")]:
    if not c.exists():
        print(f"Erreur : {label} introuvable : {c}")
        sys.exit(1)

out_dir = Path(args.out_dir) if args.out_dir else Path("limatch_out") / f"{cloud1.stem}__{cloud2.stem}"
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/home/b085164/PDM_Romain_Defferrard/ESO-PDM")
if args.repo_root:
    sys.path.insert(0, args.repo_root)

from navtools_PDM.pipeline import run_limatch_api, get_repo_root

repo_root   = get_repo_root()
limatch_cfg = args.cfg or str(repo_root / "Patcher/submodules/limatch/configs/MLS_F2B_1_APX.yml")

print(f"cloud1        : {cloud1}")
print(f"cloud2        : {cloud2}")
print(f"out_dir       : {out_dir}")
#print(f"uncertainty_r : {args.uncertainty_r}")
print(f"limatch_cfg   : {limatch_cfg}")
print()

try:
    run_limatch_api(
        repo_root=repo_root,
        limatch_cfg_path=limatch_cfg,
        cloud1=cloud1,
        cloud2=cloud2,
        out_dir=out_dir,
        #cfg_overrides={"uncertainty_r": args.uncertainty_r},
    )
    print(f"\n✓ Done — résultats dans {out_dir}")
except Exception as e:
    print(f"\n✗ FAIL : {type(e).__name__}: {e}")
    sys.exit(1)