"""
s2s_limatch_from_manifest.py
----------------------------
Génère un manifest de paires depuis des groupes définis manuellement,
puis lance LiMatch avec uncertainty_r spécifique par groupe.
Reprend là où il s'est arrêté.

Usage:
    python s2s_limatch_from_manifest.py <cropped_dir> [--out-root <path>] [--limatch-cfg <path>] [--rerun-failed]
"""

import sys
import argparse
import itertools
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("cropped_dir")
parser.add_argument("--out-root", default=None)
parser.add_argument("--limatch-cfg", default=None)
parser.add_argument("--repo-root", default=None)
parser.add_argument("--rerun-failed", action="store_true")
args = parser.parse_args()

cropped_dir = Path(args.cropped_dir)
assert cropped_dir.exists(), f"Dossier introuvable : {cropped_dir}"

out_root = Path(args.out_root) if args.out_root else cropped_dir.parent / "limatch_pairs"
out_root.mkdir(parents=True, exist_ok=True)
manifest_path = out_root / "pairs_manifest.csv"

sys.path.insert(0, "/home/b085164/PDM_Romain_Defferrard/ESO-PDM")
if args.repo_root:
    sys.path.insert(0, args.repo_root)

from navtools_PDM.pipeline import run_limatch_api, get_repo_root

repo_root   = get_repo_root()
limatch_cfg = args.limatch_cfg or str(repo_root / "Patcher/submodules/limatch/configs/MLS.yml")

print(f"cropped_dir : {cropped_dir}")
print(f"out_root    : {out_root}")
print(f"limatch_cfg : {limatch_cfg}")
print()

# ==============================================================
# GROUPES — (scan_ids, uncertainty_r)
# ==============================================================
GROUPS = [
    #([1000, 2000], 15), done
    #([2000, 3000], 10), done
    #([1000, 3000], 20),
    ([6000, 7000], 12),
    ([7000, 8000], 12),
    ([5000, 6000], 28),
    ([1000, 4000], 45),
    ([2000, 4000], 35),
    
    #([1000, 2000, 4000, 5000, 6000, 7000, 8000], 60),
]

# ==============================================================
# CRÉER OU RECHARGER LE MANIFEST
# ==============================================================
if manifest_path.exists():
    print(f"Manifest existant — rechargement : {manifest_path}")
    df = pd.read_csv(manifest_path)
    if args.rerun_failed:
        n_reset = int((df["status"] == "fail").sum())
        df.loc[df["status"] == "fail", "status"] = "pending"
        print(f"  {n_reset} paires fail → pending (--rerun-failed)")
        df.to_csv(manifest_path, index=False)
else:
    rows = []
    seen = set()
    for scans, uncertainty in GROUPS:
        for a, b in itertools.combinations(scans, 2):
            key = tuple(sorted([a, b]))
            if key in seen:
                # garder le uncertainty_r le plus petit si doublon inter-groupes
                continue
            seen.add(key)
            a_name = f"merged_{a}_VUX_PUCK"
            b_name = f"merged_{b}_VUX_PUCK"
            pair_name = f"{a_name}__{b_name}"
            rows.append({
                "cloud1":        str(cropped_dir / f"{a_name}.las"),
                "cloud2":        str(cropped_dir / f"{b_name}.las"),
                "pair_name":     pair_name,
                "out_dir":       str(out_root / pair_name),
                "uncertainty_r": uncertainty,
                "status":        "pending",
            })
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"Manifest créé : {manifest_path} ({len(df)} paires)")

n_total = len(df)
print(f"Paires : {n_total} total | done={int((df['status']=='done').sum())} | fail={int((df['status']=='fail').sum())} | pending={int((df['status']=='pending').sum())}\n")

# ==============================================================
# BOUCLE LIMATCH
# ==============================================================
for idx, row in df.iterrows():
    if row["status"] in ("done", "fail"):
        continue

    pair_name   = row["pair_name"]
    cloud1      = Path(row["cloud1"])
    cloud2      = Path(row["cloud2"])
    out_dir     = Path(row["out_dir"])
    uncertainty = float(row["uncertainty_r"])

    skip = False
    for c, label in [(cloud1, "cloud1"), (cloud2, "cloud2")]:
        if not c.exists():
            print(f"[SKIP] {pair_name} — {label} introuvable : {c}")
            df.at[idx, "status"] = "fail"
            df.to_csv(manifest_path, index=False)
            skip = True
            break
    if skip:
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_overrides = {"uncertainty_r": uncertainty}
    print(f"[{idx+1}/{n_total}] {pair_name}  (uncertainty_r={uncertainty})")

    try:
        run_limatch_api(
            repo_root=repo_root,
            limatch_cfg_path=limatch_cfg,
            cloud1=cloud1,
            cloud2=cloud2,
            out_dir=out_dir,
            cfg_overrides=cfg_overrides,
        )
        df.at[idx, "status"] = "done"
        print(f"  ✓ done")
    except Exception as e:
        df.at[idx, "status"] = "fail"
        print(f"  ✗ FAIL : {type(e).__name__}: {e}")

    df.to_csv(manifest_path, index=False)

n_done   = int((df["status"] == "done").sum())
n_failed = int((df["status"] == "fail").sum())
print(f"\n{'='*50}")
print(f"Terminé : {n_done}/{n_total} réussies | {n_failed} échecs")
print(f"Manifest : {manifest_path}")