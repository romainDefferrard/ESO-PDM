"""
diagnose_sdc_timestamps.py
--------------------------
Analyse les sauts temporels négatifs dans tous les fichiers SDC d'un dossier.
Usage:
    python diagnose_sdc_timestamps.py /media/b085164/LaCie/2026spring_RD/ECCR/VUX_SDC/HA
"""

import sys
import os
import struct
import numpy as np
from pathlib import Path

def read_sdc_timestamps(sdc_file, chunk_records=2_000_000):
    """Lit uniquement les timestamps d'un fichier SDC (rapide)."""
    sdc_file = Path(sdc_file)

    with open(sdc_file, "rb") as f:
        header_info = f.read(8)
        size_of_header = struct.unpack("<I", header_info[:4])[0]

    record_dtype = np.dtype([
        ("t",   "<f8"),
        ("f1",  "<f4"), ("f2",  "<f4"),
        ("x",   "<f4"), ("y",   "<f4"), ("z",   "<f4"),
        ("u6",  "<u2"), ("u7",  "<u2"), ("u8",  "u1"),
        ("u9",  "u1"),  ("u10", "u1"),  ("u11", "<u2"),
        ("u12", "u1"),  ("u13", "u1"),  ("f14", "<f4"),
        ("u15", "<u2"),
    ])

    file_size = os.path.getsize(sdc_file)
    payload_size = file_size - size_of_header
    record_size = record_dtype.itemsize
    record_count = payload_size // record_size

    timestamps = []
    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)
        while True:
            data = np.fromfile(f, dtype=record_dtype, count=chunk_records)
            if len(data) == 0:
                break
            valid = np.isfinite(data["t"])
            timestamps.append(data["t"][valid])

    return np.concatenate(timestamps) if timestamps else np.array([])


def analyse_file(sdc_path):
    t = read_sdc_timestamps(sdc_path)
    if len(t) == 0:
        return {"file": sdc_path.name, "error": "vide ou illisible"}

    n = len(t)
    dt = np.diff(t)
    bad_idx = np.where(dt < 0)[0]
    n_bad = len(bad_idx)

    result = {
        "file":    sdc_path.name,
        "n_pts":   n,
        "t_min":   float(np.min(t)),
        "t_max":   float(np.max(t)),
        "n_jumps": n_bad,
        "jumps":   []
    }

    for i in bad_idx[:10]:  # max 10 sauts affichés
        result["jumps"].append({
            "idx":     int(i),
            "pct":     float(i / n * 100),
            "t_before": float(t[i]),
            "t_after":  float(t[i + 1]),
            "delta_s":  float(dt[i]),
        })

    return result


def main(sdc_dir):
    sdc_dir = Path(sdc_dir)
    files = sorted(sdc_dir.glob("*.sdc"))

    if not files:
        print(f"Aucun fichier .sdc trouvé dans {sdc_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC TIMESTAMPS SDC — {sdc_dir}")
    print(f"  {len(files)} fichiers trouvés")
    print(f"{'='*70}\n")

    total_jumps = 0

    for sdc_path in files:
        print(f"  Lecture : {sdc_path.name} ...", end="", flush=True)
        res = analyse_file(sdc_path)
        print(f" {res.get('n_pts', '?')} pts")

        if "error" in res:
            print(f"    !! ERREUR : {res['error']}\n")
            continue

        print(f"    t_min={res['t_min']:.2f}  t_max={res['t_max']:.2f}  "
              f"durée={res['t_max']-res['t_min']:.2f}s")
        print(f"    Sauts négatifs : {res['n_jumps']}")

        if res['n_jumps'] > 0:
            total_jumps += res['n_jumps']
            for j in res['jumps']:
                print(f"      idx={j['idx']:>8d} ({j['pct']:5.1f}%)  "
                      f"{j['t_before']:.4f} -> {j['t_after']:.4f}  "
                      f"Δt={j['delta_s']:.4f}s")
            if res['n_jumps'] > 10:
                print(f"      ... ({res['n_jumps'] - 10} sauts supplémentaires non affichés)")
        print()

    print(f"{'='*70}")
    print(f"  Total sauts négatifs : {total_jumps} sur {len(files)} fichiers")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_sdc_timestamps.py <dossier_sdc>")
        sys.exit(1)
    main(sys.argv[1])
