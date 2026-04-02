"""
build_bbox.py
=============
Génère les fichiers chunk_bbox.csv dans chaque sous-dossier merged_* de chunks_root.
Utilise la position du VEHICULE (trajectoire SBET interpolée) et non les points LiDAR.

La bbox trajectoire fait ~15m x quelques mètres → seuls les vrais croisements
physiques de la trajectoire génèrent un overlap, pas les rues parallèles.

Usage :
    conda run -n limatch python build_bbox.py
"""

import numpy as np
import pandas as pd
import laspy
from pathlib import Path
from pyproj import Transformer

# ==============================================================================
# CONFIG — adapter si nécessaire
# ==============================================================================

CHUNKS_ROOT = Path("/media/b085164/Elements/CALIB_26_02_25/georef_ALL_traj_outage_2/scenario_combined/chunks_15m")
SBET_PATH   = Path("/media/b085164/Elements/CALIB_26_02_25/ODyN_calib/Outage_2_305645_306120/traj_outage/outage_2.out")
EPSG_OUT    = "EPSG:2056"   # LV95
TIME_FIELD  = "gps_time"

# ==============================================================================
# CHARGEMENT TRAJECTOIRE SBET
# ==============================================================================

def load_sbet_xy(sbet_path: Path, epsg_out: str):
    """
    Charge la trajectoire SBET et retourne (time, x_map, y_map) en EPSG:epsg_out.
    """
    data = np.fromfile(sbet_path, dtype=np.float64).reshape(-1, 17)
    t    = data[:, 0]
    lat  = np.degrees(data[:, 1])   # radians → degrés
    lon  = np.degrees(data[:, 2])
    alt  = data[:, 3]

    # WGS84 → LV95
    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x_map, y_map, _ = transformer.transform(lon, lat, alt)

    order = np.argsort(t)
    return t[order], x_map[order], y_map[order]


# ==============================================================================
# MAIN
# ==============================================================================

print(f"Chargement trajectoire : {SBET_PATH}")
t_trj, x_trj, y_trj = load_sbet_xy(SBET_PATH, EPSG_OUT)
print(f"  {len(t_trj)} poses | t=[{t_trj[0]:.1f}, {t_trj[-1]:.1f}]s")

merged_dirs = sorted(
    [d for d in CHUNKS_ROOT.iterdir()
     if d.is_dir() and d.name.startswith("merged_")]
)
print(f"\n{len(merged_dirs)} dossiers merged trouvés\n")

for merged_dir in merged_dirs:
    las_files = sorted(merged_dir.glob("chunk_*.las"))
    if not las_files:
        print(f"  [skip] {merged_dir.name} — aucun chunk LAS")
        continue

    bbox_rows = []

    for las_path in las_files:
        try:
            # Lire uniquement les timestamps GPS (pas les coordonnées XYZ)
            with laspy.open(las_path) as reader:
                for pts in reader.chunk_iterator(500_000):
                    if not len(pts):
                        continue
                    try:
                        t_pts = np.asarray(pts[TIME_FIELD], dtype=np.float64)
                    except Exception:
                        t_pts = np.asarray(getattr(pts, TIME_FIELD), dtype=np.float64)
                    break   # on veut juste t_start / t_end

            # Bornes temporelles du chunk
            with laspy.open(las_path) as reader:
                all_t = []
                for pts in reader.chunk_iterator(500_000):
                    if not len(pts):
                        continue
                    try:
                        all_t.append(np.asarray(pts[TIME_FIELD], dtype=np.float64))
                    except Exception:
                        all_t.append(np.asarray(getattr(pts, TIME_FIELD), dtype=np.float64))

            if not all_t:
                continue

            t_all  = np.concatenate(all_t)
            t_start = float(t_all.min())
            t_end   = float(t_all.max())

            # Interpoler la position VEHICULE sur la fenêtre temporelle
            t_query  = np.linspace(t_start, t_end, 20)
            t_query  = np.clip(t_query, t_trj[0], t_trj[-1])
            x_vehicle = np.interp(t_query, t_trj, x_trj)
            y_vehicle = np.interp(t_query, t_trj, y_trj)

            bbox_rows.append({
                "chunk_file": las_path.name,
                "t_start":    t_start,
                "t_end":      t_end,
                "x_min":      float(x_vehicle.min()),
                "x_max":      float(x_vehicle.max()),
                "y_min":      float(y_vehicle.min()),
                "y_max":      float(y_vehicle.max()),
            })

        except Exception as e:
            print(f"  [WARN] {las_path.name}: {e}")

    if bbox_rows:
        df = pd.DataFrame(bbox_rows).sort_values("t_start")
        out = merged_dir / "chunk_bbox.csv"
        df.to_csv(out, index=False)
        print(f"  {merged_dir.name}: {len(bbox_rows)} chunks → {out.name}")
    else:
        print(f"  [skip] {merged_dir.name} — aucune bbox générée")

print("\nDone.")