# ============================================================
# CELL 0 — Imports
# ============================================================
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import rowcol


# ============================================================
# CELL 1 — Paramètres
# ============================================================

GCP_PATH   = "/media/b085164/Elements/ECCR/GCP/gcp_LV95_NF02.txt"
GRID_PATH  = ("/home/b085164/miniconda3/envs/limatch/lib/python3.9"
              "/site-packages/pyproj/proj_dir/share/proj/CH"
              "/ch_swisstopo_chgeo2004_ETRS89_LN02.tif")

# Dossier contenant les nuages géoréférencés avec l'outage (trajectoire dégradée)
# Ces nuages DOIVENT avoir les extra bytes lasvec_x, lasvec_y, lasvec_z
OUTAGE_GEOREF_DIR = "/media/b085164/Elements/ECCR/georef_VUX_zone_1/HA"

# Scénarios : nom → dossier contenant les nuages géoréférencés
SCENARIOS = {
    "INS only":  "/media/b085164/Elements/ECCR/scenarios/ins_only/HA",
    "F2B":       "/media/b085164/Elements/ECCR/scenarios/f2b/HA",
    "Combined":  "/media/b085164/Elements/ECCR/scenarios/combined/HA",
}

# Picks CloudCompare : position des GCPs dans les nuages OUTAGE
# Format : (gcp_id, x_picked, y_picked, z_picked)
# Colle tes lignes [Picked] CC ici :
CC_PICKS = [
    ("5008", 2532886.665, 1154716.171, 457.835),
    # ("5009", ...),
]

SEARCH_RADIUS_M = 0.5   # rayon pour trouver les voisins autour du pick
MIN_NEIGHBORS   = 3


# ============================================================
# CELL 2 — Chargement et conversion des GCPs NF02 → ellipsoïdal
# ============================================================

gcp_raw = pd.read_csv(GCP_PATH, sep=r'\s+', header=None,
                      names=["id", "x", "y", "z_nf02"])

lv95_to_etrs = Transformer.from_crs("EPSG:2056", "EPSG:4258", always_xy=True)

with rasterio.open(GRID_PATH) as src:
    grid = src.read(1)
    lons, lats, _ = lv95_to_etrs.transform(
        gcp_raw["x"].values, gcp_raw["y"].values, gcp_raw["z_nf02"].values
    )
    rows, cols = rowcol(src.transform, lons, lats)
    N = grid[np.array(rows), np.array(cols)]

gcp_raw["N"]       = N
gcp_raw["z_ellips"] = gcp_raw["z_nf02"] + N

# Index par id pour lookup rapide
gcp = gcp_raw.set_index("id")

print(f"{len(gcp)} GCPs convertis  |  N moyen = {N.mean():.3f} m")
print(gcp[["x","y","z_nf02","N","z_ellips"]].head())


# ============================================================
# CELL 3 — Chargement des nuages outage + extraction des vecteurs laser
# ============================================================

def load_las_files(directory: str):
    """Charge tous les .las/.laz d'un dossier en un seul array."""
    paths = sorted(Path(directory).glob("*.las")) + \
            sorted(Path(directory).glob("*.laz"))
    if not paths:
        raise FileNotFoundError(f"Aucun .las/.laz dans {directory}")

    all_x, all_y, all_z, all_t = [], [], [], []
    all_lx, all_ly, all_lz    = [], [], []

    for p in paths:
        las = laspy.read(str(p))
        all_x.append(np.array(las.x))
        all_y.append(np.array(las.y))
        all_z.append(np.array(las.z))
        all_t.append(np.array(las.gps_time))
        try:
            all_lx.append(np.array(las["lasvec_x"], dtype=np.float32))
            all_ly.append(np.array(las["lasvec_y"], dtype=np.float32))
            all_lz.append(np.array(las["lasvec_z"], dtype=np.float32))
        except Exception:
            all_lx.append(np.full(len(las.x), np.nan, dtype=np.float32))
            all_ly.append(np.full(len(las.x), np.nan, dtype=np.float32))
            all_lz.append(np.full(len(las.x), np.nan, dtype=np.float32))

        print(f"  {p.name} : {len(las.x):,} pts")

    return {
        "x":  np.concatenate(all_x),
        "y":  np.concatenate(all_y),
        "z":  np.concatenate(all_z),
        "t":  np.concatenate(all_t),
        "lx": np.concatenate(all_lx),
        "ly": np.concatenate(all_ly),
        "lz": np.concatenate(all_lz),
    }


print(f"Chargement nuages outage depuis {OUTAGE_GEOREF_DIR} ...")
cloud_ref = load_las_files(OUTAGE_GEOREF_DIR)
tree_ref  = cKDTree(np.column_stack((cloud_ref["x"], cloud_ref["y"], cloud_ref["z"])))
print(f"Total : {len(cloud_ref['x']):,} points")


# ============================================================
# CELL 4 — Extraction des vecteurs laser pour chaque GCP
# ============================================================

def extract_laser_vectors(cc_picks, cloud, tree, gcp_df,
                          search_radius=SEARCH_RADIUS_M,
                          min_neighbors=MIN_NEIGHBORS):
    """
    Pour chaque pick CC :
      1. Trouve les voisins dans le nuage outage
      2. Extrait le vecteur laser moyen (frame body) + timestamp
      3. Merge avec les coordonnées terrain du GCP (ellipsoïdal)
    """
    rows = []
    for gcp_id, x_pick, y_pick, z_pick in cc_picks:

        if gcp_id not in gcp_df.index:
            print(f"  [warn] GCP {gcp_id} introuvable dans le fichier GCP")
            continue

        idxs = tree.query_ball_point([x_pick, y_pick, z_pick], r=search_radius)

        if len(idxs) < min_neighbors:
            print(f"  [warn] GCP {gcp_id} : {len(idxs)} voisins < {min_neighbors} → ignoré")
            continue

        idxs = np.array(idxs)

        # Dédoublonnage : un point peut apparaître plusieurs fois si les
        # nuages se chevauchent — on garde les indices uniques par timestamp
        _, unique_idx = np.unique(cloud["t"][idxs], return_index=True)
        idxs = idxs[unique_idx]

        lx = cloud["lx"][idxs].mean()
        ly = cloud["ly"][idxs].mean()
        lz = cloud["lz"][idxs].mean()
        norm = np.sqrt(lx**2 + ly**2 + lz**2)
        if norm > 1e-6:
            lx, ly, lz = lx/norm, ly/norm, lz/norm

        t_mean = cloud["t"][idxs].mean()

        gcp_row = gcp_df.loc[gcp_id]
        rows.append({
            "gcp_id":      gcp_id,
            "x_gcp":       gcp_row["x"],
            "y_gcp":       gcp_row["y"],
            "z_gcp":       gcp_row["z_ellips"],
            "x_pick":      x_pick,
            "y_pick":      y_pick,
            "z_pick":      z_pick,
            "t_mean":      t_mean,
            "lx":          lx,
            "ly":          ly,
            "lz":          lz,
            "n_neighbors": len(idxs),
        })
        print(f"  GCP {gcp_id} : {len(idxs)} voisins | t={t_mean:.3f}s | "
              f"lvec=({lx:.3f},{ly:.3f},{lz:.3f})")

    return pd.DataFrame(rows)


df_lasvec = extract_laser_vectors(CC_PICKS, cloud_ref, tree_ref, gcp)
print(f"\n{len(df_lasvec)} GCPs avec vecteurs laser extraits")
display(df_lasvec)


# ============================================================
# CELL 5 — Estimation des erreurs par scénario
# ============================================================

def estimate_errors_from_lasvec(scenario_dir: str, df_lv: pd.DataFrame,
                                 search_radius: float = SEARCH_RADIUS_M,
                                 dt_s: float = 0.5) -> pd.DataFrame:
    """
    Pour chaque vecteur laser extrait :
      1. Charge le nuage du scénario
      2. Retrouve les points au même timestamp (± dt_s) près de la position pick
      3. Calcule l'erreur vs GCP terrain
    """
    print(f"\n  Chargement {scenario_dir} ...")
    cloud_scen = load_las_files(scenario_dir)

    rows = []
    for _, row in df_lv.iterrows():
        # Filtre temporel d'abord (rapide)
        t_mask = np.abs(cloud_scen["t"] - row["t_mean"]) < dt_s
        if t_mask.sum() == 0:
            print(f"  [warn] GCP {row['gcp_id']} : aucun point dans ±{dt_s}s")
            rows.append(_nan_row(row["gcp_id"]))
            continue

        idx_t = np.where(t_mask)[0]

        # Puis filtre spatial sur ces points-là
        pts_t = np.column_stack((
            cloud_scen["x"][idx_t],
            cloud_scen["y"][idx_t],
            cloud_scen["z"][idx_t],
        ))
        tree_t  = cKDTree(pts_t)
        # Rayon élargi pour couvrir la dérive éventuelle du scénario
        near    = tree_t.query_ball_point(
            [row["x_pick"], row["y_pick"], row["z_pick"]],
            r=search_radius * 20
        )

        if len(near) == 0:
            print(f"  [warn] GCP {row['gcp_id']} : aucun point spatial dans le scénario")
            rows.append(_nan_row(row["gcp_id"]))
            continue

        near = np.array(near)

        # Position estimée = centroïde
        x_est = pts_t[near, 0].mean()
        y_est = pts_t[near, 1].mean()
        z_est = pts_t[near, 2].mean()

        e_x = x_est - row["x_gcp"]
        e_y = y_est - row["y_gcp"]
        e_z = z_est - row["z_gcp"]

        rows.append({
            "gcp_id":   row["gcp_id"],
            "x_est":    x_est,
            "y_est":    y_est,
            "z_est":    z_est,
            "e_x":      e_x,
            "e_y":      e_y,
            "e_z":      e_z,
            "e_planim": np.sqrt(e_x**2 + e_y**2),
            "e_3d":     np.sqrt(e_x**2 + e_y**2 + e_z**2),
        })

    return pd.DataFrame(rows)


def _nan_row(gcp_id):
    return {"gcp_id": gcp_id,
            "x_est": np.nan, "y_est": np.nan, "z_est": np.nan,
            "e_x": np.nan, "e_y": np.nan, "e_z": np.nan,
            "e_planim": np.nan, "e_3d": np.nan}


results = {}
for scenario_name, scenario_dir in SCENARIOS.items():
    print(f"\n=== {scenario_name} ===")
    df_err = estimate_errors_from_lasvec(scenario_dir, df_lasvec)
    df_err["scenario"] = scenario_name
    results[scenario_name] = df_err
    rmse_p = np.sqrt((df_err["e_planim"]**2).mean())
    rmse_3 = np.sqrt((df_err["e_3d"]**2).mean())
    print(f"  RMSE planim = {rmse_p:.3f} m  |  RMSE 3D = {rmse_3:.3f} m")


# ============================================================
# CELL 6 — Tableau récapitulatif
# ============================================================

summary = []
for name, df_err in results.items():
    summary.append({
        "Scenario":        name,
        "N GCPs":          int(df_err["e_planim"].notna().sum()),
        "RMSE X [m]":      np.sqrt((df_err["e_x"]**2).mean()),
        "RMSE Y [m]":      np.sqrt((df_err["e_y"]**2).mean()),
        "RMSE Z [m]":      np.sqrt((df_err["e_z"]**2).mean()),
        "RMSE planim [m]": np.sqrt((df_err["e_planim"]**2).mean()),
        "RMSE 3D [m]":     np.sqrt((df_err["e_3d"]**2).mean()),
        "Max planim [m]":  df_err["e_planim"].max(),
    })

df_summary = pd.DataFrame(summary).set_index("Scenario")
display(df_summary.round(3))


# ============================================================
# CELL 7 — Plot
# ============================================================

gcp_ids = df_lasvec["gcp_id"].values
n_scen  = len(results)
width   = 0.8 / n_scen

COLORS = {
    "INS only": "#888888",
    "F2B":      "#B51F1F",
    "Combined": "#007480",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle("GCP errors per scenario", fontsize=12)

for ax, comp, label in zip(axes,
    ["e_planim", "e_z", "e_3d"],
    ["Planimetric error [m]", "Vertical error [m]", "3D error [m]"]):

    x = np.arange(len(gcp_ids))
    for i, (name, df_err) in enumerate(results.items()):
        vals = []
        for gid in gcp_ids:
            match = df_err.loc[df_err["gcp_id"] == gid, comp]
            vals.append(float(match.values[0]) if len(match) else np.nan)

        ax.bar(x + i * width, vals, width=width,
               color=COLORS.get(name, "grey"),
               label=name, alpha=0.85, edgecolor="none")

    ax.set_xticks(x + width * (n_scen - 1) / 2)
    ax.set_xticklabels(gcp_ids, rotation=45, fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", color="grey", ls="--", alpha=0.3)

axes[0].legend(fontsize=9)
plt.tight_layout()
plt.savefig("gcp_errors_per_scenario.png", dpi=200, bbox_inches="tight")
plt.show()