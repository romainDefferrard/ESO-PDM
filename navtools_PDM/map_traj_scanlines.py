#!/usr/bin/env python3
# ==============================================================
# map_traj_scanlines.py
#
# Carte Folium :
#   - Trajectoire ref (ODyN ref) — monochrome rouge
#   - Trajectoire F2B — colorisée par scanline (même palette)
#   - Scanlines : segment de la traj ref par fenêtre manifest,
#     avec tooltip (scan_id, filename, t_start, t_end, durée)
#   - Légende avec toutes les scanlines et leurs couleurs
#   - MeasureControl pour mesurer les distances
# ==============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium import plugins

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

OUT_REF_PATH  = Path("/media/b085164/LaCie/2026spring_RD/ECCR/ODyN/base/out/ODyN_GNSS_INS.out")
OUT_F2B_PATH  = Path("/media/b085164/LaCie/2026spring_RD/ECCR/ODyN/zone_3_final/APX/F2B/out/F2B_3_APX.out")
MANIFEST_PATH = Path("/media/b085164/LaCie/2026spring_RD/ECCR/georef_ALL_zone_3_outage/APX/merged/ALL/merged_manifest.csv")
OUTPUT_HTML   = Path("/home/b085164/PDM_Romain_Defferrard/ESO-PDM/map_traj_scanlines.html")

LABEL_REF = "ODyN ref"
LABEL_F2B = "ODyN F2B APX"

MAX_DISPLAY_POINTS = 5000

# Palette partagée scanlines <-> traj F2B
PALETTE = [
    "#e67e22", "#8e44ad", "#16a085", "#c0392b", "#2c3e50",
    "#d35400", "#1abc9c", "#6c3483", "#117a65", "#7f8c8d",
]

# --------------------------------------------------------------
# SBET helpers
# --------------------------------------------------------------

SBET_DTYPE = np.dtype([
    ("time",    np.float64), ("lat",   np.float64), ("lon",    np.float64),
    ("alt",     np.float64), ("vx",    np.float64), ("vy",     np.float64),
    ("vz",      np.float64), ("roll",  np.float64), ("pitch",  np.float64),
    ("heading", np.float64), ("wander",np.float64), ("ax",     np.float64),
    ("ay",      np.float64), ("az",    np.float64), ("wx",     np.float64),
    ("wy",      np.float64), ("wz",    np.float64),
])

def load_sbet(path: Path) -> pd.DataFrame:
    print(f"  Chargement {path.name}...")
    return pd.DataFrame(np.fromfile(path, dtype=SBET_DTYPE))

def decimate(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[np.linspace(0, len(arr) - 1, n).astype(int)]

def get_traj_arrays(df: pd.DataFrame, n: int):
    lats  = np.degrees(decimate(df["lat"].to_numpy(), n))
    lons  = np.degrees(decimate(df["lon"].to_numpy(), n))
    times = decimate(df["time"].to_numpy(), n)
    return lats, lons, times

# --------------------------------------------------------------
# Colormap partagé : scan_id -> {color, t_start, t_end, filename}
# --------------------------------------------------------------

def build_colormap(manifest: pd.DataFrame) -> dict:
    result = {}
    for i, row in manifest.iterrows():
        result[int(row["scan_id"])] = {
            "color":    PALETTE[i % len(PALETTE)],
            "t_start":  float(row["t_start"]),
            "t_end":    float(row["t_end"]),
            "filename": row["filename"],
        }
    return result

def color_for_time(t: float, colormap: dict):
    """Retourne (color, scan_id) pour un timestamp, gris si hors fenetre."""
    for scan_id, info in colormap.items():
        if info["t_start"] <= t <= info["t_end"]:
            return info["color"], scan_id
    return "#aaaaaa", None

# --------------------------------------------------------------
# Trajectoire ref — monochrome
# --------------------------------------------------------------

def add_trajectory(group, lats, lons, times, color, label):
    for i in range(len(lats) - 1):
        folium.PolyLine(
            locations=[(lats[i], lons[i]), (lats[i+1], lons[i+1])],
            color=color, weight=2.5, opacity=0.85,
            tooltip=f"{label} | t = {times[i]:.3f} s",
        ).add_to(group)

# --------------------------------------------------------------
# Trajectoire F2B — colorisée par scanline
# --------------------------------------------------------------

def add_trajectory_colored(group, lats, lons, times, colormap, label):
    for i in range(len(lats) - 1):
        t_mid = (times[i] + times[i + 1]) / 2.0
        color, scan_id = color_for_time(t_mid, colormap)
        scan_label = f"scan {scan_id}" if scan_id is not None else "hors outage"
        folium.PolyLine(
            locations=[(lats[i], lons[i]), (lats[i+1], lons[i+1])],
            color=color, weight=3.5, opacity=0.9,
            tooltip=f"{label} | {scan_label} | t = {times[i]:.3f} s",
        ).add_to(group)

# --------------------------------------------------------------
# Scanlines — segment de la traj ref par fenetre manifest
# --------------------------------------------------------------

def extract_segment(df: pd.DataFrame, t_start: float, t_end: float):
    t = df["time"].to_numpy()
    mask = (t >= t_start) & (t <= t_end)
    sub = df.loc[mask]
    if len(sub) == 0:
        return None
    lats = np.degrees(sub["lat"].to_numpy())
    lons = np.degrees(sub["lon"].to_numpy())
    idx  = np.linspace(0, len(lats) - 1, min(len(lats), 200)).astype(int)
    return list(zip(lats[idx], lons[idx]))


def add_scanlines(m, df_ref: pd.DataFrame, manifest: pd.DataFrame, colormap: dict):
    fg = folium.FeatureGroup(name="Scanlines (ref)", show=True)

    for scan_id, info in colormap.items():
        t_start  = info["t_start"]
        t_end    = info["t_end"]
        color    = info["color"]
        filename = info["filename"]
        dur      = t_end - t_start

        coords = extract_segment(df_ref, t_start, t_end)
        if coords is None:
            print(f"  [WARN] scan_id={scan_id} : aucun point dans [{t_start:.1f}, {t_end:.1f}]")
            continue

        tooltip_text = (
            f"<b>Scan {scan_id}</b><br>"
            f"{filename}<br>"
            f"t_start = {t_start:.3f} s<br>"
            f"t_end   = {t_end:.3f} s<br>"
            f"duree   = {dur:.1f} s"
        )

        folium.PolyLine(
            locations=coords,
            color=color, weight=4, opacity=0.75,
            tooltip=folium.Tooltip(tooltip_text, sticky=True),
        ).add_to(fg)

        folium.CircleMarker(
            location=coords[0],
            radius=5, color=color, fill=True, fill_opacity=0.9,
            tooltip=folium.Tooltip(f"Debut scan {scan_id}", sticky=False),
        ).add_to(fg)

    fg.add_to(m)
    print(f"  {len(colormap)} scanlines ajoutees.")

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    print("Chargement des trajectoires...")
    df_ref = load_sbet(OUT_REF_PATH)
    df_f2b = load_sbet(OUT_F2B_PATH)

    print("Chargement manifest...")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  {len(manifest)} scanlines dans le manifest")

    t_min = float(manifest["t_start"].min())
    t_max = float(manifest["t_end"].max())
    print(f"  Fenetre manifest : {t_min:.1f} -> {t_max:.1f} s")

    colormap = build_colormap(manifest)

    # Filtrer F2B sur la zone manifest (+-60s de marge)
    mask_f2b = (df_f2b["time"] >= t_min - 60) & (df_f2b["time"] <= t_max + 60)
    df_f2b_zone = df_f2b.loc[mask_f2b].copy()
    if len(df_f2b_zone) == 0:
        print("  [WARN] F2B : aucun point dans la fenetre manifest, utilisation complete")
        df_f2b_zone = df_f2b

    lats_ref, lons_ref, t_ref = get_traj_arrays(df_ref, MAX_DISPLAY_POINTS)
    lats_f2b, lons_f2b, t_f2b = get_traj_arrays(df_f2b_zone, MAX_DISPLAY_POINTS)

    mask_c = (df_ref["time"] >= t_min) & (df_ref["time"] <= t_max)
    sub_c  = df_ref.loc[mask_c]
    center_lat = float(np.degrees(sub_c["lat"].mean())) if len(sub_c) else float(lats_ref.mean())
    center_lon = float(np.degrees(sub_c["lon"].mean())) if len(sub_c) else float(lons_ref.mean())

    print("Construction de la carte Folium...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15, max_zoom=22,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/">CARTO</a>'
        ),
    )

    # Traj ref — rouge monochrome
    fg_ref = folium.FeatureGroup(name=LABEL_REF, show=True)
    add_trajectory(fg_ref, lats_ref, lons_ref, t_ref, "#e74c3c", LABEL_REF)
    fg_ref.add_to(m)

    # Traj F2B — colorisee par scanline
    fg_f2b = folium.FeatureGroup(name=LABEL_F2B, show=True)
    add_trajectory_colored(fg_f2b, lats_f2b, lons_f2b, t_f2b, colormap, LABEL_F2B)
    fg_f2b.add_to(m)

    # Scanlines sur la traj ref
    add_scanlines(m, df_ref, manifest, colormap)

    folium.LayerControl(collapsed=False).add_to(m)

    plugins.MeasureControl(
        position="bottomright",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
    ).add_to(m)

    # Legende avec toutes les scanlines et leurs couleurs
    scanline_rows = "".join(
        f'<span style="color:{info["color"]};font-size:17px;">&#9644;</span> '
        f'<b>Scan {sid}</b> '
        f'<span style="color:#888;font-size:11px;">{info["filename"]}</span><br>'
        for sid, info in colormap.items()
    )

    legend_html = f"""
    <div style="position:fixed;top:30px;right:30px;z-index:1000;
                background:white;padding:11px 15px;border-radius:7px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.25);font-size:13px;
                line-height:2.0;max-height:92vh;overflow-y:auto;min-width:240px;">
      <div style="font-weight:bold;font-size:14px;margin-bottom:4px;">Legende</div>
      <span style="color:#e74c3c;font-size:17px;">&#9644;</span> {LABEL_REF}<br>
      <hr style="margin:5px 0;">
      <div style="font-weight:bold;">{LABEL_F2B}</div>
      <span style="font-size:11px;color:#666;">Colorise par scanline (meme couleur ci-dessous)</span><br>
      <span style="color:#aaaaaa;font-size:17px;">&#9644;</span>
      <span style="font-size:12px;"> Hors fenetre manifest</span>
      <hr style="margin:5px 0;">
      <div style="font-weight:bold;">Scanlines (ref)</div>
      <span style="font-size:11px;color:#666;">Survol = scan_id, fichier, t GPS</span><br>
      {scanline_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUTPUT_HTML))
    print(f"\nSauvegarde -> {OUTPUT_HTML}")


if __name__ == "__main__":
    main()