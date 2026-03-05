# georef_colorize_o3d.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from io_utils import read_txt_cloud, require_cols

"""
python georef_colorize_o3d.py \
  --in_txt out_err/est_with_error.txt --field e_3d \
  --out_ply out_err/est_colored_e3d.ply --vmin 0 --vmax 0.5 --show
"""

def colorize(values: np.ndarray, vmin: float, vmax: float, cmap: str = "YlGnBu"):
    s = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    return cm(s)[:, :3]  # RGB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, type=Path, help="TXT with x,y,z and error field (e_3d or e_xy)")
    ap.add_argument("--field", default="e_3d", help="Field to colorize (default e_3d)")
    ap.add_argument("--delim", default=",")
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--out_ply", required=True, type=Path, help="Output PLY (colored)")
    ap.add_argument("--show", action="store_true", help="Show interactive Open3D viewer")
    args = ap.parse_args()

    _, col, data = read_txt_cloud(args.in_txt, delimiter=args.delim)
    require_cols(col, ["x", "y", "z", args.field], path_hint=str(args.in_txt))

    xyz = data[:, [col["x"], col["y"], col["z"]]]
    val = data[:, col[args.field]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    rgb = colorize(val, args.vmin, args.vmax, cmap="YlGnBu")
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    args.out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out_ply), pcd)

    if args.show:
        o3d.visualization.draw_geometries([pcd])

    print("Wrote:", args.out_ply)

if __name__ == "__main__":
    main()