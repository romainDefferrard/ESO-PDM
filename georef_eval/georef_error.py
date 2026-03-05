# georef_error.py
import argparse
from pathlib import Path
from typing import Union, Optional
import numpy as np
from tqdm import tqdm
import os

from io_utils import read_txt_cloud, write_txt_cloud, require_cols

"""
python georef_error.py \
  --ref cloud_ref.txt --est cloud_odyn.txt --out out_err \
  --delim "," --time_tol 1e-8 --vmax 0.5 --range_max 20

"""

#!/usr/bin/env python3
# georef_error.py
# Input TXT must have header: gps_time,x,y,z  (delimiter default ',')
# Outputs:
# - georef_error_matches.csv
# - est_with_error.txt  (gps_time,x,y,z,e_3d,e_xy,ez)  <-- for Open3D colorize
# - 3 plots



def _build_cols_from_schema(schema: str):
    schema = schema.strip().lower()
    built = []
    i = 0
    while i < len(schema):
        c = schema[i]
        if c == "t":
            built.append("gps_time")
        elif c in ["x", "y", "z"]:
            built.append(c)
        elif c == "l":
            if i + 1 >= len(schema) or schema[i+1] not in ["x", "y", "z"]:
                raise ValueError(f"Bad schema near 'l' in {schema}. Use e.g. txyzlxyz")
            built.append("l" + schema[i+1])  # lx/ly/lz
            i += 1
        else:
            raise ValueError(f"Unknown schema char '{c}' in {schema}")
        i += 1
    return built

def read_txt(
    path: Path,
    delim: str = ",",
    no_header: bool = False,
    schema: str = "txyzlxyz",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
):
    header = None
    rows = []  # list[list[float]]

    # decide column mapping now (no_header)
    if no_header:
        built = _build_cols_from_schema(schema)
        col = {n: j for j, n in enumerate(built)}
    else:
        col = None  # built after reading header

    file_size = os.path.getsize(path)

    with path.open("r", encoding="utf-8") as f, tqdm(
        total=file_size, unit="B", unit_scale=True, desc=f"Reading {path.name}"
    ) as pbar:
        for ln in f:
            pbar.update(len(ln.encode("utf-8")))
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue

            # header case
            if header is None and not no_header:
                header = [c.strip() for c in ln.split(delim)]
                col = {n: i for i, n in enumerate(header)}
                for k in ["gps_time", "x", "y", "z"]:
                    if k not in col:
                        raise ValueError(f"{path}: missing column '{k}' (found: {header})")
                continue

            parts = ln.split(delim)

            # time window filter (assumes gps_time is col 0 for no_header; or use col mapping for header)
            t = float(parts[0]) if no_header else float(parts[col["gps_time"]])

            if tmin is not None and t < tmin:
                continue
            if tmax is not None and t > tmax:
                break  # assumes sorted by gps_time

            # parse full row once (all floats)
            rows.append([float(v) for v in parts])

    if not rows:
        raise ValueError(f"{path}: no data lines found in requested time window")

    data = np.asarray(rows, dtype=np.float64)
    return col, data

def write_txt(path: Path, header: list[str], data: np.ndarray, delim: str = ","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(delim.join(header) + "\n")
        for row in data:
            f.write(delim.join(f"{v:.10f}" for v in row) + "\n")

def match_by_time(t_ref, xyz_ref, t_est, xyz_est, tol_s: float):
    oR = np.argsort(t_ref)
    tR = t_ref[oR]
    xyzR = xyz_ref[oR]

    oE = np.argsort(t_est)
    tE = t_est[oE]
    xyzE = xyz_est[oE]

    idx = np.searchsorted(tR, tE, side="left")
    i0 = np.clip(idx - 1, 0, len(tR) - 1)
    i1 = np.clip(idx,     0, len(tR) - 1)

    d0 = np.abs(tR[i0] - tE)
    d1 = np.abs(tR[i1] - tE)
    ib = np.where(d1 < d0, i1, i0)
    dt = np.abs(tR[ib] - tE)

    m = dt <= tol_s
    return tE[m], xyzR[ib[m]], xyzE[m], dt[m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--est", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delim", default=",")
    ap.add_argument("--time_tol", type=float, default=0.001)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--no_header", action="store_true", help="Input TXT has no header")
    ap.add_argument("--schema", default="txyzlxyz", help="Column schema if no header. Default: t x y z lx ly lz")
    ap.add_argument("--o3d_only", action="store_true", help="Only write est_with_error.txt (no CSV, no plots)")
    args = ap.parse_args()

    colR, dR = read_txt(args.ref, args.delim, no_header=args.no_header, schema=args.schema, tmin=args.tmin, tmax=args.tmax)
    colE, dE = read_txt(args.est, args.delim, no_header=args.no_header, schema=args.schema, tmin=args.tmin, tmax=args.tmax)

    tR = dR[:, colR["gps_time"]]
    pR = dR[:, [colR["x"], colR["y"], colR["z"]]]

    tE = dE[:, colE["gps_time"]]
    pE = dE[:, [colE["x"], colE["y"], colE["z"]]]

    # Optional time window (reduces memory/time massively on huge files)
    if args.tmin is not None or args.tmax is not None:
        mR = np.ones_like(tR, dtype=bool)
        mE = np.ones_like(tE, dtype=bool)
        if args.tmin is not None:
            mR &= (tR >= args.tmin)
            mE &= (tE >= args.tmin)
        if args.tmax is not None:
            mR &= (tR <= args.tmax)
            mE &= (tE <= args.tmax)
        tR, pR = tR[mR], pR[mR]
        tE, pE = tE[mE], pE[mE]

    t, pref, pest, dt = match_by_time(tR, pR, tE, pE, args.time_tol)
    if len(t) < 50:
        raise RuntimeError(f"Too few matches: {len(t)} (increase --time_tol?)")

    d = pest - pref
    ex, ey, ez = d[:, 0], d[:, 1], d[:, 2]
    e_xy = np.sqrt(ex**2 + ey**2)
    e_3d = np.sqrt(ex**2 + ey**2 + ez**2)

    args.out.mkdir(parents=True, exist_ok=True)


    # 2) TXT for Open3D colorize (estimated points + error fields)
    out_txt = args.out / "est_with_error.txt"
    header = ["gps_time","x","y","z","e_3d","e_xy","ez"]
    data = np.column_stack([t, pest, e_3d, e_xy, ez])
    write_txt(out_txt, header, data, delim=args.delim)

    # If user only wants Open3D, stop here
    if args.o3d_only:
        print("Wrote:", out_txt)
        return
    # 3) plots
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(e_3d, bins=120)
    plt.xlabel("3D error [m]"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.out / "hist_3d.png", dpi=200)
    plt.close()

    o = np.argsort(t)
    plt.figure()
    plt.plot(t[o], e_3d[o])
    plt.xlabel("gps_time [s]"); plt.ylabel("3D error [m]")
    plt.tight_layout()
    plt.savefig(args.out / "time_3d.png", dpi=200)
    plt.close()

    plt.figure()
    sc = plt.scatter(pref[:, 0], pref[:, 1], c=e_xy, s=1, vmin=0.0, vmax=args.vmax)
    plt.axis("equal")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.colorbar(sc, label="XY error [m]")
    plt.tight_layout()
    plt.savefig(args.out / "map_xy_error.png", dpi=250)
    plt.close()

    print("OK")
    print("TXT for Open3D:", out_txt)

if __name__ == "__main__":
    main()