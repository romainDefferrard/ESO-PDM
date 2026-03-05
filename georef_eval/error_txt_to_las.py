#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import laspy


def read_error_txt(path: Path, delim: str = ","):
    """
    Expects header with at least:
      gps_time,x,y,z,e_3d
    Optional:
      e_xy,ez
    """
    # read header
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(delim)
    header = [h.strip() for h in header]
    col = {name: i for i, name in enumerate(header)}

    required = ["gps_time", "x", "y", "z", "e_3d"]
    for k in required:
        if k not in col:
            raise ValueError(f"{path}: missing column '{k}' in header {header}")

    data = np.loadtxt(str(path), delimiter=delim, skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]  # single row

    return col, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, type=Path)
    ap.add_argument("--out_las", required=True, type=Path)
    ap.add_argument("--delim", default=",")
    ap.add_argument("--scale", type=float, default=0.001, help="LAS coordinate scale (meters)")
    args = ap.parse_args()

    col, data = read_error_txt(args.in_txt, delim=args.delim)

    t = data[:, col["gps_time"]].astype(np.float64)
    x = data[:, col["x"]].astype(np.float64)
    y = data[:, col["y"]].astype(np.float64)
    z = data[:, col["z"]].astype(np.float64)

    e3d = data[:, col["e_3d"]].astype(np.float32)
    exy = data[:, col["e_xy"]].astype(np.float32) if "e_xy" in col else None
    ez  = data[:, col["ez"]].astype(np.float32) if "ez" in col else None

    # LAS 1.4 + point format 6 includes gps_time
    hdr = laspy.LasHeader(point_format=6, version="1.4")

    # scales/offsets: important so coordinates don’t overflow int storage
    hdr.scales = np.array([args.scale, args.scale, args.scale], dtype=np.float64)
    hdr.offsets = np.array([float(x.min()), float(y.min()), float(z.min())], dtype=np.float64)

    las = laspy.LasData(hdr)

    las.x = x
    las.y = y
    las.z = z
    las.gps_time = t

    # Add Extra Bytes for errors (CloudCompare will see these as scalar fields)
    las.add_extra_dim(laspy.ExtraBytesParams(name="e_3d", type=np.float32))
    las["e_3d"] = e3d

    if exy is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="e_xy", type=np.float32))
        las["e_xy"] = exy

    if ez is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="ez", type=np.float32))
        las["ez"] = ez

    args.out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(args.out_las))
    print("Wrote:", args.out_las)
    print("Points:", len(las.x))
    print("Extra dims:", [d.name for d in las.point_format.extra_dimensions])


if __name__ == "__main__":
    main()