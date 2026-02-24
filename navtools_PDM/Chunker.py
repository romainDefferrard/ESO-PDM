import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

from pointCloudGeoref import load_config, load_and_prepare_trajectory, trajectory_positions_mapping


def cumulative_distance_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)
    return np.concatenate(([0.0], np.cumsum(ds)))


def invert_s_to_t(s: np.ndarray, t: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    return np.interp(s_query, s, t)


def build_s_of_t(cfg: Dict[str, Any], epsg_out: str = "EPSG:2056"):
    trj = load_and_prepare_trajectory(cfg)
    P = trajectory_positions_mapping(trj, epsg_out=epsg_out)  # Nx3 in map
    t = np.asarray(trj.time, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    P = P[order]

    s = cumulative_distance_xy(P[:, 0], P[:, 1])
    return t, s


def chunk_txt_by_distance(
    txt_path: Union[str, Path],
    cfg_georef_path: Union[str, Path],
    out_dir: Union[str, Path],
    L: float,
    epsg_out: str = "EPSG:2056",
    delimiter: str = ",",
    skiprows: int = 0,
    min_points: int = 1000,
    sort_txt_by_time: bool = True,
    out_prefix: str = "chunk_",
):
    txt_path = Path(txt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = np.loadtxt(txt_path, delimiter=delimiter, skiprows=skiprows)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    if sort_txt_by_time:
        pts = pts[np.argsort(pts[:, 0])]

    t_pts = pts[:, 0].astype(np.float64)
    tmin_pts = float(t_pts[0])
    tmax_pts = float(t_pts[-1])

    cfg = load_config(cfg_georef_path)
    t_trj, s_trj = build_s_of_t(cfg, epsg_out=epsg_out)

    keep = (t_trj >= tmin_pts) & (t_trj <= tmax_pts)
    if int(keep.sum()) < 2:
        raise RuntimeError("Trajectory time range does not overlap TXT gps_time range.")

    t_trj = t_trj[keep]
    s_trj = s_trj[keep]

    s_start = float(s_trj[0])
    s_end = float(s_trj[-1])
    starts = np.arange(s_start, s_end, L, dtype=np.float64)

    written = 0
    for cid, s0 in enumerate(starts):
        s1 = s0 + L
        if s0 >= s_end:
            break

        t0, t1 = invert_s_to_t(s_trj, t_trj, np.array([s0, min(s1, s_end)], dtype=np.float64))

        mask = (t_pts >= t0) & (t_pts < t1)
        if int(mask.sum()) < min_points:
            continue

        out_path = out_dir / f"{out_prefix}{cid:05d}.txt"
        np.savetxt(out_path, pts[mask], delimiter=delimiter, fmt="%.10f")
        written += 1

    print(f"Chunk Creation Done. Wrote {written} chunks to: {out_dir}")