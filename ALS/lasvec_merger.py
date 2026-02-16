import numpy as np

def merge_vectors_exact_gpstime_csv(
    pts_csv_in: str,
    vec_csv_in: str,
    csv_out: str,
    time_col_pts: int = 0,
    time_col_vec: int = 0,
    vec_cols: tuple[int, int, int] = (1, 2, 3),
    fill_value: float = np.nan
) -> None:

    pts = np.loadtxt(pts_csv_in, delimiter=",")
    vec = np.loadtxt(vec_csv_in, delimiter=",")

    if pts.ndim == 1: pts = pts[None, :]
    if vec.ndim == 1: vec = vec[None, :]

    t_pts = pts[:, time_col_pts]
    t_vec = vec[:, time_col_vec]
    v = vec[:, vec_cols]

    # dictionnaire gps_time → vecteur
    vec_dict = {t: vv for t, vv in zip(t_vec, v)}

    merged_v = np.full((pts.shape[0], 3), fill_value)

    for i, t in enumerate(t_pts):
        if t in vec_dict:
            merged_v[i] = vec_dict[t]

    # garantir au moins 7 colonnes
    if pts.shape[1] < 7:
        pad = np.full((pts.shape[0], 7 - pts.shape[1]), fill_value)
        out = np.hstack([pts, pad])
    else:
        out = pts.copy()

    out[:, 4:7] = merged_v

    np.savetxt(csv_out, out, delimiter=",")


"""
merge_vectors_exact_gpstime(
    "points.txt",
    "vectors.txt",
    "points_with_vectors.txt"
)
"""