from pathlib import Path
from typing import Optional
import re
import numpy as np
import laspy as lp

SCAN_RE = re.compile(r"(\d{6})")


def extract_scan_id(p: Path) -> Optional[int]:
    """
    Example:
      250220_094545_VUX-HA1_pcd.las -> 94545
      250220_094545_VUX1-LR_pcd.las -> 94545
    """
    m = SCAN_RE.search(p.stem)
    if not m:
        return None
    return int(m.group(1))


def list_clouds(dir_path: Path) -> list[Path]:
    exts = {".txt", ".las", ".laz"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def index_by_scan(files: list[Path]) -> tuple[dict[int, Path], list[tuple[str, str]]]:
    idx = {}
    skipped = []
    for p in files:
        sid = extract_scan_id(p)
        if sid is None:
            skipped.append((p.name, "no scan id"))
            continue
        if sid in idx:
            skipped.append((p.name, f"duplicate scan={sid} (already {idx[sid].name})"))
            continue
        idx[sid] = p
    return idx, skipped


def load_txt(path: Path, delimiter: str = ",", skiprows: int = 0) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def merge_two_txt(
    a: Path,
    b: Path,
    out: Path,
    *,
    delimiter: str = ",",
    skiprows: int = 0,
    sort_by_time: bool = True,
    float_fmt: str = "%.10f",
) -> None:
    A = load_txt(a, delimiter=delimiter, skiprows=skiprows)
    B = load_txt(b, delimiter=delimiter, skiprows=skiprows)

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Column mismatch: {a.name} has {A.shape[1]} cols, {b.name} has {B.shape[1]} cols."
        )

    M = np.vstack([A, B])

    if sort_by_time:
        M = M[np.argsort(M[:, 0])]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, M, delimiter=delimiter, fmt=float_fmt)
    print(f"{a.name} + {b.name} -> {out.name} ({len(A)} + {len(B)} rows)")

def merge_two_las(
    a: Path,
    b: Path,
    out: Path,
    *,
    sort_by_time: bool = True,
) -> None:
    las_a = lp.read(a)
    las_b = lp.read(b)

    dim_a = list(las_a.point_format.dimension_names)
    dim_b = list(las_b.point_format.dimension_names)

    if dim_a != dim_b:
        raise ValueError(
            f"LAS dimension mismatch:\n"
            f"  {a.name}: {dim_a}\n"
            f"  {b.name}: {dim_b}"
        )

    n_a = len(las_a.points)
    n_b = len(las_b.points)

    # Merge real-world coordinates, not raw integer storage
    x = np.concatenate([np.asarray(las_a.x), np.asarray(las_b.x)])
    y = np.concatenate([np.asarray(las_a.y), np.asarray(las_b.y)])
    z = np.concatenate([np.asarray(las_a.z), np.asarray(las_b.z)])

    has_gps_time = "gps_time" in dim_a
    if has_gps_time:
        gps_time = np.concatenate([
            np.asarray(las_a.gps_time),
            np.asarray(las_b.gps_time)
        ])

    has_lasvec = all(d in dim_a for d in ("lasvec_x", "lasvec_y", "lasvec_z"))
    if has_lasvec:
        lasvec_x = np.concatenate([np.asarray(las_a["lasvec_x"]), np.asarray(las_b["lasvec_x"])])
        lasvec_y = np.concatenate([np.asarray(las_a["lasvec_y"]), np.asarray(las_b["lasvec_y"])])
        lasvec_z = np.concatenate([np.asarray(las_a["lasvec_z"]), np.asarray(las_b["lasvec_z"])])

    # sort if needed
    if sort_by_time and has_gps_time:
        order = np.argsort(gps_time)
        x = x[order]
        y = y[order]
        z = z[order]
        gps_time = gps_time[order]
        if has_lasvec:
            lasvec_x = lasvec_x[order]
            lasvec_y = lasvec_y[order]
            lasvec_z = lasvec_z[order]

    # create a fresh header
    header = lp.LasHeader(point_format=1, version="1.4")
    header.scales = las_a.header.scales
    header.offsets = np.array([np.min(x), np.min(y), np.min(z)])

    if has_lasvec:
        header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_x", type=np.float32))
        header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_y", type=np.float32))
        header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_z", type=np.float32))

    merged = lp.LasData(header)
    merged.x = x
    merged.y = y
    merged.z = z

    if has_gps_time:
        merged.gps_time = gps_time

    if has_lasvec:
        merged["lasvec_x"] = lasvec_x.astype(np.float32)
        merged["lasvec_y"] = lasvec_y.astype(np.float32)
        merged["lasvec_z"] = lasvec_z.astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write(out)
    print(f"{a.name} + {b.name} -> {out.name} ({n_a} + {n_b} pts)")

def merge_cloud_pairs(
    dir_a: Path,
    dir_b: Path,
    out_dir: Path,
    delimiter: str = ",",
    skiprows: int = 0,
    sort_by_time: bool = True,
    out_prefix: str = "merged_",
    out_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files_a = list_clouds(dir_a)
    files_b = list_clouds(dir_b)

    idx_a, skipped_a = index_by_scan(files_a)
    idx_b, skipped_b = index_by_scan(files_b)

    scans = sorted(set(idx_a) & set(idx_b))
    if not scans:
        print("No matching scans found. Check filenames / SCAN_RE / file extensions.")
        if skipped_a:
            print("Skipped A:", skipped_a[:10])
        if skipped_b:
            print("Skipped B:", skipped_b[:10])
        print("Files A:", [p.name for p in files_a[:10]])
        print("Files B:", [p.name for p in files_b[:10]])
        return

    ok = 0
    for scan in scans:
        a = idx_a[scan]
        b = idx_b[scan]

        if a.suffix.lower() != b.suffix.lower():
            raise ValueError(
                f"Extension mismatch for scan {scan}: {a.name} vs {b.name}"
            )

        if a.suffix.lower() == ".txt":
            out = out_dir / f"{out_prefix}{scan}{out_suffix}.txt"
            print(f"\n[Merging TXT] files {a} and {b}")
            merge_two_txt(
                a, b, out,
                delimiter=delimiter,
                skiprows=skiprows,
                sort_by_time=sort_by_time,
            )

        elif a.suffix.lower() in (".las", ".laz"):
            out = out_dir / f"{out_prefix}{scan}{out_suffix}{a.suffix.lower()}"
            print(f"\n[Merging LAS] files {a} and {b}")
            merge_two_las(
                a, b, out,
                sort_by_time=sort_by_time,
            )

        else:
            raise ValueError(f"Unsupported file type: {a.suffix}")

        ok += 1

    missing_b = sorted(set(idx_a) - set(idx_b))
    missing_a = sorted(set(idx_b) - set(idx_a))

    print("\n--- Summary ---")
    print(f"[Merging] Pairs found: {len(scans)} | Merged: {ok}")
    print(f"[Merging] Output dir: {out_dir}")
    if missing_b:
        print(f"Missing in B (first 20): {missing_b[:20]}" + (" ..." if len(missing_b) > 20 else ""))
    if missing_a:
        print(f"Missing in A (first 20): {missing_a[:20]}" + (" ..." if len(missing_a) > 20 else ""))


def merge_txt_pairs(*args, **kwargs):
    return merge_cloud_pairs(*args, **kwargs)