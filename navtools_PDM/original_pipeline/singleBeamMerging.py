from pathlib import Path
from typing import Optional
import re
import numpy as np
import laspy as lp

SCAN_RE = re.compile(r"^(\d{6})_(\d{6})")

def extract_scan_id(p: Path) -> Optional[int]:
    """
    Example:
      260225_124306_VUX-HA1_pcd.las -> 124306
      260225_124306_VUX1-LR_pcd.las -> 124306
    """
    m = SCAN_RE.match(p.stem)
    if not m:
        return None
    return int(m.group(2))


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

    if sort_by_time:
        raise NotImplementedError(
            "sort_by_time=True is too memory expensive for large LAS merges. "
            "Use sort_by_time=False for streaming merge."
        )

    with lp.open(a) as fa, lp.open(b) as fb:
        header_a = fa.header
        header_b = fb.header

        dim_a = list(header_a.point_format.dimension_names)
        dim_b = list(header_b.point_format.dimension_names)

        if dim_a != dim_b:
            raise ValueError(
                f"LAS dimension mismatch:\n"
                f"  {a.name}: {dim_a}\n"
                f"  {b.name}: {dim_b}"
            )

        has_lasvec = all(d in dim_a for d in ("lasvec_x", "lasvec_y", "lasvec_z"))

        # build output header from A
        header = lp.LasHeader(
            point_format=header_a.point_format,
            version=header_a.version,
        )
        header.scales = np.minimum(
            np.array(header_a.scales, dtype=np.float64),
            np.array(header_b.scales, dtype=np.float64),
        )
        mins_a = np.array(header_a.mins, dtype=np.float64)
        mins_b = np.array(header_b.mins, dtype=np.float64)
        header.offsets = np.floor(np.minimum(mins_a, mins_b))

        if has_lasvec and "lasvec_x" not in list(header.point_format.dimension_names):
            header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_x", type=np.float32))
            header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_y", type=np.float32))
            header.add_extra_dim(lp.ExtraBytesParams(name="lasvec_z", type=np.float32))

        out.parent.mkdir(parents=True, exist_ok=True)

        n_a = header_a.point_count
        n_b = header_b.point_count

        def reencode_chunk(points, header_out):
            """
            Re-encode a chunk into the output header so X/Y/Z use the correct
            scales/offsets and do not get spatially shifted.
            """
            las_in = lp.ScaleAwarePointRecord(
                points.array,
                point_format=points.point_format,
                scales=points.scales,
                offsets=points.offsets,
            )

            las_out = lp.LasData(header_out)

            # coordinates re-encoded with output header
            las_out.x = las_in.x
            las_out.y = las_in.y
            las_out.z = las_in.z

            # copy all other dimensions except raw/internal XYZ
            for dim in header_out.point_format.dimension_names:
                if dim in ("X", "Y", "Z", "x", "y", "z"):
                    continue
                if dim in points.array.dtype.names:
                    las_out[dim] = points[dim]

            return las_out

        with lp.open(out, mode="w", header=header) as writer:
            for points in fa.chunk_iterator(1_000_000):
                las_out = reencode_chunk(points, header)
                writer.write_points(las_out.points)

            for points in fb.chunk_iterator(1_000_000):
                las_out = reencode_chunk(points, header)
                writer.write_points(las_out.points)

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
    start_idx: Optional[int] = None,
    step: int = 1000,
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
    for i, scan in enumerate(scans):
        a = idx_a[scan]
        b = idx_b[scan]
        scan_label = start_idx + i * step if start_idx is not None else scan
        if a.suffix.lower() != b.suffix.lower():
            raise ValueError(
                f"Extension mismatch for scan {scan}: {a.name} vs {b.name}"
            )

        if a.suffix.lower() == ".txt":
            out = out_dir / f"{out_prefix}{scan_label}{out_suffix}.txt"
            print(f"\n[Merging TXT] files {a} and {b}")
            merge_two_txt(
                a, b, out,
                delimiter=delimiter,
                skiprows=skiprows,
                sort_by_time=sort_by_time,
            )

        elif a.suffix.lower() in (".las", ".laz"):
            out = out_dir / f"{out_prefix}{scan_label}{out_suffix}{a.suffix.lower()}"
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

