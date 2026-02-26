from pathlib import Path
from typing import Optional
import re
import numpy as np

SCAN_RE = re.compile(r"_(\d{3,6})(?=_)")  # capture un bloc de digits entouré par underscores: _0100_

def extract_scan_id(p: Path) -> Optional[int]:
    """
    Retourne un int (0100, 000100 -> 100) à partir du filename.
    On prend le PREMIER groupe '_<digits>_' trouvé.
    """
    m = SCAN_RE.search(p.name)
    if not m:
        return None
    return int(m.group(1))

def list_txt(dir_path: Path) -> list[Path]:
    it = dir_path.glob("*.txt")
    return sorted([p for p in it if p.is_file()])


def index_by_scan(files: list[Path]) -> tuple[dict[str, Path], list[tuple[str, str]]]:
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
        M = M[np.argsort(M[:, 0])]  # gps_time in col 0

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, M, delimiter=delimiter, fmt=float_fmt)
    print(f"{a.name} + {b.name} -> {out.name} ({len(A)} + {len(B)} rows)")


def merge_txt_pairs(
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

    files_a = list_txt(dir_a)
    files_b = list_txt(dir_b)

    idx_a, skipped_a = index_by_scan(files_a)
    idx_b, skipped_b = index_by_scan(files_b)

    scans = sorted(set(idx_a) & set(idx_b))
    if not scans:
        print("No matching scans found. Check filenames / SCAN_RE.")
        if skipped_a: print("Skipped A:", skipped_a[:10])
        if skipped_b: print("Skipped B:", skipped_b[:10])
        return

    ok = 0
    for scan in scans:
        a = idx_a[scan]
        b = idx_b[scan]
        out = out_dir / f"{out_prefix}{scan}{out_suffix}.txt"
        print(f"\n [Merging] files {a} and {b}")
        merge_two_txt(
            a, b, out,
            delimiter=delimiter,
            skiprows=skiprows,
            sort_by_time=sort_by_time,
        )
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


if __name__ == "__main__":
    dir_HA = Path("/media/b085164/Elements/PCD_SAM/test/HA")
    dir_LR = Path("/media/b085164/Elements/PCD_SAM/test/LR")
    out_dir = Path("/media/b085164/Elements/PCD_SAM/test/merged")

    merge_txt_pairs(
        dir_HA,
        dir_LR,
        out_dir,
        delimiter=",",
        out_prefix="merged_",
        out_suffix="_HA_LR",
    )