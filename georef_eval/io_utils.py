# io_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

def read_txt_cloud(path: Path, delimiter: str = ",", comment: str = "#"):
    with path.open("r", encoding="utf-8") as f:
        header = None
        rows = []
        for line in f:
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            if header is None:
                header = line
                continue
            rows.append(line)

    if header is None:
        raise ValueError(f"No header found in {path}. First non-comment line must be column names.")
    col_names = [c.strip() for c in header.split(delimiter)]
    col = {n: i for i, n in enumerate(col_names)}

    if not rows:
        raise ValueError(f"No data in {path}")

    data = np.array([[float(v) for v in r.split(delimiter)] for r in rows], dtype=np.float64)
    if data.shape[1] != len(col_names):
        raise ValueError(f"{path}: header has {len(col_names)} cols, data has {data.shape[1]}.")
    return col_names, col, data


def write_txt_cloud(path: Path, col_names, data: np.ndarray, delimiter: str = ",", comment_header: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if comment_header:
            for ln in comment_header.splitlines():
                f.write(f"# {ln}\n")
        f.write(delimiter.join(col_names) + "\n")
        for row in data:
            f.write(delimiter.join(f"{v:.10f}" for v in row) + "\n")


def require_cols(col_index: Dict[str, int], required: List[str], path_hint: str = ""):
    missing = [c for c in required if c not in col_index]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path_hint}. Present: {list(col_index.keys())}")