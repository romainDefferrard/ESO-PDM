"""
crop_ref_odyn.py
----------------
Pre-crops ODyN ref + all method targets (outage_traj, F2B, COMBINED)
to the time window of each outage.

Run once — produces small cropped files that rmse_streaming.py reads instantly.

Usage:
    python crop_ref_odyn.py
"""

from pathlib import Path
import numpy as np
import laspy
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR = Path("/media/b085164/Elements/CALIB_26_02_25")
OUT_ROOT = BASE_DIR / "georef_errors"

CHUNK_SIZE = 2_000_000

# One entry per outage
OUTAGES = [
    {
        "name":     "outage_1",
        "t_start":  305120.0,
        "duration": 580.0,
        "buffer_s": 10.0,
        "sources": [
            {
                "name":     "ref_ODyN",
                "dir":      BASE_DIR / "georef_ALL_traj_ODyN/merged",
                "manifest": BASE_DIR / "georef_ALL_traj_ODyN/merged/merged_manifest.csv",
            },
        ],
    },
    # {
    #     "name":     "outage_2",
    #     "t_start":  305645.0,
    #     "duration": 475.0,
    #     "buffer_s": 10.0,
    #     "sources": [
    #         {
    #             "name":     "ref_ODyN",
    #             "dir":      BASE_DIR / "georef_ALL_traj_ODyN/merged",
    #             "manifest": BASE_DIR / "georef_ALL_traj_ODyN/merged/merged_manifest.csv",
    #         },
    #     ],
    # },
    # {
    #     "name":     "outage_3",
    #     "t_start":  306290.0,
    #     "duration": 355.0,
    #     "buffer_s": 10.0,
    #     "sources": [
    #         {
    #             "name":     "ref_ODyN",
    #             "dir":      BASE_DIR / "georef_ALL_traj_ODyN/merged",
    #             "manifest": BASE_DIR / "georef_ALL_traj_ODyN/merged/merged_manifest.csv",
    #         },
    #     ],
    # },
]

# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "scan_id":  int(row["scan_id"]),
                "filename": row["filename"],
                "t_start":  float(row["t_start"]),
                "t_end":    float(row["t_end"]),
            })
    return rows


def crop_las_to_window(src_path: Path, dst_path: Path, tmin: float, tmax: float) -> int:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with laspy.open(src_path) as reader:
        header = reader.header
        with laspy.open(dst_path, mode="w", header=header) as writer:
            for points in reader.chunk_iterator(CHUNK_SIZE):
                t = np.asarray(points.gps_time, dtype=np.float64)
                if len(t) > 0 and t[0] > tmax:
                    break
                mask = (t >= tmin) & (t <= tmax)
                if not np.any(mask):
                    continue
                writer.write_points(points[mask])
                kept += int(mask.sum())
    return kept


def get_time_bounds(path: Path) -> tuple:
    t_min, t_max = np.inf, -np.inf
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            t = np.asarray(chunk.gps_time, dtype=np.float64)
            t_min = min(t_min, float(t.min()))
            t_max = max(t_max, float(t.max()))
    return t_min, t_max


def write_manifest(path: Path, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scan_id", "filename", "t_start", "t_end"])
        writer.writeheader()
        writer.writerows(rows)


def crop_source(outage_name: str, source: dict, tmin: float, tmax: float):
    src_name = source["name"]
    src_dir  = source["dir"]
    manifest_path = source["manifest"]

    out_dir = OUT_ROOT / outage_name / "cropped" / src_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("  [%s] reading manifest: %s", src_name, manifest_path)

    if not manifest_path.exists():
        logging.warning("  [%s] manifest not found — skipping", src_name)
        return

    manifest = read_manifest(manifest_path)
    manifest_rows = []

    for scan in manifest:
        if scan["t_end"] < tmin or scan["t_start"] > tmax:
            continue

        src  = src_dir / scan["filename"]
        dst  = out_dir / scan["filename"]

        if not src.exists():
            logging.warning("  [%s] scan %d not found: %s", src_name, scan["scan_id"], src)
            continue

        logging.info("  [%s] scan %5d  [%.3f, %.3f]",
                     src_name, scan["scan_id"], scan["t_start"], scan["t_end"])

        kept = crop_las_to_window(src, dst, tmin, tmax)
        logging.info("    -> %d pts  saved: %s", kept, dst.name)

        if kept == 0:
            dst.unlink(missing_ok=True)
            continue

        t0, t1 = get_time_bounds(dst)
        manifest_rows.append({
            "scan_id":  scan["scan_id"],
            "filename": scan["filename"],
            "t_start":  round(t0, 6),
            "t_end":    round(t1, 6),
        })

    if manifest_rows:
        mpath = out_dir / "merged_manifest.csv"
        write_manifest(mpath, manifest_rows)
        logging.info("  [%s] manifest written: %s (%d scans)", src_name, mpath, len(manifest_rows))
    else:
        logging.warning("  [%s] no scans kept!", src_name)


def main():
    for outage in OUTAGES:
        name    = outage["name"]
        tmin    = outage["t_start"] - outage["buffer_s"]
        tmax    = outage["t_start"] + outage["duration"] + outage["buffer_s"]

        logging.info("")
        logging.info("=" * 60)
        logging.info("Outage: %s  window [%.3f, %.3f]", name, tmin, tmax)
        logging.info("=" * 60)

        for source in outage["sources"]:
            crop_source(name, source, tmin, tmax)

    logging.info("")
    logging.info("Done. Update cfg_outage_X.yml to use cropped dirs:")
    for outage in OUTAGES:
        name = outage["name"]
        logging.info("  %s -> ref_dir: %s/%s/cropped/ref_ODyN", name, OUT_ROOT, name)
        logging.info("         methods dir: %s/%s/cropped/<method>", OUT_ROOT, name)


if __name__ == "__main__":
    main()