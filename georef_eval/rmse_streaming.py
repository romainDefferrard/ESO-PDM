import argparse
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
import yaml
from tqdm import tqdm


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


class GroupedTxtReader:
    """
    Read a sorted txt cloud and yield groups of identical gps_time.

    Expected line format:
        gps_time,x,y,z,...
    """

    def __init__(self, path: Path, delim: str, tmin: float, tmax: float, pbar=None):
        self.path = Path(path)
        self.delim = delim
        self.tmin = tmin
        self.tmax = tmax
        self.pbar = pbar

        self.f = self.path.open("r", encoding="utf-8", errors="ignore")
        self._pending = None
        self._done = False

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def _readline(self):
        line = self.f.readline()
        if line and self.pbar is not None:
            self.pbar.update(len(line))
        return line

    def _parse_line(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        arr = np.fromstring(line, sep=self.delim)
        if arr.size < 4:
            return None

        t = float(arr[0])
        xyz = arr[1:4].astype(np.float64)
        return t, xyz

    def _next_record(self):
        while True:
            line = self._readline()
            if not line:
                self._done = True
                return None

            rec = self._parse_line(line)
            if rec is None:
                continue

            t, xyz = rec

            if t < self.tmin:
                continue

            if t > self.tmax:
                self._done = True
                return None

            return t, xyz

    def next_group(self):
        if self._done:
            return None

        if self._pending is None:
            rec = self._next_record()
            if rec is None:
                return None
            t0, xyz0 = rec
        else:
            t0, xyz0 = self._pending
            self._pending = None

        pts = [xyz0]

        while True:
            rec = self._next_record()
            if rec is None:
                break

            t, xyz = rec
            if t == t0:
                pts.append(xyz)
            else:
                self._pending = (t, xyz)
                break

        return t0, np.vstack(pts)


def create_las_writer(out_las: Path, scale: float, offset_xyz: np.ndarray):
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([scale, scale, scale], dtype=np.float64)
    hdr.offsets = np.asarray(offset_xyz, dtype=np.float64)

    hdr.add_extra_dim(laspy.ExtraBytesParams(name="e3d", type=np.float32))

    out_las.parent.mkdir(parents=True, exist_ok=True)
    return laspy.open(out_las, mode="w", header=hdr)


def write_las_chunk(writer, t: np.ndarray, xyz: np.ndarray, e3d: np.ndarray):
    las = laspy.ScaleAwarePointRecord.zeros(len(t), header=writer.header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.gps_time = t
    las["e3d"] = e3d.astype(np.float32)
    writer.write_points(las)


def compute_summary_from_temp_e3d(temp_path: Path, accum: Dict[str, float]) -> Dict[str, float]:
    e3d = np.fromfile(temp_path, dtype=np.float32)
    if e3d.size == 0:
        raise RuntimeError("No matched points written to temporary e3d file.")

    summary = {
        "N": float(accum["n"]),
        "RMSE": float(math.sqrt(accum["sum_sq"] / accum["n"])),
        "mean": float(accum["sum_e"] / accum["n"]),
        "median": float(np.median(e3d)),
        "p95": float(np.percentile(e3d, 95)),
        "p99": float(np.percentile(e3d, 99)),
        "max": float(accum["max_e"]),
        "%>0.1m": float(accum["gt_01"] / accum["n"] * 100.0),
        "%>0.5m": float(accum["gt_05"] / accum["n"] * 100.0),
        "%>1.0m": float(accum["gt_10"] / accum["n"] * 100.0),
        "dx_mean": float(accum["sum_dx"] / accum["n"]),
        "dy_mean": float(accum["sum_dy"] / accum["n"]),
        "dz_mean": float(accum["sum_dz"] / accum["n"]),
        "dx_std": float(math.sqrt(accum["sum_dx2"] / accum["n"] - (accum["sum_dx"] / accum["n"]) ** 2)),
        "dy_std": float(math.sqrt(accum["sum_dy2"] / accum["n"] - (accum["sum_dy"] / accum["n"]) ** 2)),
        "dz_std": float(math.sqrt(accum["sum_dz2"] / accum["n"] - (accum["sum_dz"] / accum["n"]) ** 2)),
    }
    return summary


def format_summary(rows: List[Dict[str, object]]) -> str:
    cols = [
        ("target", 42),
        ("N_ref_raw", 12),
        ("N_tgt_raw", 12),
        ("N_match", 12),
        ("ref_dup_skip", 14),
        ("tgt_dup_skip", 14),
        ("time_only_ref", 14),
        ("time_only_tgt", 14),
        ("RMSE", 10),
        ("p95", 10),
        ("p99", 10),
        ("max", 10),
        ("%>0.5m", 10),
        ("%>1.0m", 10),
    ]

    def fnum(x, w):
        if x is None:
            return " " * w
        if isinstance(x, (int, float)):
            if abs(x - int(x)) < 1e-9 and w >= 6:
                s = f"{int(x)}"
            else:
                s = f"{x:.4f}"
        else:
            s = str(x)
        return s[:w].ljust(w)

    lines = []
    header = " ".join([c[0].ljust(c[1]) for c in cols])
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(" ".join([fnum(r.get(name), w) for name, w in cols]))
    return "\n".join(lines)

def process_pair_streaming(
    ref_path: Path,
    tgt_path: Path,
    out_las: Path,
    delim: str,
    tmin: float,
    tmax: float,
    scale: float,
) -> Dict[str, float]:
    total_size = os.path.getsize(ref_path) + os.path.getsize(tgt_path)

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Streaming {tgt_path.name}") as pbar:
        ref_reader = GroupedTxtReader(ref_path, delim, tmin, tmax, pbar=pbar)
        tgt_reader = GroupedTxtReader(tgt_path, delim, tmin, tmax, pbar=pbar)

        writer = None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tmp_path = Path(tmp.name)
        tmp.close()

        accum = {
            "n": 0,
            "sum_e": 0.0,
            "sum_sq": 0.0,
            "max_e": 0.0,
            "gt_01": 0,
            "gt_05": 0,
            "gt_10": 0,
            "sum_dx": 0.0,
            "sum_dy": 0.0,
            "sum_dz": 0.0,
            "sum_dx2": 0.0,
            "sum_dy2": 0.0,
            "sum_dz2": 0.0,
        }

        counts = {
            "N_ref_raw": 0,
            "N_tgt_raw": 0,
            "ref_dup_skip": 0,
            "tgt_dup_skip": 0,
            "time_only_ref": 0,
            "time_only_tgt": 0,
            "N_match": 0,
        }

        gR = ref_reader.next_group()
        gT = tgt_reader.next_group()

        with open(tmp_path, "ab") as ftmp:
            while gR is not None and gT is not None:
                tR, xyzR = gR
                tT, xyzT = gT

                counts["N_ref_raw"] += len(xyzR)
                counts["N_tgt_raw"] += len(xyzT)

                # Skip duplicated timestamps in ref
                if len(xyzR) > 1:
                    counts["ref_dup_skip"] += len(xyzR)
                    gR = ref_reader.next_group()
                    continue

                # Skip duplicated timestamps in tgt
                if len(xyzT) > 1:
                    counts["tgt_dup_skip"] += len(xyzT)
                    gT = tgt_reader.next_group()
                    continue

                if tR == tT:
                    dxyz = xyzT[0] - xyzR[0]
                    e3d = np.linalg.norm(dxyz)

                    # Initialize LAS writer at first matched point
                    if writer is None:
                        offset_xyz = xyzR[0].copy()
                        writer = create_las_writer(out_las, scale, offset_xyz)

                    counts["N_match"] += 1

                    accum["n"] += 1
                    accum["sum_e"] += e3d
                    accum["sum_sq"] += e3d * e3d
                    accum["max_e"] = max(accum["max_e"], e3d)
                    accum["gt_01"] += int(e3d > 0.1)
                    accum["gt_05"] += int(e3d > 0.5)
                    accum["gt_10"] += int(e3d > 1.0)

                    accum["sum_dx"] += dxyz[0]
                    accum["sum_dy"] += dxyz[1]
                    accum["sum_dz"] += dxyz[2]
                    accum["sum_dx2"] += dxyz[0] ** 2
                    accum["sum_dy2"] += dxyz[1] ** 2
                    accum["sum_dz2"] += dxyz[2] ** 2

                    np.asarray([e3d], dtype=np.float32).tofile(ftmp)

                    write_las_chunk(
                        writer,
                        np.asarray([tR], dtype=np.float64),
                        xyzR.reshape(1, 3),
                        np.asarray([e3d], dtype=np.float32),
                    )

                    gR = ref_reader.next_group()
                    gT = tgt_reader.next_group()

                elif tR < tT:
                    counts["time_only_ref"] += 1
                    gR = ref_reader.next_group()

                else:
                    counts["time_only_tgt"] += 1
                    gT = tgt_reader.next_group()

        if writer is not None:
            writer.close()

        ref_reader.close()
        tgt_reader.close()

    if accum["n"] == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No matched unique timestamps found between:\n  {ref_path}\n  {tgt_path}"
        )

    metrics = compute_summary_from_temp_e3d(tmp_path, accum)
    tmp_path.unlink(missing_ok=True)

    metrics.update(counts)
    return metrics

def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ref_txt = Path(cfg["reference"]["path"])
    targets = cfg["targets"]

    t0 = float(cfg["time"]["outage_start"])
    dur = float(cfg["time"]["outage_duration"])
    buf = float(cfg["time"]["buffer"])
    tmin = t0 - buf
    tmax = t0 + dur + buf

    delim = cfg.get("io", {}).get("delim", ",")
    scale = float(cfg.get("las", {}).get("scale", 0.001))

    logging.info("Reference: %s", ref_txt)
    logging.info("Window: [%.3f , %.3f] (start=%.3f dur=%.1f buffer=%.1f)", tmin, tmax, t0, dur, buf)

    summary_rows = []

    for i, tgt in enumerate(targets, start=1):
        tgt_path = Path(tgt["path"])
        out_las = Path(tgt["outfile"]).with_suffix(".las")

        logging.info("")
        logging.info("Target %d/%d: %s", i, len(targets), tgt_path)
        logging.info("Output LAS: %s", out_las)

        metrics = process_pair_streaming(
            ref_path=ref_txt,
            tgt_path=tgt_path,
            out_las=out_las,
            delim=delim,
            tmin=tmin,
            tmax=tmax,
            scale=scale,
        )

        logging.info(
            "Matched=%d | ref_dup_skip=%d | tgt_dup_skip=%d | ref_only=%d | tgt_only=%d",
            int(metrics["N_match"]),
            int(metrics["ref_dup_skip"]),
            int(metrics["tgt_dup_skip"]),
            int(metrics["time_only_ref"]),
            int(metrics["time_only_tgt"]),
        )

        logging.info(
            "RMSE_3D=%.4f m | p95=%.4f m | p99=%.4f m | max=%.4f m",
            metrics["RMSE"], metrics["p95"], metrics["p99"], metrics["max"]
        )

        summary_rows.append({
            "target": str(tgt_path),
            **metrics,
        })

    logging.info("")
    logging.info("========== RMSE SUMMARY ==========")
    print(format_summary(summary_rows))
    logging.info("==================================")

    out_csv = Path(args.config).with_suffix(".summary.csv")
    with out_csv.open("w", encoding="utf-8") as f:
        keys = list(summary_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    logging.info("Wrote summary CSV: %s", out_csv)


if __name__ == "__main__":
    main()