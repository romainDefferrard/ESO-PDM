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
        self._acc_bytes = 0

    def close(self):
        try:
            if self.pbar is not None and self._acc_bytes > 0:
                self.pbar.update(self._acc_bytes)
                self._acc_bytes = 0
            self.f.close()
        except Exception:
            pass

    def _readline(self):
        line = self.f.readline()
        if line and self.pbar is not None:
            self._acc_bytes += len(line)
            if self._acc_bytes >= 1_000_000:
                self.pbar.update(self._acc_bytes)
                self._acc_bytes = 0
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

    extra = laspy.ExtraBytesParams(name="e3d", type=np.float32)
    if hasattr(hdr, "add_extra_dim"):
        hdr.add_extra_dim(extra)
    else:
        hdr.add_extra_dimension(extra)

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


def times_match(t_ref: float, t_tgt: float, exact: bool, atol: float) -> bool:
    if exact:
        return t_ref == t_tgt
    return abs(t_ref - t_tgt) <= atol


def init_accumulators():
    return {
        "n": 0,
        "sum_e": 0.0,
        "sum_sq": 0.0,
        "max_e": 0.0,
        "min_e": float("inf"),
        "gt_01": 0,
        "gt_05": 0,
        "gt_10": 0,
        "gt_20": 0,
        "gt_50": 0,
        "sum_dx": 0.0,
        "sum_dy": 0.0,
        "sum_dz": 0.0,
        "sum_dx2": 0.0,
        "sum_dy2": 0.0,
        "sum_dz2": 0.0,
        "sum_abs_dx": 0.0,
        "sum_abs_dy": 0.0,
        "sum_abs_dz": 0.0,
        "sum_planim": 0.0,
        "sum_planim_sq": 0.0,
        "max_planim": 0.0,
        "sum_dz_abs": 0.0,
        "first_match_time": None,
        "last_match_time": None,
    }


def update_accumulators(accum: Dict[str, float], dxyz: np.ndarray, e3d: float, t: float):
    dx, dy, dz = map(float, dxyz)
    e2d = float(np.linalg.norm(dxyz[:2]))

    accum["n"] += 1
    accum["sum_e"] += e3d
    accum["sum_sq"] += e3d * e3d
    accum["max_e"] = max(accum["max_e"], e3d)
    accum["min_e"] = min(accum["min_e"], e3d)

    accum["gt_01"] += int(e3d > 0.1)
    accum["gt_05"] += int(e3d > 0.5)
    accum["gt_10"] += int(e3d > 1.0)
    accum["gt_20"] += int(e3d > 2.0)
    accum["gt_50"] += int(e3d > 5.0)

    accum["sum_dx"] += dx
    accum["sum_dy"] += dy
    accum["sum_dz"] += dz

    accum["sum_dx2"] += dx * dx
    accum["sum_dy2"] += dy * dy
    accum["sum_dz2"] += dz * dz

    accum["sum_abs_dx"] += abs(dx)
    accum["sum_abs_dy"] += abs(dy)
    accum["sum_abs_dz"] += abs(dz)

    accum["sum_planim"] += e2d
    accum["sum_planim_sq"] += e2d * e2d
    accum["max_planim"] = max(accum["max_planim"], e2d)

    accum["sum_dz_abs"] += abs(dz)

    if accum["first_match_time"] is None:
        accum["first_match_time"] = t
    accum["last_match_time"] = t


def finalize_metrics(accum: Dict[str, float], e3d_all: np.ndarray) -> Dict[str, float]:
    n = accum["n"]
    if n == 0:
        raise RuntimeError("No matched points available to compute metrics.")

    mean_dx = accum["sum_dx"] / n
    mean_dy = accum["sum_dy"] / n
    mean_dz = accum["sum_dz"] / n

    metrics = {
        "N_match": int(n),
        "RMSE_3D": float(math.sqrt(accum["sum_sq"] / n)),
        "Mean_3D": float(accum["sum_e"] / n),
        "Median_3D": float(np.median(e3d_all)),
        "P90_3D": float(np.percentile(e3d_all, 90)),
        "P95_3D": float(np.percentile(e3d_all, 95)),
        "P99_3D": float(np.percentile(e3d_all, 99)),
        "Max_3D": float(accum["max_e"]),
        "Min_3D": float(accum["min_e"]),
        "Pct_gt_0.1m": float(accum["gt_01"] / n * 100.0),
        "Pct_gt_0.5m": float(accum["gt_05"] / n * 100.0),
        "Pct_gt_1.0m": float(accum["gt_10"] / n * 100.0),
        "Pct_gt_2.0m": float(accum["gt_20"] / n * 100.0),
        "Pct_gt_5.0m": float(accum["gt_50"] / n * 100.0),
        "Bias_dx": float(mean_dx),
        "Bias_dy": float(mean_dy),
        "Bias_dz": float(mean_dz),
        "Std_dx": float(math.sqrt(max(0.0, accum["sum_dx2"] / n - mean_dx ** 2))),
        "Std_dy": float(math.sqrt(max(0.0, accum["sum_dy2"] / n - mean_dy ** 2))),
        "Std_dz": float(math.sqrt(max(0.0, accum["sum_dz2"] / n - mean_dz ** 2))),
        "Mean_abs_dx": float(accum["sum_abs_dx"] / n),
        "Mean_abs_dy": float(accum["sum_abs_dy"] / n),
        "Mean_abs_dz": float(accum["sum_abs_dz"] / n),
        "Mean_2D": float(accum["sum_planim"] / n),
        "RMSE_2D": float(math.sqrt(accum["sum_planim_sq"] / n)),
        "Max_2D": float(accum["max_planim"]),
        "Mean_abs_dz_only": float(accum["sum_dz_abs"] / n),
        "First_match_time": float(accum["first_match_time"]),
        "Last_match_time": float(accum["last_match_time"]),
    }
    return metrics


def write_metrics_txt(out_txt: Path, target_path: Path, metrics: Dict[str, float], counts: Dict[str, int], out_las: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"target_file: {target_path}")
    lines.append(f"output_las: {out_las}")
    lines.append("")

    count_order = [
        "N_ref_raw",
        "N_tgt_raw",
        "N_match",
        "ref_no_match",
        "ref_dup_skip",
        "tgt_dup_skip",
        "tgt_extra_skip",
    ]

    for k in count_order:
        if k in counts:
            lines.append(f"{k}: {counts[k]}")

    if counts.get("N_ref_raw", 0) > 0:
        lines.append(f"ref_match_rate_percent: {100.0 * counts['N_match'] / counts['N_ref_raw']:.6f}")
    if counts.get("N_tgt_raw", 0) > 0:
        lines.append(f"tgt_used_rate_percent: {100.0 * counts['N_match'] / counts['N_tgt_raw']:.6f}")

    lines.append("")

    metric_order = [
        "RMSE_3D",
        "Mean_3D",
        "Median_3D",
        "P90_3D",
        "P95_3D",
        "P99_3D",
        "Max_3D",
        "Min_3D",
        "Pct_gt_0.1m",
        "Pct_gt_0.5m",
        "Pct_gt_1.0m",
        "Pct_gt_2.0m",
        "Pct_gt_5.0m",
        "Bias_dx",
        "Bias_dy",
        "Bias_dz",
        "Std_dx",
        "Std_dy",
        "Std_dz",
        "Mean_abs_dx",
        "Mean_abs_dy",
        "Mean_abs_dz",
        "Mean_2D",
        "RMSE_2D",
        "Max_2D",
        "Mean_abs_dz_only",
        "First_match_time",
        "Last_match_time",
    ]

    for k in metric_order:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"{k}: {v:.10f}")
            else:
                lines.append(f"{k}: {v}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def format_summary(rows: List[Dict[str, object]]) -> str:
    cols = [
        ("target_name", 34),
        ("N_ref_raw", 12),
        ("N_tgt_raw", 12),
        ("N_match", 12),
        ("ref_no_match", 14),
        ("tgt_extra_skip", 16),
        ("ref_match_%", 12),
        ("RMSE_3D", 10),
        ("P95_3D", 10),
        ("P99_3D", 10),
        ("Max_3D", 10),
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

def flush_las_buffer(writer, buf_t, buf_xyz, buf_e3d):
    if len(buf_t) == 0:
        return

    t = np.asarray(buf_t, dtype=np.float64)
    xyz = np.asarray(buf_xyz, dtype=np.float64)
    e3d = np.asarray(buf_e3d, dtype=np.float32)

    las = laspy.ScaleAwarePointRecord.zeros(len(t), header=writer.header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.gps_time = t
    las["e3d"] = e3d
    writer.write_points(las)

    buf_t.clear()
    buf_xyz.clear()
    buf_e3d.clear()



def process_pair_streaming(
    ref_path: Path,
    tgt_path: Path,
    out_las: Path,
    out_metrics_txt: Path,
    delim: str,
    tmin: float,
    tmax: float,
    scale: float,
    exact_time: bool,
    time_atol: float,
) -> Dict[str, object]:
    total_size = os.path.getsize(ref_path) + os.path.getsize(tgt_path)

    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Streaming pair: {ref_path.name} + {tgt_path.name}"
    ) as pbar:
        ref_reader = GroupedTxtReader(ref_path, delim, tmin, tmax, pbar=pbar)
        tgt_reader = GroupedTxtReader(tgt_path, delim, tmin, tmax, pbar=pbar)

        writer = None
        buf_t = []
        buf_xyz = []
        buf_e3d = []
        flush_every = 100_000

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tmp_path = Path(tmp.name)
        tmp.close()

        accum = init_accumulators()

        counts = {
            "N_ref_raw": 0,
            "N_tgt_raw": 0,
            "N_match": 0,
            "ref_dup_skip": 0,
            "tgt_dup_skip": 0,
            "ref_no_match": 0,
            "tgt_extra_skip": 0,
        }

        gR = ref_reader.next_group()
        gT = tgt_reader.next_group()

        with open(tmp_path, "ab") as ftmp:
            while gR is not None:
                tR, xyzR = gR
                counts["N_ref_raw"] += len(xyzR)

                if len(xyzR) > 1:
                    counts["ref_dup_skip"] += len(xyzR)
                    gR = ref_reader.next_group()
                    continue

                while gT is not None:
                    tT, xyzT = gT
                    counts["N_tgt_raw"] += len(xyzT)

                    if len(xyzT) > 1:
                        counts["tgt_dup_skip"] += len(xyzT)
                        gT = tgt_reader.next_group()
                        continue

                    if exact_time:
                        if tT < tR:
                            counts["tgt_extra_skip"] += len(xyzT)
                            gT = tgt_reader.next_group()
                            continue
                    else:
                        if tT < tR - time_atol:
                            counts["tgt_extra_skip"] += len(xyzT)
                            gT = tgt_reader.next_group()
                            continue

                    break

                if gT is None:
                    counts["ref_no_match"] += 1
                    gR = ref_reader.next_group()
                    continue

                tT, xyzT = gT

                if times_match(tR, tT, exact_time, time_atol):
                    dxyz = xyzT[0] - xyzR[0]
                    e3d = float(np.linalg.norm(dxyz))

                    if writer is None:
                        offset_xyz = xyzR[0].copy()
                        writer = create_las_writer(out_las, scale, offset_xyz)

                    counts["N_match"] += 1
                    update_accumulators(accum, dxyz, e3d, tR)

                    buf_t.append(tR)
                    buf_xyz.append(xyzR[0].copy())
                    buf_e3d.append(e3d)

                    if len(buf_t) >= flush_every:
                        np.asarray(buf_e3d, dtype=np.float32).tofile(ftmp)
                        if writer is not None:
                            flush_las_buffer(writer, buf_t, buf_xyz, buf_e3d)

                    gR = ref_reader.next_group()
                    gT = tgt_reader.next_group()

                else:
                    counts["ref_no_match"] += 1
                    gR = ref_reader.next_group()

            while gT is not None:
                tT, xyzT = gT
                counts["N_tgt_raw"] += len(xyzT)
                if len(xyzT) > 1:
                    counts["tgt_dup_skip"] += len(xyzT)
                else:
                    counts["tgt_extra_skip"] += len(xyzT)
                gT = tgt_reader.next_group()

            # final flush
            if len(buf_t) > 0:
                np.asarray(buf_e3d, dtype=np.float32).tofile(ftmp)
                if writer is not None:
                    flush_las_buffer(writer, buf_t, buf_xyz, buf_e3d)

        if writer is not None:
            writer.close()

        ref_reader.close()
        tgt_reader.close()

    if accum["n"] == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No matched reference points found between:\n  {ref_path}\n  {tgt_path}"
        )

    e3d_all = np.fromfile(tmp_path, dtype=np.float32)
    tmp_path.unlink(missing_ok=True)

    metrics = finalize_metrics(accum, e3d_all)

    result = {
        "target_name": tgt_path.name,
        "target_path": str(tgt_path),
        **counts,
        **metrics,
    }
    result["ref_match_%"] = 100.0 * result["N_match"] / result["N_ref_raw"] if result["N_ref_raw"] > 0 else 0.0

    write_metrics_txt(out_metrics_txt, tgt_path, metrics, counts, out_las)

    logging.info("Point statistics for current pair:")
    logging.info("   Reference raw points : %d", counts["N_ref_raw"])
    logging.info("   Target raw points    : %d", counts["N_tgt_raw"])
    logging.info("   Matched pairs        : %d", counts["N_match"])
    logging.info("   Ref without match    : %d", counts["ref_no_match"])
    logging.info("   Ref duplicates skip  : %d", counts["ref_dup_skip"])
    logging.info("   Tgt duplicates skip  : %d", counts["tgt_dup_skip"])
    logging.info("   Tgt extra skipped    : %d", counts["tgt_extra_skip"])
    logging.info("   Ref match rate       : %.4f%%", result["ref_match_%"])
    if counts["N_tgt_raw"] > 0:
        logging.info("   Target used rate     : %.4f%%", 100.0 * counts["N_match"] / counts["N_tgt_raw"])

    return result


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

    exact_time = bool(cfg.get("matching", {}).get("exact_time", True))
    time_atol = float(cfg.get("matching", {}).get("time_atol", 0.0))

    logging.info("Reference: %s", ref_txt)
    logging.info(
        "Window: [%.3f , %.3f] (start=%.3f dur=%.1f buffer=%.1f)",
        tmin, tmax, t0, dur, buf
    )
    logging.info("Matching mode: exact_time=%s | time_atol=%.9f", exact_time, time_atol)

    summary_rows = []

    for i, tgt in enumerate(targets, start=1):
        tgt_path = Path(tgt["path"])
        out_base = Path(tgt["outfile"])

        out_las = out_base.with_suffix(".las")
        out_metrics = out_base.parent / f"summary_{out_las.stem}.txt"

        logging.info("")
        logging.info("Target %d/%d: %s", i, len(targets), tgt_path)
        logging.info("Output LAS: %s", out_las)
        logging.info("Output summary TXT: %s", out_metrics)

        metrics = process_pair_streaming(
            ref_path=ref_txt,
            tgt_path=tgt_path,
            out_las=out_las,
            out_metrics_txt=out_metrics,
            delim=delim,
            tmin=tmin,
            tmax=tmax,
            scale=scale,
            exact_time=exact_time,
            time_atol=time_atol,
        )

        logging.info(
            "RMSE_3D=%.4f m | P95=%.4f m | P99=%.4f m | Max=%.4f m",
            metrics["RMSE_3D"], metrics["P95_3D"], metrics["P99_3D"], metrics["Max_3D"]
        )

        summary_rows.append(metrics)

    summary_txt = Path(args.config).with_suffix(".summary.txt")
    summary_txt.write_text(format_summary(summary_rows), encoding="utf-8")

    logging.info("")
    logging.info("========== RMSE SUMMARY ==========")
    print(format_summary(summary_rows))
    logging.info("==================================")
    logging.info("Wrote global summary TXT: %s", summary_txt)


if __name__ == "__main__":
    main()