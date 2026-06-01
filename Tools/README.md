# Tools — Standalone Utility Scripts

Miscellaneous tools that do not belong to the main pipeline but are used for analysis, calibration, and future research directions.

---

## `gsd_analysis.py` — Ground Sampling Distance Analysis

Measures the Ground Sampling Distance (GSD) of MLS point clouds as the nearest-neighbour distance between points. Analyses the full cloud and subsets by scanner source (VUX vs PUCK) and by range bin.

```bash
# Single file
python Tools/gsd_analysis.py --las /path/to/merged_1000_VUX_PUCK.las --every 500

# All merged_*.las in a folder
python Tools/gsd_analysis.py --dir /path/to/merged/ALL --every 1000 --out gsd_results.csv
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--las` | — | Path to a single LAS file |
| `--dir` | — | Folder containing `merged_*.las` files |
| `--glob` | `merged_*.las` | Glob pattern for `--dir` mode |
| `--every` | `1000` | Subsample: keep 1 point every N for NN query |
| `--out` | `gsd_results.csv` | Output CSV path |

**What it measures:**
- Builds a KDTree on the point cloud
- Queries the 2nd nearest neighbour for each sampled point
- Reports RMSE, Q50, Q90, STD of NN distances

**Subsets analysed:**
- `ALL` — all points combined
- `VUX` — points with `scanner_src == 2`
- `PUCK` — points with `scanner_src == 1`

**Range bins** (hardcoded, edit in `main()` if needed):
- 0–10 m, 10–20 m, 20–30 m from scanner

Results are saved incrementally to CSV after each file (RAM-safe for large datasets). A weighted-average aggregation is printed at the end.

**Required LAS extra dims:** `lasvec_x`, `lasvec_y`, `lasvec_z` (used to compute range), `scanner_src`.

---

## `align_APX_AIRINS.ipynb` — APX15 / AIRINS Rotation Alignment

Aligns the body frame of the APX15 IMU with the AIRINS IMU using rotation matrices.

The alignment uses the **Fréchet mean** on the Lie group SO(3), which minimises the sum of squared geodesic distances between a set of rotation matrices. This is the proper Riemannian generalisation of the Euclidean mean for rotations.

**Method:**
1. Collect a set of relative rotation matrices R_AIRINS_to_APX from overlapping segments
2. Compute the Fréchet mean of these rotations on SO(3)
3. Use the result as the body-to-body rotation R_bb for trajectory alignment

**Input:** SBET trajectories from both INS systems, GPS time window for alignment.

---

## `leverarm_puck.ipynb` — PUCK Lever-Arm Estimation

Estimates the lever arm of the Velodyne PUCK scanner relative to the vehicle body frame using ICP fine-registration residuals from CloudCompare.

**Method:**
1. Export registration residuals from CloudCompare ICP between PUCK-georeferenced cloud and a reference cloud
2. Analyse residual distributions as a function of acquisition direction
3. Fit the lever-arm offset that best explains the systematic residual pattern

**Input:** CloudCompare ICP residual files, scanner mount calibration.

---

## `ALS_MLS_limatch.ipynb` — ALS–MLS Matching (Future Research)

Prototype notebook for matching ALS (Airborne) and MLS (Mobile) point clouds using LiMatch. This is a **future research direction** exploring cross-platform point-to-point constraints.

> Note: This notebook needs to be cleaned up and translated to English before use.

**Concept:** Use temporally co-acquired ALS and MLS clouds of the same area to establish cross-sensor correspondences that could constrain both trajectory solutions simultaneously in ODyN.
