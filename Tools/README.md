# Tools — Standalone Utility Scripts

Tools that do not belong to the main pipeline but are used for analysis, calibration, and future research directions.

---

## `gsd_analysis.py` — Ground Sampling Distance Analysis

Measures nearest-neighbour distance on MLS point clouds, split by scanner source (VUX / PUCK) and range bin (0–10 m, 10–20 m, 20–30 m). Results saved to CSV.

```bash
python Tools/gsd_analysis.py --dir /path/to/merged/ALL --every 1000 --out gsd_results.csv
```

Requires `lasvec_x/y/z` and `scanner_src` extra dims in the LAS files.

---

## `align_APX_AIRINS.ipynb` — APX15 / AIRINS Rotation Alignment

Aligns the body frame of the APX15 IMU with the AIRINS IMU using rotation matrices.

The alignment uses the **Fréchet mean** on the SO(3) group, which minimises the sum of squared geodesic distances between a set of rotation matrices.

**Method:**
1. Collect a set of relative rotation matrices R_AIRINS_to_APX
2. Compute the Fréchet mean of these rotations on SO(3)
3. Use the result as the body-to-body rotation R_bb for trajectory alignment

**Input:** SBET trajectories from both INS systems.

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

Notebook for matching ALS and MLS point clouds. This is a **future research direction** exploring cross-platform point-to-point constraints.
