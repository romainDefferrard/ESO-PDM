# ESO-PDM — LiDAR Processing Pipeline for GNSS Outage Studies

This repository contains all tools developed during a Master's project (PDM) at the ESO laboratory (EPFL) for processing Mobile Laser Scanning (MLS) and Airborne Laser Scanning (ALS) point clouds in the context of GNSS outage experiments.

## Overview

The project investigates trajectory estimation degradation during GNSS outages and its impact on LiDAR georeferencing quality. It provides:

- A **georeferencing pipeline** for multi-scanner MLS data (VUX HA, VUX LR, Velodyne PUCK)
- A **Patcher tool** for extracting overlapping scan patches (ALS & MLS), used as input for the LiMatch point-to-point matching algorithm
- **Evaluation scripts** for computing georeferencing RMSE, L2L constraints residuals, and LCP-based metrics
- **Utility tools** for GSD analysis, sensor alignment, and lever-arm estimation

## Repository Structure

```
ESO-PDM/
├── pipeline/           # MLS georeferencing & chunking pipeline (main processing)
│   ├── pipeline.py     # Orchestrator
│   ├── __main__.py     # CLI entry point
│   ├── steps/          # Pipeline steps: georef, merge, chunk, limatch
│   ├── lib/            # Core libs: trajectory, rotations, loaders, lidar
│   ├── cfg/            # Experiment pipeline configs (.yml)
│   └── scanner_cfg/    # Per-scanner calibration configs (.yml)
│
├── Patcher/            # ALS/MLS patch extraction tool + LiMatch integration
│   ├── main.py         # CLI entry point
│   ├── utils/          # Patcher modules (footprint, GUI, extractor, …)
│   ├── config/         # Per-dataset Patcher configs (.yml)
│   └── run_*.sh        # Ready-to-run launch scripts
│
├── Evaluation/         # Evaluation metrics
│   ├── georef_eval/    # RMSE computation (generate configs, stream, analyse)
│   ├── L2L_eval/       # L2L constraint residuals (S2S combined)
│   └── LCP_eval/       # LCP-based evaluation
│
├── Tools/              # Standalone utility scripts
│   ├── gsd_analysis.py         # Ground Sampling Distance analysis
│   ├── align_APX_AIRINS.ipynb  # APX15 / AIRINS rotation alignment
│   ├── leverarm_puck.ipynb     # PUCK lever-arm estimation via ICP residuals
│   └── ALS_MLS_limatch.ipynb   # ALS–MLS future research matching
│
└── navtools_PDM/       # Original pipeline (reference, not actively maintained)
```

## Quick Start

### 1. Run the pipeline

```bash
cd ESO-PDM
python -m pipeline -c pipeline/cfg/pipeline_outage_1.yml
```

See [`pipeline/README.md`](pipeline/README.md) for full documentation.

### 2. Run Patcher

```bash
cd ESO-PDM
python Patcher/main.py --yml Patcher/config/MLS_Epalinges_config.yml
# or use the provided shell scripts:
bash Patcher/run_Epalinges.sh
```

See [`Patcher/README.md`](Patcher/README.md) for full documentation.

### 3. Evaluate georeferencing RMSE

```bash
cd Evaluation/georef_eval
python generate_rmse_configs.py --config gen_config_outage_1.yml
bash <output_dir>/INS_only/run_rmse_outage_1_INS_only.sh
# Then open analyse_rmse.ipynb
```

See [`Evaluation/README.md`](Evaluation/README.md) for full documentation.

## Dependencies

The project uses Python 3.9+. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:

- `numpy`, `scipy` — numerical processing
- `laspy` — LAS/LAZ point cloud I/O
- `pyproj` — coordinate transformations (ECEF ↔ LV95 / EPSG:2056)
- `pandas` — manifests, CSV results
- `PyQt6` — GUI (Patcher)
- `matplotlib` — plots (Patcher GUI, evaluation notebooks)
- `shapely` — 2D footprint geometry (Patcher)
- `scikit-image` — contour detection (Patcher footprint)
- `tqdm` — progress bars
- `pyyaml` — configuration files
- `more-itertools` — iteration utilities

[LiMatch](https://github.com/EPFL-ENAC/lte-limatch) is used as a submodule under `Patcher/submodules/limatch/`.

## Data Flow

```
SDC/CSV laser vectors
      +
 SBET trajectory
      │
      ▼
[georef] → per-scanner LAS point clouds
      │
      ▼
[merge] → merged LAS per scan (VUX HA+LR → +PUCK → merged_manifest.csv)
      │
      ▼
[chunk] → spatial chunks around outage window
      │
  ┌───┴───┐
  │       │
[F2B]   [S2S via Patcher]
  │       │
  └───┬───┘
      ▼
LiMatch correspondences → ODyN bundle adjustment
      │
      ▼
New SBET trajectory
      │
      ▼
[Evaluation] → RMSE / L2L / LCP metrics
```
