# pipeline — MLS Georeferencing & Processing Pipeline

This module orchestrates the full MLS processing chain for GNSS outage experiments:
**Georef → Merge → Chunk → LiMatch (F2B and/or S2S)**.

```
┌──────────────────────────────────────────────────────────────┐
│         SDC/CSV laser vectors  +  SBET trajectory            │
└────────────────────────┬─────────────────────────────────────┘
                         │  Step 1 — georef
                         ▼
          per-scanner LAS point clouds
          (HA/   LR/   PUCK/)
                         │  Step 2 — merge
                         ▼
     merged LAS per scan  +  merged_manifest.csv
     (merged/HA_LR/  →  merged/ALL/)
                         │  Step 3 — chunk
                         ▼
      spatial chunks around outage window
      (scenario_combined/chunks_15m/)
               ┌──────────┴──────────┐
    Step 4a    │                     │   Step 4b
  chunk.limatch│                     │   steps.s2s
(F2B / Combined)                   (S2S via Patcher)
               │                     │
               └──────────┬──────────┘
                           │
                           ▼
              LiDAR_p2p.txt  →  ODyN bundle adjustment
```

---

## Key Concepts

### Manifests

The pipeline is built around **manifest CSV files** to avoid scanning entire data directories on each run. A manifest indexes each scan file with its GPS time bounds (`scan_id`, `filename`, `t_start`, `t_end`).

- **Merged cloud manifests** are created automatically by the merge step.
- **SDC/CSV laser vector manifests** must be built once using `pipeline/build_sdc_manifest.ipynb`, then the path is set in the scanner config under `manifest_path`. Once built, the manifest allows the georef step to select only the files whose time range overlaps the outage window, without reading every file.

### Outage Definition

All experiments are driven by an outage window defined as:

```yaml
outage: [t_start_GPS_s, duration_s]
```

This single field controls the georef time windowing and the chunking bounds.

### Coordinate System

All outputs use **EPSG:2056** (Swiss LV95) by default. Change via `epsg_out` in the pipeline config.

---

## Running the Pipeline

```bash
# From the repo root
python -m pipeline -c pipeline/cfg/pipeline_outage_1.yml
```

The `-c` argument takes any pipeline config YAML (see `cfg/` for examples).

---

## Directory Layout

```
pipeline/
├── pipeline.py         # Orchestrator logic
├── __main__.py         # CLI entry point
├── _log.py             # Logging helpers
├── steps/
│   ├── georef.py       # Step 1: LiDAR georeferencing
│   ├── merge.py        # Step 2: Merge multi-scanner clouds per scan
│   ├── chunk.py        # Step 3: Spatial chunking around outage window
│   ├── limatch.py      # Step 4: LiMatch F2B / S2S correspondences
│   └── s2s_chunks.py   # S2S spatial chunking helper
├── lib/
│   ├── trajectory.py   # Trajectory loading & interpolation
│   ├── rotations.py    # Rotation math (quaternions, DCM, R1/R2/R3)
│   ├── loaders.py      # SBET binary loader
│   └── lidar.py        # LiDAR utility functions
├── cfg/                # Experiment pipeline configs
└── scanner_cfg/        # Per-scanner calibration & mount configs
```

---

## Pipeline Config Reference

### Top-level fields

```yaml
scenario_name: "georef_ALL_traj_outage_1/AIRINS/georef_S2S"
# Used as the output subdirectory under root_out_dir.
# Nested paths are supported (e.g. "dataset/method/variant").
```

### `scanners`

Map of scanner keys to their config YAML paths (relative to `pipeline/` or absolute).

```yaml
scanners:
  ha_cfg:   "scanner_cfg/scanner_HA_CALIB.yml"
  lr_cfg:   "scanner_cfg/scanner_LR_CALIB.yml"
  puck_cfg: "scanner_cfg/scanner_PUCK_CALIB.yml"
```

Keys are arbitrary but referenced in `merge.vux_scanners` and `merge.puck_scanner`.

### `trajectory`

```yaml
trajectory:
  type: "SBET"
  path: "/path/to/trajectory.out"
  cfg:
    type: "SBET"
```

Currently only SBET binary format is supported.

### `outage`

```yaml
outage: [305120.0, 580.0]   # [t_start_GPS_s, duration_s]
```

Drives the georef time window filter and the chunker bounds.

### `georef_time_window`

```yaml
georef_time_window:
  enable: true
  margin_s: 10.0     # Buffer added on each side of the outage window
```

When enabled, only laser vector files whose time range overlaps `[t_start - margin_s, t_end + margin_s]` are georeferenced. Requires `manifest_path` in the scanner config.

### `distance_filtering`

```yaml
distance_filtering:
  enable: true
  max_distance_m: 30        # Horizontal distance threshold [m]
  map_epsg: "EPSG:2056"
  filter_trj: null
```

Filters out georeferenced points that are more than `max_distance_m` away from the vehicle trajectory. The distance is computed in the map plane (EPSG:2056) as the 2D horizontal separation between each point and the SBET position interpolated at the same GPS timestamp — it is **not** a range in the scanner frame.

### `paths`

```yaml
paths:
  root_out_dir: "/media/.../output_root"   # All outputs go here
  limatch_cfg:  "Patcher/submodules/limatch/configs/MLS_F2B_1.yml"
  patcher_cfg:  "Patcher/config/MLS_Epalinges_config.yml"
```

### `steps`

Toggle each step on/off independently:

```yaml
steps:
  georef: true    # Step 1
  merge:  true    # Step 2
  chunk:  false   # Step 3
  s2s:    false   # Step 4 (S2S variant)
```

---

### Step 1 — `georef`

No dedicated block in the config; parameters come from the scanner config YAMLs.

**What it does:**
1. Loads the SBET trajectory
2. For each declared scanner, builds a merged georef config and calls `georef.run_scanner()`
3. Reads raw SDC or CSV laser vector files, applies time-sync, leap second correction, time-window filtering, and projects each point into EPSG:2056 using the mount calibration (R_sensor2body, lever arm)
4. Outputs one LAS (or TXT) per input scan file, under `<root_out_dir>/<scenario>/<scanner_name>/`

#### Scanner config (`scanner_cfg/*.yml`)

```yaml
scanner_name: "HA"            # Must match the directory name used in outputs

lasvec:
  type: 'SDC'                 # 'SDC' or 'CSV'
  cols: [0, 3, 4, 5]          # Column indices for [time, x, y, z] in CSV
  path: '/path/to/sdc/folder' # Folder (scanned for *.sdc) or single file

leapsec: 18                   # GPS leap seconds to add to timestamps

mount:
  R_mount:                    # 3×3 rotation matrix — sensor frame → body frame
    - [ 0.3838, -0.7431, -0.5481]
    - [-0.4263, -0.6691,  0.6087]
    - [-0.8192,  0.0000, -0.5736]

  boresight:
    roll:  0.00506            # Boresight angles [rad]
    pitch: 0.00151
    yaw:   0.00085
  lever_arm: [-0.365, 0.331, -0.247]   # Lever arm [x, y, z] in body frame [m]

output_defaults:
  type: 'LAS'                 # 'LAS' or 'ASCII'
  lasvec: true                # Store raw laser vectors as extra dims
  lasvec_to_body: true        # Express laser vectors in body frame

manifest_path: "/path/to/manifests/manifest_HA.csv"
# Required for georef_time_window filtering.
# Build once with pipeline/build_sdc_manifest.ipynb.
```

---

### Step 2 — `merge`

Merges per-scanner point clouds into unified per-scan clouds.

```yaml
merge:
  preset: "all"               # "all" | "vux_only" | "puck_on_existing"
  vux_scanners: ["ha_cfg", "lr_cfg"]   # Keys merged pairwise (HA + LR)
  puck_scanner: "puck_cfg"             # Key for the PUCK scanner
  output_suffix: "_VUX_PUCK"          # Appended to merged file stems
  scanner_src_vux:  2                  # scanner_src value for VUX points
  scanner_src_puck: 1                  # scanner_src value for PUCK points
  chunk_size: 10000000                 # Points per chunk for LAS I/O
  cleanup: true                        # After merge, delete the individual
                                       # scanner dirs (HA/, LR/, PUCK/) and
                                       # the intermediate HA_LR/ folder to
                                       # free disk space
```

**Presets:**
- `all` — merge VUX HA+LR pairwise, then interleave PUCK by GPS time
- `vux_only` — only merge HA+LR
- `puck_on_existing` — add PUCK to an already-merged VUX folder

**Outputs:**
- `merged/HA_LR/` — VUX merged clouds + `merged_manifest.csv`
- `merged/ALL/` — VUX+PUCK clouds + `merged_manifest.csv`

The `merged_manifest.csv` is created automatically and indexes each file with its GPS time bounds. It is used by the chunk step to select only the relevant scan files for a given outage window.

---

### Step 3 — `chunk`

Splits the merged point clouds into spatial chunks around the outage window. These chunks are the direct input for the LiMatch matching step, which produces the point-to-point correspondences used in **F2B** (Front-to-Back) and **Combined** (F2B + spatial crossings) trajectory estimation scenarios.

```yaml
chunk:
  source: "generate"          # "generate" | "existing"
  existing_root: null         # Path to reuse existing chunks (source=existing)
  output_root: null           # null = <root_out_dir>/<scenario>/scenario_combined
  length_m: 15.0              # Chunk length in curvilinear meters
  epsg_out: "EPSG:2056"
  reference_scanner: "HA"     # Scanner used to load the trajectory for chunking
```

**What it does:**
1. Loads the trajectory and computes the cumulative curvilinear distance along the vehicle path
2. Reads the `merged_manifest.csv` and selects all merged scan files whose time range `[t_start, t_end]` overlaps the window `[t_outage - margin_s, t_outage + duration + margin_s]`
3. For each selected scan, splits the point cloud into chunks of `length_m` metres of curvilinear distance
4. Writes one sub-directory per scan with `chunk_XXXX.las` files + `chunk_bbox.csv` (bounding boxes used by spatial crossing detection)

#### `chunk.limatch` — LiMatch on chunks (F2B and Combined)

This sub-block runs [LiMatch](https://github.com/EPFL-ENAC/lte-limatch) on the generated chunks to produce point-to-point correspondences for ODyN.

```yaml
chunk:
  ...
  limatch:
    enabled: true

    # ── F2B consecutive pairs ──────────────────────────────────
    neighbor_k: 1
    # Within each scan pass, every chunk i is matched with chunk i+1,
    # i+2, ..., up to i+k.
    #   k=1 → only consecutive pairs (i / i+1)        ← pure F2B
    #   k=2 → consecutive + skip-one (i/i+1 + i/i+2) ← denser F2B

    # ── Cross-scan matching ────────────────────────────────────
    do_cross_scan: false
    # If true, matches the last chunk of scan pass N with the first
    # chunk of scan pass N+1 (sequential cross-scan link between passes).

    # ── Spatial crossings (Combined scenario) ─────────────────
    do_spatial_crossings: true
    # If true, detects and matches chunk pairs from *different* scan
    # passes that spatially overlap — i.e. the vehicle drove through
    # the same area at different times. This is the S2S component of
    # the Combined scenario (F2B + spatial crossings).
    #
    # Detection uses the chunk_bbox.csv bounding boxes:
    # two chunks are candidate crossings if their 2D bounding boxes
    # overlap (with a tolerance of crossing_overlap_margin_m) AND
    # their sequential indices differ by more than crossing_min_separation.

    crossing_min_separation: 30
    # Minimum sequential-index gap between two chunks to be considered
    # a spatial crossing. Chunks closer than this in the sequence are
    # assumed to belong to the same or adjacent passes and are already
    # covered by F2B / do_cross_scan — they are excluded here to avoid
    # redundant pairs.

    crossing_overlap_margin_m: 3.0
    # Tolerance [m] added to each side of a chunk's bounding box when
    # testing for spatial overlap. Compensates for small georeferencing
    # errors so that genuinely overlapping chunks are not missed due to
    # a slight bbox mismatch.

    # ── LiMatch uncertainty radius override ───────────────────
    # Three mutually exclusive forms — choose one:

    uncertainty_r_min: 0.0
    uncertainty_r_max: 2.0
    # Hollow-ring search: LiMatch only accepts correspondences whose
    # distance to the nearest point falls in [r_min, r_max].
    # Useful to exclude near-zero matches (duplicate/static points)
    # while keeping an upper bound. Both must be given together;
    # they override uncertainty_r from the LiMatch yml.

    # uncertainty_r: 2.0
    # Scalar override: replaces uncertainty_r in the LiMatch yml.

    # (no entry) → use the value from the LiMatch yml unchanged
```

---

### Step 4b — `s2s` (Scan-to-Scan via Patcher)

Runs the full S2S pipeline: Patcher extracts overlapping time windows, then LiMatch is applied on each pair.

```yaml
s2s:
  output_root: null           # null = <root_out_dir>/<scenario>/s2s
  patcher_out_root: null      # null = <output_root>/patcher_output

  pc_dir_override: null       # Override path to merged clouds for Patcher
  pc_dir_suffix: "_VUX_PUCK.las"

  L:            20.0          # Spatial chunk length [m]
  min_last_m:   null          # null = 2/3 * L (auto)
  min_points:   500
  min_time_sep: 30.0          # Skip pairs with time separation < X s

  limatch:
    enabled: true
    uncertainty_r: 2.0
    # uncertainty_r_min: 0.0
    # uncertainty_r_max: 3.0
```

---

### `merge_correspondences`

Merges all LiMatch correspondence files (F2B + crossings, or S2S) into a single file for ODyN.

```yaml
merge_correspondences:
  enabled: true
  output_file: null   # null = <limatch_root>/LiDAR_p2p.txt
```

---

## Output Directory Structure

```
<root_out_dir>/
└── <scenario_name>/
    ├── HA/                     # Georeferenced HA scans
    ├── LR/                     # Georeferenced LR scans
    ├── PUCK/                   # Georeferenced PUCK scans
    ├── merged/
    │   ├── HA_LR/              # VUX merged clouds + merged_manifest.csv
    │   └── ALL/                # VUX+PUCK clouds + merged_manifest.csv
    ├── scenario_combined/
    │   └── chunks_15m/
    │       ├── merged_XXXX/    # Chunks per scan (chunk_XXXX.las + chunk_bbox.csv)
    │       └── limatch/        # LiMatch outputs (F2B, crossings)
    ├── s2s/
    │   ├── patcher_output/     # Patcher extracted pairs
    │   └── limatch/            # LiMatch S2S outputs
    └── tmp/
        └── generated_configs/  # Auto-generated georef YAMLs (debug)
```

---

## Building SDC/CSV Manifests

Before running the pipeline for the first time on a dataset, build the manifest for each scanner:

1. Open `pipeline/build_sdc_manifest.ipynb`
2. Set the path to your SDC/CSV folder and the output CSV path
3. Run the notebook — it scans each file and records `(filename, t_min, t_max)`
4. Add the manifest path to the scanner config:

```yaml
manifest_path: "/path/to/manifests/manifest_HA.csv"
```

This is a one-time operation per dataset.

---

## Standalone Georef

The georef step can also be run standalone without a full pipeline config:

```bash
python -m pipeline.steps.georef -c /path/to/georef_config.yml
```

A standalone georef config has a slightly different structure (see `navtools_PDM/PDM_configs/` for examples).
