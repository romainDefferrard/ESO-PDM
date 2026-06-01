# pipeline — MLS Georeferencing & Processing Pipeline

This module orchestrates the full MLS processing chain for GNSS outage experiments:
**Georef → Merge → Chunk → LiMatch (F2B and/or S2S)**.

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

A pipeline config is a YAML file that controls all four steps. Below is an annotated reference.

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

Drives the georef time window filter, the chunker bounds, and the RMSE evaluation window.

### `georef_time_window`

```yaml
georef_time_window:
  enable: true
  margin_s: 10.0     # Buffer added on each side of the outage window
```

When enabled, only laser vector files whose time overlaps `[t_start - margin_s, t_end + margin_s]` are georeferenced. Requires `manifest_path` in the scanner config.

### `distance_filtering`

```yaml
distance_filtering:
  enable: true
  max_distance_m: 30        # Keep only points within 30 m of the vehicle
  map_epsg: "EPSG:2056"     # Projection used for distance computation
  filter_trj: null
```

### `paths`

```yaml
paths:
  root_out_dir: "/media/.../output_root"   # All outputs go here
  limatch_cfg:  "Patcher/submodules/limatch/configs/MLS_F2B_1.yml"
  patcher_cfg:  "Patcher/config/MLS_Epalinges_config.yml"   # required if s2s.run_patcher=true
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
    - [...]
    - [...]
    - [...]
  boresight:
    roll:  0.005              # Boresight angles [rad]
    pitch: 0.0015
    yaw:   0.0008
  lever_arm: [-0.364, 0.330, -0.247]   # Lever arm [x, y, z] in body frame [m]

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
  cleanup: true                        # Delete individual scanner dirs after merge
```

**Presets:**
- `all` — merge VUX HA+LR pairwise, then interleave PUCK by GPS time
- `vux_only` — only merge HA+LR
- `puck_on_existing` — add PUCK to an already-merged VUX folder

**Outputs:**
- `merged/HA_LR/` — VUX merged clouds + `merged_manifest.csv`
- `merged/ALL/` — VUX+PUCK clouds + `merged_manifest.csv`

The `merged_manifest.csv` is created automatically and indexes each file with its GPS time bounds. It is used by the chunk step to avoid reading all files.

---

### Step 3 — `chunk`

Splits merged point clouds into spatial chunks around the outage window.

```yaml
chunk:
  source: "generate"          # "generate" | "existing"
  existing_root: null         # Path to reuse existing chunks (source=existing)
  output_root: null           # null = <root_out_dir>/<scenario>/scenario_combined
  length_m: 15.0              # Chunk length in curvilinear meters
  epsg_out: "EPSG:2056"
  reference_scanner: "HA"     # Scanner used to load the trajectory for chunking

  limatch:                    # Optional: run LiMatch F2B after chunking
    enabled: true
    neighbor_k: 1             # Number of consecutive neighbors (k+1 pairs)
    do_cross_scan: false      # Match last chunk of scan N with first of scan N+1
    do_spatial_crossings: true
    crossing_min_separation: 30    # Min time gap (s) between crossing pair scans
    crossing_overlap_margin_m: 3.0

    # Uncertainty radius override (choose one form):
    uncertainty_r_min: 0.0    # Hollow ring: [r_min, r_max]
    uncertainty_r_max: 2.0    # (both must be given together)
    # uncertainty_r: 2.0      # Scalar override
    # (no entry) → use the value from the LiMatch yml unchanged
```

**What it does:**
1. Loads the trajectory and computes curvilinear distance
2. Selects merged scans that overlap `[t_outage - margin, t_outage + duration + margin]` using the manifest
3. Splits each selected scan into chunks of `length_m` meters
4. Writes one sub-directory per scan with `chunk_XXXX.las` files + `chunk_bbox.csv`

---

### Step 4a — `chunk.limatch` (F2B)

Runs LiMatch on F2B (Forward-to-Backward) chunk pairs within each scan, and optionally on spatial crossing pairs between scans.

Activated by setting `chunk.limatch.enabled: true`.

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
