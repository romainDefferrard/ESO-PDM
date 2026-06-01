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

### Matching Strategies

Three strategies produce L2L correspondences for ODyN, each exploiting a different aspect of the acquisition geometry:

- **F2B (Front-to-Back)** — matches each spatial chunk with its immediate successor(s) along the trajectory, exploiting the forward overlap naturally induced by the VUX scanners' scanning geometry. Geometry-independent and applicable to any acquisition, but overlap can be narrow on featureless road sections.
- **S2S (Scan-to-Scan)** — matches spatially aligned chunks from different scan passes (back-and-forth drives, parking laps, road crossings). Provides strong constraints where crossing geometry is available, but requires at least two passes over the same area and assumes the accumulated drift between passes is small enough for footprints to overlap. This condition is satisfied by the AIRINS but not guaranteed for the APX15.
- **Combined (F2B + S2S)** — integrates both sets of correspondences in a single ODyN run. For **AIRINS**, F2B and S2S crossing chunks are extracted simultaneously from the degraded point cloud. For **APX15**, the large accumulated drift requires a sequential approach: F2B is applied first, the corrected trajectory is used to re-georeference a new point cloud, and S2S crossings are then extracted from this intermediate cloud. If the remaining drift after F2B is still too large for Patcher to detect overlaps automatically, use `Evaluation/L2L_eval/L2L_S2S.ipynb` instead: it allows manually specifying the GPS time windows of the scan lines to match and adjusting the LiMatch config accordingly.


**The Combined approach via `chunk.limatch` (`do_spatial_crossings: true`) is strongly preferred over the standalone `s2s` step: it reuses the existing F2B chunks without a full Patcher run, and is significantly faster.**

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
```

Used as the output subdirectory under `root_out_dir`. Nested paths are supported (e.g. `"dataset/method/variant"`).

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
  margin_s: 10.0
```

When enabled, only laser vector files whose time range overlaps `[t_start - margin_s, t_end + margin_s]` are georeferenced. Requires `manifest_path` in the scanner config.

### `distance_filtering`

```yaml
distance_filtering:
  enable: true
  max_distance_m: 30
  map_epsg: "EPSG:2056"
  filter_trj: null
```

Filters out georeferenced points that are more than `max_distance_m` away from the vehicle trajectory. The distance is computed in the map plane (EPSG:2056) as the 2D horizontal separation between each georeferenced point and the SBET position interpolated at the same GPS timestamp — it is **not** a range in the scanner frame.

### `paths`

```yaml
paths:
  root_out_dir: "/media/.../output_root"
  limatch_cfg:  "Patcher/submodules/limatch/configs/MLS_F2B_1.yml"
  patcher_cfg:  "Patcher/config/MLS_Epalinges_config.yml"
```

### `steps`

Toggle each step on/off independently:

```yaml
steps:
  georef: true
  merge:  true
  chunk:  false
  s2s:    false
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
scanner_name: "HA"

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
  lever_arm: [-0.365, 0.331, -0.247]   # [x, y, z] in body frame [m]

output_defaults:
  type: 'LAS'                 # 'LAS' or 'ASCII'
  lasvec: true                # Store raw laser vectors as extra dims
  lasvec_to_body: true        # Express laser vectors in body frame

manifest_path: "/path/to/manifests/manifest_HA.csv"
```

#### Building SDC/CSV Manifests

Before running the pipeline for the first time on a dataset, build the manifest for each scanner. This is a **one-time operation per dataset**.

1. Open `pipeline/build_sdc_manifest.ipynb`
2. Set the path to your SDC/CSV folder and the output CSV path
3. Run the notebook — it scans each file and records `(filename, t_min, t_max)`
4. Set the resulting path in the scanner config under `manifest_path`

The manifest enables the georef step to select only the files whose time range overlaps the outage window, without reading every file in the folder.

---

### Step 2 — `merge`

Merges per-scanner point clouds into unified per-scan clouds.

```yaml
merge:
  preset: "all"
  vux_scanners: ["ha_cfg", "lr_cfg"]
  puck_scanner: "puck_cfg"
  output_suffix: "_VUX_PUCK"
  scanner_src_vux:  2
  scanner_src_puck: 1
  chunk_size: 10000000
  cleanup: true
```

- **`preset`** — `"all"`: merge VUX HA+LR pairwise then interleave PUCK by GPS time; `"vux_only"`: only merge HA+LR; `"puck_on_existing"`: add PUCK to an already-merged VUX folder
- **`vux_scanners`** — scanner keys to merge pairwise
- **`puck_scanner`** — scanner key for the Velodyne PUCK
- **`output_suffix`** — appended to merged file stems (e.g. `merged_1000_VUX_PUCK.las`)
- **`scanner_src_vux` / `scanner_src_puck`** — integer tag stored in the `scanner_src` extra dim of the merged LAS to identify point origin
- **`chunk_size`** — points per chunk during LAS I/O (memory control)
- **`cleanup`** — if `true`, deletes the individual scanner dirs (`HA/`, `LR/`, `PUCK/`) and the intermediate `HA_LR/` folder after the final merge to free disk space

**Outputs:**
- `merged/HA_LR/` — VUX merged clouds + `merged_manifest.csv`
- `merged/ALL/` — VUX+PUCK clouds + `merged_manifest.csv`

---

### Step 3 — `chunk`

Splits the merged point clouds into spatial chunks around the outage window. These chunks are the input for [LiMatch](https://github.com/ESO-EPFL/limatch), which produces point-to-point correspondences for the **F2B** (Front-to-Back) and **Combined** (F2B + spatial crossings) trajectory estimation scenarios.

```yaml
chunk:
  source: "generate"
  existing_root: null
  output_root: null
  length_m: 15.0
  epsg_out: "EPSG:2056"
  reference_scanner: "HA"
```

- **`source`** — `"generate"`: run the chunker; `"existing"`: skip and reuse chunks at `existing_root`
- **`existing_root`** — path to an existing chunks directory (only used when `source: existing`)
- **`output_root`** — output directory; `null` → `<root_out_dir>/<scenario>/scenario_combined`
- **`length_m`** — chunk length in curvilinear metres along the vehicle trajectory. L=15m was empirically determined to provide sufficient spatial overlap for the VUX scanners' forward scanning geometry
- **`epsg_out`** — map CRS for trajectory projection
- **`reference_scanner`** — which scanner's trajectory is used to compute curvilinear distance

**What it does:**
1. Loads the trajectory and computes the cumulative curvilinear distance along the vehicle path
2. Reads `merged_manifest.csv` and selects scan files whose time range `[t_start, t_end]` overlaps the window `[t_outage - margin_s,  t_outage + duration + margin_s]`
3. Splits each selected scan into chunks of `length_m` metres
4. Writes one sub-directory per scan with `chunk_XXXX.las` files + `chunk_bbox.csv` (spatial bounding boxes used for crossing detection)

#### `chunk.limatch` — LiMatch on chunks (F2B and Combined)

Runs [LiMatch](https://github.com/ESO-EPFL/limatch) on the generated chunks to produce point-to-point correspondences for ODyN.

**1. F2B — consecutive chunk pairs**

```yaml
chunk:
  limatch:
    enabled: true
    neighbor_k: 1
    do_cross_scan: false
```

- **`neighbor_k`** — within each scan pass, every chunk `i` is matched with chunks `i+1`, `i+2`, …, `i+k`.
  - `k=1` → only consecutive pairs (`i` / `i+1`) — pure F2B
  - `k=2` → consecutive + skip-one pairs (`i`/`i+1` and `i`/`i+2`) — denser F2B
- **`do_cross_scan`** — if `true`, also matches the last chunk of scan pass N with the first chunk of scan pass N+1, creating a sequential link between consecutive passes.

**2. S2S crossings — Combined scenario**

```yaml
    do_spatial_crossings: true
    crossing_min_separation: 30
    crossing_overlap_margin_m: 3.0
    uncertainty_r_min: 0.0
    uncertainty_r_max: 2.0
    # uncertainty_r: 2.0
```

- **`do_spatial_crossings`** — if `true`, detects and matches chunk pairs from *different* scan passes that cover the same area. This is the S2S component of the **Combined** scenario. Detection uses `chunk_bbox.csv` bounding boxes: two chunks are candidates if their 2D bounding boxes overlap (within `crossing_overlap_margin_m`) and their sequential indices differ by more than `crossing_min_separation`.
- **`crossing_min_separation`** — minimum sequential-index gap between two chunks to be considered a spatial crossing, ensuring that only temporally distant (and therefore truly independent) passes are matched.
- **`crossing_overlap_margin_m`** — tolerance in metres added to each side of a bounding box when testing for spatial overlap. Compensates for small georeferencing errors so that genuinely overlapping chunks are not missed.

**LiMatch uncertainty radius override**

Three mutually exclusive forms — choose one or omit entirely:

- **`uncertainty_r_min` + `uncertainty_r_max`** — defines a hollow-ring (annulus) search: correspondences are only accepted if their nearest-neighbour distance falls in `[r_min, r_max]`. Useful to exclude near-zero matches and bound the search radius. Both values must be provided together; they override `uncertainty_r` from the LiMatch yml. **Note: using the annular search requires that LiMatch's `get_candidates()` function supports `uncertainty_r_min` / `uncertainty_r_max` — verify this in your LiMatch version before use.**
- **`uncertainty_r`** — scalar override: replaces `uncertainty_r` in the LiMatch yml.
- *(no entry)* — uses the value from the LiMatch yml unchanged.

**Recommended LiMatch parameters (from thesis, Appendix 8.1)**

Tiling and keypoint limits are disabled — chunks are already spatially bounded and contain a manageable number of points.

| Config | `rsc_thr` | `lcd_r` | `uncertainty_r` |
|---|---|---|---|
| AIRINS — F2B and S2S | 0.5 m | 1.0 m | 1.5 m (scalar) |
| APX15 — F2B | 0.5 m | 1.0 m | 1.5 m (scalar) |
| APX15 — S2S (after F2B re-georef) | 0.8 m | 2.0 m | annular `[r_min, r_max]` |

For APX15 S2S, the larger residual drift after F2B correction requires a wider `rsc_thr` and `lcd_r`, and the annular search to constrain the search to the expected inter-scan offset and reduce computation time.

---

### Step 4b — `s2s` (Scan-to-Scan via Patcher)

Runs the standalone S2S pipeline: Patcher detects temporal overlaps between scan passes, extracts the corresponding point cloud sub-windows, and runs [LiMatch](https://github.com/ESO-EPFL/limatch) on each pair. This produces S2S correspondences independently of the chunk step — as opposed to `do_spatial_crossings` in `chunk.limatch`, which produces S2S correspondences from the same chunks used for F2B.

```yaml
s2s:
  output_root: null
  patcher_out_root: null
  pc_dir_override: null
  pc_dir_suffix: "_VUX_PUCK.las"
  L: 20.0
  min_time_sep: 30.0
  epsg: "EPSG:2056"
  limatch:
    enabled: true
    uncertainty_r: 1.5
    # uncertainty_r_min: 0.0
    # uncertainty_r_max: 3.0
    max_kpts: 10000
```

- **`output_root`** — root output directory; `null` → `<root_out_dir>/<scenario>/s2s`
- **`patcher_out_root`** — where Patcher writes extracted pairs; `null` → `<output_root>/patcher_output`
- **`pc_dir_override`** — explicit path to the merged clouds folder for Patcher; `null` → auto-resolved from `merged/ALL` of the current scenario
- **`pc_dir_suffix`** — suffix appended to `{flight_id}` in the merged file template (e.g. `merged_{flight_id}_VUX_PUCK.las`)
- **`L`** — spatial chunk length [m] for splitting each Patcher-extracted overlap before LiMatch. L=20m used in thesis experiments.
- **`min_time_sep`** — pairs of scan passes whose temporal separation is less than this value [s] are skipped (avoids matching nearly simultaneous passes that are not independent)
- **`epsg`** — map CRS for spatial chunking
- **`limatch.uncertainty_r`** / **`uncertainty_r_min` + `uncertainty_r_max`** — same override logic as for `chunk.limatch` (see above)
- **`limatch.max_kpts`** — optional override for the maximum number of keypoints per cloud passed to LiMatch; omit to use the LiMatch yml value

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
