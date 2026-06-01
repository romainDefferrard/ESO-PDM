# Patcher — ALS & MLS Patch Extraction Tool

Patcher was developed as part of a semester project at the ESO laboratory (EPFL, Spring 2025). It extracts overlapping point cloud patches between flight lines or scan passes and feeds them to the [LiMatch](https://github.com/EPFL-ENAC/lte-limatch) point-to-point matching algorithm. It supports both ALS (Airborne Laser Scanning) and MLS (Mobile Laser Scanning) modes.

## Running Patcher

```bash
# From the repo root
python Patcher/main.py --yml Patcher/config/MLS_Epalinges_config.yml

# Or use a ready-made script
bash Patcher/run_Epalinges.sh
bash Patcher/run_Arpette.sh
bash Patcher/run_Vallet.sh
```

Patcher can also be called programmatically from within the pipeline:

```python
from Patcher.main import run_patcher_from_config
run_patcher_from_config("Patcher/config/MLS_Epalinges_config.yml")
```

---

## Pipeline Steps

```
load_data()
    ↓
generate_footprint()         # Estimate LiDAR footprints from point cloud bounds
    ↓
generate_patches()           # ALS only: create rectangular patches from overlap zones
    ↓
launch_gui()                 # Interactive PyQt6 GUI for patch review and selection
    ↓
extract_als() / extract_mls()   # Extract point subsets per pair
    ↓
run_limatch()                # Optional: launch LiMatch on extracted pairs
```

---

## Configuration Reference

### Mode and project

```yaml
SCAN_MODE: "ALS"    # "ALS" or "MLS"
PRJ_NAME:  "Arpette"
```

### Paths

```yaml
PC_DIR:     "./data/Arpette/ARPETTE_LV95_HELL_1560II_CH1_211020_{flight_id}"
# Template string: {flight_id} is replaced for each flight.
# MLS example: "/media/.../merged/ALL/merged_{flight_id}_VUX_PUCK.las"

OUTPUT_DIR: "./output/Arpette/"
OUTPUT_PC_FMT: "txt"   # "las" | "laz" | "txt" | "txyzs"
```

### Grid (raster)

```yaml
GRID_RES:    25    # Raster cell size [m], used for footprint computation
GRID_BUFFER: 0     # Extra buffer around data bounds [m]
```

### ALS patch generation

```yaml
PAIR_MODE: "successive"
# "successive": only generate patches between adjacent flight lines
# "all": generate patches between all possible pairs

PATCH_DIMS: [400, 400, 1000]   # [length, width, height] in meters

POINTCLOUD_DOWNSAMPLING: 100   # Keep 1 point every N for footprint estimation
EXTRACTION_MODE: "independent" # Only supported mode
```

### LiMatch integration

```yaml
LIMATCH_CFG: "/path/to/limatch/configs/ALS_close_range.yml"
```

---

## ALS Mode

In ALS mode, Patcher:
1. Loads all flight files (one per `flight_id` in `PC_DIR`)
2. Estimates the 2D footprint of each flight by projecting points onto a raster grid
3. Detects overlap zones between flights using the `PAIR_MODE` strategy
4. Generates rectangular patches covering each overlap zone
5. Opens a **GUI** where you can inspect patches, disable unwanted ones, and trigger extraction
6. Extracts one LAS/TXT file per (patch, flight) pair into `OUTPUT_DIR/Flights_A_B/patch_N_flight_A.ext`

### Output structure (ALS)

```
OUTPUT_DIR/
└── Flights_001_002/
    ├── patch_0_flight_001.txt
    ├── patch_0_flight_002.txt
    ├── patch_1_flight_001.txt
    └── patch_1_flight_002.txt
```

---

## MLS Mode

In MLS mode, Patcher:
1. Loads all scan files (georeferenced merged LAS)
2. Estimates temporal overlap between scans based on their spatial footprint — no physical patches are generated
3. Opens a **GUI** showing the overlap zones on a map
4. Extracts one LAS file per scan per overlap time window (reads each scan file only once for all its windows)

This "grouped extraction" strategy is designed for memory efficiency: each large merged LAS is read once and all time windows for that scan are extracted in a single pass.

### Output structure (MLS)

```
OUTPUT_DIR/
└── Flights_{A}_{B}/        # Normalized pair directory (A < B always)
    ├── Patch_from_scan_A_with_B.las
    └── Patch_from_scan_B_with_A.las
```

---

## GUI Overview

The GUI (PyQt6) shows:
- The raster footprint of each flight/scan
- Overlap zones between pairs
- Generated patches (ALS) or temporal overlap zones (MLS)

Controls available:
- Enable/disable individual patches
- Set output directory
- Toggle LiMatch execution after extraction
- Start extraction

---

## Config Examples

### ALS — Arpette

```yaml
SCAN_MODE: "ALS"
PRJ_NAME: "Arpette"
OUTPUT_DIR: "./output/Arpette/"
PC_DIR: "./data/Arpette/ARPETTE_LV95_HELL_1560II_CH1_211020_{flight_id}"
OUTPUT_PC_FMT: "txt"
GRID_RES: 25
GRID_BUFFER: 0
PAIR_MODE: "successive"
PATCH_DIMS: [400, 400, 1000]
POINTCLOUD_DOWNSAMPLING: 100
EXTRACTION_MODE: "independent"
LIMATCH_CFG: "/path/to/limatch/configs/ALS_close_range.yml"
```

### MLS — Epalinges

```yaml
SCAN_MODE: "MLS"
PRJ_NAME: "Epalinges"
OUTPUT_DIR: "./output/Epalinges/"
PC_DIR: "/media/.../merged/ALL/merged_{flight_id}_VUX_PUCK.las"
OUTPUT_PC_FMT: "las"
GRID_RES: 5
GRID_BUFFER: 10
CRS: "EPSG:2056"
LIMATCH_CFG: "/path/to/limatch/configs/MLS_F2B_1.yml"
```

---

## Integration with the Main Pipeline

The main pipeline (`pipeline/`) calls Patcher automatically when `steps.s2s: true`. In that case, the Patcher config is specified under `paths.patcher_cfg` in the pipeline YAML:

```yaml
paths:
  patcher_cfg: "Patcher/config/MLS_Epalinges_config.yml"

steps:
  s2s: true
```

The pipeline overrides `PC_DIR` and `OUTPUT_DIR` automatically from the scenario context.

---

## Submodules

[LiMatch](https://github.com/EPFL-ENAC/lte-limatch) lives at `Patcher/submodules/limatch/`. Make sure submodules are initialized:

```bash
git submodule update --init --recursive
```
