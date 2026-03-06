#!/usr/bin/env bash
set -euo pipefail

cd /home/b085164/PDM_Romain_Defferrard/ESO-PDM

echo "=============================="
echo "GEOREF REF"
echo "=============================="
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/ref_HA.yml
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/ref_LR.yml

echo "=============================="
echo "GEOREF OUTAGE ONLY"
echo "=============================="
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/outage_only_HA.yml
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/outage_only_LR.yml

echo "=============================="
echo "GEOREF SCAN2SCAN"
echo "=============================="
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/scan2scan_HA.yml
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/scan2scan_LR.yml

echo "=============================="
echo "GEOREF CHUNK2CHUNK"
echo "=============================="
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/chunk2chunk_HA.yml
python -m navtools_PDM.pointCloudGeoref -c navtools_PDM/cfg_georef_test/chunk2chunk_LR.yml

echo "=============================="
echo "MERGE"
echo "=============================="
python - <<'PY'
from pathlib import Path
from navtools_PDM.singleBeamMerging import merge_txt_pairs

pairs = [
    (
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/ref/HA"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/ref/LR"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/ref/merged"),
    ),
    (
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/outage_only/HA"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/outage_only/LR"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/outage_only/merged"),
    ),
    (
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_scan2scan/HA"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_scan2scan/LR"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_scan2scan/merged"),
    ),
    (
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_chunk2chunk/HA"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_chunk2chunk/LR"),
        Path("/media/b085164/Elements/PCD_SAM/Georef_v6/limatch_chunk2chunk/merged"),
    ),
]

for ha_dir, lr_dir, out_dir in pairs:
    merge_txt_pairs(
        ha_dir,
        lr_dir,
        out_dir,
        delimiter=",",
        out_prefix="merged_",
        out_suffix="_HA_LR",
    )
PY

