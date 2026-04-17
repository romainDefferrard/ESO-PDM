#!/bin/bash
# Auto-generated — georef + merge batch
# 6 runs: 3 outages x 2 methods (F2B / COMBINED)

set -euo pipefail

cd "/home/b085164/PDM_Romain_Defferrard/ESO-PDM"

PYTHON="${PYTHON:-python3}"
LOG_DIR="/media/b085164/Elements/CALIB_26_02_25/logs/georef_batch"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=============================="
echo " Georef batch — $(date)"
echo "=============================="


echo ""
echo "[$(date +%H:%M:%S)] ===== outage_2_F2B ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/configs_georef/pipeline_outage_2_F2B.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_2_F2B_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_2_F2B failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_2_F2B DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_2_COMBINED ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/configs_georef/pipeline_outage_2_COMBINED.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_2_COMBINED_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_2_COMBINED failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_2_COMBINED DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_3_F2B ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/configs_georef/pipeline_outage_3_F2B.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_3_F2B_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_3_F2B failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_3_F2B DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_3_COMBINED ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/configs_georef/pipeline_outage_3_COMBINED.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_3_COMBINED_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_3_COMBINED failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_3_COMBINED DONE"

echo ""
echo "=============================="
echo " ALL DONE — $(date)"
echo "=============================="