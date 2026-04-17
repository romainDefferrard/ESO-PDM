#!/bin/bash
# Auto-generated — georef + merge batch
# 6 runs: 3 outages x 2 methods (F2B / COMBINED)
#
# Output structure:
#   georef_ALL_traj_outage_X/georef_F2B/
#   georef_ALL_traj_outage_X/georef_COMBINED/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Adjust REPO_ROOT if needed — should be the root of your ESO-PDM repo
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
LOG_DIR="/media/b085164/Elements/CALIB_26_02_25/logs/georef_batch"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=============================="
echo " Georef batch — $(date)"
echo " Repo root : ${REPO_ROOT}"
echo "=============================="

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_1_F2B ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_1_F2B.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_1_F2B_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_1_F2B failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_1_F2B DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_1_COMBINED ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_1_COMBINED.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_1_COMBINED_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_1_COMBINED failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_1_COMBINED DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_2_F2B ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_2_F2B.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_2_F2B_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_2_F2B failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_2_F2B DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_2_COMBINED ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_2_COMBINED.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_2_COMBINED_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_2_COMBINED failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_2_COMBINED DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_3_F2B ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_3_F2B.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_3_F2B_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_3_F2B failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_3_F2B DONE"

echo ""
echo "[$(date +%H:%M:%S)] ===== outage_3_COMBINED ====="
"${PYTHON}" -m navtools_PDM.pipeline -c "/home/b085164/PDM_Romain_Defferrard/ESO-PDM/navtools_PDM/PDM_configs/georef_batch/pipeline_outage_3_COMBINED.yml" \
    2>&1 | tee "${LOG_DIR}/pipeline_outage_3_COMBINED_${TIMESTAMP}.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[ERROR] outage_3_COMBINED failed"; exit 1; }
echo "[$(date +%H:%M:%S)] outage_3_COMBINED DONE"

echo ""
echo "=============================="
echo " ALL DONE — $(date)"
echo "=============================="