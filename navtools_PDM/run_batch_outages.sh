#!/bin/bash
# ==============================================================
# run_batch_outages.sh
# Order: outage_1 -> outage_2 -> outage_3
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/navtools_PDM/pipeline.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/pipeline.py" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "[ERROR] Cannot locate repo root from ${SCRIPT_DIR}"
    exit 1
fi

cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
PIPELINE_MODULE="navtools_PDM.pipeline"

CONFIG_1="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_1.yml"
CONFIG_2="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_2.yml"
CONFIG_3="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_3.yml"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_1="${LOG_DIR}/pipeline_outage_1_${TIMESTAMP}.log"
LOG_2="${LOG_DIR}/pipeline_outage_2_${TIMESTAMP}.log"
LOG_3="${LOG_DIR}/pipeline_outage_3_${TIMESTAMP}.log"

echo "=============================================="
echo " Batch pipeline — $(date)"
echo " Repo root : ${REPO_ROOT}"
echo " Python    : ${PYTHON}"
echo "=============================================="

run_step() {
    local label="$1"
    local config="$2"
    local log="$3"

    echo ""
    echo "[$(date +%H:%M:%S)] Starting ${label} — config: ${config}"
    echo "Log: ${log}"
    echo "----------------------------------------------"

    "${PYTHON}" -m "${PIPELINE_MODULE}" -c "${config}" 2>&1 | tee "${log}"
    local exit_code=${PIPESTATUS[0]}

    if [ "${exit_code}" -ne 0 ]; then
        echo ""
        echo "[ERROR] ${label} failed (exit ${exit_code}). See ${log}"
        echo "Aborting batch."
        exit "${exit_code}"
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] ${label} DONE"
}

run_step "OUTAGE 1" "${CONFIG_1}" "${LOG_1}"
run_step "OUTAGE 2" "${CONFIG_2}" "${LOG_2}"
run_step "OUTAGE 3" "${CONFIG_3}" "${LOG_3}"

echo ""
echo "=============================================="
echo " ALL DONE — $(date)"
echo " Outage 1 log : ${LOG_1}"
echo " Outage 2 log : ${LOG_2}"
echo " Outage 3 log : ${LOG_3}"
echo "=============================================="