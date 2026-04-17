#!/bin/bash
# ==============================================================
# run_batch_outages.sh
# Order: outage_1 -> outage_2 -> outage_3 -> S2S chunks
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
S2S_MODULE="navtools_PDM.s2s_chunks"

CONFIG_1="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_1.yml"
CONFIG_2="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_2.yml"
CONFIG_3="${REPO_ROOT}/navtools_PDM/PDM_configs/pipeline_outage_3.yml"

S2S_LIMATCH_CFG="${REPO_ROOT}/Patcher/submodules/limatch/configs/MLS_F2B_1.yml"
S2S_SBET="/media/b085164/Elements/CALIB_26_02_25/ODyN_calib/Outage_1_305120_305700/traj_outage/outage_1.out"
S2S_OUT_ROOT="/media/b085164/Elements/CALIB_26_02_25/georef_ALL_traj_outage_1/s2s_chunks"
S2S_L="20.0"

#S2S_PAIRS_DIR_1="/media/b085164/Elements/CALIB_26_02_25/georef_ALL_traj_outage_1/Patcher/cropped/pairs_1"
S2S_PAIRS_DIR_2="/media/b085164/Elements/CALIB_26_02_25/georef_ALL_traj_outage_1/Patcher/cropped/pairs_2"
S2S_PAIRS_DIR_3="/media/b085164/Elements/CALIB_26_02_25/georef_ALL_traj_outage_1/Patcher/cropped/pairs_3"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_1="${LOG_DIR}/pipeline_outage_1_${TIMESTAMP}.log"
LOG_2="${LOG_DIR}/pipeline_outage_2_${TIMESTAMP}.log"
LOG_3="${LOG_DIR}/pipeline_outage_3_${TIMESTAMP}.log"
LOG_S2S="${LOG_DIR}/s2s_chunks_${TIMESTAMP}.log"

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

run_s2s_chunks() {
    local log="$1"

    echo ""
    echo "[$(date +%H:%M:%S)] Starting S2S CHUNKS"
    echo "Log: ${log}"
    echo "----------------------------------------------"
    echo "pairs_dirs:"
    #echo "  - ${S2S_PAIRS_DIR_1}"
    echo "  - ${S2S_PAIRS_DIR_2}"
    echo "  - ${S2S_PAIRS_DIR_3}"
    echo "sbet:        ${S2S_SBET}"
    echo "limatch_cfg: ${S2S_LIMATCH_CFG}"
    echo "out_root:    ${S2S_OUT_ROOT}"
    echo "L:           ${S2S_L}"

    "${PYTHON}" -m "${S2S_MODULE}" \
        --pairs_dirs "${S2S_PAIRS_DIR_2}" "${S2S_PAIRS_DIR_3}" \
        --sbet "${S2S_SBET}" \
        --limatch_cfg "${S2S_LIMATCH_CFG}" \
        --out_root "${S2S_OUT_ROOT}" \
        --L "${S2S_L}" \
        --epsg "EPSG:2056" \
        --repo_root "${REPO_ROOT}" \
        2>&1 | tee "${log}"
    local exit_code=${PIPESTATUS[0]}

    if [ "${exit_code}" -ne 0 ]; then
        echo ""
        echo "[ERROR] S2S CHUNKS failed (exit ${exit_code}). See ${log}"
        echo "Aborting batch."
        exit "${exit_code}"
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] S2S CHUNKS DONE"
}

run_step "OUTAGE 1" "${CONFIG_1}" "${LOG_1}"
run_step "OUTAGE 2" "${CONFIG_2}" "${LOG_2}"
run_step "OUTAGE 3" "${CONFIG_3}" "${LOG_3}"
run_s2s_chunks "${LOG_S2S}"

echo ""
echo "=============================================="
echo " ALL DONE — $(date)"
echo " Outage 1 log : ${LOG_1}"
echo " Outage 2 log : ${LOG_2}"
echo " Outage 3 log : ${LOG_3}"
echo " S2S log      : ${LOG_S2S}"
echo "=============================================="
