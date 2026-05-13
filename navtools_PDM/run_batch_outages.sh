#!/bin/bash
set -e

ROOT="/home/b085164/PDM_Romain_Defferrard/ESO-PDM"
CONDA_ENV="limatch"

cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"


echo "============================================================"
echo "[1/2] pipeline_outage_1_apx.yml"
echo "============================================================"


python -m navtools_PDM.pipeline -c "navtools_PDM/PDM_configs/pipeline_outage_1_apx.yml"


echo "Done."