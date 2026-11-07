#!/bin/bash
#SBATCH --job-name=x500_smoke
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/common/scripts/logs/%x_%j.out
#SBATCH --error=/mnt/weka/sgrigoryan/dataset_codebase/common/scripts/logs/%x_%j.err
#SBATCH --partition=research8
#SBATCH --exclude=gpu06,gpu08
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
RUN_DIR="/mnt/weka/sgrigoryan/dataset_codebase/common/_runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export OPTIX_CACHE_PATH="$RUN_DIR/optix-cache"
cd /mnt/weka/sgrigoryan/dataset_codebase/common/pipeline
python smoke_x500.py --out "$RUN_DIR/output" ${SMOKE_ARGS:-}
