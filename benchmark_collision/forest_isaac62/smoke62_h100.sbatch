#!/bin/bash
# Isaac Sim 6.2 smoke test of the drone stack (X500 + PX4 SITL + Pegasus via isaac62_compat + contact reports), small flat scene.
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu02,gpu05,gpu06,gpu09
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=00:30:00
#SBATCH -J isaac62-smoke-h100
#SBATCH -o /home/sgrigoryan/Desktop/Benchmark_collision_codebase_isaac62/logs/smoke62_%j.out
set -uo pipefail
HERE=/home/sgrigoryan/Desktop/Benchmark_collision_codebase_isaac62
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/sgrigoryan/isaac62/env
export PATH="$CONDA_PREFIX/bin:$PATH"   # a self-resubmitted job inherits an already-active env: conda activate then skips it and base python would come first
export OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD=/mnt/weka/sgrigoryan/Benchmark_collision_work_isaac62/tmp_${SLURM_JOB_ID}; mkdir -p $TMPD
export XDG_CACHE_HOME=$TMPD/cache HOME=$TMPD/home CUDA_CACHE_PATH=$TMPD/cuda-cache; mkdir -p $HOME
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# known bad GPUs (same list as job.sbatch): Vulkan DEVICE_LOST at start-up on ae9778d4 (jobs 380967, 381211)
BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"
BAD_GPUS="$BAD_GPUS $(awk '{print $1}' /mnt/weka/sgrigoryan/dataset_codebase/common/scripts/bad_gpus.txt 2>/dev/null)"
GPU_UUID=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); GPU_UUID=${GPU_UUID#GPU-}
echo "gpu $(hostname) $GPU_UUID"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU -- resubmitted as $(sbatch --parsable "$HERE/smoke62_h100.sbatch"); holding this GPU 4 min"; sleep 240; exit 76;; esac; done
python -c "import importlib.metadata as m; print('isaacsim', m.version('isaacsim'))"
python $HERE/smoke62.py --out /mnt/weka/sgrigoryan/Benchmark_collision_work_isaac62/smoke62_${SLURM_JOB_ID} --max-sim-s 30
echo "smoke exit $?"
pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i 0 " 2>/dev/null
