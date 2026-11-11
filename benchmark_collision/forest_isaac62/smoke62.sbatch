#!/bin/bash
# Isaac Sim 6.2 smoke test of the drone stack (X500 + PX4 SITL + Pegasus via isaac62_compat + contact reports), small flat scene.
#SBATCH --partition=rtx
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:00:00
#SBATCH -J isaac62-smoke
#SBATCH -o /home/sgrigoryan/Desktop/Benchmark_collision_codebase_isaac62/logs/smoke62_%j.out
set -uo pipefail
HERE=/home/sgrigoryan/Desktop/Benchmark_collision_codebase_isaac62
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/sgrigoryan/isaac62/env
export OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD=/mnt/weka/sgrigoryan/Benchmark_collision_work_isaac62/tmp_${SLURM_JOB_ID}; mkdir -p $TMPD
export XDG_CACHE_HOME=$TMPD/cache HOME=$TMPD/home CUDA_CACHE_PATH=$TMPD/cuda-cache; mkdir -p $HOME
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import importlib.metadata as m; print('isaacsim', m.version('isaacsim'))"
python $HERE/smoke62.py --out /mnt/weka/sgrigoryan/Benchmark_collision_work_isaac62/smoke62_${SLURM_JOB_ID} --max-sim-s 60
echo "smoke exit $?"
pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i 0 " 2>/dev/null
