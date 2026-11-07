#!/bin/bash
#SBATCH --partition=research8
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=00:40:00
# usage: sbatch [--nodelist=gpuNN] preview.sbatch <mission> <outdir> [env]
set -uo pipefail
CB=/mnt/weka/sgrigoryan/dataset_codebase
OUT="$2"; mkdir -p "$OUT"
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
T="$OUT/tmp_${SLURM_JOB_ID}"; mkdir -p "$T"
export XDG_CACHE_HOME="$T/cache" OPTIX_CACHE_PATH="$T/optix-cache" HOME="$T/home" CUDA_CACHE_PATH="$T/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$T/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"  # crash Isaac Sim at start-up
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1); U=${U#GPU-}
for b in $BAD_GPUS; do case "$U" in "$b"*) echo "BAD GPU $(hostname) $U"; exit 76;; esac; done
cd $CB/common/pipeline
python preview_stills.py --env "${3:-Forest}" --mission "$1" --out "$OUT" > "$OUT/preview_${SLURM_JOB_ID}.log" 2>&1
echo "rc=$? node=$(hostname)"
