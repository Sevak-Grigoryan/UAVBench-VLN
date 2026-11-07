#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=01:30:00
# usage: sbatch probe.sbatch <script.py> <outdir> [extra args...]
set -uo pipefail
OUT="$2"; mkdir -p "$OUT"
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
T="$OUT/tmp_${SLURM_JOB_ID}"; mkdir -p "$T"
export XDG_CACHE_HOME="$T/cache" OPTIX_CACHE_PATH="$T/optix-cache" HOME="$T/home" CUDA_CACHE_PATH="$T/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$T/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697"
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1); U=${U#GPU-}
for b in $BAD_GPUS; do case "$U" in "$b"*) echo "BAD GPU $(hostname) $U"; exit 76;; esac; done
cd "$(dirname "$1")"
S="$1"; shift 2; python "$S" --out "$OUT" "$@" > "$OUT/run_${SLURM_JOB_ID}.log" 2>&1; grep -A8 RESULT "$OUT/run_${SLURM_JOB_ID}.log"
echo "rc=$? node=$(hostname) gpu=$U"
