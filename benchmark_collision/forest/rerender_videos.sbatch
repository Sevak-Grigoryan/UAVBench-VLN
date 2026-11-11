#!/bin/bash
# Re-render collision-test videos only (no flights):  sbatch rerender_videos.sbatch <scene> <work root> "<tests>"
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
set -uo pipefail
SCENE="$1"; ROOT="$2"; TESTS="$3"
HERE=/home/sgrigoryan/Desktop/Benchmark_collision_codebase
CB=/mnt/weka/sgrigoryan/dataset_codebase
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD="$ROOT/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache"
export HOME="$TMPD/home" CUDA_CACHE_PATH="$TMPD/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$TMPD/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
cd $CB/common/pipeline

BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"
BAD_GPUS="$BAD_GPUS $(awk '{print $1}' $CB/common/scripts/bad_gpus.txt 2>/dev/null)"
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ "$(echo "$UUIDS" | grep -c GPU-)" -gt 1 ]; then UUIDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); fi
GPU_UUID=${UUIDS#GPU-}
echo "gpu $(hostname) $GPU_UUID"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU $(hostname) $GPU_UUID -- skipped"; exit 76;; esac; done

# re-render of the collision-test videos with the fixture-aware side camera (render_debug.py, 2026-11-10) into video_v2/;
# the first videos (video/) are kept.
for n in $TESTS; do
  TW="$ROOT/colltest_$n"
  [ -f "$TW/video_v2/collision_test.mp4" ] && continue
  for att in 1 2; do
    L="$TW/video_v2_${SLURM_JOB_ID}_${att}.log"
    python $HERE/render_debug.py --env Forest --scene "$SCENE" --flight "$TW/flight" --out "$TW/video_v2" --camera side \
      --max-seconds 12 --stride 2 --hold-s 2 --bank-samples 0 --name collision_test > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
  echo "colltest_$n video_v2: $(ls "$TW/video_v2/collision_test.mp4" 2>/dev/null || echo MISSING) $(grep -h 'side camera' "$L")"
done
echo "job done"
