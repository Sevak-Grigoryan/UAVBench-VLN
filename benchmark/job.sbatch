#!/bin/bash
# One GPU job of a benchmark episode (adapted from dataset_codebase/common/scripts/x500_job.sbatch; same start-up guards).
#   flight:  sbatch job.sbatch flight <mission> <work_dir> <vehicle_id>
#   slices:  sbatch job.sbatch slices <mission> <work_dir> "start:end,start:end,..."   (rendered concurrently on one GPU)
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:30:00
set -uo pipefail
MODE="$1"; MISSION="$2"; WORK="$3"
HERE=/home/sgrigoryan/Desktop/Benchmark_codebase
CB=/mnt/weka/sgrigoryan/dataset_codebase
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD="$WORK/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache"
# /home is nearly full: every cache Isaac Sim / the driver writes under $HOME goes to weka instead
export HOME="$TMPD/home" CUDA_CACHE_PATH="$TMPD/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$TMPD/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
cd $CB/common/pipeline

# GPUs on which Isaac Sim crashes at start-up every time (same list as the dataset jobs): exit 76 = resubmit elsewhere
BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"
BAD_GPUS="$BAD_GPUS $(awk '{print $1}' $CB/common/scripts/bad_gpus.txt 2>/dev/null)"
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ "$(echo "$UUIDS" | grep -c GPU-)" -gt 1 ]; then UUIDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); fi
GPU_UUID=${UUIDS#GPU-}
echo "gpu $(hostname) $GPU_UUID"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU $(hostname) $GPU_UUID -- skipped"; exit 76;; esac; done

if [ "$MODE" = slices ]; then
  render_one() {
    local s=$1 e=$2 att
    for att in 1 2; do
      sleep $((RANDOM % 30))
      local L="$TMPD/attempt_$(printf %06d "$s")_${att}.log"
      XDG_CACHE_HOME="$TMPD/cache_$s" OPTIX_CACHE_PATH="$TMPD/optix_$s" \
        python $HERE/render.py --env Mountain --telemetry "$WORK/flight/telemetry.json" --start "$s" --end "$e" \
        --out "$WORK/slices/slice_$(printf %06d "$s").mp4" --mission "$MISSION" > "$L" 2>&1
      local rc=$?
      if [ $rc -eq 0 ] || grep -q "\[slice\] booted" "$L"; then return $rc; fi
      echo "slice $s attempt $att: Isaac Sim start-up failure (rc=$rc)"
    done
    return 75
  }
  pids=()
  for se in ${4//,/ }; do render_one "${se%%:*}" "${se##*:}" & pids+=($!); done
  rc_all=0
  for p in "${pids[@]}"; do wait "$p" || rc_all=$?; done
  exit $rc_all
fi

if [ "$MODE" = flight ]; then
  export X500_NO_BAKE=1   # physics-only flight needs no baked ground textures
  for attempt in 1 2; do
    sleep $((RANDOM % 45))
    L="$TMPD/attempt_${attempt}.log"
    python $HERE/fly.py --env Mountain --mission "$MISSION" --out "$WORK/flight" --vehicle-id "$4" > "$L" 2>&1
    rc=$?
    pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i $4 " 2>/dev/null
    if [ $rc -eq 0 ] || grep -q "recording starts" "$L"; then exit $rc; fi
    echo "attempt $attempt: Isaac Sim start-up failure (rc=$rc)"
  done
  exit 75
fi
echo "unknown mode $MODE"; exit 2
