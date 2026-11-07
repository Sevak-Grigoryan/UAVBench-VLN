#!/bin/bash
# One GPU job of the X500 dataset pipeline. Submitted by <Env>/scripts/manage_missions.sh.
#   flight:  sbatch x500_job.sbatch <Env> flight <mission> <vehicle_id>
#   slice:   sbatch x500_job.sbatch <Env> slice  <mission> <start> <end>
#SBATCH --partition=research8
#SBATCH --exclude=gpu05,gpu06
## (2026-10-15) only gpu05 is excluded as a whole and gpu09 are excluded as a whole (several different GPUs crashed Isaac Sim at start-up, 2026-10-15); elsewhere bad GPUs are skipped one by one via BAD_GPUS below
# 2026-10-11: gpu03, gpu04, gpu08 start Isaac Sim fine (preview jobs 321714-17); gpu02 still DEVICE_LOST.
# Previously only gpu04 (driver 590.48): on the 580.105 nodes and on gpu09 (580.173) Isaac Sim crashed at start-up
# with ERROR_DEVICE_LOST on 86 of 86 attempts on 2026-10-09 evening (gpu06, 580.126, likewise).
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:30:00
set -uo pipefail
ENV="$1"; MODE="$2"; MISSION="$3"
CB=/mnt/weka/sgrigoryan/dataset_codebase
RUN="$CB/$ENV/_runs/$MISSION"
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD="$RUN/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache"
# /home is a shared 877 GB volume that sat at 100% on 2026-10-09; keep every cache/log Isaac Sim and
# the NVIDIA driver write under $HOME (~/.nv, ~/.nvidia-omniverse, shader caches) on weka instead.
if [ "${X500_HOME_REDIRECT:-1}" = 1 ]; then
  export HOME="$TMPD/home"; mkdir -p "$HOME"
  export CUDA_CACHE_PATH="$TMPD/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$TMPD/gl-cache"
  mkdir -p "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
fi
cd $CB/common/pipeline

# GPUs on which Isaac Sim crashed at start-up (Vulkan ERROR_DEVICE_LOST) on every attempt, 2026-10-12 evening:
# ~900 crashes on these 8 vs 0-1 successes; every other GPU of gpu03/07/08 starts fine. Skip them quickly
# (exit 76, not counted as a try by the manager) instead of burning 2 x 3 min and a retry.
BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ "$(echo "$UUIDS" | grep -c GPU-)" -gt 1 ]; then UUIDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); fi
GPU_UUID=${UUIDS#GPU-}
echo "gpu $(hostname) $GPU_UUID"
# plus GPUs found bad later (render workers add a GPU here after Isaac Sim crashed at start-up on it, 2026-10-23)
BAD_GPUS="$BAD_GPUS $(cat $CB/common/scripts/bad_gpus.txt 2>/dev/null | awk '{print $1}')"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU $(hostname) $GPU_UUID -- skipped"; exit 76;; esac; done
# GPU load every 10 s (decides whether two renders could share one GPU)
nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > "$TMPD/gpu_util.csv" 2>/dev/null &

if [ "$MODE" = slices ]; then
  # several slices of one mission rendered CONCURRENTLY on this one GPU (a single render keeps an H100 only ~37 % busy,
  # 2026-10-15 measurement): $4 = "start:end,start:end,..."; each slice has its own caches, logs and start-up retry
  render_one() {
    local s=$1 e=$2 att
    for att in 1 2; do
      sleep $((RANDOM % 30))
      local L="$TMPD/attempt_$(printf %06d "$s")_${att}.log"
      XDG_CACHE_HOME="$TMPD/cache_$s" OPTIX_CACHE_PATH="$TMPD/optix_$s" \
        python render_slice.py --env "$ENV" --telemetry "$RUN/flight/telemetry.json" --start "$s" --end "$e" \
        --out "$RUN/slices/slice_$(printf %06d "$s").mp4" --mission "$MISSION" > "$L" 2>&1
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
  export X500_NO_BAKE=1  # Mountain: physics-only flight needs no baked ground textures (other envs ignore it)
  CMD=(python run_x500_mission.py --env "$ENV" --mission "$MISSION" --out "$RUN/flight" --vehicle-id "$4")
  STARTED="recording starts"
else
  CMD=(python render_slice.py --env "$ENV" --telemetry "$RUN/flight/telemetry.json" --start "$4" --end "$5"
       --out "$RUN/slices/slice_$(printf %06d "$4").mp4" --mission "$MISSION")
  STARTED="\[slice\] booted"
fi

# Isaac Sim on this cluster often dies at start-up with Vulkan ERROR_DEVICE_LOST / a segfault before
# any of our code runs (driver 580.105 nodes, 2026-10-09). Retry that once here; the manager
# resubmits to a fresh allocation if both attempts crash. A failure after start-up is not retried.
for attempt in 1 2; do
  sleep $((RANDOM % 45))
  ATT_LOG="$TMPD/attempt_${attempt}.log"
  # Log straight to a file (no tee): with a pipe, PX4 -- a child of the python process -- inherited
  # stdout and kept the pipe open after python exited, so a finished job looked hung for hours.
  "${CMD[@]}" > "$ATT_LOG" 2>&1
  rc=$?
  [ "$MODE" = flight ] && pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i $4 " 2>/dev/null  # stray PX4 of this flight
  if [ $rc -eq 0 ] || grep -q "$STARTED" "$ATT_LOG"; then exit $rc; fi
  echo "attempt $attempt: Isaac Sim start-up failure (rc=$rc)"
done
exit 75
