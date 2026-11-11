#!/bin/bash
# ONE GPU job for the collision-aware Forest benchmark (adapted from Benchmark_codebase/job.sbatch; same start-up guards):
# flies the benchmark episode and every collision test one after another (one Isaac Sim process each), then renders the
# episode frames (dataset render_slice.py, unchanged), the opt-in external debug video of the sideways test and one
# video per collision test.
#   sbatch job.sbatch <scene> <episode mission> <work root> "<test names>"
# Units already done (FLIGHT_DONE / slice mp4 / debug mp4 present) are skipped, so a resubmitted job continues.
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=09:00:00
set -uo pipefail
SCENE="$1"; EPISODE="$2"; ROOT="$3"; TESTS="$4"
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

# fly <mission> <dir> <vehicle id>: two attempts against Isaac Sim start-up failures
fly() {
  local m=$1 d=$2 vid=$3 att L rc
  [ -f "$d/flight/FLIGHT_DONE" ] && { echo "$m: already flown ($(cat "$d/flight/FLIGHT_DONE"))"; return 0; }
  mkdir -p "$d"
  for att in 1 2; do
    L="$d/flight_attempt_${SLURM_JOB_ID}_${att}.log"
    X500_NO_BAKE=1 python $HERE/run_flight.py --env Forest --scene "$SCENE" --mission "$m" --out "$d/flight" --vehicle-id "$vid" > "$L" 2>&1
    rc=$?
    pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i $vid " 2>/dev/null
    echo "$m attempt $att rc=$rc: $(grep -h -m1 '\[x500\] RESULT' "$L" | cut -c1-300)"
    if [ -f "$d/flight/FLIGHT_DONE" ] || grep -q "recording starts" "$L"; then return 0; fi
  done
}

fly "$EPISODE" "$ROOT/$EPISODE" 0
vid=1
for n in $TESTS; do fly "colltest_$n" "$ROOT/colltest_$n" $vid; vid=$(( vid % 9 + 1 )); done

# episode frames -> slices (dataset renderer, RGB, same scene)
EW="$ROOT/$EPISODE"
if [ -f "$EW/flight/FLIGHT_OK" ]; then
  read F0 F1 < <(python3 $HERE/window.py "$EW/flight")
  echo "episode window [$F0, $F1)"
  pids=()
  for ((s = F0; s < F1; s += 900)); do
    e=$((s + 900 < F1 ? s + 900 : F1)); out="$EW/slices/slice_$(printf %06d $s).mp4"
    [ -f "$out" ] && continue
    ( for att in 1 2; do
        L="$EW/slice_${s}_${SLURM_JOB_ID}_${att}.log"
        XDG_CACHE_HOME="$TMPD/cache_$s" OPTIX_CACHE_PATH="$TMPD/optix_$s" \
          python render_slice.py --env Forest --telemetry "$EW/flight/telemetry.json" --start "$s" --end "$e" --out "$out" \
          --mission "$SCENE" > "$L" 2>&1 && break
        grep -q "\[slice\] booted" "$L" && break
      done ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi

# opt-in debug video (sideways test)
DW="$ROOT/colltest_debug_sideways"
if [ -f "$DW/flight/FLIGHT_DONE" ] && [ ! -f "$DW/debug/debug_external_onboard.mp4" ]; then
  for att in 1 2; do
    L="$DW/debug_${SLURM_JOB_ID}_${att}.log"
    python $HERE/render_debug.py --env Forest --scene "$SCENE" --flight "$DW/flight" --out "$DW/debug" > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
fi

# one video per collision test: last 12 s of the test's episode up to the contact, external camera following from the side
# (re-created fixtures, contact marked) next to the onboard view; every 2nd frame (15 fps), last frame held 2 s
for n in $TESTS; do
  [ "$n" = debug_sideways ] && continue
  TW="$ROOT/colltest_$n"
  [ -f "$TW/flight/FLIGHT_DONE" ] && [ -f "$TW/flight/telemetry.json" ] || continue
  [ -f "$TW/video/collision_test.mp4" ] && continue
  for att in 1 2; do
    L="$TW/video_${SLURM_JOB_ID}_${att}.log"
    python $HERE/render_debug.py --env Forest --scene "$SCENE" --flight "$TW/flight" --out "$TW/video" --camera side \
      --max-seconds 12 --stride 2 --hold-s 2 --bank-samples 0 --name collision_test > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
  echo "colltest_$n video: $(ls "$TW/video/collision_test.mp4" 2>/dev/null || echo MISSING)"
done
echo "job done"
